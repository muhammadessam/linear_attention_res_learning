import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from factory.neurons_factory import create_neuron


###############################################
# Stage12 Feature Extractor (Stage-1/2)
###############################################


class PatchEmbedInit(nn.Module):
    def __init__(self, in_channels=1, embed_dims=192, neuron_type="LIF", surrogate_function="sigmoid", neuron_args=None):
        super().__init__()
        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 2)
        self.proj_lif = create_neuron(neuron_type=neuron_type, surrogate_type=surrogate_function, **(neuron_args or {}))

        self.proj1_conv = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj1_bn = nn.BatchNorm2d(embed_dims)
        self.proj1_lif = create_neuron(neuron_type=neuron_type, surrogate_type=surrogate_function, **(neuron_args or {}))
        self.proj1_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.proj2_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj2_bn = nn.BatchNorm2d(embed_dims)
        self.proj2_lif = create_neuron(neuron_type=neuron_type, surrogate_type=surrogate_function, **(neuron_args or {}))

        self.proj_res_conv = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=1, stride=2, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = create_neuron(neuron_type=neuron_type, surrogate_type=surrogate_function, **(neuron_args or {}))

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif(x).flatten(0, 1).contiguous()

        x_feat = x
        H1, W1 = H, W

        x = self.proj1_conv(x)
        x = self.proj1_bn(x).reshape(T, B, -1, H1, W1).contiguous()
        x = self.proj1_lif(x).flatten(0, 1).contiguous()
        x = self.proj1_maxpool(x)

        H2, W2 = x.shape[-2], x.shape[-1]

        x = self.proj2_conv(x)
        x = self.proj2_bn(x).reshape(T, B, -1, H2, W2).contiguous()
        x = self.proj2_lif(x)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat).reshape(T, B, -1, H2, W2).contiguous()
        x_feat = self.proj_res_lif(x_feat)

        x = x + x_feat
        return x


class TokenQKAttention(nn.Module):
    def __init__(self, dim, num_heads=8, neuron_type="LIF", surrogate_function="sigmoid", neuron_args=None, dropout=0.2):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = create_neuron(neuron_type=neuron_type, surrogate_type=surrogate_function, **(neuron_args or {}))

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = create_neuron(neuron_type=neuron_type, surrogate_type=surrogate_function, **(neuron_args or {}))

        self.attn_lif = create_neuron(neuron_type=neuron_type, surrogate_type=surrogate_function, **(neuron_args or {}))
        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = create_neuron(neuron_type=neuron_type, surrogate_type=surrogate_function, **(neuron_args or {}))

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x):
        T, B, C, H, W = x.shape
        dtype = x.dtype
        x = x.flatten(3)  # [T,B,C,N]
        T, B, C, N = x.shape
        xb = x.flatten(0, 1)

        q = self.q_lif(self.q_bn(self.q_conv(xb)).reshape(T, B, C, N))
        k = self.k_lif(self.k_bn(self.k_conv(xb)).reshape(T, B, C, N))

        q = q.reshape(T, B, self.num_heads, C // self.num_heads, N)
        k = k.reshape(T, B, self.num_heads, C // self.num_heads, N)

        attn = self.attn_lif(torch.sum(q, dim=3, keepdim=True))
        attn = self.attn_dropout(attn.to(dtype))
        x = torch.mul(attn, k).reshape(T, B, C, N)

        x = self.proj_bn(self.proj_conv(x.flatten(0, 1))).reshape(T, B, C, N)
        x = self.proj_lif(x)
        x = self.proj_dropout(x.to(dtype))

        x = x.reshape(T, B, C, H, W)
        return x


class ChannelQKAttention(nn.Module):
    def __init__(self, dim, num_heads=8, neuron_type="LIF", surrogate_function="sigmoid", neuron_args=None, dropout=0.2):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = create_neuron(neuron_type=neuron_type, surrogate_type=surrogate_function, **(neuron_args or {}))

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = create_neuron(neuron_type=neuron_type, surrogate_type=surrogate_function, **(neuron_args or {}))

        self.attn_lif = create_neuron(neuron_type=neuron_type, surrogate_type=surrogate_function, **(neuron_args or {}))
        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = create_neuron(neuron_type=neuron_type, surrogate_type=surrogate_function, **(neuron_args or {}))

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x):
        T, B, C, H, W = x.shape
        dtype = x.dtype
        x = x.flatten(3)
        T, B, C, N = x.shape
        xb = x.flatten(0, 1)

        q = self.q_lif(self.q_bn(self.q_conv(xb)).reshape(T, B, C, N))
        k = self.k_lif(self.k_bn(self.k_conv(xb)).reshape(T, B, C, N))

        q = q.reshape(T, B, self.num_heads, self.head_dim, N)
        channel_attn = self.attn_lif(torch.sum(q, dim=4, keepdim=True))
        channel_attn = self.attn_dropout(channel_attn.to(dtype))
        k = k.reshape(T, B, self.num_heads, self.head_dim, N)
        x = torch.mul(channel_attn, k).reshape(T, B, C, N)

        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x).reshape(T, B, C, H, W)
        x = self.proj_lif(x)
        x = self.proj_dropout(x.to(dtype))
        return x


class MLP(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, neuron_type="LIF", surrogate_function="sigmoid", neuron_args=None, dropout=0.2
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm2d(hidden_features)
        self.fc1_lif = create_neuron(neuron_type=neuron_type, surrogate_type=surrogate_function, **(neuron_args or {}))
        self.fc2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm2d(out_features)
        self.fc2_lif = create_neuron(neuron_type=neuron_type, surrogate_type=surrogate_function, **(neuron_args or {}))
        self.c_hidden = hidden_features
        self.c_output = out_features
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        T, B, C, H, W = x.shape
        dtype = x.dtype
        z = self.fc1_conv(x.flatten(0, 1))
        z = self.fc1_bn(z).reshape(T, B, self.c_hidden, H, W).contiguous()
        z = self.fc1_lif(z)
        z = self.fc2_conv(z.flatten(0, 1))
        z = self.fc2_bn(z).reshape(T, B, C, H, W).contiguous()
        z = self.fc2_lif(z)
        z = self.dropout(z.to(dtype))
        return z


class TokenSpikingTransformer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        neuron_type="LIF",
        surrogate_function="sigmoid",
        neuron_args=None,
        dropout=0.2,
        attention_type: str = "token",
    ):
        super().__init__()
        attn_type = (attention_type or "token").lower()
        if attn_type not in {"token", "channel"}:
            raise ValueError("attention_type must be 'token' or 'channel'")
        if attn_type == "token":
            self.tssa = TokenQKAttention(dim, num_heads, neuron_type, surrogate_function, neuron_args, dropout)
        else:
            self.tssa = ChannelQKAttention(dim, num_heads, neuron_type, surrogate_function, neuron_args, dropout)
        self.mlp = MLP(
            dim,
            int(dim * mlp_ratio),
            neuron_type=neuron_type,
            surrogate_function=surrogate_function,
            neuron_args=neuron_args,
            out_features=dim,
            dropout=dropout,
        )

    def forward(self, x):
        x = x + self.tssa(x)
        x = x + self.mlp(x)
        return x


class PatchEmbeddingStage(nn.Module):
    def __init__(self, in_channels=3, embed_dims=512, neuron_type="LIF", surrogate_function="sigmoid", neuron_args=None):
        super().__init__()
        self.proj3_conv = nn.Conv2d(in_channels, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj3_bn = nn.BatchNorm2d(embed_dims)
        self.proj3_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.proj3_lif = create_neuron(neuron_type=neuron_type, surrogate_type=surrogate_function, **(neuron_args or {}))
        self.proj4_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj4_bn = nn.BatchNorm2d(embed_dims)
        self.proj4_lif = create_neuron(neuron_type=neuron_type, surrogate_type=surrogate_function, **(neuron_args or {}))
        self.proj_res_conv = nn.Conv2d(in_channels, embed_dims, kernel_size=1, stride=2, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = create_neuron(neuron_type=neuron_type, surrogate_type=surrogate_function, **(neuron_args or {}))

    def forward(self, x):
        T, B, C, H, W = x.shape
        x_main = x.flatten(0, 1)
        x_res = x_main
        x_main = self.proj3_conv(x_main)
        x_main = self.proj3_bn(x_main).reshape(T, B, -1, H, W).contiguous()
        x_main = self.proj3_lif(x_main).flatten(0, 1).contiguous()
        x_main = self.proj3_maxpool(x_main)
        H1, W1 = x_main.shape[-2], x_main.shape[-1]
        x_main = self.proj4_conv(x_main)
        x_main = self.proj4_bn(x_main).reshape(T, B, -1, H1, W1).contiguous()
        x_main = self.proj4_lif(x_main)
        x_res = self.proj_res_conv(x_res)
        x_res = self.proj_res_bn(x_res).reshape(T, B, -1, H1, W1).contiguous()
        x_res = self.proj_res_lif(x_res)
        x = x_main + x_res
        return x


class Stage12FeatureExtractorStageSelect(nn.Module):
    def __init__(
        self,
        T: int = 4,
        in_channels: int = 1,
        num_classes: int = 10,
        embed_dims: Tuple[int, int] = (96, 192),
        num_heads=(8, 8),
        mlp_ratio: float = 4.0,
        depths=(1, 2),
        is_video: bool = True,
        neuron_type: str = "LIF",
        surrogate_function: str = "sigmoid",
        neuron_args: dict | None = None,
        dropout: float = 0.2,
        attention_type: str | None = None,
        apply_stages: int = 2,
    ) -> None:
        super().__init__()
        neuron_args = neuron_args or {}
        assert len(num_heads) == 2 and len(depths) == 2 and len(embed_dims) == 2
        if apply_stages not in (1, 2):
            raise ValueError("apply_stages must be 1 or 2")
        dim_s1, dim_s2 = int(embed_dims[0]), int(embed_dims[1])
        self.T = T
        self.is_video = is_video
        self.num_classes = num_classes
        self.apply_stages = apply_stages
        self.attention_type = ("token" if is_video else "channel") if attention_type is None else attention_type.lower()

        self.patch_embed1 = PatchEmbedInit(
            in_channels=in_channels, embed_dims=dim_s1, neuron_type=neuron_type, surrogate_function=surrogate_function, neuron_args=neuron_args
        )
        self.stage1 = nn.ModuleList(
            [
                TokenSpikingTransformer(
                    dim=dim_s1,
                    num_heads=num_heads[0],
                    mlp_ratio=mlp_ratio,
                    neuron_type=neuron_type,
                    surrogate_function=surrogate_function,
                    neuron_args=neuron_args,
                    dropout=dropout,
                    attention_type=self.attention_type,
                )
                for _ in range(depths[0])
            ]
        )

        self.patch_embed2 = PatchEmbeddingStage(
            in_channels=dim_s1, embed_dims=dim_s2, neuron_type=neuron_type, surrogate_function=surrogate_function, neuron_args=neuron_args
        )
        self.stage2 = nn.ModuleList(
            [
                TokenSpikingTransformer(
                    dim=dim_s2,
                    num_heads=num_heads[1],
                    mlp_ratio=mlp_ratio,
                    neuron_type=neuron_type,
                    surrogate_function=surrogate_function,
                    neuron_args=neuron_args,
                    dropout=dropout,
                    attention_type=self.attention_type,
                )
                for _ in range(depths[1])
            ]
        )

        self.head_s1 = nn.Linear(dim_s1, num_classes) if num_classes > 0 else nn.Identity()
        self.head_s2 = nn.Linear(dim_s2, num_classes) if num_classes > 0 else nn.Identity()

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_video:
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        x = self.patch_embed1(x)
        for blk in self.stage1:
            x = blk(x)
        if self.apply_stages == 1:
            return x
        x = self.patch_embed2(x)
        for blk in self.stage2:
            x = blk(x)
        return x

    def forward(self, x: torch.Tensor):
        if x.dim() == 4:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        features = self.extract_features(x)
        feats_tbC = features.flatten(3).mean(3)
        feats_bC = feats_tbC.mean(0)
        logits = self.head_s1(feats_bC) if self.apply_stages == 1 else self.head_s2(feats_bC)
        return logits, features


###############################################
# Align and Adapter
###############################################


class AlignGrid2d(nn.Module):
    def __init__(self, mode: str = "avg", target_hw: Optional[Tuple[int, int]] = None):
        super().__init__()
        assert mode in {"avg", "max", "interp"}
        self.mode = mode
        self.target_hw = target_hw

    @staticmethod
    def _gcd(a: int, b: int) -> int:
        return math.gcd(int(a), int(b))

    def forward(self, a: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert a.dim() == 5 and v.dim() == 5
        Ta, Ba, Ca, Ha, Wa = a.shape
        Tv, Bv, Cv, Hv, Wv = v.shape
        assert Ta == Tv and Ba == Bv and Ca == Cv
        if self.target_hw is None:
            Ht = self._gcd(Ha, Hv) or min(Ha, Hv)
            Wt = self._gcd(Wa, Wv) or min(Wa, Wv)
        else:
            Ht, Wt = self.target_hw
        if self.mode == "avg":
            a2 = F.adaptive_avg_pool2d(a.flatten(0, 1), (Ht, Wt)).reshape(Ta, Ba, Ca, Ht, Wt)
            v2 = F.adaptive_avg_pool2d(v.flatten(0, 1), (Ht, Wt)).reshape(Tv, Bv, Cv, Ht, Wt)
        elif self.mode == "max":
            a2 = F.adaptive_max_pool2d(a.flatten(0, 1), (Ht, Wt)).reshape(Ta, Ba, Ca, Ht, Wt)
            v2 = F.adaptive_max_pool2d(v.flatten(0, 1), (Ht, Wt)).reshape(Tv, Bv, Cv, Ht, Wt)
        else:
            a2 = F.interpolate(a.flatten(0, 1), size=(Ht, Wt), mode="bilinear", align_corners=False).reshape(Ta, Ba, Ca, Ht, Wt)
            v2 = F.interpolate(v.flatten(0, 1), size=(Ht, Wt), mode="bilinear", align_corners=False).reshape(Tv, Bv, Cv, Ht, Wt)
        return a2, v2


class QKFormerPatchAdapter(nn.Module):
    def __init__(
        self,
        in_channels=1,
        embed_dims: Tuple[int, int] = (96, 192),
        T=4,
        apply_stages=2,
        add_rpe=True,
        attention_type: str = "token",
        neuron_type="LIF",
        surrogate_function="sigmoid",
        neuron_args=None,
    ):
        super().__init__()
        self.T = T
        self.apply_stages = apply_stages
        self.backbone = Stage12FeatureExtractorStageSelect(
            T=T,
            in_channels=in_channels,
            num_classes=0,
            embed_dims=embed_dims,
            num_heads=(8, 8),
            depths=(1, 2),
            is_video=True,
            neuron_type=neuron_type,
            surrogate_function=surrogate_function,
            neuron_args=neuron_args or {},
            dropout=0.2,
            attention_type=attention_type,
            apply_stages=apply_stages,
        )
        self.target_c = int(embed_dims[1])
        in_c = int(embed_dims[0]) if apply_stages == 1 else self.target_c
        if in_c != self.target_c:
            self.unify_conv = nn.Conv2d(in_c, self.target_c, kernel_size=1, stride=1, bias=False)
            self.unify_bn = nn.BatchNorm2d(self.target_c)
            self.unify_lif = create_neuron(neuron_type=neuron_type, surrogate_type=surrogate_function, **(neuron_args or {}))
        else:
            self.unify_conv = None
        self.add_rpe = add_rpe
        if add_rpe:
            self.rpe_conv = nn.Conv2d(self.target_c, self.target_c, kernel_size=3, stride=1, padding=1, bias=False)
            self.rpe_bn = nn.BatchNorm2d(self.target_c)
            self.rpe_lif = create_neuron(neuron_type=neuron_type, surrogate_type=surrogate_function, **(neuron_args or {}))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        _, feats = self.backbone(x)
        if self.unify_conv is not None:
            T, B, C, H, W = feats.shape
            z = self.unify_conv(feats.flatten(0, 1))
            z = self.unify_bn(z).reshape(T, B, self.target_c, H, W)
            z = self.unify_lif(z)
            feats = z
        if self.add_rpe:
            T, B, C, H, W = feats.shape
            res = feats
            y = self.rpe_conv(feats.flatten(0, 1))
            y = self.rpe_bn(y).reshape(T, B, self.target_c, H, W)
            y = self.rpe_lif(y)
            feats = res + y
        return feats


###############################################
# CCSSA (Spatial-Temporal) and Fusion Block
###############################################


class SpatialAudioVisualSSA(nn.Module):
    def __init__(self, dim, step=10, num_heads=16, neuron_type="LIF", surrogate_function="sigmoid", neuron_args=None):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.25
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = create_neuron(neuron_type=neuron_type, surrogate_type=surrogate_function, **(neuron_args or {}))
        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = create_neuron(neuron_type=neuron_type, surrogate_type=surrogate_function, **(neuron_args or {}))
        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = create_neuron(neuron_type=neuron_type, surrogate_type=surrogate_function, **(neuron_args or {}))
        self.attn_lif = create_neuron(neuron_type=neuron_type, surrogate_type=surrogate_function, **(neuron_args or {}))
        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = create_neuron(neuron_type=neuron_type, surrogate_type=surrogate_function, **(neuron_args or {}))

    def forward(self, x, y):
        T, B, C, N = x.shape
        xq = x.flatten(0, 1)
        yk = y.flatten(0, 1)
        q = self.q_lif(self.q_bn(self.q_conv(xq))).reshape(T, B, C, N).transpose(-2, -1)
        k = self.k_lif(self.k_bn(self.k_conv(yk))).reshape(T, B, C, N).transpose(-2, -1)
        v = self.v_lif(self.v_bn(self.v_conv(yk))).reshape(T, B, C, N).transpose(-2, -1)
        q = q.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)
        k = k.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)
        v = v.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)
        attn = q @ k.transpose(-2, -1)
        x = (attn @ v) * self.scale
        x = x.transpose(3, 4).reshape(T, B, C, N)
        x = self.attn_lif(x.flatten(0, 1))
        x = self.proj_lif(self.proj_bn(self.proj_conv(x))).reshape(T, B, C, N)
        return x


class TemporalAudioVisualSSA(nn.Module):
    def __init__(self, dim, step=10, num_heads=16, neuron_type="LIF", surrogate_function="sigmoid", neuron_args=None):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.25
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = create_neuron(neuron_type=neuron_type, surrogate_type=surrogate_function, **(neuron_args or {}))
        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = create_neuron(neuron_type=neuron_type, surrogate_type=surrogate_function, **(neuron_args or {}))
        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = create_neuron(neuron_type=neuron_type, surrogate_type=surrogate_function, **(neuron_args or {}))
        self.attn_lif = create_neuron(neuron_type=neuron_type, surrogate_type=surrogate_function, **(neuron_args or {}))
        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = create_neuron(neuron_type=neuron_type, surrogate_type=surrogate_function, **(neuron_args or {}))

    def forward(self, x, y):
        T, B, C, N = x.shape
        xq = x.flatten(0, 1)
        yk = y.flatten(0, 1)
        q = self.q_lif(self.q_bn(self.q_conv(xq))).reshape(T, B, C, N).permute(3, 1, 0, 2)
        k = self.k_lif(self.k_bn(self.k_conv(yk))).reshape(T, B, C, N).permute(3, 1, 0, 2)
        v = self.v_lif(self.v_bn(self.v_conv(yk))).reshape(T, B, C, N).permute(3, 1, 0, 2)
        q = q.reshape(N, B, T, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)
        k = k.reshape(N, B, T, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)
        v = v.reshape(N, B, T, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)
        attn = q @ k.transpose(-2, -1)
        x = (attn @ v) * self.scale
        x = x.reshape(N, B, T, C).permute(2, 1, 3, 0)
        x = self.attn_lif(x.flatten(0, 1))
        x = self.proj_lif(self.proj_bn(self.proj_conv(x))).reshape(T, B, C, N)
        return x


class SpatialTemporalAudioVisualSSA(nn.Module):
    def __init__(self, dim, step=10, num_heads=16, neuron_type="LIF", surrogate_function="sigmoid", neuron_args=None):
        super().__init__()
        self.spatial_attn = SpatialAudioVisualSSA(
            dim, step=step, num_heads=num_heads, neuron_type=neuron_type, surrogate_function=surrogate_function, neuron_args=neuron_args
        )
        self.temporal_attn = TemporalAudioVisualSSA(
            dim, step=step, num_heads=num_heads, neuron_type=neuron_type, surrogate_function=surrogate_function, neuron_args=neuron_args
        )

    def forward(self, x, y):
        a = self.spatial_attn(x, y)
        b = self.temporal_attn(x, y)
        T, B, C, N = a.shape
        a_reduced = a.mean(dim=3)
        b_reduced = b.mean(dim=0)
        a_exp = a_reduced.unsqueeze(-1).expand(-1, -1, -1, N)
        b_exp = b_reduced.unsqueeze(0).expand(T, -1, -1, -1)
        return a_exp * b_exp


class TokenChannelLayerNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.ln = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 1, 3, 2)
        x = self.ln(x)
        x = x.permute(0, 1, 3, 2)
        return x


class TokenMLP1D(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, neuron_type="LIF", surrogate_function="sigmoid", neuron_args=None):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Conv1d(dim, hidden, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.lif1 = create_neuron(neuron_type=neuron_type, surrogate_type=surrogate_function, **(neuron_args or {}))
        self.fc2 = nn.Conv1d(hidden, dim, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm1d(dim)
        self.lif2 = create_neuron(neuron_type=neuron_type, surrogate_type=surrogate_function, **(neuron_args or {}))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, C, N = x.shape
        z = self.fc1(x.flatten(0, 1))
        z = self.bn1(z).reshape(T, B, -1, N)
        z = self.lif1(z)
        z = self.fc2(z.flatten(0, 1))
        z = self.bn2(z).reshape(T, B, C, N)
        z = self.lif2(z)
        return z


class AudioVisualBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        step: int = 10,
        mlp_ratio: float = 4.0,
        alpha: float = 0.5,
        neuron_type="LIF",
        surrogate_function="sigmoid",
        neuron_args=None,
    ):
        super().__init__()
        self.alpha = alpha
        self.norm_x = TokenChannelLayerNorm(dim)
        self.norm_y = TokenChannelLayerNorm(dim)
        self.attn = SpatialTemporalAudioVisualSSA(
            dim, step=step, num_heads=num_heads, neuron_type=neuron_type, surrogate_function=surrogate_function, neuron_args=neuron_args
        )
        self.mlp_norm = TokenChannelLayerNorm(dim)
        self.mlp = TokenMLP1D(dim, mlp_ratio=mlp_ratio, neuron_type=neuron_type, surrogate_function=surrogate_function, neuron_args=neuron_args)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        st = self.attn(self.norm_x(x), self.norm_y(y))
        z = x + self.alpha * st
        z = z + self.mlp(self.mlp_norm(z))
        return z, st


###############################################
# Main Model (Bi-directional)
###############################################


class SpikformerCCSSA_QKFusion_ST(nn.Module):
    def __init__(
        self,
        num_classes=10,
        step=4,
        in_channels_audio=1,
        in_channels_video=1,
        embed_dims=192,
        num_heads=8,
        mlp_ratio=4.0,
        align_mode="avg",
        target_hw=None,
        apply_stages=2,
        add_rpe=True,
        alpha=0.5,
        neuron_type="LIF",
        surrogate_function="sigmoid",
        neuron_args=None,
    ):
        super().__init__()
        self.T = step
        s1_dim = max(embed_dims // 2, 1)
        self.enc_a = QKFormerPatchAdapter(
            in_channels=in_channels_audio,
            embed_dims=(s1_dim, embed_dims),
            T=step,
            apply_stages=apply_stages,
            add_rpe=add_rpe,
            attention_type="channel",
            neuron_type=neuron_type,
            surrogate_function=surrogate_function,
            neuron_args=neuron_args or {},
        )
        self.enc_v = QKFormerPatchAdapter(
            in_channels=in_channels_video,
            embed_dims=(s1_dim, embed_dims),
            T=step,
            apply_stages=apply_stages,
            add_rpe=add_rpe,
            attention_type="token",
            neuron_type=neuron_type,
            surrogate_function=surrogate_function,
            neuron_args=neuron_args or {},
        )
        self.align = AlignGrid2d(mode=align_mode, target_hw=target_hw)

        # Two directional blocks: audio<-video and video<-audio
        self.block_av = AudioVisualBlock(
            dim=embed_dims,
            num_heads=num_heads,
            step=step,
            mlp_ratio=mlp_ratio,
            alpha=alpha,
            neuron_type=neuron_type,
            surrogate_function=surrogate_function,
            neuron_args=neuron_args or {},
        )
        self.block_va = AudioVisualBlock(
            dim=embed_dims,
            num_heads=num_heads,
            step=step,
            mlp_ratio=mlp_ratio,
            alpha=alpha,
            neuron_type=neuron_type,
            surrogate_function=surrogate_function,
            neuron_args=neuron_args or {},
        )

        self.head = nn.Linear(embed_dims * 2, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, xa: torch.Tensor, xv: torch.Tensor):
        if xa.dim() == 4:
            xa = xa.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        if xv.dim() == 4:
            xv = xv.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        fa = self.enc_a(xa)
        fv = self.enc_v(xv)
        fa, fv = self.align(fa, fv)
        x = fa.flatten(3)
        y = fv.flatten(3)

        # Bi-directional fusion
        tokens_a, _ = self.block_av(x, y)  # audio enhanced by visual
        tokens_v, _ = self.block_va(y, x)  # visual enhanced by audio

        # Simple symmetric fusion (average); alternative: concat/add then project
        fused_tokens = torch.cat((tokens_v, tokens_a), dim=2)

        # if used addition donot forget to change the head input dim to embed_dims
        # fused_tokens = tokens_a + tokens_v

        x_cls = fused_tokens.mean(dim=3).mean(dim=0)
        logits = self.head(x_cls)
        return logits, fused_tokens
