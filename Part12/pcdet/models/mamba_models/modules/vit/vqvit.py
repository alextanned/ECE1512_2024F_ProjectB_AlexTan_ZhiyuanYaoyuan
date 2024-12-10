import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from timm.models.swin_transformer import BasicLayer
from torch import nn

from .fsq.fsq import FSQ


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega 

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0]) 
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])

    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """Args:
    grid_size: int of the grid height and width
    Returns:
    pos_embed: [grid_size*grid_size, embed_dim] or
    [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: Tuple[int, int],
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[nn.Module] = None,
        flatten: bool = True,
        bias: bool = True,
    ):
        """
        Args:
            img_size: input image size
            patch_size: patch size for swin transformer
            in_chans: number of input channels
            embed_dim: embedding dimension for swin transformer
            norm_layer: normalization layer
            flatten: whether to flatten the output
            bias: whether to use bias for the first conv layer
        """
        super().__init__()
        self.img_size: Tuple[int, int] = img_size
        self.patch_size: int = patch_size
        self.grid_size: Tuple[int, int] = (
            img_size[0] // patch_size,
            img_size[1] // patch_size,
        )
        self.num_patches: int = self.grid_size[0] * self.grid_size[1]
        self.flatten: bool = flatten

        self.first_conv = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor):
        """x: (B, C, H, W)"""
        _, _, H, W = x.shape
        assert (
            H == self.img_size[0]
        ), f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        assert (
            W == self.img_size[1]
        ), f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        x = self.first_conv(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class VQEncoder(nn.Module):
    def __init__(
        self,
        img_size: Tuple[int, int],
        patch_size: int = 8,
        in_chans: int = 40,
        embed_dim: int = 256,
        num_heads: int = 4,
        depth: int = 6,
        codebook_dim: int = 1024,
        init_method: str = "xavier_uniform",
        bias: bool = True,
    ):
        """
        Args:
            img_size: input image size
            patch_size: patch size for swin transformer
            in_chans: number of input channels
            embed_dim: embedding dimension for swin transformer
            num_heads: number of heads for swin transformer
            depth: number of layers for swin transformer
            codebook_dim: embedding dimension for codebook
            init_method: initialization method for weights
            bias: whether to use bias for the pre-quantization linear layer
        """
        super().__init__()

        norm_layer = nn.LayerNorm
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans, embed_dim, norm_layer=norm_layer
        )
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.h = img_size[0] // patch_size
        self.w = img_size[1] // patch_size

        self.blocks = BasicLayer(
            embed_dim,
            embed_dim,
            input_resolution=(img_size[0] // patch_size, img_size[1] // patch_size),
            depth=depth,
            num_heads=num_heads,
            window_size=8,
            drop=0.0,
            attn_drop=0.0,
            downsample=None,
        )

        self.norm = nn.Sequential(norm_layer(embed_dim), nn.GELU())
        self.pre_quant = nn.Linear(embed_dim, codebook_dim, bias=bias)

        self.init_method: str = init_method
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], (self.h, self.w), cls_token=False
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.patch_embed.first_conv.weight.data
        fan_in = w.size(0) * w.size(-1) * w.size(-2)
        fan_out = w.size(1)
        if self.init_method == "xavier_uniform":
            nn.init.xavier_uniform_(w.view([fan_in, fan_out]))
        elif self.init_method == "palm":
            nn.init.normal_(w, mean=0.0, std=math.sqrt(1 / fan_in))
        elif self.init_method == "gpt":
            nn.init.normal_(w, mean=0.0, std=math.sqrt(1 / (3 * fan_in)))
        else:
            raise NotImplementedError
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            fan_in = m.weight.size(0)
            if self.init_method == "xavier_uniform":
                nn.init.xavier_uniform_(
                    m.weight
                )
            elif self.init_method == "palm":
                nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(1 / fan_in))
            elif self.init_method == "gpt":
                nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(1 / (3 * fan_in)))
            else:
                raise NotImplementedError
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)
        x = x + self.pos_embed

        x = self.blocks(x)
        x = self.norm(x)
        x = self.pre_quant(x)

        return x


class VQDecoder(nn.Module):
    def __init__(
        self,
        img_size: Tuple[int, int],
        patch_size: int = 8,
        in_chans: int = 40,
        embed_dim: int = 256,
        num_heads: int = 4,
        depth: int = 6,
        codebook_dim: int = 1024,
        bias_init: float | None = -5.0,
        init_method: str = "xavier_uniform",
    ):
        """
        Args:
            img_size: input image size
            patch_size: patch size for swin transformer
            in_chans: number of input channels
            embed_dim: embedding dimension for swin transformer
            num_heads: number of heads for swin transformer
            depth: number of layers for swin transformer
            codebook_dim: embedding dimension for codebook
            bias_init: bias initialization for the final linear layer
        """
        super().__init__()

        norm_layer = nn.LayerNorm
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.h = img_size[0] // patch_size
        self.w = img_size[1] // patch_size

        self.decoder_embed = nn.Linear(codebook_dim, embed_dim, bias=True)

        num_patches = self.h * self.w
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim), requires_grad=False
        ) 
        self.blocks = BasicLayer(
            embed_dim,
            embed_dim,
            input_resolution=(self.h, self.w),
            depth=depth,
            num_heads=num_heads,
            window_size=8,
            drop=0.0,
            attn_drop=0.0,
            downsample=None,
        )

        self.norm = nn.Sequential(norm_layer(embed_dim), nn.GELU())
        self.pred = nn.Linear(embed_dim, patch_size**2 * in_chans, bias=True)
        self.init_method: str = init_method
        self.initialize_weights()
        if bias_init is not None:
            nn.init.constant_(self.pred.bias, bias_init)

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], (self.h, self.w), cls_token=False
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            fan_in = m.weight.size(0)
            if self.init_method == "xavier_uniform":
                nn.init.xavier_uniform_(
                    m.weight
                )
            elif self.init_method == "palm":
                nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(1 / fan_in))
            elif self.init_method == "gpt":
                nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(1 / (3 * fan_in)))
            else:
                raise NotImplementedError
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def unpatchify(self, x):
        p = self.patch_size
        h, w = self.h, self.w
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, w * p))

        return imgs

    def forward(self, x):
        x = self.decoder_embed(x)

        x = x + self.pos_embed

        x = self.blocks(x)
        x = self.norm(x)

        x = self.pred(x)
        x = self.unpatchify(x)

        return x


class VQViT(nn.Module):
    def __init__(
        self,
        img_size: Tuple[int, int] = (512, 512),
        in_chans: int = 50,
        patch_size: int = 8,
        num_heads: int = 16,
        depth: int = 6,
        embed_dim: int = 256,
        init_method: str = "gpt",
        bias_init: float | None = -5.0,
        fsq_levels: list[int] = [8, 5, 5, 5],
    ):
        """
        Args:
            img_size: input image size
            in_chans: number of input channels
            patch_size: patch size for swin transformer
            num_heads: number of heads for swin transformer
            depth: number of layers for swin transformer
            embed_dim: embedding dimension for swin transformer
            codebook_dim: embedding dimension for codebook
            codebook_size: number of embeddings in codebook
            cosine_similarity: whether to use cosine similarity for vector quantization
            dead_limit: number of iterations to wait before declaring a code as dead
            dead_percent_limit: percentage of codebook to be dead before reinitializing
            min_n_iter_per_codebook: the minimum number of iterations we run optimization for every codebook
                regardless of whether the codes are considered dead or not.
            reservoir_size_multiplier: the size of the reservoir is the number of codes in the codebook times this value
            reservoir_frac_update_per_iter: the fraction of reservior that gets updated per iteration.
                The actual number of codes being updated in reservior will be
                int(n_e * reservoir_size_multiplier // reservoir_frac_update_per_iter). Defaults to 100.
            use_faiss: whether to use faiss for vector quantization
            init_method: initialization method for weights
            vq_rec: weight for reconstruction loss
            vq_commit: weight for commitment loss
            vq_codebook: weight for codebook loss
        """
        super().__init__()

        assert fsq_levels is not None
        self.quantizer = FSQ(fsq_levels)
        self.codebook_dim = len(fsq_levels)
        self.codebook_size = 1
        for l in fsq_levels:
            self.codebook_size *= l

        self.encoder = VQEncoder(
            img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            codebook_dim=self.codebook_dim,
            init_method=init_method,
            bias=False,
        )
        self.decoder = VQDecoder(
            img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            codebook_dim=self.codebook_dim,
            bias_init=bias_init,
            init_method=init_method,
        )

    def forward_encoder(self, x: torch.Tensor):
        z_enc = self.encoder(x)
        z_q, codes_idx = self.quantizer(z_enc)
        return z_q, codes_idx

    def forward_logits(self, x: torch.Tensor):
        z_enc = self.encoder(x)
        z_q, codes_idx = self.quantizer(z_enc)
        x_rec_logits = self.decoder(z_q)
        return x_rec_logits, codes_idx

    def forward(self, x: torch.Tensor):

        x_rec_logits, codes_idx = self.forward_logits(x)
        x_rec_prob = torch.sigmoid(x_rec_logits)

        loss_rec = F.binary_cross_entropy_with_logits(x_rec_logits, x, reduction="none")
        loss_rec = loss_rec.mean()

        loss_total = loss_rec

        metas = {
            "det/0/loss_total": loss_total.item(),
            "det/0/loss_rec": loss_rec.item(),
            "det/0/rec_iou": ((x_rec_prob >= 0.5) & (x >= 0.5)).sum().item()
            / max(((x_rec_prob >= 0.5) | (x >= 0.5)).sum().item(), 1),
        }
        for c_idx in range(x_rec_prob.shape[1]):
            rec_iou_c = (
                (x_rec_prob[:, c_idx] >= 0.5) & (x[:, c_idx] >= 0.5)
            ).sum().item() / (
                max(
                    ((x_rec_prob[:, c_idx] >= 0.5) | (x[:, c_idx] >= 0.5)).sum().item(),
                    1,
                )
            )
            metas[f"det/0/rec_iou_{c_idx}"] = rec_iou_c
            metas[f"det/0/rec_recall_{c_idx}"] = (
                (x_rec_prob[:, c_idx] >= 0.5) & (x[:, c_idx] >= 0.5)
            ).sum().item() / (x.shape[-1] * x.shape[-2])
            metas[f"det/0/rec_pos_frac_{c_idx}"] = (x[:, c_idx] >= 0.5).sum().item() / (
                x.shape[-1] * x.shape[-2]
            )
        info = {
            "rec_prob": x_rec_prob,
            "rec_logits": x_rec_logits,
            "codes_idx": codes_idx,
        }
        return loss_total, metas, info
