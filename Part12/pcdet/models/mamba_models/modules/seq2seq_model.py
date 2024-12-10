import math
import torch
from torch import nn

from dataclasses import dataclass

from pcdet.utils.commu_utils import get_rank

from .vit.vqvit import get_2d_sincos_pos_embed
from .conv_gru.conv_gru import ConvGRUCell2D
from mamba_ssm import Mamba


def cosine_decay(current_step, total_steps, initial_value, final_value):
    """
    Calculates the decayed value using a cosine decay schedule.

    Args:
        current_step (int): The current training step.
        total_steps (int): The total number of training steps.
        initial_value (float): The initial value at step 0.
        final_value (float): The final value at the last step.

    Returns:
        float: The decayed value.
    """
    if current_step >= total_steps:
        return final_value
    cosine_decay = 0.5 * (1 + math.cos(math.pi * current_step / total_steps))
    decayed = final_value + (initial_value - final_value) * cosine_decay
    return decayed


def gumbel_softmax_sample(logits, codebook, temperature, eps=1e-10):
    """
    Differentiably samples a single codebook entry per sample using the
    Straight-Through Gumbel-Softmax estimator, and returns the sampled indices.

    Args:
        logits (torch.Tensor): Logits of shape (N, C), where C is the number of classes.
        codebook (torch.Tensor): Embeddings of shape (C, D).
        temperature (float): The temperature parameter for the Gumbel-Softmax distribution.
        eps (float): Small constant for numerical stability (default: 1e-10).

    Returns:
        output (torch.Tensor): Samples from the codebook of shape (N, D) with gradients.
        indices (torch.Tensor): Indices of the sampled codebook entries, shape (N,)
    """
    # Sample Gumbel noise
    U = torch.rand_like(logits)
    U = U.clamp(min=eps, max=1 - eps)
    G = -torch.log(-torch.log(U))

    # Compute the logits with Gumbel noise
    gumbel_logits = (logits + G) / temperature  # Shape: (N, C)

    # Softmax probabilities (for gradient computation)
    y_soft = torch.softmax(gumbel_logits, dim=1)  # Shape: (N, C)

    # Hard one-hot encoded vectors (for forward pass)
    _, indices = torch.max(gumbel_logits, dim=1)  # Shape: (N,)
    y_hard = torch.zeros_like(logits).scatter_(
        1, indices.unsqueeze(1), 1.0
    )  # Shape: (N, C)

    # Straight-Through Estimator
    y = y_hard - y_soft.detach() + y_soft  # Shape: (N, C)

    # Compute the output by selecting codebook entries
    output = torch.matmul(y, codebook)  # Shape: (N, D)

    return output, indices


class Seq2SeqModel(nn.Module):
    def __init__(self, cfg):
        """Initialize the ConvLSTM cell"""
        super().__init__()
        self.cfg = cfg

        self.memory_shape = tuple(
            [s // self.cfg.patch_size for s in self.img_size][::-1]
        )

        if getattr(cfg, "fsq_levels", None) is not None:
            self.out_channels = 1
            for l in cfg.fsq_levels:
                self.out_channels *= l
            self.in_channels = len(cfg.fsq_levels)
        else:
            self.out_channels = cfg.seq2seq_out_channels
            self.in_channels = cfg.seq2seq_in_channels

        self.decoder_embed = nn.Sequential(
            nn.Linear(self.in_channels, self.cfg.seq2seq_embed_dim, bias=True),
            nn.LayerNorm(self.cfg.seq2seq_embed_dim),
        )
        self.pos_embed = nn.Parameter(
            torch.zeros((1,) + self.memory_shape + (self.cfg.seq2seq_embed_dim,)),
            requires_grad=False,
        )  # fixed sin-cos embedding

        # self.seq2seq = ConvGRUCell2D(in_channels=cfg.seq2seq_embed_dim)
        self.seq2seq = Mamba(
                d_model=cfg.seq2seq_embed_dim, # Model dimension d_model
                d_state=16,  # SSM state expansion factor
                d_conv=4,    # Local convolution width
                expand=2,    # Block expansion factor
            )
        self.pred = nn.Conv2d(
            self.cfg.seq2seq_embed_dim,
            self.out_channels,
            kernel_size=1,
            stride=1,
            bias=True,
        )  # decoder to patch
        self.initialize_weights()

    @property
    def img_size(self):
        x_size = int(
            round(
                (self.cfg.voxel_cfg.x_max - self.cfg.voxel_cfg.x_min)
                / self.cfg.voxel_cfg.step
            )
        )
        y_size = int(
            round(
                (self.cfg.voxel_cfg.y_max - self.cfg.voxel_cfg.y_min)
                / self.cfg.voxel_cfg.step
            )
        )
        return x_size, y_size

    def forward(self, code: torch.Tensor, memory: torch.Tensor):

        batch_size = code.shape[0]
        x = self.decoder_embed(code.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        pos_emb = self.pos_embed.expand(batch_size, -1, -1, -1).permute(0, 3, 1, 2)
        x = x + pos_emb

        memory = self.seq2seq(x, memory)

        logits = self.pred(memory)

        return logits, memory

    def train_iter(
        self, input_code: torch.Tensor, memory: torch.Tensor, gt_code_idx: torch.Tensor
    ):

        batch_size, _, height, width = input_code.shape
        pred, memory = self.forward(input_code, memory) 
        pred = pred.permute(0, 2, 3, 1).reshape(batch_size, height * width, -1)

        pred = pred.flatten(0, 1) 
        gt_code_idx = gt_code_idx.flatten(0, 2)

        mask = torch.ones_like(gt_code_idx).float()

        loss = self.loss(pred, mask, gt_code_idx.long())
        accuracy = (pred.max(dim=-1)[1] == gt_code_idx)[mask > 0].float().mean()

        metas = {
            "seq2seq/loss_mask_prediction": loss.item(),
            "seq2seq/accuracy_mask_prediction": accuracy.item(),
            "memory": memory,
        }

        pred_code_idcs = torch.argmax(pred, dim=-1).detach()
        pred_code_idcs = pred_code_idcs.reshape(batch_size, height, width)
        metas["pred_code_idx"] = pred_code_idcs
        metas["logits"] = pred

        return loss, metas

    def predict(
        self,
        input_code: torch.Tensor,
        memory: torch.Tensor,
        codebook: torch.nn.Embedding,
        temperature: float = 0.0,
    ):
        batch_size, _, height, width = input_code.shape
        pred, memory = self.forward(input_code, memory)  # (B, C, H, W)
        pred = pred.permute(0, 2, 3, 1).reshape(batch_size, height * width, -1)

        if temperature == 0:
            # Argmax behavior
            sample_ids = torch.argmax(pred, dim=-1)
        else:
            # Adjust logits with temperature
            logits = pred / temperature
            sample_ids = torch.distributions.Categorical(logits=logits).sample()

        x = codebook(sample_ids)
        code_idx = sample_ids.clone()

        return x, code_idx, memory

    def loss(self, x, mask, target):
        return (
            torch.nn.functional.cross_entropy(
                x, target, reduction="none", label_smoothing=0.1
            )
            * mask
        ).sum() / mask.sum()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.memory_shape, cls_token=False
        )
        pos_embed = pos_embed.reshape(
            (1,) + self.memory_shape + (self.cfg.seq2seq_embed_dim,)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


@dataclass
class SceneMemory:

    features: torch.Tensor | None = None  
    chunk_id: list[tuple] | None = None 
    t_world_vehicle: torch.Tensor | None = None

    def empty(self) -> "SceneMemory":
        print(f"[rank={get_rank()}] Emptying cache!")
        return SceneMemory()

    def is_empty(self) -> bool:
        return self.features is None

    def is_new_chunk(self, chunk_id: list[tuple]) -> bool:
        assert len(chunk_id) == len(self.chunk_id)
        for i, c in enumerate(chunk_id):
            if (c[:2] != self.chunk_id[i][:2]):
                return True
            if c[-1] != self.chunk_id[i][-1] + 1:
                return True
        return False

    def transform_features(
        self,
        new_t_world_vehicle: torch.Tensor,
        voxel_cfg: dict,
        padding: torch.Tensor = None,
        delta_theta_emb_fn: nn.Module | None = None,
    ) -> torch.Tensor:
        assert not self.is_empty()
        assert new_t_world_vehicle.shape == self.t_world_vehicle.shape
        if padding is not None:
            assert padding.shape[0] == self.features.shape[1]

        t_old_new = (
            torch.linalg.inv(self.t_world_vehicle) @ new_t_world_vehicle
        ) 

        delta_theta_emb = None
        if delta_theta_emb_fn is not None:
            delta_theta = torch.atan2(t_old_new[:, 1, 0], t_old_new[:, 0, 0])[
                :, None
            ] 
            delta_theta_emb = delta_theta_emb_fn(delta_theta.float())

        B, C, H, W = self.features.shape

        x_min, x_max = voxel_cfg.x_min, voxel_cfg.x_max
        y_min, y_max = voxel_cfg.y_min, voxel_cfg.y_max

        # Create grid of coordinates in world coordinates
        device = self.features.device
        ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
        ys = ys.view(1, H, W).repeat(B, 1, 1).to(device) + 0.5
        xs = xs.view(1, H, W).repeat(B, 1, 1).to(device) + 0.5
        grid_x = xs * ((voxel_cfg.x_max - voxel_cfg.x_min) / W) + voxel_cfg.x_min
        grid_y = voxel_cfg.y_max - ys * ((voxel_cfg.y_max - voxel_cfg.y_min) / H)


        ones = torch.ones_like(grid_x)
        zeros = torch.zeros_like(grid_x)
        grid_coords = torch.stack([grid_x, grid_y, zeros, ones], dim=-1)


        p_sample_new = grid_coords.view(B, -1, 4, 1)


        p_sample_old = t_old_new[:, None].to(p_sample_new.dtype) @ p_sample_new 
        p_sample_old = p_sample_old.squeeze(-1)


        x_trans = p_sample_old[:, :, 0]
        y_trans = p_sample_old[:, :, 1]


        x_trans_norm = 2 * ((x_trans - x_min) / (x_max - x_min)) - 1
        y_trans_norm = 2 * ((y_max - y_trans) / (y_max - y_min)) - 1


        grid_sample_coords = torch.stack((x_trans_norm, y_trans_norm), dim=-1) 
        grid_sample_coords = grid_sample_coords.view(B, H, W, 2) 

        mask = (grid_sample_coords < -1) | (grid_sample_coords > 1)
        mask = mask.any(dim=-1)
        mask = mask.unsqueeze(1).expand(-1, C, -1, -1)

        transformed_features = torch.nn.functional.grid_sample(
            self.features, grid_sample_coords, align_corners=False, padding_mode="zeros"
        )

        if padding is not None:
            padding = padding.to(transformed_features.device).view(1, C, 1, 1)
            transformed_features = torch.where(mask, padding, transformed_features)

        if delta_theta_emb is not None:
            transformed_features = (transformed_features + delta_theta_emb[:, :, None, None])

        return transformed_features