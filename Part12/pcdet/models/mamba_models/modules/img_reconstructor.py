from pcdet.models.mamba_models.modules.vit.vqvit import VQViT
import torch
from torch import nn


def _sample_logistic(shape, out=None):
    u = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)

    return torch.log(u) - torch.log(1 - u)


def _sigmoid_sample(logits, tau=1):
    """Implementation of Bernouilli reparametrization based on Maddison et al. 2017"""
    logistic_noise = _sample_logistic(logits.size(), out=logits.data.new())
    y = logits + logistic_noise
    return torch.sigmoid(y / tau)


def gumbel_sigmoid(logits, tau=1, hard=False):
    y_soft = _sigmoid_sample(logits, tau=tau)
    if hard:
        y_hard = torch.where(
            y_soft > 0.5, torch.ones_like(y_soft), torch.zeros_like(y_soft)
        )
        y = y_hard.data - y_soft.data + y_soft
    else:
        y = y_soft
    return y


class Reconstructor(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        in_channel: int,
        patch_size: int,
        depth: int,
        embed_dim: int,
        fsq_levels: list[int],
    ):

        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_channel

        self.vqvit = VQViT(
            img_size=img_size,
            in_chans=self.in_chans,
            patch_size=self.patch_size,
            depth=depth,
            embed_dim=embed_dim,
            fsq_levels=fsq_levels,
        )

    @torch.jit.unused
    def train_iter(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        self.train()

        loss_total, metas, info = self.vqvit(x)

        return loss_total, metas

    def get_codebook_entry(self, indices, shape=None):
        out = self.vqvit.quantizer.indices_to_codes(indices)
        if shape is not None:
            out = out.reshape(shape + (-1,))
        return out

    def quantize(self, x):
        return self.vqvit.quantizer(self.vqvit.encoder(x))

    def decode(self, x, return_logits=True):
        x_rec_logits = self.vqvit.decoder(x)

        if return_logits:
            return x_rec_logits
        else:
            return gumbel_sigmoid(x_rec_logits, hard=True)
