import imageio
import torch
from torch import nn
import numpy as np
import kornia
import torchvision.transforms.functional as TF
from PIL import ImageDraw, ImageFont

from pcdet.models.mamba_models.modules.img_reconstructor import Reconstructor
from pcdet.models.mamba_models.utils.label_raster import create_bev_image
from tools.viz_utils import plot_output

from .modules.seq2seq_model import (
    SceneMemory,
    Seq2SeqModel,
)


def add_text(img, text):
    """Helper function to add larger text in the top-left corner of a PIL image."""
    img_pil = TF.to_pil_image(img)
    draw = ImageDraw.Draw(img_pil)

    font_size = 20
    try:
        font = ImageFont.truetype("arial.ttf", font_size * 2)
    except IOError:
        font = ImageFont.load_default()

    draw.text((10, 10), text, fill=(255, 255, 255), font=font)
    return TF.to_tensor(img_pil)


def soft_iou_torch(y_pred: torch.Tensor, y_true: torch.Tensor, sigmoid=True):
    assert torch.all(torch.logical_or(y_true == 0.0, y_true == 1.0))
    assert not torch.any(torch.isnan(y_pred))
    assert not torch.any(torch.isnan(y_true))
    if sigmoid:
        y_pred = 1.0 / (1.0 + torch.exp(-y_pred))
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum() - intersection
    return intersection / union


class SceneMemoryModel(nn.Module):
    def __init__(self, cfg: dict, num_class: int, dataset_cfg: dict):
        super().__init__()

        self.register_buffer("global_step", torch.LongTensor(1).zero_())
        self.total_iters = 1
        self.cfg = cfg
        self.stage = self.cfg["stage"]

        self.img_size = (
            int(round((self.cfg.voxel_cfg.x_max - self.cfg.voxel_cfg.x_min)
                    / self.cfg.voxel_cfg.step
                )
            ),
            int(
                round(
                    (self.cfg.voxel_cfg.y_max - self.cfg.voxel_cfg.y_min)
                    / self.cfg.voxel_cfg.step
                )
            ),
        )

        self.seq2seq = None
        self.init_memory = None
        self.train_scene_memory = SceneMemory()
        self.eval_scene_memory = SceneMemory()
        self.viz_scene_memory = SceneMemory()
        self.viz_outputs = []
        self.delta_theta_embed = None

        self.eval_out = {}


        if self.stage in ["FSQ", "SEQ2SEQ"]:
            self.reconstructor = Reconstructor(
                img_size=self.img_size[::-1],
                in_channel=len(self.cfg.class_cfg.active_classes),
                patch_size=cfg.patch_size,
                depth=cfg.depth,
                embed_dim=cfg.embed_dim,
                fsq_levels=cfg.fsq_levels,
            )
            aug_list = []
            aug_list.extend(
                [
                    kornia.augmentation.RandomVerticalFlip(),
                    kornia.augmentation.RandomHorizontalFlip(),
                ]
            )
            self.aug = nn.Sequential(*aug_list)

        if self.stage == "SEQ2SEQ":

            for p in self.reconstructor.parameters():
                p.requires_grad = False
            self.reconstructor.eval()

            self.memory_shape = tuple(
                [s // self.cfg.patch_size for s in self.img_size][::-1]
            )
            self.init_memory = nn.Parameter(
                torch.zeros(
                    (1, cfg.seq2seq_embed_dim, *self.memory_shape),
                    dtype=torch.float32,
                ),
                requires_grad=True,
            )
            self.padding_emb = nn.Parameter(
                torch.zeros([cfg.seq2seq_embed_dim], dtype=torch.float32),
                requires_grad=True,
            )

            for p in self.reconstructor.parameters():
                p.requires_grad = False
            self.reconstructor.eval()

            self.seq2seq_model = Seq2SeqModel(cfg)

            if self.cfg.transform_features:
                self.delta_theta_embed = nn.Sequential(
                    nn.Linear(1, cfg.seq2seq_embed_dim, bias=False),
                    nn.LayerNorm(cfg.seq2seq_embed_dim),
                    nn.ReLU(),
                )

            if getattr(self.cfg, "reconstructor_weights_path", None) is not None:
                state_dict = torch.load(
                    self.cfg.reconstructor_weights_path, map_location="cpu"
                )
                self.load_state_dict(state_dict["model_state"], strict=False)
            else:
                print("WARNING: No reconstructor weights loaded, this won't work well!")

    def update_global_step(self):
        self.global_step += 1

    def make_label_image(self, gt_boxes):
        gt_boxes = gt_boxes.clone()
        gt_boxes[..., -1] = gt_boxes[..., -1] - 1
        raster_image = create_bev_image(gt_boxes, self.cfg.voxel_cfg)
        return raster_image

    def aug_labels(self, gt_boxes: torch.Tensor):
        traj_noise_std = torch.tensor(self.cfg.aug_box_std[:3], dtype=torch.float32)[None, None]
        box_noise_std = torch.tensor(self.cfg.aug_box_std[3:], dtype=torch.float32)[None]

        p_drop_box = torch.rand(1).item() * self.cfg.max_p_drop_box
        p_aug_box = torch.rand(1).item() * self.cfg.max_p_noise_box

        keep_mask = (torch.rand(gt_boxes.shape[1])[None].expand(gt_boxes.shape[0], -1) > p_drop_box)
        if not keep_mask.any():
            keep_mask[:, 0] = True
        gt_boxes_keep = gt_boxes[keep_mask].reshape(gt_boxes.shape[0], -1, gt_boxes.shape[2])
        noise_mask = (torch.rand(gt_boxes_keep.shape[1])[None].expand(gt_boxes.shape[0], -1) < p_aug_box)
        trajectory_noise = torch.randn_like(gt_boxes_keep[:, :, :3]) * traj_noise_std.to(gt_boxes.device)
        trajectory_noise[~noise_mask] = 0.0
        gt_boxes_keep[:, :, :3] = gt_boxes_keep[:, :, :3] + trajectory_noise
        box_noise = torch.randn_like(gt_boxes_keep[:, :, 3:6]) * box_noise_std.to(gt_boxes.device)
        box_noise[~noise_mask] = 0.0
        gt_boxes_keep[:, :, 3:6] = gt_boxes_keep[:, :, 3:6] + box_noise

        return gt_boxes_keep

    @torch.jit.unused
    def forward(
        self,
        batch_dict,
        scene_memory=None,
    ):
        if scene_memory is None:
            scene_memory = self.train_scene_memory
        if self.stage == "FSQ":
            x = self.make_label_image(batch_dict["gt_boxes"])
            if self.training:
                self.reconstructor.train()
                x = self.aug(x)
                total_loss, metas = self.reconstructor.train_iter(x)
            else:
                x_hat, _ = self.reconstructor.vqvit.forward_logits(x)
                return x, x_hat
        elif self.stage == "DET_FSQ":
            if self.training:
                self.reconstructor.train()
                self.perception_model.eval()
                total_loss, metas = self.reconstructor.train_iter(batch_dict)
            else:
                x_hat, info = self.reconstructor.forward_det(batch_dict)
                pred_dicts, recall_dicts = self.perception_model.post_processing(x_hat)
                return pred_dicts, recall_dicts
        elif self.stage == "SEQ2SEQ":
            keys = [k for k in batch_dict.keys() if k.startswith("item")]
            keys = sorted(keys, key=lambda x: int(x.split("_")[-1]))
            assert "item_0" in keys

            num_timesteps = len(keys)

            total_loss = 0.0
            metas = {}

            for timestep in range(num_timesteps):

                sequence_id = batch_dict[f"item_{timestep}"]["sequence_name"].tolist()
                batch_size = len(sequence_id)
                assert batch_size > 0
                frame_id = [
                    int(f)
                    for f in batch_dict[f"item_{timestep}"]["sample_idx"].tolist()
                ]
                dataset_id = ["wod" for _ in range(batch_size)]

                chunk_ids = list(zip(dataset_id, sequence_id, frame_id, strict=True))
                print(f"Chunk ids = {chunk_ids}")
                if not scene_memory.is_empty() and scene_memory.is_new_chunk(chunk_ids):
                    scene_memory = scene_memory.empty()

                batch_dict_t = batch_dict[keys[timestep]]
                label_raster = self.make_label_image(batch_dict_t["gt_boxes"])
                input_label_raster = self.make_label_image(
                    self.aug_labels(batch_dict_t["gt_boxes"])
                )
                if self.training:
                    self.reconstructor.eval()
                    self.seq2seq_model.train()
                with torch.no_grad():
                    code, code_idx = self.reconstructor.quantize(torch.concat([label_raster, input_label_raster], dim=0))
                    input_code = code[label_raster.shape[0] :]
                    gt_code_idx = code_idx[: label_raster.shape[0]]
                    x_size, y_size = self.img_size
                    w, h = x_size // self.cfg.patch_size, y_size // self.cfg.patch_size
                    input_code_t = input_code.reshape(batch_size, h, w, -1).permute( 0, 3, 1, 2)
                    gt_code_idx_t = gt_code_idx.reshape(batch_size, h, w)

                track_t_world_vehicle = batch_dict_t["vehicle_to_world"]
                if scene_memory.is_empty():
                    memory_t = self.init_memory.expand(batch_size, -1, -1, -1)
                else:
                    if self.cfg.transform_features:
                        memory_t = scene_memory.transform_features(
                            track_t_world_vehicle,
                            voxel_cfg=self.cfg.voxel_cfg,
                            padding=self.padding_emb,
                            delta_theta_emb_fn=self.delta_theta_embed,
                        )
                    else:
                        memory_t = scene_memory.features
                loss_t, metas_t = self.seq2seq_model.train_iter(
                    input_code_t, memory_t, gt_code_idx_t
                )
                scene_memory.features = metas_t.pop("memory")
                if self.training:
                    del metas_t["pred_code_idx"]
                    del metas_t["logits"]
                assert track_t_world_vehicle is not None
                scene_memory.t_world_vehicle = track_t_world_vehicle
                total_loss = total_loss + loss_t

                scene_memory.chunk_id = chunk_ids

                for k in metas_t:
                    if k not in metas:
                        metas[k] = []
                    metas[k].append(metas_t[k])

            # no more backprop
            scene_memory.features = scene_memory.features.detach()
            if not self.training:
                assert num_timesteps == 1
                pred_code_idx = metas_t["pred_code_idx"]
                codes = self.reconstructor.vqvit.quantizer.indices_to_codes(pred_code_idx)
                codes = codes.reshape(codes.shape[:2] + (-1,)).permute(0, 2, 1)
                decoder_logits = self.reconstructor.decode(codes)
                return input_label_raster, label_raster, decoder_logits

            for k in metas:
                metas[k] = sum(metas[k]) / num_timesteps

        return {"loss": total_loss}, metas, {"loss": total_loss}

    @torch.jit.unused
    def eval_iter(
        self,
        batch_dict: dict,
        iteration: int,
    ):
        self.eval()
        if iteration == 0:
            self.eval_scene_memory = self.eval_scene_memory.empty()
        metrics = {}
        det_outputs = None
        if self.stage == "FSQ":
            sequence_id = batch_dict["sequence_name"].tolist()
            batch_size = len(sequence_id)
            assert batch_size > 0
            x, x_hat = self.forward(batch_dict)
            soft_iou = soft_iou_torch(x_hat, x, sigmoid=True).item()
            metrics["soft_iou"] = soft_iou
        elif self.stage == "SEQ2SEQ":
            input_label_raster, label_raster, logits = self.forward(
                {"item_0": batch_dict}, scene_memory=self.eval_scene_memory
            )
            input_soft_iou = soft_iou_torch(input_label_raster, label_raster, sigmoid=False).item()
            pred_soft_iou = soft_iou_torch(logits, label_raster, sigmoid=True).item()

            metrics["pred_soft_iou"] = pred_soft_iou
            metrics["input_soft_iou"] = input_soft_iou
        return metrics, det_outputs

    @torch.jit.unused
    def viz_iter(
        self,
        batch_dict: dict,
        out_dir: str,
        iteration: int,
    ):
        self.eval()
        if iteration == 0:
            self.viz_scene_memory = self.viz_scene_memory.empty()

        sequence_id = batch_dict["sequence_name"].tolist()
        batch_size = len(sequence_id)
        assert batch_size > 0
        frame_id = [int(f) for f in batch_dict["sample_idx"].tolist()]
        dataset_id = ["wod" for _ in range(batch_size)]
        chunk_ids = list(zip(dataset_id, sequence_id, frame_id, strict=True))

        if self.stage == "FSQ":
            x, x_hat = self.forward(batch_dict)
            for bi in range(len(chunk_ids)):
                out_name = "_".join([str(x) for x in chunk_ids[bi]])
                img = x[bi].detach().cpu().numpy().transpose(1, 2, 0)
                img_hat = (torch.sigmoid(x_hat[bi]).detach().cpu().numpy().transpose(1, 2, 0))
                img = (img * 255).astype(np.uint8)
                img_hat = (img_hat * 255).astype(np.uint8)
                imageio.imsave(f"{out_dir}/{out_name}_gt.png", img)
                imageio.imsave(f"{out_dir}/{out_name}_hat.png", img_hat)
        elif self.stage == "SEQ2SEQ":
            input_label_raster, label_raster, logits = self.forward(
                {"item_0": batch_dict}, scene_memory=self.viz_scene_memory
            )
            for bi in range(input_label_raster.shape[0]):
                out_name = "_".join([str(x) for x in chunk_ids[bi]])
                input_img = (input_label_raster[bi].detach().cpu().numpy().transpose(1, 2, 0))
                gt_img = label_raster[bi].detach().cpu().numpy().transpose(1, 2, 0)
                pred_img = (torch.sigmoid(logits[bi]).detach().cpu().numpy().transpose(1, 2, 0))
                input_img = (input_img * 255).astype(np.uint8)
                gt_img = (gt_img * 255).astype(np.uint8)
                pred_img = (pred_img * 255).astype(np.uint8)
                imageio.imsave(f"{out_dir}/{out_name}_input.png", input_img)
                imageio.imsave(f"{out_dir}/{out_name}_gt.png", gt_img)
                imageio.imsave(f"{out_dir}/{out_name}_pred.png", pred_img)
