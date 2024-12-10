import matplotlib.pyplot as plt
from typing import Sequence, Optional, Tuple, Union
import os
import torch
from matplotlib.colors import Colormap, to_rgb
from matplotlib.axes import Axes
import numpy as np
from scipy.stats import binned_statistic_2d
from enum import IntEnum

from matplotlib.patches import Arrow, Rectangle

def colorize_lidar(
    lidar_sweep: torch.Tensor,
    lidar_feature_idx: int,
    cmap: Colormap = plt.cm.jet,
    imin: float = 0,
    imax: float = 50,
) -> torch.Tensor:

    lidar_feature_clipped = ((lidar_sweep[:, lidar_feature_idx] - imin) / (imax - imin)).clamp(min=0.0, max=1.0)
    colors_rgba = torch.tensor(cmap(lidar_feature_clipped))
    return colors_rgba


def voxelize_lidar(lidar_sweep: torch.Tensor, voxel_size_m: float, lidar_feature_idx: int):
    if lidar_sweep.size(0) == 0:
        return lidar_sweep

    x_min, x_max = torch.min(lidar_sweep[:, 0]), torch.max(lidar_sweep[:, 0])
    y_min, y_max = torch.min(lidar_sweep[:, 1]), torch.max(lidar_sweep[:, 1])
    x_bins, y_bins = (
        np.arange(x_min, x_max + voxel_size_m + 1e-3, voxel_size_m).tolist(),
        np.arange(y_min, y_max + voxel_size_m + 1e-3, voxel_size_m).tolist(),
    )

    counts = binned_statistic_2d(
        lidar_sweep[:, 0].numpy(),
        lidar_sweep[:, 1].numpy(),
        lidar_sweep[:, lidar_feature_idx].numpy(),
        statistic="count",
        bins=[x_bins, y_bins],
    )[0]
    values = binned_statistic_2d(
        lidar_sweep[:, 0].numpy(),
        lidar_sweep[:, 1].numpy(),
        lidar_sweep[:, lidar_feature_idx].numpy(),
        statistic="median",
        bins=[x_bins, y_bins],
    )[0]

    x_grid, y_grid = torch.meshgrid(
        [torch.tensor(x_bins) + voxel_size_m / 2.0, torch.tensor(y_bins) + voxel_size_m / 2.0]
    )
    nz_coords = torch.tensor(counts.nonzero()).t()
    x = x_grid[nz_coords[:, 0], nz_coords[:, 1]]
    y = y_grid[nz_coords[:, 0], nz_coords[:, 1]]
    v = torch.from_numpy(np.array(values[nz_coords[:, 0], nz_coords[:, 1]]))

    new_lidar_sweep = torch.zeros((len(x), lidar_sweep.size(1)))
    new_lidar_sweep[:, 0] = x
    new_lidar_sweep[:, 1] = y
    new_lidar_sweep[:, lidar_feature_idx] = v
    return new_lidar_sweep


def plot_lidar(
    ax: Axes,
    lidar: Sequence[torch.Tensor],
    alpha: float = 0.1,
    color: Optional[Tuple[float, float, float, float]] = None,
    lidar_feature_idx: int = 2,
    imin=-2.0,
    imax=6.0,
    voxelize: bool = True,
    voxel_size_m: float = 0.25,
    pt_size: float = 1.0,
):

    colors: torch.Tensor
    for pre_sweeps in lidar[:-1]:
        if voxelize:
            pre_sweeps = voxelize_lidar(pre_sweeps, voxel_size_m=voxel_size_m, lidar_feature_idx=lidar_feature_idx)

        if color:
            colors = torch.tensor([color] * len(pre_sweeps))
        else:
            colors = colorize_lidar(pre_sweeps, lidar_feature_idx=lidar_feature_idx, imin=imin, imax=imax)
        colors[:, -1] = alpha
        ax.scatter(pre_sweeps[:, 0], pre_sweeps[:, 1], s=1, c=colors)

    # Plot last sweep
    sweep = lidar[-1]
    if voxelize:
        sweep = voxelize_lidar(sweep, voxel_size_m=voxel_size_m, lidar_feature_idx=lidar_feature_idx)

    if color:
        colors = torch.tensor([color] * len(sweep))
    else:
        colors = colorize_lidar(sweep, lidar_feature_idx=lidar_feature_idx, imin=imin, imax=imax)
    colors[:, -1] = alpha

    ax.scatter(sweep[:, 0], sweep[:, 1], s=pt_size, c=colors)

def plot_box(
    ax: Axes,
    x: float,
    y: float,
    yaw: float,
    length: float,
    width: float,
    color: Union[Tuple[float, ...], str] = "red",
    face_color: Union[Tuple[float, ...], str] = "none",
    alpha: float = 1.0,
    label: Optional[str] = None,
    string: Optional[str] = "",
    rec_line_width: float = 4.0,
    hatch: Optional[str] = None,
    linestyle: Optional[str] = "-",
    fill: bool = False,
    plot_orientation: bool = True,
    fontsize: float = 12.0,
    rotate_text: bool = True,
):

    dx = np.cos(yaw) * length - np.sin(yaw) * width
    dy = np.sin(yaw) * length + np.cos(yaw) * width
    # Plot rectangle
    ax.add_patch(
        Rectangle(
            (x - dx / 2, y - dy / 2),
            length,
            width,
            np.rad2deg(yaw),
            edgecolor=color,
            facecolor=face_color,
            lw=rec_line_width,
            alpha=alpha,
            label=label,
            hatch=hatch,
            linestyle=linestyle,
            fill=fill,
        )
    )

    if not rotate_text:
        dx = length
        dy = width
    ax.text(
        x - dx / 2,
        y - dy / 2 - 0.5,
        string,
        rotation=np.rad2deg(yaw) if rotate_text else 0,
        color=color,
        ha="left",
        va="top",
        weight="extra bold",
        fontsize=fontsize,
    )
    # Plot orientation
    if plot_orientation:
        ax.add_patch(
            Arrow(
                x,
                y,
                np.cos(yaw) * length / 2,
                np.sin(yaw) * length / 2,
                edgecolor=color,
                facecolor=color,
                alpha=alpha,
                capstyle="projecting",
                lw=1,
            )
        )


def plot_output(out_bi, labels, point_cloud, voxel_cfg, out_dir, out_name):
    x_min = voxel_cfg.x_min
    x_max = voxel_cfg.x_max
    y_min = voxel_cfg.y_min
    y_max = voxel_cfg.y_max
    xsize = x_max - x_min
    ysize = y_max - y_min
    fig, ax = plt.subplots(figsize=(xsize * 0.1, ysize * 0.1), dpi=100)
    boxes = out_bi["pred_boxes"]
    scores = out_bi["pred_scores"]

    if point_cloud is not None:
        plot_lidar(ax, [point_cloud.cpu()], color=None, alpha=0.2, pt_size=3)

    for label in labels:
        # label = (x, y, z, l, w, h, yaw, class_id)
        plot_box(
            ax,
            label[0].item(), 
            label[1].item(),
            label[6].item(),
            label[3].item(),
            label[4].item(),
            color="gray",
            face_color="gray",
            alpha=1.0,
        )

    for box, score in zip(boxes, scores):
        if score.item() > 0.2:
            plot_box(
                ax,
                box[0].item(),
                box[1].item(),
                box[6].item(),
                box[3].item(),
                box[4].item(),
                color="yellow",
                face_color="none",
                alpha=score.item(),
                string=f"{score:.2f}",
            )

    ax.set_xlim((x_min, x_max))
    ax.set_ylim((y_min, y_max))
    facecolor = "black"
    ax.set_facecolor(facecolor)
    fig.set_facecolor(facecolor)
    ax.set_axis_off()
    fig.tight_layout()

    out = os.path.join(out_dir, out_name)
    print(f"Saving to {out}")
    plt.savefig(out)
    plt.close("all")
