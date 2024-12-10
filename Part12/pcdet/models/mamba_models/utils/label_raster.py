import torch
import numpy as np
import easydict


def create_bev_image(bboxes, voxel_cfg):
    """
    Create a BEV image of bounding boxes.

    Args:
        bboxes (torch.Tensor): Tensor of shape (B, N, 8) containing bounding boxes.
                               Each bounding box is represented as (x, y, z, l, w, h, theta, class_id).
        voxel_cfg (VoxelConfig): Configuration object containing:
                                 - x_min, x_max, y_min, y_max: Boundaries of the BEV space.
                                 - step: The resolution of each voxel.

    Returns:
        torch.Tensor: BEV images of shape (B, 3, H, W), where H and W are the dimensions of the grid.
    """
    H = int((voxel_cfg.y_max - voxel_cfg.y_min) / voxel_cfg.step)
    W = int((voxel_cfg.x_max - voxel_cfg.x_min) / voxel_cfg.step)

    device = bboxes.device

    bev_images = torch.zeros(
        (bboxes.shape[0], 3, H, W), dtype=torch.uint8, device=device
    )

    for b in range(bboxes.shape[0]):
        for bbox in bboxes[b]:
            x, y, _, l, w, _, theta, class_id = bbox
            if class_id.int().item() not in {0, 1, 2} or bbox.sum() == 0:
                continue

            class_id = int(class_id)

            # Compute the corners of the bounding box
            cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
            dx = torch.tensor([l / 2, l / 2, -l / 2, -l / 2], device=device)
            dy = torch.tensor([w / 2, -w / 2, -w / 2, w / 2], device=device)
            corners_x = x + cos_theta * dx - sin_theta * dy
            corners_y = y + sin_theta * dx + cos_theta * dy

            # Convert corners to grid space
            grid_corners_x = ((corners_x - voxel_cfg.x_min) / voxel_cfg.step).long()
            grid_corners_y = ((corners_y - voxel_cfg.y_min) / voxel_cfg.step).long()

            # Clip to grid boundaries
            grid_corners_x = torch.clamp(grid_corners_x, 0, W - 1)
            grid_corners_y = torch.clamp(grid_corners_y, 0, H - 1)

            # Fill the polygon in the BEV image
            bev_images[b, class_id] = fill_polygon(
                bev_images[b, class_id], grid_corners_x, grid_corners_y
            )

    return bev_images.float()


def fill_polygon(bev_channel, grid_x, grid_y):
    """
    Fill a polygon on a BEV channel using rasterization.

    Args:
        bev_channel (torch.Tensor): 2D tensor representing a single BEV channel.
        grid_x (torch.Tensor): Tensor of x-coordinates of the polygon's vertices.
        grid_y (torch.Tensor): Tensor of y-coordinates of the polygon's vertices.

    Returns:
        torch.Tensor: Updated BEV channel with the polygon filled.
    """
    import cv2  # OpenCV for rasterization

    # Convert torch tensors to numpy arrays
    points = np.array(
        [[grid_x[i].item(), grid_y[i].item()] for i in range(len(grid_x))],
        dtype=np.int32,
    )

    canvas = bev_channel.detach().cpu().numpy()

    cv2.fillPoly(canvas, [points], 1)  # Fill the polygon with value 1

    return torch.tensor(canvas).to(bev_channel)  # Modifications are in-place


# Testing
def test_create_bev_image():
    voxel_cfg = easydict.EasyDict(
        {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 100, "step": 0.05}
    )

    # Create a batch of bounding boxes
    bboxes = torch.tensor(
        [
            [
                [10, 20, 0, 4, 2, 1, 0, 0],  # Class 0
                [50, 50, 0, 10, 5, 1, 0.5, 1],  # Class 1
                [70, 80, 0, 6, 3, 1, 1.0, 2],  # Class 2
                [0, 0, 0, 0, 0, 0, 0, 0],  # Masked
            ]
        ]
    )  # Shape: (1, 4, 8)

    bev_images = create_bev_image(bboxes, voxel_cfg)

    # Assert the shape is correct
    # assert bev_images.shape == (1, 3, 100, 100)

    # Check that class-specific channels have data
    assert bev_images[0, 0].sum() > 0  # Class 0 channel
    assert bev_images[0, 1].sum() > 0  # Class 1 channel
    assert bev_images[0, 2].sum() > 0  # Class 2 channel

    # Save the BEV images for visual inspection
    import cv2
    import os

    os.makedirs("bev_images", exist_ok=True)
    for c in range(3):
        bev_image = bev_images[0, c].numpy() * 255  # Scale to 0-255
        bev_image = bev_image.astype(np.uint8)
        cv2.imwrite(f"bev_images/bev_class_{c}.png", bev_image)

    # Also, save a combined image
    combined_bev = bev_images[0].numpy().transpose(1, 2, 0) * 255  # Shape: (H, W, 3)
    combined_bev = combined_bev.astype(np.uint8)
    cv2.imwrite("bev_images/combined_bev.png", combined_bev)

    print("All tests passed!")


if __name__ == "__main__":
    test_create_bev_image()
