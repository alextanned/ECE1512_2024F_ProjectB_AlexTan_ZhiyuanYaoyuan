import torch
import numpy as np
from functools import partial
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler as _DistributedSampler

from pcdet.utils import common_utils

from .dataset import DatasetTemplate
from .kitti.kitti_dataset import KittiDataset
from .nuscenes.nuscenes_dataset import NuScenesDataset
from .waymo.waymo_dataset import WaymoChunkDataset, WaymoDataset
from .pandaset.pandaset_dataset import PandasetDataset
from .lyft.lyft_dataset import LyftDataset
from .once.once_dataset import ONCEDataset
from .argo2.argo2_dataset import Argo2Dataset
from .custom.custom_dataset import CustomDataset

__all__ = {
    'DatasetTemplate': DatasetTemplate,
    'KittiDataset': KittiDataset,
    'NuScenesDataset': NuScenesDataset,
    'WaymoDataset': WaymoDataset,
    'WaymoChunkDataset': WaymoChunkDataset,
    'PandasetDataset': PandasetDataset,
    'LyftDataset': LyftDataset,
    'ONCEDataset': ONCEDataset,
    'CustomDataset': CustomDataset,
    'Argo2Dataset': Argo2Dataset
}


class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

class ChunkSampler(DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, num_sequential_chunks: int = 1, batch_size: int = 1):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.items = getattr(self.dataset, "items", None)
        assert self.items is not None # list of lists
        self.chunk_length = getattr(self.dataset, "chunk_length", None)
        assert self.chunk_length is not None
        self.per_seq_indices = getattr(self.dataset, "per_seq_indices", None)
        assert self.per_seq_indices is not None
        self.batch_size = batch_size

        self.all_chunk_groups = []
        for seq in self.per_seq_indices:
            per_seq_indices = dataset.per_seq_indices[seq]
            for i in range(0, len(per_seq_indices) - num_sequential_chunks*self.chunk_length + 1):
                self.all_chunk_groups.extend(
                    [per_seq_indices[i:i + num_sequential_chunks*self.chunk_length:self.chunk_length]]
                )

    def __iter__(self):
        if self.shuffle:
            np.random.seed(self.epoch)
            np.random.shuffle(self.all_chunk_groups)

        per_rank_chunks_groups = self.all_chunk_groups[self.rank::self.num_replicas]
        batch_inds = [[] for _ in range(self.batch_size)]
        for i, group in enumerate(per_rank_chunks_groups):
            batch_ind = i % self.batch_size
            batch_inds[batch_ind].extend(group)

        inds = [item for combined in zip(*batch_inds) for item in combined]
        return iter(inds)


def build_dataloader(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4, seed=None,
                     logger=None, training=True, merge_all_iters_to_one_epoch=False, total_epochs=0):

    dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=root_path,
        training=training,
        logger=logger,
    )

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training:
            if isinstance(dataset, WaymoChunkDataset):
                sampler = ChunkSampler(dataset, num_replicas=world_size, rank=rank, shuffle=getattr(dataset_cfg, "SHUFFLE", False), num_sequential_chunks=dataset_cfg.num_sequential_chunks, batch_size=batch_size)
            else:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None
        if isinstance(dataset, WaymoChunkDataset):
            sampler = ChunkSampler(dataset, num_replicas=1, rank=0, shuffle=getattr(dataset_cfg, "SHUFFLE", False), num_sequential_chunks=dataset_cfg.num_sequential_chunks, batch_size=batch_size)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler is None) and training and getattr(dataset_cfg, "SHUFFLE", True),
        collate_fn=dataset.collate_batch,
        drop_last=False, sampler=sampler, timeout=0, worker_init_fn=partial(common_utils.worker_init_fn, seed=seed),
    )

    return dataset, dataloader, sampler
