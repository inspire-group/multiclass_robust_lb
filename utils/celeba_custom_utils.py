'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
from __future__ import print_function
import os
import os.path
import numpy as np
import sys
import torch

import csv
from collections import namedtuple
from typing import Any, Callable, List, Optional, Tuple, Union

import PIL

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_file_from_google_drive, extract_archive, verify_str_arg

CSV = namedtuple("CSV", ["header", "index", "data"])

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

class CelebA(VisionDataset):
    """`Large-scale CelebFaces Attributes (CelebA) Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ Dataset.
    Uses identities as classes
    """

    base_folder = "celeba"
    # There currently does not appear to be a easy way to extract 7z in python (without introducing additional
    # dependencies). The "in-the-wild" (not aligned+cropped) images are only in 7z, so they are not available
    # right now.
    file_list = [
        # File ID                                      MD5 Hash                            Filename
        ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        # ("0B7EVK8r0v71pbWNEUjJKdDQ3dGc","b6cd7e93bc7a96c2dc33f819aa3ac651", "img_align_celeba_png.7z"),
        # ("0B7EVK8r0v71peklHb0pGdDl6R28", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_celeba.7z"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
        ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
        # ("0B7EVK8r0v71pTzJIdlJWdHczRlU", "063ee6ddb681f96bc9ca28c6febb9d1a", "list_landmarks_celeba.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]

    def __init__(
        self,
        root: str, args,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        np_array=False
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = split
        self.target_type = ['identity']
        
        if not self.target_type and self.target_transform is not None:
            raise RuntimeError("target_transform is specified but target_type is empty")
        
        self.np_array = np_array
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[verify_str_arg(split.lower(), "split", ("train", "valid", "test", "all"))]
        splits = self._load_csv("list_eval_partition.txt")
        identity = self._load_csv("identity_CelebA.txt")
        mask = slice(None) if split_ is None else (splits.data == split_).squeeze()

        if mask == slice(None):  # if split == "all"
            self.filename = splits.index
        else:
            self.filename = [splits.index[i] for i in torch.squeeze(torch.nonzero(mask))]
        self.identity = identity.data[mask][:, 0]
        unique_id = np.unique(self.identity)
        self.total_classes = len(unique_id)

        # some identities are exclusively within a single partition,
        # relabel identities so that for the split we are looking at we have
        # identity labels 0 ... self.total_classes
        for i, id in enumerate(unique_id):
            idx = np.where(self.identity == id)[0]
            self.identity[idx] = i

        self.files_paths = []
        for i in range(len(self.identity)):
            X = os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[i])
            self.files_paths.append(X)
        self.files_paths = np.array(self.files_paths)

        self.data_paths, self.targets, self.num_samples_per_class = self._multi_c_filter(args)
        self.data_paths = self.data_paths.reshape(-1)

    def _multi_c_filter(self, args):
        if args.use_all_classes:
            class_list = list(range(self.total_classes))
        else:
            class_list = args.classes
        num_class = len(class_list)

        vs_all = ('all' in class_list)
        targets_arr = np.array(self.identity)
        X_c = []
        
        for i in range(num_class):
            c = class_list[i]
            if c == 'all':
                continue
            idx = np.where(targets_arr==c)[0]
            X_c.append(self.files_paths[idx])

        if vs_all:
            all_dat = []
            all_classes = list(range(self.total_classes))
            for i in range(self.total_classes):
                if i in class_list:
                    continue
                c = all_classes[i]
                idx = np.where(targets_arr==c)[0]
                all_dat.append(self.files_paths[idx])
            all_dat = np.concatenate(all_dat, axis=0)
            X_c.append(all_dat)

        X_lens = []
        for x in X_c:
            X_lens.append(len(x))

        if args.balanced:
            num_samples = min(*X_lens,args.num_samples)
        else:
            num_samples = args.num_samples

        num_samples_per_class = []
        for i in range(num_class):
            X_c[i] = X_c[i][:num_samples]
            num_samples_per_class.append(len(X_c[i]))
        
        curr_data = np.concatenate(X_c, axis=0)

        Y_curr = np.zeros(len(curr_data))
        start = 0
        for i in range(num_class):
            Y_curr[start:start+num_samples_per_class[i]] = i
            start += num_samples_per_class[i]

        curr_labels = Y_curr.tolist()
        curr_labels = [int(x) for x in curr_labels]
        return curr_data, curr_labels, num_samples_per_class

    def _load_csv(
        self,
        filename: str,
        header: Optional[int] = None,
    ) -> CSV:
        with open(os.path.join(self.root, self.base_folder, filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

        if header is not None:
            headers = data[header]
            data = data[header + 1 :]
        else:
            headers = []

        indices = [row[0] for row in data]
        data = [row[1:] for row in data]
        data_int = [list(map(int, i)) for i in data]

        return CSV(headers, indices, torch.tensor(data_int))

    def _check_integrity(self) -> bool:
        for (_, md5, filename) in self.file_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            _, ext = os.path.splitext(filename)
            # Allow original archive to be deleted (zip and 7z)
            # Only need the extracted images
            if ext not in [".zip", ".7z"] and not check_integrity(fpath, md5):
                return False

        # Should check a hash of the images
        return os.path.isdir(os.path.join(self.root, self.base_folder, "img_align_celeba"))

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        for (file_id, md5, filename) in self.file_list:
            download_file_from_google_drive(file_id, os.path.join(self.root, self.base_folder), filename, md5)

        extract_archive(os.path.join(self.root, self.base_folder, "img_align_celeba.zip"))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X_path, target = self.data_paths[index], self.targets[index]
        X = PIL.Image.open(X_path)

        if self.np_array:
            X = np.asarray(X)

        if self.transform is not None:
            X = self.transform(X)

        return X, target, 0, 0, 0

    def __len__(self) -> int:
        return len(self.targets)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return "\n".join(lines).format(**self.__dict__)
