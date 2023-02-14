'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
from __future__ import print_function
from socket import if_indextoname
from PIL import Image
import os
import os.path
import numpy as np
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from .io_utils import matching_file_name, degree_file_name, distance_file_name, global_matching_file_name

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

class cifar10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, args, train=True, transform=None, target_transform=None,
                 download=False, np_array=False, dropping=False, training_time=False, total_classes=10):

        super(cifar10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.training_time = training_time
        self.train = train  # training set or test set
        self.np_array = np_array
        self.total_classes = total_classes

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

        self.data, self.targets, self.num_samples_per_class = self._multi_c_filter(args)

    def _multi_c_filter(self, args):
        if args.use_all_classes:
            class_list = list(range(self.total_classes))
        else:
            class_list = args.classes
        num_class = len(class_list)

        vs_all = ('all' in class_list)

        targets_arr = np.array(self.targets)

        X_c = []
        
        for i in range(num_class):
            c = class_list[i]
            if c == 'all':
                continue
            idx = np.where(targets_arr==c)[0]
            X_c.append(self.data[idx])

        if vs_all:
            all_dat = []
            all_classes = list(range(self.total_classes))
            for i in range(self.total_classes):
                if i in class_list:
                    continue
                c = all_classes[i]
                idx = np.where(targets_arr==c)[0]
                all_dat.append(self.data[idx])
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

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        #easy_indc = self.easy_idx[index]

        #matched_idx_curr = int(self.matched_idx[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if not self.np_array:
            img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        #return img, target, index, easy_indc, matched_idx_curr
        return img, target, index, None, None

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

    # def _load_meta(self):
    #     path = os.path.join(self.root, self.base_folder, self.meta['filename'])
    #     if not check_integrity(path, self.meta['md5']):
    #         raise RuntimeError('Dataset metadata file not found or corrupted.' +
    #                            ' You can use download=True to download it')
    #     with open(path, 'rb') as infile:
    #         if sys.version_info[0] == 2:
    #             data = pickle.load(infile)
    #         else:
    #             data = pickle.load(infile, encoding='latin1')
    #         self.classes = data[self.meta['key']]
    #     self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    # def __getitem__(self, index):
    #     """
    #     Args:
    #         index (int): Index

    #     Returns:
    #         tuple: (image, target) where target is index of the target class.
    #     """
    #     img, target = self.data[index], self.targets[index]

    #     # doing this so that it is consistent with all other datasets
    #     # to return a PIL Image
    #     img = Image.fromarray(img)

    #     if self.transform is not None:
    #         img = self.transform(img)

    #     if self.target_transform is not None:
    #         target = self.target_transform(target)

    #     return img, target


    # def __len__(self):
    #     return len(self.data)

    # def _check_integrity(self):
    #     root = self.root
    #     for fentry in (self.train_list + self.test_list):
    #         filename, md5 = fentry[0], fentry[1]
    #         fpath = os.path.join(root, self.base_folder, filename)
    #         if not check_integrity(fpath, md5):
    #             return False
    #     return True

    # def download(self):
    #     if self._check_integrity():
    #         print('Files already downloaded and verified')
    #         return
    #     download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    # def extra_repr(self):
    #     return "Split: {}".format("Train" if self.train is True else "Test")

class cifar100(cifar10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """

    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }

    def __init__(self, root, args, train=True, transform=None, target_transform=None,
                 download=False, np_array=False, dropping=False, training_time=False):
        self.total_classes = 100
        super(cifar100, self).__init__(root, args, train, transform, target_transform,
                 download, np_array, dropping, training_time, total_classes=self.total_classes)
        
