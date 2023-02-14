import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from .mnist_custom_utils import MNIST, FashionMNIST
from .cifar_custom_utils import cifar10, cifar100
from .celeba_custom_utils import CelebA
import numpy as np


# def data_augmentation(input_images, max_rot=25, horizontal_flip=True, width_shift_range=0.2, height_shift_range=0.2):
#     def sometimes(aug): return iaa.Sometimes(0.5, aug)
#     seq = iaa.Sequential([
#         iaa.Fliplr(0.3),  # horizontally flip 50% of the images
#         # iaa.GaussianBlur(sigma=(0, 1.0)), # blur images with a sigma of 0 to 3.0
#         sometimes(iaa.Affine(
#             # scale images to 80-120% of their size, individually per axis
#             scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
#             # translate by -20 to +20 percent (per axis
#             translate_percent={"x": (-width_shift_range, width_shift_range),
#                                "y": (-height_shift_range, height_shift_range)},
#             cval=(0, 1),  # if mode is constant, use a cval between 0 and 255
#         )),
#         sometimes(iaa.Affine(
#             rotate=(-max_rot, max_rot),  # rotate by -45 to +45 degrees
#         ))
#     ])
#     return seq.augment_images(input_images)

def get_num_batches(num_samples, batch_size):
    num_batches = num_samples / batch_size
    if num_batches - int(num_batches) > 0:
        num_batches = int(num_batches) + 1
    else: 
        num_batches = int(num_batches)
    return num_batches


class NumpyDataLoader(object):

    def __init__(self, numpy_dataset, datadim, batch_size=100):
        self.dataset = numpy_dataset
        self.num_samples = numpy_dataset.num_samples_per_class
        self.class_end_idx = np.cumsum(self.num_samples)
        self.class_start_idx = [0] + list(self.class_end_idx)
        self.num_classes = numpy_dataset.total_classes
        self.batch_size = batch_size
        self.num_batches_per_class = [get_num_batches(n, self.batch_size) for n in self.num_samples]
        self.datadim = datadim
    
    def loader(self, c):
        '''dataloader for class c'''
        start = self.class_start_idx[c]
        for batch_idx in range(self.num_batches_per_class[c]):
            batch_x = []
            for i in range(start, start+self.batch_size):
                if i >= self.class_end_idx[c]:
                    break
                X, y, _, _, _ = self.dataset[i]
                batch_x.append(X)
            yield (np.array(batch_x) / 255).reshape(-1, self.datadim)
            start += self.batch_size

    def __len__(self):
        return len(self.dataset)

def load_dataset(args, data_dir, training_time):
    if args.dataset_in == 'CIFAR-10':
        loader_train, loader_test, data_details = load_cifar_dataset(args, data_dir, training_time)
    elif 'MNIST' in args.dataset_in:
        loader_train, loader_test, data_details = load_mnist_dataset(args, data_dir, training_time)
    else:
        raise ValueError('No support for dataset %s' % args.dataset)

    return loader_train, loader_test, data_details


def load_mnist_dataset(args, data_dir, training_time):
    # MNIST data loaders
    if args.dataset_in == 'MNIST':
        trainset = datasets.MNIST(root=data_dir, train=True,
                                    download=False, transform=transforms.ToTensor(),
                                    training_time=training_time)
        testset = datasets.MNIST(root=data_dir, train=False,
                                    download=False, transform=transforms.ToTensor(),
                                    training_time=training_time)
    elif args.dataset_in == 'fMNIST':
        trainset = datasets.FashionMNIST(root=data_dir, train=True,
                                        download=False, transform=transforms.ToTensor(),
                                        training_time=training_time)
        testset = datasets.FashionMNIST(root=data_dir, train=False,
                                        download=False, transform=transforms.ToTensor(),
                                        training_time=training_time)
    
    loader_train = torch.utils.data.DataLoader(trainset, 
                                batch_size=args.batch_size,
                                shuffle=True)

    loader_test = torch.utils.data.DataLoader(testset, 
                                batch_size=args.test_batch_size,
                                shuffle=False)
    data_details = {'n_channels':1, 'h_in':28, 'w_in':28, 'scale':255.0}
    return loader_train, loader_test, data_details


def load_cifar_dataset(args, data_dir, training_time):
    # CIFAR-10 data loaders
    trainset = datasets.CIFAR10(root=data_dir, train=True,
                                download=True, 
                                transform=transforms.Compose([
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, 4),
                                    transforms.ToTensor()
                                ]))
    loader_train = torch.utils.data.DataLoader(trainset, 
                                batch_size=args.batch_size,
                                shuffle=True)

    testset = datasets.CIFAR10(root=data_dir,
                                train=True,
                                download=False, transform=transforms.ToTensor())
    loader_test = torch.utils.data.DataLoader(testset, 
                                batch_size=args.test_batch_size,
                                shuffle=False)
    data_details = {'n_channels':3, 'h_in':32, 'w_in':32, 'scale':255.0}
    return loader_train, loader_test, data_details


def load_dataset_custom(args, data_dir, training_time):
    if args.dataset_in == 'CIFAR-10':
        loader_train, loader_test, data_details = load_cifar_dataset_custom(args, data_dir, training_time)
    elif 'MNIST' in args.dataset_in:
        loader_train, loader_test, data_details = load_mnist_dataset_custom(args, data_dir, training_time)
    else:
        raise ValueError('No support for dataset %s' % args.dataset)

    return loader_train, loader_test, data_details

def load_mnist_dataset_custom(args, data_dir, training_time):
    # MNIST data loaders
    if args.dataset_in == 'MNIST':
        trainset = MNIST(root=data_dir, args=args, train=True,
                                    download=False, 
                                    transform=transforms.ToTensor(),
                                    dropping=args.dropping,
                                    training_time=training_time)
        testset = MNIST(root=data_dir, args=args, train=False,
                            download=False, 
                            transform=transforms.ToTensor(),
                            dropping=args.dropping,
                            training_time=training_time)
    elif args.dataset_in == 'fMNIST':
        trainset = FashionMNIST(root=data_dir, args=args, train=True,
                                    download=True, 
                                    transform=transforms.ToTensor(),
                                    dropping=args.dropping,
                                    training_time=training_time)
        testset = FashionMNIST(root=data_dir, args=args, train=False,
                                    download=True, 
                                    transform=transforms.ToTensor(),
                                    dropping=args.dropping,
                                    training_time=training_time)

    loader_train = torch.utils.data.DataLoader(trainset, 
                                batch_size=args.batch_size,
                                shuffle=True)

    loader_test = torch.utils.data.DataLoader(testset, 
                                batch_size=args.test_batch_size,
                                shuffle=False)
    data_details = {'n_channels':1, 'h_in':28, 'w_in':28, 'scale':255.0}
    return loader_train, loader_test, data_details

def load_cifar_dataset_custom(args, data_dir, training_time):
    # CIFAR-10 data loaders
    trainset = cifar10(root=data_dir, args=args, train=True,
                                download=False, 
                                transform=transforms.Compose([
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, 4),
                                    transforms.ToTensor()
                                ]),
                                dropping=args.dropping,
                                training_time=training_time)
    loader_train = torch.utils.data.DataLoader(trainset, 
                                batch_size=args.batch_size,
                                shuffle=True)

    testset = cifar10(root=data_dir, args=args,
                                train=False,
                                download=False, 
                                transform=transforms.ToTensor(),
                                dropping=args.dropping,
                                training_time=training_time)

    loader_test = torch.utils.data.DataLoader(testset, 
                                batch_size=args.test_batch_size,
                                shuffle=False)
    data_details = {'n_channels':3, 'h_in':32, 'w_in':32, 'scale':255.0}
    return loader_train, loader_test, data_details

def load_dataset_numpy(args, data_dir, training_time):
    if args.dataset_in == 'MNIST':
        trainset = MNIST(root=data_dir, args=args, train=True,
                            download=True,
                            np_array=True,
                            training_time=training_time)
        testset = MNIST(root=data_dir, args=args, train=False,
                                download=False,
                                np_array=True,
                                training_time=training_time)
        data_details = {'n_channels':1, 'h_in':28, 'w_in':28, 'scale':255.0}

    elif args.dataset_in == 'fMNIST':
        trainset = FashionMNIST(root=data_dir, args=args, train=True,
                            download=True,
                            np_array=True,
                            training_time=training_time)
        testset = FashionMNIST(root=data_dir, args=args, train=False,
                                download=True,
                                np_array=True,
                                training_time=training_time)
        data_details = {'n_channels':1, 'h_in':28, 'w_in':28, 'scale':255.0}
        
    elif args.dataset_in == 'CIFAR-10':
        trainset = cifar10(root=data_dir, args=args, train=True,
                            download=True, 
                            np_array=True,
                            training_time=training_time)
        testset = cifar10(root=data_dir, args=args, train=False,
                                download=True,
                                np_array=True,
                                training_time=training_time)
        data_details = {'n_channels':3, 'h_in':32, 'w_in':32, 'scale':255.0}

    elif args.dataset_in == 'CIFAR-100':
        trainset = cifar100(root=data_dir, args=args, train=True,
                            download=True, 
                            np_array=True,
                            training_time=training_time)
        testset = cifar100(root=data_dir, args=args, train=False,
                                download=True,
                                np_array=True,
                                training_time=training_time)
        data_details = {'n_channels':3, 'h_in':32, 'w_in':32, 'scale':255.0}

    elif args.dataset_in == 'CelebA':
        trainset = CelebA(root=data_dir, args=args, split='train',
                            download=True, 
                            np_array=True)
        testset = CelebA(root=data_dir, args=args, split='test',
                            download=True, 
                            np_array=True)
        data_details = {'n_channels':3, 'h_in':178, 'w_in':218, 'scale':255.0}

    return trainset, testset, data_details

def load_dataset_tensor(args, data_dir, training_time):
    if args.dataset_in == 'MNIST':
        trainset = MNIST(root=data_dir, args=args, train=True,
                            download=False,
                            transform=transforms.ToTensor(),
                            training_time=training_time)
        testset = MNIST(root=data_dir, args=args, train=False,
                                download=False,
                                transform=transforms.ToTensor(),
                                training_time=training_time)
        data_details = {'n_channels':1, 'h_in':28, 'w_in':28, 'scale':255.0}
    elif args.dataset_in == 'fMNIST':
        trainset = FashionMNIST(root=data_dir, args=args, train=True,
                            download=False,
                            transform=transforms.ToTensor(),
                            training_time=training_time)
        testset = FashionMNIST(root=data_dir, args=args, train=True,
                                download=False,
                                transform=transforms.ToTensor(),
                                training_time=training_time)
        data_details = {'n_channels':1, 'h_in':28, 'w_in':28, 'scale':255.0}
    elif args.dataset_in == 'CIFAR-10':
        trainset = cifar10(root=data_dir, args=args, train=True,
                            download=False,
                            transform=transforms.ToTensor(),
                            training_time=training_time)
        testset = cifar10(root=data_dir, args=args,
                                train=False,
                                download=False,
                                transform=transforms.ToTensor(),
                                training_time=training_time)
        data_details = {'n_channels':3, 'h_in':32, 'w_in':32, 'scale':255.0}
    return trainset, testset, data_details