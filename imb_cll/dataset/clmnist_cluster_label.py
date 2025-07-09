import codecs
import os
import os.path
import sys
import warnings
from typing import Dict
from urllib.error import URLError
import numpy as np
import torch
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip
from torchvision.datasets.utils import download_and_extract_archive, check_integrity
from torchvision.datasets.vision import VisionDataset
from .base_dataset import BaseDataset
from sklearn.cluster import KMeans


class CLMNIST(VisionDataset, BaseDataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``MNIST/raw/train-images-idx3-ubyte``
            and  ``MNIST/raw/t10k-images-idx3-ubyte`` exist.
        train (bool, optional): If True, creates dataset from ``train-images-idx3-ubyte``,
            otherwise from ``t10k-images-idx3-ubyte``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    mirrors = [
        "http://yann.lecun.com/exdb/mnist/",
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
    ]

    resources = [
        ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
    ]

    training_file = "training.pt"
    test_file = "test.pt"
    classes = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data


    def __init__(
        self,
        root=None,
        train=True,
        data_type=None,
        # transform=transforms.ToTensor(),
        transform=None,
        validate=False,
        target_transform=None,
        download=True,
        kmean_cluster=None,
        max_train_samples=None,
        multi_label=False,
        augment=False,
        imb_type=None,
        imb_factor=1.0,
        pretrain=None,
        seed=1126,
        input_dataset=None,
        transition_bias=1.0,
        setup_type=None
    ):
        self.root = root
        self.data_type = data_type
        self.num_classes = 10
        self.input_dim = 1 * 28 * 28
        self.multi_label = multi_label
        self.input_dataset = input_dataset
        self.imb_type = imb_type
        self.imb_factor = imb_factor
        self.kmean_cluster = kmean_cluster # Number of clustering with K mean method.
        self.transition_bias = transition_bias
        self.setup_type = setup_type
        # self.mean, self.std = 0.1307, 0.3081
        
        super(CLMNIST, self).__init__(
            root, train, transform, target_transform)
        
        self.train = train  # training set or test set
        self.validate = validate
        self.pretrain = pretrain
        self.seed = seed

        if self.input_dataset == 'mnist':
            self.input_dataset = self.input_dataset.upper()
        elif self.input_dataset == 'kmnist':
            self.input_dataset = self.input_dataset.upper()
        elif self.input_dataset == 'fashionmnist':
            self.input_dataset = 'FashionMNIST'
        
        if self.input_dataset == 'MNIST':
            self.mean, self.std = 0.1307, 0.3081
        elif self.input_dataset == 'FashionMNIST':
            self.mean, self.std = 0.2860, 0.3530
        elif self.input_dataset == 'KMNIST':
            self.mean, self.std = 0.2860, 0.3530
        else:
            raise NotImplementedError

        if self.input_dataset in ('MNIST', 'FashionMNIST'):
            if self._check_legacy_exist():
                self.data, self.targets = self._load_legacy_data()
                return

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self.data, self.targets = self._load_data()

        if self.data_type == 'train':
            if self.imb_type is not None:
                self.img_num_list, self.img_max = self.get_img_num_per_cls(self.num_classes, self.imb_type, self.imb_factor)
                self.gen_imbalanced_data(self.img_num_list)
            
            if max_train_samples: #limit the size of the training dataset to max_train_samples
                train_len = min(len(self.data), max_train_samples)
                self.data = self.data[:train_len]
                self.targets = self.targets[:train_len]

            if self.setup_type == "setup 1":
                self.gen_complementary_target()
            elif self.setup_type == "setup 2":
                self.gen_bias_complementary_label()

        self.idx_train = len(self.data)
        if self.data_type == 'train' and not validate:
            if augment:
                self.transform=Compose([
                    RandomHorizontalFlip(),
                    RandomCrop(28, 4, padding_mode='reflect'),
                    ToTensor(),
                    Normalize(mean=self.mean, std=self.std),
                ])
            else:
                self.transform=Compose([
                ToTensor(),
                Normalize(mean=self.mean, std=self.std),
            ])

        else:
            self.transform=Compose([
                ToTensor(),
                Normalize(mean=self.mean, std=self.std),
            ])
        if self.data_type =='train':
            if self.kmean_cluster != 0:
                self.k_mean_targets = self.features_space()
                print("Done: K_Mean Cluster")
        

    def _check_legacy_exist(self):
        processed_folder_exists = os.path.exists(self.processed_folder)
        if not processed_folder_exists:
            return False

        return all(
            check_integrity(os.path.join(self.processed_folder, file)) for file in (self.training_file, self.test_file)
        )

    def _load_legacy_data(self):
        # This is for BC only. We no longer cache the data in a custom binary, but simply read from the raw data
        # directly.
        data_file = self.training_file if self.train else self.test_file
        return torch.load(os.path.join(self.processed_folder, data_file))

    def _load_data(self):
        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
        data = read_image_file(os.path.join(self.raw_folder, image_file))

        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
        targets = read_label_file(os.path.join(self.raw_folder, label_file))

        return data, targets

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.data_type == 'train':
            if self.imb_type is not None:
                img, targets, true_targets, k_mean_target = self.data[index], self.targets[index], self.true_targets[index], self.k_mean_targets[index]
                img = Image.fromarray(img, mode='L')
            else:
                img, targets, true_targets, k_mean_target = self.data[index], self.targets[index], self.true_targets[index], self.k_mean_targets[index]
                img = Image.fromarray(img.numpy(), mode='L')
        if self.data_type == 'test':
            img, targets = self.data[index], int(self.targets[index])
            img = Image.fromarray(img.numpy(), mode='L')
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            targets = self.target_transform(targets)

        if self.data_type == 'train':
            return img, targets, true_targets, k_mean_target, self.img_max
        else:
            return img, targets

    def __len__(self) -> int:
        return len(self.data)
    
    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "processed")

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self) -> bool:
        return all(
            check_integrity(os.path.join(self.raw_folder, os.path.splitext(os.path.basename(url))[0]))
            for url, _ in self.resources
        )

    def download(self) -> None:
        """Download the MNIST data if it doesn't exist already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        # download files
        for filename, md5 in self.resources:
            for mirror in self.mirrors:
                url = f"{mirror}{filename}"
                try:
                    print(f"Downloading {url}")
                    download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)
                except URLError as error:
                    print(f"Failed to download (trying next):\n{error}")
                    continue
                finally:
                    print()
                break
            else:
                raise RuntimeError(f"Error downloading {filename}")

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"
    
    @torch.no_grad()
    def features_space(self):
        if self.data_type == "train":
            model_simsiam = resnet18()
            
            if self.input_dataset in ('MNIST', 'FashionMNIST', 'KMNIST'):
                num_channel = 1
            else:
                num_channel = 3
            
            model_simsiam.conv1 = nn.Conv2d(num_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
            # model_simsiam.maxpool = nn.Identity()

            transform=Compose([
                ToTensor(),
                Normalize(mean=self.mean, std=self.std),
            ])
            ###NOTED: Need to create imbalanced dataset first and get the idx of training()
            tensor = torch.stack([transform(self.data[i]) for i in range(0, self.idx_train)])
            ds = torch.utils.data.TensorDataset(tensor)
            dl = torch.utils.data.DataLoader(ds, batch_size=1024, shuffle=False)
            print(self.pretrain)

            checkpoint = torch.load(self.pretrain, map_location="cpu")
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder up to before the embedding layer
                if k.startswith('module.encoder') and not k.startswith('module.encoder.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
            model_simsiam.load_state_dict(state_dict, strict=False)
            model_simsiam.fc = nn.Identity()
            model_simsiam.cpu()

            features = []

            model_simsiam.eval()
            for input in dl:
                # features.append(F.normalize(model_simsiam(input.cpu())).cpu())
                features.append(F.normalize(model_simsiam(torch.cat(input).cpu())).cpu())
            features = torch.cat(features, dim=0).cpu().detach().numpy()

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=self.kmean_cluster, random_state=1126)
        cluster_labels = kmeans.fit_predict(features)

        classes, class_counts = np.unique(cluster_labels, return_counts=True)
        sorted_list = sorted(class_counts, reverse=True)
        print("The number of each sample into each cluster is {}".format(sorted_list))

        return cluster_labels

class CLFashionMNIST(CLMNIST):
    """`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``FashionMNIST/raw/train-images-idx3-ubyte``
            and  ``FashionMNIST/raw/t10k-images-idx3-ubyte`` exist.
        train (bool, optional): If True, creates dataset from ``train-images-idx3-ubyte``,
            otherwise from ``t10k-images-idx3-ubyte``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    mirrors = ["http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"]

    resources = [
        ("train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
        ("train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
        ("t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79"),
        ("t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310"),
    ]
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


class CLKMNIST(CLMNIST):
    """`Kuzushiji-MNIST <https://github.com/rois-codh/kmnist>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``KMNIST/raw/train-images-idx3-ubyte``
            and  ``KMNIST/raw/t10k-images-idx3-ubyte`` exist.
        train (bool, optional): If True, creates dataset from ``train-images-idx3-ubyte``,
            otherwise from ``t10k-images-idx3-ubyte``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    mirrors = ["http://codh.rois.ac.jp/kmnist/dataset/kmnist/"]

    resources = [
        ("train-images-idx3-ubyte.gz", "bdb82020997e1d708af4cf47b453dcf7"),
        ("train-labels-idx1-ubyte.gz", "e144d726b3acfaa3e44228e80efcd344"),
        ("t10k-images-idx3-ubyte.gz", "5c965bf0a639b31b8f53240b1b52f4d7"),
        ("t10k-labels-idx1-ubyte.gz", "7320c461ea6c1c855c0b718fb2a4b134"),
    ]
    classes = ["o", "ki", "su", "tsu", "na", "ha", "ma", "ya", "re", "wo"]
    
def get_int(b: bytes) -> int:
    return int(codecs.encode(b, "hex"), 16)


SN3_PASCALVINCENT_TYPEMAP = {
    8: torch.uint8,
    9: torch.int8,
    11: torch.int16,
    12: torch.int32,
    13: torch.float32,
    14: torch.float64,
}


def read_sn3_pascalvincent_tensor(path: str, strict: bool = True) -> torch.Tensor:
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
    Argument may be a filename, compressed filename, or file object.
    """
    # read
    with open(path, "rb") as f:
        data = f.read()
    # parse
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert 1 <= nd <= 3
    assert 8 <= ty <= 14
    torch_type = SN3_PASCALVINCENT_TYPEMAP[ty]
    s = [get_int(data[4 * (i + 1) : 4 * (i + 2)]) for i in range(nd)]

    num_bytes_per_value = torch.iinfo(torch_type).bits // 8
    # The MNIST format uses the big endian byte order. If the system uses little endian byte order by default,
    # we need to reverse the bytes before we can read them with torch.frombuffer().
    needs_byte_reversal = sys.byteorder == "little" and num_bytes_per_value > 1
    parsed = torch.frombuffer(bytearray(data), dtype=torch_type, offset=(4 * (nd + 1)))
    if needs_byte_reversal:
        parsed = parsed.flip(0)

    assert parsed.shape[0] == np.prod(s) or not strict
    return parsed.view(*s)


def read_label_file(path: str) -> torch.Tensor:
    x = read_sn3_pascalvincent_tensor(path, strict=False)
    if x.dtype != torch.uint8:
        raise TypeError(f"x should be of dtype torch.uint8 instead of {x.dtype}")
    if x.ndimension() != 1:
        raise ValueError(f"x should have 1 dimension instead of {x.ndimension()}")
    return x.long()


def read_image_file(path: str) -> torch.Tensor:
    x = read_sn3_pascalvincent_tensor(path, strict=False)
    if x.dtype != torch.uint8:
        raise TypeError(f"x should be of dtype torch.uint8 instead of {x.dtype}")
    if x.ndimension() != 3:
        raise ValueError(f"x should have 3 dimension instead of {x.ndimension()}")
    return x