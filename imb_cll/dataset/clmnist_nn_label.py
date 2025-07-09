import codecs
import os
import os.path
import sys
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.error import URLError
from PIL import Image
import os
import os.path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision.models import resnet18
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip
from .base_dataset import BaseDataset

from tqdm.auto import tqdm

class NCLMNIST(VisionDataset, BaseDataset):
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
    
    def __init__(self,
        root=None,
        train=True,
        data_type=None,
        transform=None,
        validate=False,
        target_transform=None,
        download=True,
        max_train_samples=None,
        multi_label=False,
        imb_type=None,
        imb_factor=1.0,
        num_comp=1,
        num_neighbors=64,
        pretrain=None,
        seed=1126,
        noise=0,
        num_iter=100,
        weight=None,
        input_dataset=None,
    ):
        self.root = root
        self.data_type = data_type
        self.num_classes = 10
        self.input_dim = 1 * 28 * 28
        self.multi_label = multi_label
        self.input_dataset = input_dataset
        self.imb_type = imb_type
        self.imb_factor = imb_factor
        
        super(NCLMNIST, self).__init__(
            root, train, transform, target_transform)
        
        self.train = train  # training set or test set
        self.validate = validate
        self.pretrain = pretrain
        self.seed = seed
        self.noise = noise
        self.num_iter = num_iter
        self.weight = weight

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

        self.rng = np.random.default_rng(self.seed)
        self.idx = self.rng.permutation(len(self.targets))
        self.idx_train = self.idx[:len(self.data)]

        self.num_comp = num_comp
        self.num_neighbors = num_neighbors

        if self.data_type in ("train", "val"):
            self.comp_labels = self.generate_multi_compl_labels()

        if self.data_type == "train" and not validate:
            ol = []
            for i in range(len(self.data)):
                ol.append(self.targets[self.idx_train[i]])
            ol = torch.LongTensor(ol)
            K = 10
            # # T = (torch.ones(K,K) - torch.eye(K)) / (K-1)
            # # soft_cl = T[ol]

            # orig_cl = F.nll_loss(-self.comp_labels, self.comp)
            # noise_cl = F.nll_loss(-self.comp_labels, ol)
            # res_cl = (1 - orig_cl - noise_cl) / (K-2)

            # # margin calculation
            # comp_ = self.comp_labels.clone().scatter(1, ol.unsqueeze(1), 1)
            # min_except_ol = torch.min(comp_, dim=-1)[0]
            # margin = min_except_ol - F.nll_loss(-self.comp_labels, ol, reduction="none") 
            # margin_avg = margin.mean()

            # # least_cl = torch.min(self.comp_labels)[1]
            least_cl = (ol == torch.min(self.comp_labels, dim=-1)[1]).float().mean()
            print(f"{least_cl:.4f}")
            # print(f"{least_cl:.4f} {margin_avg:.4f} {orig_cl:.4f} {noise_cl:.4f} {res_cl:.4f}")

        if self.data_type == "train" and not validate:
            self.transform=Compose([
                # RandomHorizontalFlip(),
                # RandomCrop(32, 4, padding_mode='reflect'),
                ToTensor(),
                Normalize(mean=self.mean, std=self.std),
            ])
        else:
            self.transform=Compose([
                ToTensor(),
                Normalize(mean=self.mean, std=self.std),
            ])      

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
                img, target = self.data[self.idx_train[index]], self.targets[self.idx_train[index]]
                img = Image.fromarray(img, mode='L')
            else:
                img, target = self.data[self.idx_train[index]], self.targets[self.idx_train[index]]
                img = Image.fromarray(img.numpy(), mode='L')
        if self.data_type == 'test':
            img, target = self.data[index], int(self.targets[index])
            img = Image.fromarray(img.numpy(), mode='L')
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.data_type == 'train':
            return img, target, self.comp_labels[index]
        else:
            return img, target

    def __len__(self):
        if self.data_type == "train":
            return len(self.idx_train)
        else:
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
    def generate_multi_compl_labels(self):
        if self.data_type == "train" and not self.validate:
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

            # tensor = torch.stack([transform(self.data[i]) for i in range(0, len(self.idx_train))])
            tensor = torch.stack([transform(self.data[i]) for i in self.idx_train])
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
            model_simsiam.cuda()

            feature = []

            model_simsiam.eval()
            for x in dl:
                feature.append(F.normalize(model_simsiam(torch.cat(x).cuda())).cpu())
            feature = torch.cat(feature)
            self.feature = feature

            # Dist, Idx, neigh = get_kNN(feature.numpy(), 1025)
            # self.Idx = Idx

        K = max(self.targets) + 1
        T = (torch.ones(K,K) - torch.eye(K)) / (K-1)
        if self.noise > 0:
            T = (1-self.noise) * T + self.noise * torch.ones(K,K)/K

        y_oh = F.one_hot(torch.LongTensor(self.targets), K).float()
        g_cpu = torch.Generator()
        g_cpu.manual_seed(self.seed)
        comp = torch.multinomial(y_oh.matmul(torch.Tensor(T)), self.num_comp, generator=g_cpu)

        if self.data_type == "train":
            comp = comp[self.idx_train]
            if self.validate:
                comp = F.one_hot(comp.squeeze(),K).float()
                return comp

        self.comp = comp.squeeze()
        comp_oh = F.one_hot(comp,K).float().sum(dim=1)
        if self.num_neighbors == 0:
            return comp_oh

        smooth_comp = comp_oh.clone()

        sim_matrix = feature @ feature.T
        sim_matrix = (-(2-2*sim_matrix)).exp()
        sim_matrix.fill_diagonal_(0)

        val, ind = torch.topk(sim_matrix, self.num_neighbors, dim=-1)

        if self.weight == "distance":
            # Distance-based
            W = torch.zeros_like(sim_matrix).scatter_(1, ind, val)
        elif self.weight == "rank":
            # Rank-based
            W = torch.zeros_like(sim_matrix)
            for i in range(ind.size(1)):
                W.scatter_(1, ind[:,i:i+1], 1/(i+1))

        Z = W.sum(dim=1, keepdim=True)
        W = W / Z
 
        alpha = 0.9
        W = W.to_sparse()
        for i in tqdm(range(self.num_iter)):
            smooth_comp = alpha * W @ smooth_comp + (1-alpha) * comp_oh
 
        smooth_comp = smooth_comp / smooth_comp.sum(dim=1, keepdim=True)
 
        return smooth_comp
    

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