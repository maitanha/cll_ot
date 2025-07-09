from PIL import Image
import os
import os.path
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision.models import resnet18
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip, RandAugment
from .base_dataset import BaseDataset
from sklearn.cluster import KMeans
from imb_cll.utils.autoaugment import AutoAugment, Cutout

def _cifar100_to_cifar20(target):
    # obtained from cifar_test script
    _dict = {
        0: 4,
        1: 1,
        2: 14,
        3: 8,
        4: 0,
        5: 6,
        6: 7,
        7: 7,
        8: 18,
        9: 3,
        10: 3,
        11: 14,
        12: 9,
        13: 18,
        14: 7,
        15: 11,
        16: 3,
        17: 9,
        18: 7,
        19: 11,
        20: 6,
        21: 11,
        22: 5,
        23: 10,
        24: 7,
        25: 6,
        26: 13,
        27: 15,
        28: 3,
        29: 15,
        30: 0,
        31: 11,
        32: 1,
        33: 10,
        34: 12,
        35: 14,
        36: 16,
        37: 9,
        38: 11,
        39: 5,
        40: 5,
        41: 19,
        42: 8,
        43: 8,
        44: 15,
        45: 13,
        46: 14,
        47: 17,
        48: 18,
        49: 10,
        50: 16,
        51: 4,
        52: 17,
        53: 4,
        54: 2,
        55: 0,
        56: 17,
        57: 4,
        58: 18,
        59: 17,
        60: 10,
        61: 3,
        62: 2,
        63: 12,
        64: 12,
        65: 16,
        66: 12,
        67: 1,
        68: 9,
        69: 19,
        70: 2,
        71: 10,
        72: 0,
        73: 1,
        74: 16,
        75: 12,
        76: 9,
        77: 13,
        78: 15,
        79: 13,
        80: 16,
        81: 19,
        82: 2,
        83: 4,
        84: 6,
        85: 19,
        86: 5,
        87: 5,
        88: 8,
        89: 19,
        90: 18,
        91: 1,
        92: 2,
        93: 15,
        94: 6,
        95: 0,
        96: 17,
        97: 8,
        98: 14,
        99: 13,
    }

    return _dict[target]

class CLCIFAR10(VisionDataset, BaseDataset):
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

    def __init__(self,
        root=None,
        train=True,
        data_type=None,
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
        setup_type=None,
        aug_type = None
    ):
        self.root = root
        self.data_type = data_type
        self.num_classes = 10
        self.input_dim = 3 * 32 * 32
        self.multi_label = multi_label
        self.input_dataset = input_dataset
        self.imb_type = imb_type
        self.imb_factor = imb_factor
        self.kmean_cluster = kmean_cluster # Number of clustering with K mean method.
        self.transition_bias = transition_bias
        self.setup_type = setup_type

        super(CLCIFAR10, self).__init__(
            root, train, transform, target_transform)
        
        self.train = train
        self.validate = validate
        self.pretrain = pretrain
        self.seed = seed

        if seed is None:
            raise RuntimeError('Seed is not specified.')

        if self.data_type == "train" and imb_factor > 0 and not imb_type in ["exp", "step"]:
            raise RuntimeError(f'Imb_type method {imb_type} is invalid.')
        
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.data_type in ("train", "val"):
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        if self.data_type =="train":
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

            # if self.setup_type == "transition_bias":
            #     self.gen_bias_complementary_label()
            # elif self.setup_type == "ordinary_imb":
            #     self.gen_complementary_target()
        
        # self.rng = np.random.default_rng(self.seed)
        # self.idx = self.rng.permutation(len(self.data))

        self.idx_train = len(self.data)
        # print("The range of index {}".format(self.idx_train[:10]))

        self.mean, self.std = [0.4914, 0.4822, 0.4465], [0.247,  0.2435, 0.2616]
        # self.mean, self.std = 0.1307, 0.3081

        # Define a custom transform to convert images to grayscale
        class GrayscaleTransform(object):
            def __call__(self, img):
                gray_img = img.convert('L')  # Convert to grayscale using 'L' mode
                return gray_img

        if self.data_type == "train" and not validate:
            if augment:
                if aug_type == "randaug":
                    self.transform=Compose([
                        # GrayscaleTransform(),  # Convert to grayscale
                        RandAugment(3, 5),
                        RandomHorizontalFlip(),
                        RandomCrop(32, 4, padding_mode='reflect'),
                        ToTensor(),
                        Normalize(mean=self.mean, std=self.std),
                    ])
                elif aug_type == "autoaug":
                    self.transform=Compose([
                        # GrayscaleTransform(),  # Convert to grayscale
                        RandomHorizontalFlip(),
                        RandomCrop(32, 4, padding_mode='reflect'),
                        AutoAugment(),
                        ToTensor(),
                        Normalize(mean=self.mean, std=self.std),
                    ])
                elif aug_type == "cutout":
                    self.transform=Compose([
                        # GrayscaleTransform(),  # Convert to grayscale
                        RandomHorizontalFlip(),
                        RandomCrop(32, 4, padding_mode='reflect'),
                        Cutout(),
                        ToTensor(),
                        Normalize(mean=self.mean, std=self.std),
                    ])
                elif aug_type == "flipflop":
                    self.transform=Compose([
                        # GrayscaleTransform(),  # Convert to grayscale
                        RandomHorizontalFlip(),
                        RandomCrop(32, 4, padding_mode='reflect'),
                        ToTensor(),
                        Normalize(mean=self.mean, std=self.std),
                    ])
            else:
                self.transform=Compose([
                # GrayscaleTransform(),  # Convert to grayscale
                ToTensor(),
                Normalize(mean=self.mean, std=self.std),
            ])

        else:
            self.transform=Compose([
                # GrayscaleTransform(),  # Convert to grayscale
                ToTensor(),
                Normalize(mean=self.mean, std=self.std),
            ])

        self._load_meta()
        if self.data_type =="train":
            if self.kmean_cluster != 0:
                self.k_mean_targets = self.features_space()
                print("Done: K_Mean Cluster")

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
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
        if self.data_type == "train":
            img, targets, true_targets, k_mean_targets = self.data[index], self.targets[index], self.true_targets[index], self.k_mean_targets[index]

        if self.data_type == "test":
            img, targets = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            targets = self.target_transform(targets)

        if self.data_type == "train":
            return img, targets, true_targets, k_mean_targets, self.img_max
        else:
            return img, targets
    
    def __len__(self):
        if self.data_type == "train":
            return len(self.data)
        else:
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
            ###NOTED: Need to create imbalanced dataset first and get the idx of training
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
    

class CLCIFAR100(CLCIFAR10):
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


class CLCIFAR20(CLCIFAR100):
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

    def __init__(self,
        root= None,
        train=True,
        data_type=None,
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
        setup_type=None,
        aug_type = None
    ):
        self.data_type = data_type
        self.num_classes = 20
        self.input_dim = 3 * 32 * 32
        self.multi_label = multi_label
        self.input_dataset = input_dataset
        self.imb_type = imb_type
        self.imb_factor = imb_factor
        self.kmean_cluster = kmean_cluster # Number of clustering with K mean method.
        self.transition_bias = transition_bias
        self.setup_type = setup_type

        self.train = train
        self.validate = validate
        self.pretrain = pretrain
        self.seed = seed
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        if seed is None:
            raise RuntimeError('Seed is not specified.')

        if self.data_type == "train" and imb_factor > 0 and not imb_type in ["exp", "step"]:
            raise RuntimeError(f'Imb_type method {imb_type} is invalid.')
        
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.data_type in ("train", "val"):
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
        
        self.targets = [_cifar100_to_cifar20(l) for l in self.targets]

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        if self.data_type =="train":
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
        self.mean, self.std = [0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]
        # self.mean, self.std = 0.1307, 0.3081

        # Define a custom transform to convert images to grayscale
        class GrayscaleTransform(object):
            def __call__(self, img):
                gray_img = img.convert('L')  # Convert to grayscale using 'L' mode
                return gray_img

        if self.data_type == "train" and not validate:
            if augment:
                if aug_type == "randaug":
                    self.transform=Compose([
                        # GrayscaleTransform(),  # Convert to grayscale
                        RandAugment(3, 5),
                        RandomHorizontalFlip(),
                        RandomCrop(32, 4, padding_mode='reflect'),
                        ToTensor(),
                        Normalize(mean=self.mean, std=self.std),
                    ])
                elif aug_type == "autoaug":
                    self.transform=Compose([
                        # GrayscaleTransform(),  # Convert to grayscale
                        RandomHorizontalFlip(),
                        RandomCrop(32, 4, padding_mode='reflect'),
                        AutoAugment(),
                        ToTensor(),
                        Normalize(mean=self.mean, std=self.std),
                    ])
                elif aug_type == "cutout":
                    self.transform=Compose([
                        # GrayscaleTransform(),  # Convert to grayscale
                        RandomHorizontalFlip(),
                        RandomCrop(32, 4, padding_mode='reflect'),
                        Cutout(),
                        ToTensor(),
                        Normalize(mean=self.mean, std=self.std),
                    ])
                elif aug_type == "flipflop":
                    self.transform=Compose([
                        # GrayscaleTransform(),  # Convert to grayscale
                        RandomHorizontalFlip(),
                        RandomCrop(32, 4, padding_mode='reflect'),
                        ToTensor(),
                        Normalize(mean=self.mean, std=self.std),
                    ])
            else:
                self.transform=Compose([
                # GrayscaleTransform(),  # Convert to grayscale
                ToTensor(),
                Normalize(mean=self.mean, std=self.std),
            ])

        else:
            self.transform=Compose([
                # GrayscaleTransform(),  # Convert to grayscale
                ToTensor(),
                Normalize(mean=self.mean, std=self.std),
            ])

        self._load_meta()
        if self.data_type =="train":
            if self.kmean_cluster != 0:
                self.k_mean_targets = self.features_space()
                print("Done: K_Mean Cluster")