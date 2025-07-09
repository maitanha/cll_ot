from torch.utils.data import Dataset
import pickle
import gdown
import os
import copy
import requests
import numpy as np
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from .base_dataset import BaseDataset
from sklearn.cluster import KMeans

from tqdm.auto import tqdm
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision.models import resnet18
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip

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

class PCLCIFAR10(Dataset, BaseDataset):
    """
    Complementary CIFAR-10 training Dataset containing 50000 images with 3 complementary labels
    and 1 ordinary label.
    """
    def __init__(
        self, 
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
        setup_type=None
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

        self.train = train
        self.validate = validate
        self.pretrain = pretrain
        self.seed = seed
        num_cl = 1
        
        if seed is None:
            raise RuntimeError('Seed is not specified.')
        
        if train == "train" and imb_factor > 0 and not imb_type in ["exp", "step"]:
            raise RuntimeError(f'Imb_type method {imb_type} is invalid.')
        
        dataset_path = f"{root}/clcifar10.pkl"
        if download and not os.path.exists(dataset_path):
            gdown.download(id="1uNLqmRUkHzZGiSsCtV2-fHoDbtKPnVt2", output=dataset_path)
        with open(dataset_path, "rb") as f:
            data = pickle.load(f)
        
        self.names = data["names"]
        self.data = data["images"]
        self.true_targets = torch.Tensor(data["ord_labels"]).view(-1)
        self.targets = torch.Tensor(data["cl_labels"])[:, :num_cl]
        # self.targets = [labels[0] for labels in data["cl_labels"]]
        self.k_mean_targets = copy.deepcopy(self.true_targets)

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        if self.data_type =="train":
            if self.imb_type is not None:
                print("Imbalance ratio: {}".format(self.imb_factor))
                # self.img_num_list, self.img_max = self.get_img_num_per_cls(self.num_classes, self.imb_type, self.imb_factor)
                # self.gen_imbalanced_data(self.img_num_list)
            
            if max_train_samples: #limit the size of the training dataset to max_train_samples
                train_len = min(len(self.data), max_train_samples)
                self.data = self.data[:train_len]
                self.targets = self.targets[:train_len]
            
            # if self.setup_type == "setup 1":
            #     self.gen_complementary_target()
            # elif self.setup_type == "setup 2":
            #     self.gen_bias_complementary_label()
        
        self.transform = transform
        self.target_transform = target_transform

        # if transform is None:
        if self.data_type== "train":
            if augment:
                self.transform = transforms.Compose(
                    [
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            [0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]
                        ),
                    ]
                )
            else:
                self.transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            [0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]
                        ),
                    ]
                )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]
                    ),
                ]
            )

        self.mean, self.std = [0.4914, 0.4822, 0.4465], [0.247,  0.2435, 0.2616]
        self.idx_train = len(self.data)
        self.root_dir = root
        self.img_max = 5000

        # self._load_meta()
        if self.data_type =="train":
            if self.kmean_cluster != 0:
                self.k_mean_targets = self.features_space()
                print("Done: K_Mean Cluster")

    def __len__(self):
        # return len(self.images)
        if self.data_type == "train":
            return len(self.data)
        else:
            return len(self.data)

    def __getitem__(self, index):

        if self.data_type == "train":
            img, targets, true_targets, k_mean_target = self.data[index], self.targets[index], self.true_targets[index], self.k_mean_targets[index]

        if self.data_type == "test":
            img, targets = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            targets = self.target_transform(targets)

        if self.data_type == "train":
            return img, targets, true_targets, k_mean_target, self.img_max
            # return img, targets, true_targets, k_mean_target
        else:
            return img, targets
    
    @torch.no_grad()
    def features_space(self):
        if self.data_type == "train":
            model_simsiam = resnet18()
            model_simsiam.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
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


class PCLCIFAR20(Dataset, BaseDataset):
    """
    Complementary CIFAR-20 training Dataset containing 50000 images with 3 complementary labels
    and 1 ordinary label.
    """

    def __init__(
        self, 
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
        setup_type=None
    ):
        self.root = root
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
        num_cl = 1

        if seed is None:
            raise RuntimeError('Seed is not specified.')
        
        if train == "train" and imb_factor > 0 and not imb_type in ["exp", "step"]:
            raise RuntimeError(f'Imb_type method {imb_type} is invalid.')
        
        dataset_path = f"{root}/clcifar20.pkl"
        if download and not os.path.exists(dataset_path):
            gdown.download(id="1PhZsyoi1dAHDGlmB4QIJvDHLf_JBsFeP", output=dataset_path)
        with open(dataset_path, "rb") as f:
            data = pickle.load(f)
        
        self.names = data["names"]
        self.data = data["images"]
        self.true_targets = torch.Tensor(data["ord_labels"]).view(-1)
        self.targets = torch.Tensor(data["cl_labels"])[:, :num_cl]
        # self.targets = [labels[0] for labels in data["cl_labels"]]
        self.k_mean_targets = copy.deepcopy(self.true_targets)

        self.targets = [_cifar100_to_cifar20(i) for i in self.targets]
        self.targets = torch.Tensor(self.targets)

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        if self.data_type =="train":
            if self.imb_type is not None:
                print("Imbalance ratio: {}".format(self.imb_factor))
                # self.img_num_list, self.img_max = self.get_img_num_per_cls(self.num_classes, self.imb_type, self.imb_factor)
                # self.gen_imbalanced_data(self.img_num_list)
            
            if max_train_samples: #limit the size of the training dataset to max_train_samples
                train_len = min(len(self.data), max_train_samples)
                self.data = self.data[:train_len]
                self.targets = self.targets[:train_len]
            
            # if self.setup_type == "setup 1":
            #     self.gen_complementary_target()
            # elif self.setup_type == "setup 2":
            #     self.gen_bias_complementary_label()

        self.transform = transform
        self.target_transform = target_transform

        # if transform is None:
        if self.data_type== "train":
            if augment:
                self.transform = transforms.Compose(
                    [
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            [0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]
                        ),
                    ]
                )
            else:
                self.transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            [0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]
                        ),
                    ]
                )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]
                    ),
                ]
            )

        self.mean, self.std = [0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]
        self.idx_train = len(self.data)
        self.root_dir = root
        self.img_max = 2500

        # self._load_meta()
        if self.data_type =="train":
            if self.kmean_cluster != 0:
                self.k_mean_targets = self.features_space()
                print("Done: K_Mean Cluster")

    def __len__(self):
        # return len(self.images)
        if self.data_type == "train":
            return len(self.data)
        else:
            return len(self.data)

    def __getitem__(self, index):

        if self.data_type == "train":
            img, targets, true_targets, k_mean_targets = self.data[index], self.targets[index], self.true_targets[index], self.k_mean_targets[index]

        if self.data_type == "test":
            img, targets = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            targets = self.target_transform(targets)

        if self.data_type == "train":
            return img, targets, true_targets, k_mean_targets, self.img_max
            # return img, targets, true_targets, k_mean_targets
        else:
            return img, targets

    @torch.no_grad()
    def features_space(self):
        if self.data_type == "train":
            model_simsiam = resnet18()
            model_simsiam.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
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