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
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip, RandAugment
from imb_cll.utils.autoaugment import AutoAugment, Cutout


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
        self.true_targets = torch.Tensor(data["ord_labels"]).view(-1).long()
        # self.targets = torch.Tensor(data["cl_labels"])[:, num_cl].long()
        self.targets = torch.Tensor(data["cl_labels"])[:, :num_cl].long()

        # self.targets = [labels[0] for labels in data["cl_labels"]]
        self.k_mean_targets = copy.deepcopy(self.true_targets)

        # self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        # self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        if self.data_type =="train":
            if self.imb_type is not None and self.imb_factor != 1:
                print("Imbalance ratio: {}".format(self.imb_factor))
                self.img_num_list, self.img_max = self.get_img_num_per_cls(self.num_classes, self.imb_type, self.imb_factor)
                self.gen_imbalanced_data(self.img_num_list)
            
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
        self.mean, self.std = [0.4914, 0.4822, 0.4465], [0.247,  0.2435, 0.2616]
        self.idx_train = len(self.data)
        self.root_dir = root
        self.img_max = 5000

        # if transform is None:
        if self.data_type== "train":
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
        setup_type=None,
        aug_type = None
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
        self.true_targets = torch.Tensor(data["ord_labels"]).view(-1).long()
        self.targets = torch.Tensor(data["cl_labels"])[:, :num_cl].long()
        if self.imb_type is not None and self.imb_factor != 1:
            self.targets = torch.Tensor(data["cl_labels"])[:, num_cl:].long()

        # self.targets = [labels[0] for labels in data["cl_labels"]]
        self.k_mean_targets = copy.deepcopy(self.true_targets)

        self.data = np.array(self.data)
        # self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        # self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        if self.data_type =="train":
            if self.imb_type is not None and self.imb_factor != 1:
                print("Imbalance ratio: {}".format(self.imb_factor))
                self.img_num_list, self.img_max = self.get_img_num_per_cls(self.num_classes, self.imb_type, self.imb_factor)
                self.gen_imbalanced_data(self.img_num_list)
            
            if max_train_samples: #limit the size of the training dataset to max_train_samples
                train_len = min(len(self.data), max_train_samples)
                self.data = self.data[:train_len]
                self.targets = self.targets[:train_len]

        self.transform = transform
        self.target_transform = target_transform
        self.mean, self.std = [0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]
        self.idx_train = len(self.data)
        self.root_dir = root
        self.img_max = 2500

        # if transform is None:
        if self.data_type== "train":
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