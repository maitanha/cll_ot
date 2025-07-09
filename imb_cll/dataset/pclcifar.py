from torch.utils.data import Dataset
import pickle
import os
import requests
import numpy as np
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from tqdm.auto import tqdm
from torchvision.models import resnet18

class PCLCIFAR10(Dataset):
    """
    Complementary CIFAR-10 training Dataset containing 50000 images with 3 complementary labels
    and 1 ordinary label.
    """
    def __init__(
        self, 
        root="./data", 
        train=None, 
        transform=None, 
        target_transform=None, 
        max_train_samples=None,
        multi_label=False,
        num_neighbors=256,
        pretrain=None,
        num_iter=100,
        weight=None,
        seed=None,
    ):
        if seed is None:
            raise RuntimeError('Seed is not specified.')

        if not train in ["train", "val", "test"]:
            raise RuntimeError(f'Training split {train} is invalid.')

        if train =="train" and num_neighbors > 0 and not weight in ["rank", "distance"]:
            raise RuntimeError(f'Weighting method {weight} is invalid.')

        if not os.path.exists(root):
            print("Creating folder " + root)
            os.mkdir(root)
        if not os.path.exists(root + "/pcl50000-folder/"):
            print("Creating folder " + root + "/pcl50000-folder/")
            os.mkdir(root + "/pcl50000-folder/")

        print("checking file...(0/2)")
        if not os.path.exists(root + "/pcl50000-folder/" + "pcl50000.pkl"):
            print("Downloading datasets")
            response = requests.get(
                "https://cll-data-collect.s3.us-west-2.amazonaws.com/pcl50000.pkl"
            )
            open(root + "/pcl50000-folder/pcl50000.pkl", "wb").write(
                response.content
            )
        print("checking file...(1/2)")

        if not os.path.exists(root + "/pcl50000-folder/test-data.pkl"):
            print("Downloading datasets")
            response = requests.get(
                "https://cll-data-collect.s3.us-west-2.amazonaws.com/cl-10000-folder/test-data.pkl"
            )
            open(root + "/pcl50000-folder/test-data.pkl", "wb").write(
                response.content
            )
        print("checking file...done")

        if train in ["train", "val"]:
            data = pickle.load(open(root + "/pcl50000-folder/pcl50000.pkl", "rb"))
        else:
            data = pickle.load(open(root + "/pcl50000-folder/test-data.pkl", "rb"))

        self.names = data["names"]
        self.images = data["images"]

        self.images = np.vstack(self.images).reshape(-1, 3, 32, 32)
        self.images = self.images.transpose((0, 2, 3, 1))  # convert to HWC

        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.idx = self.rng.permutation(len(self.images))
        self.idx_train = self.idx[:45000]
        self.idx_val = self.idx[45000:]

        self.ord_labels = data["ord_labels"]
        if train in ["train", "val"]:
            self.cl_labels = data["cl_labels"]
        self.root_dir = root
        # if transform is None:
        if train == "train":
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
        self.target_transform = target_transform
        self.train = train
        self.multi_label = multi_label
        self.input_dim = 3 * 32 * 32
        self.num_classes = 10
        self.mean, self.std = [0.4914, 0.4822, 0.4465], [0.247,  0.2435, 0.2616]
        self.num_neighbors = num_neighbors
        self.num_iter = num_iter
        self.pretrain = pretrain
        self.weight = weight

        # if self.train and max_train_samples:
        #     train_len = min(len(self.images), max_train_samples)
        #     self.images = self.images[:train_len]
        #     self.cl_labels = self.cl_labels[:train_len]

        if self.train == "train":
            self.cl_labels = self.augment_with_knn()
            # self.cl_labels = self.resample_cl()

            # Force denoise
            # for i, index in enumerate(self.idx_train):
            #     ol = self.ord_labels[index]
            #     self.cl_labels[i, ol] = 0

            # self.cl_labels = self.cl_labels / self.cl_labels.sum(dim=1, keepdim=True)

            least_cl = (torch.LongTensor(self.ord_labels)[self.idx_train] == torch.min(self.cl_labels, dim=-1)[1]).float().mean()
            
            print(f"{least_cl:.4f}")
        elif self.train == "val":
            self.cl_labels = F.one_hot(torch.LongTensor(self.cl_labels)[self.idx_val]).sum(1) / 3

    def resample_cl(self):
        T = torch.zeros(10,10)
        for index in self.idx_train:
            ol = self.ord_labels[index]
            cls = self.cl_labels[index]
            for cl in cls:
                T[ol,cl] += 1

        # for i in range(10):
        #     T[i,i] = 0
        T = T / T.sum(dim=1, keepdim=True)
        self.T = T
        comp = torch.multinomial(F.one_hot(torch.LongTensor(self.ord_labels),10)[self.idx_train].float().matmul(T), 1)

        return F.one_hot(comp,10).sum(1).float() / 1

    @torch.no_grad()
    def augment_with_knn(self):
        self.cl_labels = F.one_hot(torch.LongTensor(self.cl_labels)).sum(1) / 3
        self.cl_labels = self.cl_labels[self.idx_train]

        model_simsiam = resnet18()
        model_simsiam.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model_simsiam.maxpool = nn.Identity()

        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

        tensor = torch.stack([transform(self.images[idx]) for idx in self.idx_train])

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
        for x in tqdm(dl):
            feature.append(F.normalize(model_simsiam(torch.cat(x).cuda())).cpu())
        feature = torch.cat(feature)
        self.feature = feature

        if self.num_neighbors == 0:
            return self.cl_labels

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
        smooth_comp = self.cl_labels.clone()
        for i in tqdm(range(self.num_iter)):
            smooth_comp = alpha * W @ smooth_comp + (1-alpha) * self.cl_labels

        smooth_comp = smooth_comp / smooth_comp.sum(dim=1, keepdim=True)

        # min_val, min_ind = smooth_comp.min(dim=-1)
        # smooth_comp -= min_val.unsqueeze(-1)
        # smooth_comp = smooth_comp / smooth_comp.sum(dim=1, keepdim=True)

        return smooth_comp

    def __len__(self):
        # return len(self.images)
        if self.train == "train":
            return len(self.idx_train)
        elif self.train == "val":
            return len(self.idx_val)
        else:
            return len(self.images)

    def __getitem__(self, index):
        if self.train == "train":
            img, target = self.images[self.idx_train[index]], self.ord_labels[self.idx_train[index]]
        elif self.train == "val":
            img, target = self.images[self.idx_val[index]], self.ord_labels[self.idx_val[index]]
        else:
            img, target = self.images[index], self.ord_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.train == "train":
            return img, target, self.cl_labels[index]
        elif self.train == "val":
            return img, target, self.cl_labels[index]
        else:
            return img, target


class PCLCIFAR20(Dataset):
    """
    Complementary CIFAR-20 training Dataset containing 50000 images with 3 complementary labels
    and 1 ordinary label.
    """

    def __init__(
        self,
        root="./data",
        train="train",
        transform=None,
        target_transform=None,
        max_train_samples=None,
        multi_label=False,
        num_neighbors=256,
        pretrain=None,
        num_iter=100,
        weight=None,
        seed=None,
    ):
        if seed is None:
            raise RuntimeError('Seed is not specified.')

        if not train in ["train", "val", "test"]:
            raise RuntimeError(f'Training split {train} is invalid.')

        if train =="train" and num_neighbors > 0 and not weight in ["rank", "distance"]:
            raise RuntimeError(f'Weighting method {weight} is invalid.')

        if not os.path.exists(root + "/pcl_cifar20-folder/"):
            print("Creating folder " + root + "/pcl_cifar20-folder/")
            os.makedirs(
                root + "/pcl_cifar20-folder/",
            )

        print("checking file...(0/2)")
        if not os.path.exists(root + "/pcl_cifar20-folder/pcl_cifar20_train.pkl"):
            print("Downloading datasets")
            response = requests.get(
                "https://cll-data-collect.s3.us-west-2.amazonaws.com/cifar20/pcl_cifar20_train.pkl"
            )
            open(root + "/pcl_cifar20-folder/pcl_cifar20_train.pkl", "wb").write(
                response.content
            )
        print("checking file...(1/2)")

        if not os.path.exists(root + "/pcl_cifar20-folder/pcl_cifar20_test.pkl"):
            print("Downloading datasets")
            response = requests.get(
                "https://cll-data-collect.s3.us-west-2.amazonaws.com/cifar20/pcl_cifar20_test.pkl"
            )
            open(root + "/pcl_cifar20-folder/pcl_cifar20_test.pkl", "wb").write(
                response.content
            )
        print("checking file...done")

        if train in ["train", "val"]:
            data = pickle.load(
                open(root + "/pcl_cifar20-folder/pcl_cifar20_train.pkl", "rb")
            )
        else:
            data = pickle.load(
                open(root + "/pcl_cifar20-folder/pcl_cifar20_test.pkl", "rb")
            )

        self.names = data["filenames"]
        # self.images = []

        # for i in range(len(data["data"])):
        #     self.images.append(np.resize(data["data"][i], (3,32,32)).transpose(1,2,0))
        self.images = data["data"]

        self.images = np.vstack(self.images).reshape(-1, 3, 32, 32)
        self.images = self.images.transpose((0, 2, 3, 1))  # convert to HWC

        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.idx = self.rng.permutation(len(self.images))
        self.idx_train = self.idx[:45000]
        self.idx_val = self.idx[45000:]

        self.ord_labels = data["ord_labels"]
        if train in ["train", "val"]:
            self.cl_labels = data["cl_labels"]
        self.root_dir = root
        if train == "train":
            self.transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]
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
        self.target_transform = target_transform
        self.train = train
        self.multi_label = multi_label
        self.input_dim = 3 * 32 * 32
        self.num_classes = 20
        self.mean, self.std = [0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]

        self.num_neighbors = num_neighbors
        self.num_iter = num_iter
        self.pretrain = pretrain
        self.weight = weight

        # if self.train and max_train_samples:
        #     train_len = min(len(self.images), max_train_samples)
        #     self.images = self.images[:train_len]
        #     self.cl_labels = self.cl_labels[:train_len]

        if self.train == "train":
            self.cl_labels = self.augment_with_knn()

            # Force denoise
            # for i, index in enumerate(self.idx_train):
            #     ol = self.ord_labels[index]
            #     self.cl_labels[i, ol] = 0

            least_cl = (torch.LongTensor(self.ord_labels)[self.idx_train] == torch.min(self.cl_labels, dim=-1)[1]).float().mean()
            print(f"{least_cl:.4f}")
        elif self.train == "val":
            self.cl_labels = F.one_hot(torch.LongTensor(self.cl_labels)[self.idx_val]).sum(1) / 3

    @torch.no_grad()
    def augment_with_knn(self):
        self.cl_labels = F.one_hot(torch.LongTensor(self.cl_labels)).sum(1) / 3
        self.cl_labels = self.cl_labels[self.idx_train]

        model_simsiam = resnet18()
        model_simsiam.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model_simsiam.maxpool = nn.Identity()

        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

        tensor = torch.stack([transform(self.images[idx]) for idx in self.idx_train])

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
        for x in tqdm(dl):
            feature.append(F.normalize(model_simsiam(torch.cat(x).cuda())).cpu())
        feature = torch.cat(feature)
        self.feature = feature

        if self.num_neighbors == 0:
            return self.cl_labels

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

        # Random walk symmetrization
        Z = W.sum(dim=1, keepdim=True)
        W = W / Z
        # Diffusion symmetrization
        # Z = W.sum(dim=1, keepdim=True).sqrt()
        # W = W / Z / Z.T

        alpha = 0.9
        W = W.to_sparse()
        smooth_comp = self.cl_labels.clone()
        for i in tqdm(range(self.num_iter)):
            smooth_comp = alpha * W @ smooth_comp + (1-alpha) * self.cl_labels

        smooth_comp = smooth_comp / smooth_comp.sum(dim=1, keepdim=True)

        # min_val, min_ind = smooth_comp.min(dim=-1)
        # smooth_comp -= min_val.unsqueeze(-1)
        # smooth_comp = smooth_comp / smooth_comp.sum(dim=1, keepdim=True)

        return smooth_comp

    def __len__(self):
        # return len(self.images)
        if self.train == "train":
            return len(self.idx_train)
        elif self.train == "val":
            return len(self.idx_val)
        else:
            return len(self.images)

    def __getitem__(self, index):
        if self.train == "train":
            img, target = self.images[self.idx_train[index]], self.ord_labels[self.idx_train[index]]
        elif self.train == "val":
            img, target = self.images[self.idx_val[index]], self.ord_labels[self.idx_val[index]]
        else:
            img, target = self.images[index], self.ord_labels[index]

        img = Image.fromarray(img)

        # if self.train:
        #     # One hot encoding
        #     # if self.multi_label:
        #     #     target = torch.tensor(self.cl_labels[idx])
        #     # else:
        #     #     target = torch.tensor(self.cl_labels[idx][0])
        #     target = self.cl_labels[idx]
        #     ol = self.ord_labels[idx]
        # else:
        #     target = self.ord_labels[idx]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.train == "train":
            return img, target, self.cl_labels[index]
        elif self.train == "val":
            return img, target, self.cl_labels[index]
        else:
            return img, target
