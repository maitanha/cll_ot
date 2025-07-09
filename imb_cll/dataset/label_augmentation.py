from PIL import Image
import os
import os.path
import numpy as np
from scipy import sparse
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision.models import resnet18
from torchvision.transforms import Compose, ToTensor, Normalize, Pad, RandomCrop, RandomHorizontalFlip, RandomErasing, ToPILImage, TrivialAugmentWide
import copy
# from sklearnex.neighbors import NearestNeighbors

from utils import _cifar100_to_cifar20
from tqdm.auto import tqdm

# def get_kNN(X, k):
#     neigh = NearestNeighbors(n_neighbors=k).fit(X)
#     Dist, Idx = neigh.kneighbors(X)
#     return Dist, Idx, neigh

class CLCIFAR10(VisionDataset):
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
            root="./data",
            train=None,
            validate=False,
            transform=None,
            target_transform=None,
            download=False,
            num_comp=1,
            num_neighbors=64,
            pretrain=None,
            noise=0,
            num_iter=100,
            weight=None,
            seed=None):

        super(CLCIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training, val, or test
        self.validate = validate
        self.pretrain = pretrain
        self.noise = noise
        self.seed = seed
        self.num_iter = num_iter
        self.weight = weight

        if seed is None:
            raise RuntimeError('Seed is not specified.')

        if not train in ["train", "val", "test"]:
            raise RuntimeError(f'Training split {train} is invalid.')

        if train =="train" and num_neighbors > 0 and not weight in ["rank", "distance"]:
            raise RuntimeError(f'Weighting method {weight} is invalid.')

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train in ("train", "val"):
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

        self.rng = np.random.default_rng(self.seed)
        self.idx = self.rng.permutation(len(self.targets))
        self.idx_train = self.idx[:45000]
        self.idx_val = self.idx[45000:]

        self.num_comp = num_comp
        self.num_neighbors = num_neighbors
        self.mean, self.std = [0.4914, 0.4822, 0.4465], [0.247,  0.2435, 0.2616]
        
        if self.train in ("train", "val"):
            self.comp_labels = self.generate_multi_compl_labels()

        if self.train == "train" and not validate:
            ol = []
            for i in range(45000):
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

        if self.train == "train" and not validate:
            self.transform=Compose([
                RandomHorizontalFlip(),
                RandomCrop(32, 4, padding_mode='reflect'),
                ToTensor(),
                Normalize(mean=self.mean, std=self.std),
            ])
        else:
            self.transform=Compose([
                ToTensor(),
                Normalize(mean=self.mean, std=self.std),
            ])

        self._load_meta()
        if self.train =="train":
            self.cl_labels = self.comp_labels

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
        if self.train == "train":
            img, target = self.data[self.idx_train[index]], self.targets[self.idx_train[index]]
        elif self.train == "val":
            img, target = self.data[self.idx_val[index]], self.targets[self.idx_val[index]]
        else:
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img_ori = self.transform(img)



        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.train == "train":
            return img_ori, target, self.comp_labels[index]
        elif self.train == "val":
            return img_ori, target, self.comp_labels[index]
        else:
            return img_ori, target

    def __len__(self):
        if self.train == "train":
            return len(self.idx_train)
        elif self.train == "val":
            return len(self.idx_val)
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
    def generate_multi_compl_labels(self):
        if self.train == "train" and not self.validate:
            model_simsiam = resnet18()
            model_simsiam.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model_simsiam.maxpool = nn.Identity()

            transform=Compose([
                ToTensor(),
                Normalize(mean=self.mean, std=self.std),
            ])

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

        if self.train == "train":
            comp = comp[self.idx_train]
            if self.validate:
                comp = F.one_hot(comp.squeeze(),K).float()
                return comp
        elif self.train =="val":
            comp = comp[self.idx_val].squeeze()
            comp = F.one_hot(comp,K).float()
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

        import pdb
        pdb.set_trace()
 
        return smooth_comp

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

class CLCIFAR20(CLCIFAR10):
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
            root="./data",
            train="train",
            validate=False,
            transform=None,
            target_transform=None,
            download=False,
            num_comp=1,
            num_neighbors=64,
            pretrain=None,
            noise=0,
            num_iter=100,
            weight=None,
            seed=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training, val, or test
        self.validate = validate
        self.pretrain = pretrain
        self.noise = noise
        self.seed = seed
        self.num_iter = num_iter
        self.weight = weight

        if seed is None:
            raise RuntimeError('Seed is not specified.')

        if not train in ["train", "val", "test"]:
            raise RuntimeError(f'Training split {train} is invalid.')

        if train =="train" and num_neighbors > 0 and not weight in ["rank", "distance"]:
            raise RuntimeError(f'Weighting method {weight} is invalid.')

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train in ("train", "val"):
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

        self.rng = np.random.default_rng(self.seed)
        self.idx = self.rng.permutation(len(self.targets))
        self.idx_train = self.idx[:45000]
        self.idx_val = self.idx[45000:]

        self.num_comp = num_comp
        self.num_neighbors = num_neighbors
        self.mean, self.std = [0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]
        
        if self.train in ("train", "val"):
            self.comp_labels = self.generate_multi_compl_labels()

        if self.train == "train" and not validate:
            K = 20
            T = (torch.ones(K,K) - torch.eye(K)) / (K-1)

            g_cpu = torch.Generator()
            g_cpu.manual_seed(self.seed)
            for i in range(45000):
                ol = self.targets[self.idx_train[i]]
                noise_cl = self.comp_labels[i,ol].item()

                ## Resample
                # comp_wo_noise = self.comp_labels[i].clone()
                # comp_wo_noise[ol] = 0.
                # comp_wo_noise = F.normalize(comp_wo_noise, dim=-1, p=1)
                # effective_cl_cnt = torch.special.entr(comp_wo_noise).sum().exp()

                # effective_cl_cnt = int(effective_cl_cnt.floor() + torch.bernoulli(effective_cl_cnt-effective_cl_cnt.floor(), generator=g_cpu))
                # sampling_comp = F.one_hot(torch.multinomial(T[ol], effective_cl_cnt, generator=g_cpu), K).sum(0) / effective_cl_cnt * (1-noise_cl)

                # sampling_comp[ol] = noise_cl
                # self.comp_labels[i] = sampling_comp

                ## Remove uniform
                # self.comp_labels[i] -= self.comp_labels[i].min()
                # self.comp_labels[i] /= self.comp_labels[i].sum()

                ## Adjust temperature
                # nonoise_cl = 1 - noise_cl
                # self.comp_labels[i,ol] = 0.
                # logit = self.comp_labels[i].clamp(1e-8, 1-1e-8).log()
                # self.comp_labels[i] = F.softmax(logit) * nonoise_cl
                # self.comp_labels[i,ol] = noise_cl

            ol = []
            for i in range(45000):
                ol.append(self.targets[self.idx_train[i]])
            ol = torch.LongTensor(ol)

            # orig_cl = F.nll_loss(-self.comp_labels, self.comp)
            # noise_cl = F.nll_loss(-self.comp_labels, ol)
            # res_cl = (1 - orig_cl - noise_cl) / (K-2)

            # # margin calculation
            # comp_except_noise = F.normalize(self.comp_labels.clone().scatter(1, ol.unsqueeze(1), 0), dim=-1, p=1)
            # ent = torch.special.entr(comp_except_noise).sum(-1).exp().mean()

            # least_cl = torch.min(self.comp_labels)[1]
            least_cl = (ol == torch.min(self.comp_labels, dim=-1)[1]).float().mean()
            # ideal_cl = T[ol]
            # l1 = (comp_except_noise - ideal_cl).abs().sum(-1).mean()

            # print(f"{least_cl:.4f} {ent:.4f} {orig_cl:.4f} {noise_cl:.4f} {res_cl:.4f} {l1:.4f}")
            print(f"{least_cl:.4f}")

        if self.train == "train" and not validate:
            self.transform=Compose([
                RandomHorizontalFlip(),
                RandomCrop(32, 4, padding_mode='reflect'),
                ToTensor(),
                Normalize(mean=self.mean, std=self.std),
            ])
        else:
            self.transform=Compose([
                ToTensor(),
                Normalize(mean=self.mean, std=self.std),
            ])

        self._load_meta()
        if self.train =="train":
            self.cl_labels = self.comp_labels
