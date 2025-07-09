import numpy as np
import copy
from PIL import Image
import torchvision
from .base_dataset import BaseDataset
import torchvision.transforms as transforms
from imb_cll.utils.autoaugment import AutoAugment, Cutout


class CLCIFAR10(BaseDataset, torchvision.datasets.CIFAR10):
    def __init__(
        self,
        root="./data/cifar10",
        data_type=None,
        train=True,
        transform=None,
        target_transform=None,
        download=True,
        max_train_samples=None,
        multi_label=False,
        augment=False,
        imb_type=None,
        imb_factor=1.0,
    ):
        self.num_classes = 10
        self.input_dim = 3 * 32 * 32
        self.multi_label = multi_label
        self.data_type = data_type
        if transform is None:
            if train:
                if augment:
                    transform = transforms.Compose(
                        [
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomCrop(32, padding=4),
                            # AutoAugment(),
                            # Cutout(),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                [0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]
                            ),
                        ]
                    )
                else:
                    transform = transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Normalize(
                                [0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]
                            ),
                        ]
                    )
            else:
                transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            [0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]
                        ),
                    ]
                )
        super(CLCIFAR10, self).__init__(
            root, train, transform, target_transform, download)
        # np.random.seed(rand_number)

        if train:
            if imb_type is not None:
                self.img_num_list = self.get_img_num_per_cls(self.num_classes, imb_type, imb_factor)
                self.gen_imbalanced_data(self.img_num_list)

            if max_train_samples: #limit the size of the training dataset to max_train_samples
                train_len = min(len(self.data), max_train_samples)
                self.data = self.data[:train_len]
                self.targets = self.targets[:train_len]
            self.gen_complementary_target()

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, true_target) where target is index of the target class.
        """ 
        if self.data_type == "train":
            img, target, true_target = self.data[index], self.targets[index], self.true_targets[index]
        elif self.data_type == "test":
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.data_type == "train":
            return img, target, true_target
        else:
            return img, target
        