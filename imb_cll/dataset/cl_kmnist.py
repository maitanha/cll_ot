import torchvision
from .base_dataset import BaseDataset
import torchvision.transforms as transforms


class CLKMNIST(BaseDataset, torchvision.datasets.KMNIST):
    def __init__(
        self,
        root="./data/kmnist",
        train=True,
        transform=transforms.ToTensor(),
        target_transform=None,
        download=True,
        max_train_samples=None,
        multi_label=False,
    ):
        super(CLKMNIST, self).__init__(
            root, train, transform, target_transform, download
        )
        self.num_classes = 10
        self.input_dim = 1 * 28 * 28
        self.multi_label = multi_label
        if train:
            if max_train_samples:
                train_len = min(len(self.data), max_train_samples)
                self.data = self.data[:train_len]
                self.targets = self.targets[:train_len]
            self.gen_complementary_target()