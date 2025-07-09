# ICM: Data Augmentation for Complementary-Label Learning.

## Overview
* In this example, we can run a collection of benchmark for research purpose.

## How to use
### Environment
* Python version: 3.10
* GPU: Tesla V100-SXM2

### Quick Start: ICM for CIFAR10 Training with balance scenario
* To reproduce SCL-NL training with balance scenario, utilizing ICM method
```
python train.py --algo=scl-nl --dataset_name CIFAR10 --model resnet18 --imb_type exp --imb_factor 1 --mixup true --alpha 0.2 --k_cluster 50 --icm true --data_aug true
```

* To reproduce SCL-NL training with balance scenario, utilizing MICM method
```
python train.py --algo=scl-nl --dataset_name CIFAR10 --model resnet18 --imb_type exp --imb_factor 1 --mixup true --alpha 0.2 --k_cluster 50 --micm true --data_aug true
```

### Quick Start: ICM for CIFAR10 Training with imbalanced scenario with Setup 1.
* To reproduce SCL-NL training with imbalance scenario, utilizing ICM method
```
python train.py --algo=scl-nl --dataset_name CIFAR10 --model resnet18 --imb_type exp --imb_factor 0.1 --mixup true --alpha 0.2 --k_cluster 50 --icm true --data_aug true --setup_type "setup 1"
```

* To reproduce SCL-NL training with imbalance scenario, utilizing MICM method
```
python train.py --algo=scl-nl --dataset_name CIFAR10 --model resnet18 --imb_type exp --imb_factor 0.1 --mixup true --alpha 0.2 --k_cluster 50 --micm true --data_aug true --setup_type "setup 1"
```

* As explanation in the paper, Setup 1: the imbalanced CLL comes from ordinary itself.

### Quick Start: ICM for CIFAR10 Training with imbalanced scenario with Setup 2.
* To reproduce SCL-NL training with imbalance scenario, utilizing ICM method
```
python train.py --algo=scl-nl --dataset_name CIFAR10 --model resnet18 --imb_type exp --imb_factor 1 --mixup true --alpha 0.2 --k_cluster 50 --icm true --data_aug true --setup_type "setup 2" --transition_bias 10
```

* To reproduce SCL-NL training with balance scenario, utilizing MICM method
```
python train.py --algo=scl-nl --dataset_name CIFAR10 --model resnet18 --imb_type exp --imb_factor 1 --mixup true --alpha 0.2 --k_cluster 50 --micm true --data_aug true --setup_type "setup 2" --transition_bias 10
```

* Setup 2: The imbalanced CLL is from biased transition matrix.


### Quick Start: ICM for CIFAR10 Training with imbalanced scenario with Setup 3.
* To reproduce SCL-NL training with imbalance scenario, utilizing ICM method
```
python train.py --algo=scl-nl --dataset_name CIFAR10 --model resnet18 --imb_type exp --imb_factor 0.1 --mixup true --alpha 0.2 --k_cluster 50 --icm true --data_aug true --setup_type "setup 2" --transition_bias 10
```

* To reproduce SCL-NL training with balance scenario, utilizing MICM method
```
python train.py --algo=scl-nl --dataset_name CIFAR10 --model resnet18 --imb_type exp --imb_factor 0.1 --mixup true --alpha 0.2 --k_cluster 50 --micm true --data_aug true --setup_type "setup 2" --transition_bias 10
```

* Setup 3: The imbalanced CLL is a combined of imbalanced ordinary dataset and biased transition matrix. Therefore, imb_factor 0.1 and setup_type "setup 2"

|  Parameter | Description|
|:----------:|:----------:|
| `--config` | Path to config file (specify by different dataset)|
|`--algorithm`| `SCL-NL`, `FWD`, `LW`, `SCL_EXP`|
|`--model`    | `resnet18`|
|`--dataset`    | `CIFAR10`, `CIFAR20`, `PCLCIFAR10`, `PCLCIFAR20`, `MNIST`, `KMNIST`, `FashionNIST`|
|`--imb_factor`| `1`, `0.1`, `0.02`, `0.01`|
|`--imb_exp`| `exp`, `step`|
|`--k_cluster`| The number of clustering|
