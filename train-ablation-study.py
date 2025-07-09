import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader
from imb_cll.dataset.dataset import prepare_cluster_dataset, prepare_neighbour_dataset
from imb_cll.utils.utils import AverageMeter, validate, compute_metrics_and_record, weighting_calculation, num_img_per_class, adjust_learning_rate, get_dataset_T
from imb_cll.utils.metrics import accuracy
from imb_cll.utils.cl_augmentation import mixup_cl_data, icm_data, mixup_data, mixup_criterion
from imb_cll.models.models import get_modified_resnet18, get_resnet18
from imb_cll.models.basemodels import Linear, MLP

num_workers = 4
device = "cuda"

def train_icm(args):
    dataset_name = args.dataset_name
    algo = args.algo
    model = args.model
    lr = args.lr
    weight_decay = args.weight_decay
    seed = args.seed
    alpha = args.alpha
    data_aug = True if args.data_aug.lower()=="true" else False
    aug_type = args.aug_type
    mixup = True if args.mixup.lower()=="true" else False
    mamix_intra_class = True if args.mamix_intra_class.lower()=="true" else False
    icm = True if args.icm.lower()=="true" else False
    micm = True if args.micm.lower()=="true" else False
    four_images_intra_class = True if args.four_images_intra_class.lower()=="true" else False
    cl_aug = True if args.cl_aug.lower()=="true" else False
    k_cluster = args.k_cluster
    mamix_ratio = args.mamix_ratio
    warm_epoch = args.warm_epoch
    epochs = args.n_epoch
    input_dataset = args.dataset_name

    eval_n_epoch = args.evaluate_step
    batch_size = args.batch_size
    n_weight = args.weighting
    imb_factor = args.imb_factor
    transition_bias = args.transition_bias
    setup_type = args.setup_type
    imb_type = args.imb_type
    best_acc1 = 0.
    mixup_noisy_error = 0
    cls_num_list = []
    k_mean_targets = []

    np.random.seed(seed)
    torch.manual_seed(seed)

    if data_aug:
        print("Use data augmentation.")

    if icm:
        print("Use complementary mixup intra class")
    elif cl_aug:
        print("Use mixup noise-free")

    weights, pretrain = weighting_calculation(input_dataset, imb_factor, n_weight)

    print("Use prepare_cluster_dataset")
    train_data = "train"
    trainset, input_dim, num_classes = prepare_cluster_dataset(input_dataset=input_dataset, data_type=train_data, kmean_cluster=k_cluster, max_train_samples=None, multi_label=False, 
                                    augment=data_aug, imb_type=imb_type, imb_factor=imb_factor, pretrain=pretrain, transition_bias=transition_bias, setup_type=setup_type, aug_type=aug_type)
    test_data = "test"
    testset, input_dim, num_classes = prepare_cluster_dataset(input_dataset=input_dataset, data_type=test_data, kmean_cluster=k_cluster, max_train_samples=None, multi_label=False, 
                                    augment=data_aug, imb_type=imb_type, imb_factor=imb_factor, pretrain=pretrain, transition_bias=transition_bias, setup_type=setup_type)
    

    dataset_T, class_count = get_dataset_T(trainset, num_classes)
    # Set Q for forward algorithm
    if algo in ["fwd-u", "ure-ga-u"]:
        Q = torch.full([num_classes, num_classes], 1/(num_classes-1), device=device)
        for i in range(num_classes):
            Q[i][i] = 0
    elif algo in ["fwd-r", "ure-ga"]:
        # Print the complementary label distribution T
        dataset_T = torch.tensor(dataset_T, dtype=torch.float).to(device)
        Q = dataset_T
    elif algo == "fwd-int":
        # Print the complementary label distribution T
        dataset_T = torch.tensor(dataset_T, dtype=torch.float).to(device)
        U = np.full([num_classes, num_classes], 1/(num_classes-1))
        for i in range(num_classes):
            U[i][i] = 0
        alpha_Q = 0.0
        dataset_T, class_count = get_dataset_T(trainset, num_classes)
        Q = torch.tensor(alpha_Q * U + (1-alpha_Q) * dataset_T).to(device).float()
        dataset_T = torch.tensor(dataset_T, dtype=torch.float).to(device)

    class_count = torch.tensor(class_count, dtype=torch.float).to(device)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

    if args.model == "resnet18":
        model = get_resnet18(num_classes, input_dataset).to(device)
    elif args.model == "m-resnet18":
        model = get_modified_resnet18(num_classes, input_dataset).to(device)
    elif args.model == "mlp":
        model = MLP(input_dim=input_dim,hidden_dim=args.hidden_dim,num_classes=num_classes).to(device)
    elif args.model == "linear":
        model = Linear(input_dim=input_dim,num_classes=num_classes).to(device)
    else:
        raise NotImplementedError

    for epoch in range(0, epochs):
        # learning_rate = adjust_learning_rate(epochs, epoch, lr)
        learning_rate = lr
        training_loss = 0.0
        model.train()
        
        weights, pretrain = weighting_calculation(input_dataset, imb_factor, n_weight)
        
        if epoch >= warm_epoch:
            weights = weights
            print("=> Weighting per class: {}".format(weights))
        else:
            weights = weights ** 0
            print("=> Weighting per class: {}".format(weights))

        weights = weights.to(device)
        print("Alpha value for generating Lambda with Dirichlet(alpha, alpha, alpha) distribution: {}".format(alpha))
        print("===========================================")

        # For confusion matrix
        all_preds = list()
        all_targets = list()
        cl_samples = list()
        num_samples = list()

        # Record
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        # learning_rate = lr 
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        for i, (inputs, labels, true_labels, k_mean_targets, img_max) in enumerate(trainloader):
            inputs, labels, true_labels, k_mean_targets = inputs.to(device), labels.to(device), true_labels.to(device), k_mean_targets.to(device)

            # Two kinds of output
            optimizer.zero_grad()
            outputs = model(inputs)

            if mixup:
                _input_mix, target_a, target_b, lam = icm_data(inputs, true_labels)
                # _input_mix, target_a, target_b, lam = mixup_data(inputs, true_labels)
                target_a, target_b = target_a.to(device), target_b.to(device)

                # Move only the inner tensors to the specified device
                output_mix = model(_input_mix)

                # For Loss, we use mixup output
                criterion = nn.CrossEntropyLoss(reduction='none').to(device)
                loss = mixup_criterion(criterion, output_mix, target_a,
                                   target_b, lam).mean()
            else:
                # print("=> ERM training")
                criterion = nn.CrossEntropyLoss(reduction='none').to(device)
                loss = criterion(outputs, true_labels).mean()

            acc1, acc5 = accuracy(outputs, true_labels, topk=(1, 5))
            all_targets.extend(true_labels.cpu().numpy())
            _, pred = torch.max(outputs, 1)
            all_preds.extend(pred.cpu().numpy())

            # Measure accuracy and record loss
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

            # Optimizer
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

            if i % eval_n_epoch == 0:
                output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                            epoch,
                            i,
                            len(trainloader),
                            loss=losses,
                            top1=top1,
                            top5=top5,
                            lr=learning_rate))
                print(output)
        
        compute_metrics_and_record(all_preds,
                                all_targets,
                                losses,
                                top1,
                                top5,
                                flag='Training')
        
        acc1 = validate(model, testloader, eval_n_epoch, epoch, device)
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
        print(output_best)
            
def train_nn(args):
    algo = args.algo
    model = args.model
    weight = args.weight
    lr = args.lr
    seed = args.seed
    data_aug = True if args.data_aug.lower()=="true" else False
    neighbor = True if args.neighbor.lower()=="true" else False

    epochs = args.n_epoch
    input_dataset = args.dataset_name

    eval_n_epoch = args.evaluate_step
    batch_size = args.batch_size
    n_weight = args.weighting
    imb_factor = args.imb_factor
    imb_type = args.imb_type
    best_acc1 = 0.

    np.random.seed(seed)
    torch.manual_seed(seed)

    if data_aug:
        print("Use data augmentation.")

    weights, pretrain = weighting_calculation(input_dataset, imb_factor, n_weight)
    # if neighbor:
    print("Use prepare_neighbour_dataset")
    train_data = "train"
    trainset, input_dim, num_classes = prepare_neighbour_dataset(input_dataset=input_dataset, data_type=train_data, max_train_samples=None, multi_label=False, 
                                    weight=weight, imb_type=imb_type, imb_factor=imb_factor, pretrain=pretrain)
    test_data = "test"
    testset, input_dim, num_classes = prepare_neighbour_dataset(input_dataset=input_dataset, data_type=test_data, max_train_samples=None, multi_label=False, 
                                    weight=weight, imb_type=imb_type, imb_factor=imb_factor, pretrain=pretrain)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

    if args.model == "resnet18":
        model = get_resnet18(num_classes, input_dataset).to(device)
    elif args.model == "m-resnet18":
        model = get_modified_resnet18(num_classes, input).to(device)
    elif args.model == "mlp":
        model = MLP(input_dim=input_dim,hidden_dim=args.hidden_dim,num_classes=num_classes).to(device)
    elif args.model == "linear":
        model = Linear(input_dim=input_dim,num_classes=num_classes).to(device)
    else:
        raise NotImplementedError

    for epoch in range(0, epochs):
        # learning_rate = adjust_learning_rate(epochs, epoch, lr)
        learning_rate = lr
        training_loss = 0.0
        model.train()

        weights = weights.to(device)

        # For confusion matrix
        all_preds = list()
        all_targets = list()

        # Record
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        # learning_rate = lr 
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

        for i, (inputs, true_labels, labels) in enumerate(trainloader):
            inputs, true_labels, labels = inputs.to(device), true_labels.to(device), labels.to(device)

            # Two kinds of output
            optimizer.zero_grad()
            outputs = model(inputs)

            if algo == "scl-exp":
                outputs = F.softmax(outputs, dim=1)
                labels = labels.squeeze()
                loss = -F.nll_loss(outputs.exp(), labels, weights)
            elif algo == "scl-nl":
                if neighbor:
                    # --------For soft label-------------
                    # p = (1-F.softmax(outputs,1)).clamp(1e-8,1-1e-8).log()
                    p = (1-F.softmax(outputs,1)).clamp(1e-6,1-1e-6).log()
                    loss = (-p * labels).sum(-1).mean()
                else:
                    #--------For hard label-------------
                    p = (1 - F.softmax(outputs, dim=1) + 1e-6).log()
                    labels = labels.squeeze()
                    loss = F.nll_loss(p, labels, weights)
            else:
                raise NotImplementedError

            acc1, acc5 = accuracy(outputs, true_labels, topk=(1, 5))
            all_targets.extend(true_labels.cpu().numpy())

            _, pred = torch.max(outputs, 1)
            all_preds.extend(pred.cpu().numpy())

            # Measure accuracy and record loss
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

            # Optimizer
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

            if i % eval_n_epoch == 0:
                output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                            epoch,
                            i,
                            len(trainloader),
                            loss=losses,
                            top1=top1,
                            top5=top5,
                            lr=learning_rate))
                print(output)
        
        compute_metrics_and_record(all_preds,
                                all_targets,
                                losses,
                                top1,
                                top5,
                                flag='Training')
        
        acc1 = validate(model, testloader, eval_n_epoch, epoch, device)
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
        print(output_best)
              

if __name__ == "__main__":
    print(torch.__version__)
    torch.cuda.empty_cache()
    dataset_list = [
        "CIFAR10",
        "CIFAR20",
        "PCLCIFAR10",
        "PCLCIFAR20",
        "KMNIST",
        "MNIST",
        "FashionMNIST"
    ]

    algo_list = [
        "scl-exp",
        "scl-nl",
        "scl-lin",
        "ure-ga-u",
        "ure-ga",
        "fwd-int",
        "fwd-u",
        "fwd-r",
        "lw"
    ]

    model_list = [
        "resnet18",
        "m-resnet18",
        "linear",
        "mlp"
    ]

    weight_list = [
        "rank",
        "distance"
    ]

    setup_list = [
        "setup 1",
        "setup 2"
    ]

    aug_type = [
        "randaug",
        "autoaug",
        "cutout",
        "flipflop"
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, choices=algo_list, help='Algorithm')
    parser.add_argument('--dataset_name', type=str, choices=dataset_list, help='Dataset name', default='cifar10')
    parser.add_argument('--model', type=str, choices=model_list, help='Model name', default='resnet18')
    parser.add_argument('--lr', type=float, help='Learning rate', default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, help='Random seed', default=1126)
    parser.add_argument('--data_aug', type=str, default='false')
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--evaluate_step', type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=500)
    parser.add_argument('--k_cluster', type=int, default=0)
    parser.add_argument('--n_epoch', type=int, default=300)
    parser.add_argument('--warm_epoch', type=int, default=240)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--multi_label', action='store_true')
    parser.add_argument('--imb_type', type=str, default=None)
    parser.add_argument('--imb_factor', type=float, default=1.0)
    parser.add_argument('--weighting', type=int, default=0)
    parser.add_argument('--mixup', type=str, default='false')
    parser.add_argument('--icm', type=str, default='false')
    parser.add_argument('--micm', type=str, default='false')
    parser.add_argument('--four_images_intra_class', type=str, default='false')
    parser.add_argument('--cl_aug', type=str, default='false')
    parser.add_argument('--mamix_intra_class', type=str, default='false')
    parser.add_argument('--orig_mixup', type=str, default='false')
    parser.add_argument('--mamix_ratio', type=float, default=-0.25)
    parser.add_argument('--neighbor', type=str, default='false')
    parser.add_argument('--weight', type=str, choices=weight_list, help='rank or distance')
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--transition_bias', type=float, default=1.0)
    parser.add_argument('--setup_type', type=str, choices=setup_list, help='problem setup', default='setup 1')
    parser.add_argument('--aug_type', type=str, choices=aug_type, help='augmentation type', default='flipflop')

    args = parser.parse_args()
    neighbor = True if args.neighbor.lower()=="true" else False
    if neighbor:
        train_nn(args)
    else:
        train_icm(args)
