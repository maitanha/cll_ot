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
from imb_cll.utils.cl_augmentation import mixup_cl_data, mixup_data, aug_intra_class, mamix_intra_aug, aug_intra_class_three_images, aug_intra_class_four_images, intra_class_count_error, mixup_cl_data_count_error
from imb_cll.models.models import get_modified_resnet18, get_resnet18
from imb_cll.models.basemodels import Linear, MLP
import torchvision.transforms as transforms
import wandb
import os
import json

# === Import your OT functions and feature extractor loader ===
from ot_module import get_per_batch_OT_cost, get_OT_dual_sol
from models.preact_resnet import load_pretrained_feature_extractor
from lava import get_per_batch_OT_cost, get_OT_dual_sol

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
    new_data_aug = args.new_data_aug
    mixup = True if args.mixup.lower()=="true" else False
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

    if new_data_aug == "icm":
        print("Use complementary mixup intra class")
    elif new_data_aug == "cl_aug":
        print("Use mixup noise-free")

    weights, pretrain = weighting_calculation(input_dataset, imb_factor, n_weight)

    # training hyperparameters for OT
    pruning_percentage = args.prune_perc

    feature_extractor = load_pretrained_feature_extractor(
        "cifar10_embedder_preact_resnet18.pth",
        device,
    )
    # data transformations for torch.Dataset
    if args.corruption_type == "feature":
        # no normalization for noisy features data
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_transform_selection = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_transform_selection = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

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

    # Calculate the OT cost matrix
    train_iter = iter(trainloader)
    test_iter = iter(testloader)

    # Get first batch from train and test
    x_tr, y_tr, *_ = next(train_iter)
    x_val, y_val, *_ = next(test_iter)

    # Move to device if needed
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)

    # Call OT cost function
    cost, plan, dual_sol, label_distances = get_per_batch_OT_cost(
        feature_extractor=model,  # or your embedder
        x_tr=x_tr,
        y_tr=y_tr,
        x_val=x_val,
        y_val=y_val,
        batch_size=batch_size,   # usually batch_size
        device=device
    )

    import pdb
    pdb.set_trace()

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
    
    wandb.login()
    wandb.init(project=args.dataset_name, name=f"{algo}-{dataset_name}-{imb_factor}-{lr}-{weight_decay}-{epochs}-{aug_type}-{new_data_aug}", config={"lr": lr, "weight_decay": weight_decay, "epochs": epochs, "aug_type": aug_type, "algo": algo, "new_data_aug": new_data_aug}, tags=[str(imb_factor)])
    # Ensure the logs directory exists
    os.makedirs("logs", exist_ok=True)

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

        total_count_error = 0

        for i, (inputs, labels, true_labels, k_mean_targets, img_max) in enumerate(trainloader):
            inputs, labels, true_labels, k_mean_targets = inputs.to(device), labels.to(device), true_labels.to(device), k_mean_targets.to(device)

            # Two kinds of output
            optimizer.zero_grad()
            outputs = model(inputs)

            if mixup:
                if new_data_aug == "icm":
                    #----MIXUP INTRA CLASS----
                    _input_mix, targets = aug_intra_class(inputs, labels, true_labels, k_mean_targets, device, dataset_name, alpha) # Mixup Intra Class
                    targets = targets.to(device)
                    _input_mix, targets = aug_intra_class_three_images(inputs, labels, true_labels, k_mean_targets, device, dataset_name, alpha)
                    targets = targets.to(device)
                elif new_data_aug == "four_images_intra_class":
                    _input_mix, target_a, target_b, target_c, target_d, lam1, lam2, lam3, lam4, count_error = aug_intra_class_four_images(inputs, labels, true_labels, k_mean_targets, device, dataset_name)
                    total_count_error += count_error
                elif new_data_aug == "cl_aug":  # Mixup filter by true label --> Proof of Concept
                    #----MIXUP FILTER EXTRA CLASS----
                    _input_mix, target_a, target_b, lam = mixup_cl_data(inputs, labels, true_labels, device)
                elif new_data_aug == "mamix_intra_class":
                    if len(cls_num_list) == 0:
                        img_max = img_max[0].item()
                        cls_num_list = num_img_per_class(img_max, num_classes, imb_type, imb_factor)
                    _input_mix, target_a, target_b, _, lam = mamix_intra_aug(inputs, labels, k_mean_targets, mamix_ratio, cls_num_list, device)
                else:  # Original Mixup without clustering and filtering
                    #----MIXUP ORIGINAL COUNT ERROR----
                    _input_mix, target_a, target_b, lam, count_error = mixup_cl_data_count_error(inputs, labels, true_labels, device)
                    target_a, target_b = target_a.type(torch.LongTensor),  target_b.type(torch.LongTensor)  # casting to long
                    target_a, target_b = target_a.to(device), target_b.to(device)
                    total_count_error += count_error

                # Move only the inner tensors to the specified device
                output_mix = model(_input_mix)

                if algo == "scl-lin":
                    if new_data_aug == "icm" or new_data_aug == "micm": #--For soft label----
                        p = -F.softmax(output_mix, dim=1)
                        loss = (-p * targets * weights).sum(-1).mean()
                    else: #--For hard label----
                        p = -F.softmax(output_mix, dim=1)
                        target_a = target_a.squeeze()
                        target_b = target_b.squeeze()
                        loss = (lam * (F.nll_loss(p, target_a, weights)) + (1 - lam) * (F.nll_loss(p, target_b, weights))).mean() #Soft-Label
                elif algo == "scl-exp":
                    if new_data_aug == "icm" or new_data_aug == "micm": #--For soft label----
                        p = -torch.exp(F.softmax(output_mix, dim=1))
                        loss = (-p * targets * weights).sum(-1).mean()
                    else: #--For hard label----
                        p = -torch.exp(F.softmax(output_mix, dim=1))
                        target_a = target_a.squeeze()
                        target_b = target_b.squeeze()
                        loss = (lam * (F.nll_loss(p, target_a, weights)) + (1 - lam) * (F.nll_loss(p, target_b, weights))).mean() #Soft-Label
                elif algo == "scl-nl":
                    if new_data_aug == "icm" or new_data_aug == "micm": #--For soft label----
                        p = (1 - F.softmax(output_mix, dim=1)).clamp(1e-6,1-1e-6).log()
                        loss = (-p * targets * weights).sum(-1).mean() 
                    else: #--For hard label----
                        p = (1 - F.softmax(output_mix, dim=1) + 1e-6).log()
                        target_a = target_a.squeeze()
                        target_b = target_b.squeeze()
                        loss = (lam * F.nll_loss(p, target_a, weights) + (1 - lam) * F.nll_loss(p, target_b, weights)).mean() #Soft-Label
                        # loss = (F.nll_loss(p, target_a, weights) + F.nll_loss(p, target_b, weights)).mean()  # Hard-label
                elif algo[:3] == "fwd":
                    if new_data_aug == "icm" or new_data_aug == "micm": #--For soft label----
                        q = torch.mm(F.softmax(output_mix, dim=1), Q).clamp(1e-8,1-1e-8)
                        loss = (-q.log() * targets).sum(-1).mean()
                    else: #--For hard label----
                        q = torch.mm(F.softmax(output_mix, dim=1), Q).clamp(1e-8,1-1e-8)
                        target_a = target_a.squeeze()
                        target_b = target_b.squeeze()
                        loss = (lam * F.nll_loss(q.log(), target_a) + (1 -lam) * F.nll_loss(q.log(), target_a)).mean()
                elif algo == "lw":
                    if new_data_aug == "icm" or new_data_aug == "micm": #--For soft label----
                        p = 1-F.softmax(output_mix, dim=1)
                        Q_1 = F.log_softmax(p, dim=1)
                        w_1 = torch.mul(p / (num_classes-1), Q_1)
                        loss = (-Q_1 * targets * weights).sum(-1).mean() + (-w_1 * targets * weights).sum(-1).mean()
                    else: #-----For hard label-----
                        p = 1 - F.softmax(output_mix, dim=1)
                        q = F.softmax(p, dim=1) + 1e-6
                        w = torch.mul(p / (output_mix.shape[1] - 1), q.log())
                        target_a = target_a.squeeze()
                        target_b = target_b.squeeze()
                        loss = (lam * (F.nll_loss(q.log(), target_a.long(), weights) + F.nll_loss(w, target_a.long(), weights)) + (1 - lam) * (F.nll_loss(q.log(), target_b.long(), weights) + F.nll_loss(w, target_b.long(), weights)))
                elif algo == "ure-ga":
                    if new_data_aug == "icm" or new_data_aug == "micm": #--For soft label----
                        logprob = F.log_softmax(output_mix, dim=1)
                        l = (-logprob * targets)

                        labels_count = targets.sum(0)
                        l_sum = l.sum(0)

                        loss = torch.zeros_like(l_sum)
                        idx = labels_count > 0
                        loss[idx] = -(num_classes-1) * l_sum[idx] / labels_count[idx] * class_count[idx]
                        for j in range(num_classes):
                            if labels_count[j] > 0:
                                loss += (-logprob * targets[:,j].unsqueeze(1)).sum(0) / labels_count[j] * class_count[j]

                        if torch.min(loss) > 0:
                            loss = loss.sum()
                        else:
                            loss = F.relu(-loss).sum()
                else:
                    raise NotImplementedError
            else:
                if algo == "scl-exp": 
                    p = -torch.exp(F.softmax(outputs, dim=1))
                    labels = labels.squeeze()
                    labels = labels.long()
                    loss = F.nll_loss(p, labels, weights)
                elif algo == "scl-lin":
                    #--------For hard label-------------
                    p = -F.softmax(outputs, dim=1)
                    labels = labels.squeeze()
                    labels = labels.long()
                    loss = F.nll_loss(p, labels, weights)
                elif algo == "scl-nl":
                    #--------For hard label-------------
                    p = (1 - F.softmax(outputs, dim=1) + 1e-6).log()
                    labels = labels.squeeze()
                    labels = labels.long()
                    loss = F.nll_loss(p, labels, weights)
                elif algo[:3] == "fwd":
                    q = torch.mm(F.softmax(outputs, dim=1), Q).clamp(1e-8,1-1e-8)
                    labels = labels.long()
                    loss = F.nll_loss(q.log(), labels.squeeze(), weights)
                elif algo == "lw":
                    p = 1 - F.softmax(outputs, dim=1)
                    q = F.softmax(p, dim=1) + 1e-6
                    w = torch.mul(p / (outputs.shape[1] - 1), q.log())
                    labels = labels.squeeze()
                    loss = F.nll_loss(q.log(), labels.long(), weights) + F.nll_loss(w, labels.long(), weights)
                elif algo == "ure-ga":
                    if torch.det(Q) != 0:
                        Tinv = torch.inverse(Q)
                    else:
                        Tinv = torch.pinverse(Q)

                    neglog = -F.log_softmax(outputs, dim=1)
                    labels = labels.squeeze()
                    l = labels.long()
                    counts = torch.bincount(l, minlength=num_classes).view(-1, 1)
                    lh = F.one_hot(l, num_classes).float()
                    neg_vector = torch.matmul(lh.t(), neglog)
                    loss_vec = (Tinv.to(device) * neg_vector).sum(dim=1) * class_count
                    vc = (1 / counts).nan_to_num(0).view(-1)
                    loss_vec = loss_vec * vc
                    if loss_vec.min() > 0:
                        loss = loss_vec.sum()
                    else:
                        loss = F.relu(-loss_vec).sum()
                else:
                    raise NotImplementedError

            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            all_targets.extend(labels.cpu().numpy())
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

        val_acc1 = validate(model, testloader, eval_n_epoch, epoch, device)
        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)
        output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
        print(output_best)
        wandb.log({"train_acc": acc1, "valid_acc": val_acc1, "best_acc": best_acc1.item()})
        with open(f"logs/{algo}-{dataset_name}-{aug_type}-{lr}--{weight_decay}--{imb_factor}--{new_data_aug}.json", "w") as f:
            json.dump(best_acc1.item(), f)
    wandb.finish()
                     
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

    new_data_aug = [
        "icm",
        "micm",
        "cl_aug",
        "mamix_intra_class",
        "orig_mixup",
        "none"
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
    parser.add_argument('--mamix_ratio', type=float, default=-0.25)
    parser.add_argument('--neighbor', type=str, default='false')
    parser.add_argument('--weight', type=str, choices=weight_list, help='rank or distance')
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--transition_bias', type=float, default=1.0)
    parser.add_argument('--setup_type', type=str, choices=setup_list, help='problem setup', default='setup 1')
    parser.add_argument('--new_data_aug', type=str, choices=new_data_aug, help='choose new data aug method', default='none')
    parser.add_argument('--aug_type', type=str, choices=aug_type, help='augmentation type', default='flipflop')

    args = parser.parse_args()
    neighbor = True if args.neighbor.lower()=="true" else False
    if neighbor:
        train_nn(args)
    else:
        train_icm(args)
