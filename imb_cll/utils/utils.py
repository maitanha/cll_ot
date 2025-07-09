import torch.optim as optim
import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from imb_cll.utils.metrics import shot_acc, accuracy

def get_dataset_T(dataset, num_classes):
    dataset_T = np.zeros((num_classes,num_classes))
    class_count = np.zeros(num_classes)
    for i in range(len(dataset)):
        dataset_T[dataset.true_targets[i]][dataset.targets[i]] += 1
        class_count[dataset.true_targets[i]] += 1
    for i in range(num_classes):
        dataset_T[i] /= class_count[i]
    return dataset_T, class_count

def validate(model, dataloader, eval_n_epoch, epoch, device):

    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    all_preds = list()
    all_targets = list()

    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none').to(device)

    with torch.no_grad():
        for i, (inputs, true_labels) in enumerate(dataloader):
            inputs, true_labels = inputs.to(device), true_labels.to(device)
            outputs = model(inputs)

            acc1, acc5 = accuracy(outputs, true_labels, topk=(1, 5))
            _, pred = torch.max(outputs, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(true_labels.cpu().numpy())
            loss = criterion(outputs, true_labels).mean()

            # measure accuracy and record loss
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

            if i % eval_n_epoch == 0:
                output = ('Epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        epoch,
                        i,
                        len(dataloader),
                        loss=losses,
                        top1=top1,
                        top5=top5))
                print(output)

        cls_acc_string = compute_metrics_and_record(all_preds,
                                all_targets,
                                losses,
                                top1,
                                top5,
                                flag='Testing')
        
    if cls_acc_string is not None:
            return top1.avg, cls_acc_string
    else:
        return top1.avg

def num_img_per_class(img_max, cls_num, imb_type, imb_factor):
    cls_num_list = []
    if imb_type == 'exp':
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
            cls_num_list.append(int(num))
    elif imb_type == 'step':
        for cls_idx in range(cls_num // 2):
            cls_num_list.append(int(img_max))
        for cls_idx in range(cls_num // 2):
            cls_num_list.append(int(img_max * imb_factor))
    else:
        cls_num_list.extend([int(img_max)] * cls_num)
    print("The number samples of each class: {}".format(cls_num_list))
    return cls_num_list

def weighting_calculation(dataset_name, imb_factor, n_weight):
    if dataset_name == "CIFAR10":
        if imb_factor == 0.01:
            # pretrain = "./imb_cll_pretrained/CIFAR10/CIFAR10_checkpoint_0799_-0.7960.pth.tar"
            pretrain = "./imb_cll_pretrained/CIFAR10/CIFAR10_input32_checkpoint_0799_-0.8011.pth.tar"
            weights = torch.tensor([1.44308171, 1.14900082, 1.02512971, 0.96447884, 0.93054867, 0.91227983, 0.90093544, 0.89476267, 0.89110236, 0.88867995])
            # weights = torch.tensor([0.05061136, 0.07689979, 0.12113387, 0.19503667, 0.31880537, 0.52458985, 0.86834894, 1.44262727, 2.40922305, 3.99272382])
            weights = weights ** n_weight
        elif imb_factor == 0.02:
            pretrain = "./imb_cll_pretrained/CIFAR10/CIFAR10_0.02_checkpoint_0799_-0.8122.pth.tar"
            weights = torch.tensor([1.44308171, 1.14900082, 1.02512971, 0.96447884, 0.93054867, 0.91227983, 0.90093544, 0.89476267, 0.89110236, 0.88867995])
            # weights = torch.tensor([0.05061136, 0.07689979, 0.12113387, 0.19503667, 0.31880537, 0.52458985, 0.86834894, 1.44262727, 2.40922305, 3.99272382])
            weights = weights ** n_weight
        elif imb_factor == 0.1:
            pretrain = "./imb_cll_pretrained/CIFAR10/CIFAR10_0.1_checkpoint_0799_-0.8381.pth.tar"
            weights = torch.tensor([1.44308171, 1.14900082, 1.02512971, 0.96447884, 0.93054867, 0.91227983, 0.90093544, 0.89476267, 0.89110236, 0.88867995])
            # weights = torch.tensor([0.05061136, 0.07689979, 0.12113387, 0.19503667, 0.31880537, 0.52458985, 0.86834894, 1.44262727, 2.40922305, 3.99272382])
            weights = weights ** n_weight
        elif imb_factor == 1:
            pretrain = "./balanced_cll_pretrained/CIFAR10/CIFAR10_checkpoint_0799_-0.8583.pth.tar"
            weights = torch.tensor([1.44308171, 1.14900082, 1.02512971, 0.96447884, 0.93054867, 0.91227983, 0.90093544, 0.89476267, 0.89110236, 0.88867995])
            weights = weights ** n_weight
    elif dataset_name == "PCLCIFAR10":
        if imb_factor == 0.01:
            # pretrain = "./imb_cll_pretrained/CIFAR10/CIFAR10_checkpoint_0799_-0.7960.pth.tar"
            pretrain = "./imb_cll_pretrained/CIFAR10/CIFAR10_input32_checkpoint_0799_-0.8011.pth.tar"
            weights = torch.tensor([1.44308171, 1.14900082, 1.02512971, 0.96447884, 0.93054867, 0.91227983, 0.90093544, 0.89476267, 0.89110236, 0.88867995])
            # weights = torch.tensor([0.05061136, 0.07689979, 0.12113387, 0.19503667, 0.31880537, 0.52458985, 0.86834894, 1.44262727, 2.40922305, 3.99272382])
            weights = weights ** n_weight
        elif imb_factor == 0.02:
            pretrain = "./imb_cll_pretrained/CIFAR10/CIFAR10_0.02_checkpoint_0799_-0.8122.pth.tar"
            weights = torch.tensor([1.44308171, 1.14900082, 1.02512971, 0.96447884, 0.93054867, 0.91227983, 0.90093544, 0.89476267, 0.89110236, 0.88867995])
            # weights = torch.tensor([0.05061136, 0.07689979, 0.12113387, 0.19503667, 0.31880537, 0.52458985, 0.86834894, 1.44262727, 2.40922305, 3.99272382])
            weights = weights ** n_weight
        elif imb_factor == 0.1:
            pretrain = "./imb_cll_pretrained/CIFAR10/CIFAR10_0.1_checkpoint_0799_-0.8381.pth.tar"
            weights = torch.tensor([1.44308171, 1.14900082, 1.02512971, 0.96447884, 0.93054867, 0.91227983, 0.90093544, 0.89476267, 0.89110236, 0.88867995])
            # weights = torch.tensor([0.05061136, 0.07689979, 0.12113387, 0.19503667, 0.31880537, 0.52458985, 0.86834894, 1.44262727, 2.40922305, 3.99272382])
            weights = weights ** n_weight
        elif imb_factor == 1:
            pretrain = "./balanced_cll_pretrained/CIFAR10/CIFAR10_checkpoint_0799_-0.8583.pth.tar"
            weights = torch.tensor([1.44308171, 1.14900082, 1.02512971, 0.96447884, 0.93054867, 0.91227983, 0.90093544, 0.89476267, 0.89110236, 0.88867995])
            weights = weights ** n_weight
    elif dataset_name == "PCLCIFAR20":
        if imb_factor == 0.01:
            # pretrain = "./imb_cll_pretrained/CIFAR20/CIFAR20_checkpoint_0799_-0.7825.pth.tar"
            pretrain = "./imb_cll_pretrained/CIFAR20/CIFAR20_input32_checkpoint_0799_-0.7561.pth.tar"
            weights = torch.tensor([1.20217289, 1.13594089, 1.08908254, 1.0550147,  1.02975162, 1.01078401, 0.99631446, 0.98537103, 0.97687792, 0.97037571, 0.96528757, 0.96132228, 0.95828854, 0.95592451, 0.95397714, 0.95252198, 0.95139334, 0.95050849, 0.94978578, 0.94930461])
            weights = weights ** n_weight
        elif imb_factor == 0.02:
            # pretrain = "./imb_cll_pretrained/CIFAR20/CIFAR20_checkpoint_0799_-0.7825.pth.tar"
            pretrain = "./imb_cll_pretrained/CIFAR20/CIFAR20_0.02_checkpoint_0799_-0.8322.pth.tar"
            weights = torch.tensor([1.20217289, 1.13594089, 1.08908254, 1.0550147,  1.02975162, 1.01078401, 0.99631446, 0.98537103, 0.97687792, 0.97037571, 0.96528757, 0.96132228, 0.95828854, 0.95592451, 0.95397714, 0.95252198, 0.95139334, 0.95050849, 0.94978578, 0.94930461])
            weights = weights ** n_weight
        elif imb_factor == 0.1:
            # pretrain = "./imb_cll_pretrained/CIFAR20/CIFAR20_checkpoint_0799_-0.7825.pth.tar"
            pretrain = "./imb_cll_pretrained/CIFAR20/CIFAR20_0.1_checkpoint_0799_-0.8652.pth.tar"
            weights = torch.tensor([1.20217289, 1.13594089, 1.08908254, 1.0550147,  1.02975162, 1.01078401, 0.99631446, 0.98537103, 0.97687792, 0.97037571, 0.96528757, 0.96132228, 0.95828854, 0.95592451, 0.95397714, 0.95252198, 0.95139334, 0.95050849, 0.94978578, 0.94930461])
            weights = weights ** n_weight
        elif imb_factor == 1:
            pretrain = "./balanced_cll_pretrained/CIFAR20/CIFAR20_checkpoint_0799_-0.8386.pth.tar"
            weights = torch.tensor([1.20217289, 1.13594089, 1.08908254, 1.0550147,  1.02975162, 1.01078401, 0.99631446, 0.98537103, 0.97687792, 0.97037571, 0.96528757, 0.96132228, 0.95828854, 0.95592451, 0.95397714, 0.95252198, 0.95139334, 0.95050849, 0.94978578, 0.94930461])
            weights = weights ** n_weight
    elif dataset_name == "CIFAR20":
        if imb_factor == 0.01:
            # pretrain = "./imb_cll_pretrained/CIFAR20/CIFAR20_checkpoint_0799_-0.7825.pth.tar"
            pretrain = "./imb_cll_pretrained/CIFAR20/CIFAR20_input32_checkpoint_0799_-0.7561.pth.tar"
            weights = torch.tensor([1.20217289, 1.13594089, 1.08908254, 1.0550147,  1.02975162, 1.01078401, 0.99631446, 0.98537103, 0.97687792, 0.97037571, 0.96528757, 0.96132228, 0.95828854, 0.95592451, 0.95397714, 0.95252198, 0.95139334, 0.95050849, 0.94978578, 0.94930461])
            weights = weights ** n_weight
        elif imb_factor == 0.02:
            # pretrain = "./imb_cll_pretrained/CIFAR20/CIFAR20_checkpoint_0799_-0.7825.pth.tar"
            pretrain = "./imb_cll_pretrained/CIFAR20/CIFAR20_0.02_checkpoint_0799_-0.8322.pth.tar"
            weights = torch.tensor([1.20217289, 1.13594089, 1.08908254, 1.0550147,  1.02975162, 1.01078401, 0.99631446, 0.98537103, 0.97687792, 0.97037571, 0.96528757, 0.96132228, 0.95828854, 0.95592451, 0.95397714, 0.95252198, 0.95139334, 0.95050849, 0.94978578, 0.94930461])
            weights = weights ** n_weight
        elif imb_factor == 0.1:
            # pretrain = "./imb_cll_pretrained/CIFAR20/CIFAR20_checkpoint_0799_-0.7825.pth.tar"
            pretrain = "./imb_cll_pretrained/CIFAR20/CIFAR20_0.1_checkpoint_0799_-0.8652.pth.tar"
            weights = torch.tensor([1.20217289, 1.13594089, 1.08908254, 1.0550147,  1.02975162, 1.01078401, 0.99631446, 0.98537103, 0.97687792, 0.97037571, 0.96528757, 0.96132228, 0.95828854, 0.95592451, 0.95397714, 0.95252198, 0.95139334, 0.95050849, 0.94978578, 0.94930461])
            weights = weights ** n_weight
        elif imb_factor == 1:
            pretrain = "./balanced_cll_pretrained/CIFAR20/CIFAR20_checkpoint_0799_-0.8386.pth.tar"
            weights = torch.tensor([1.20217289, 1.13594089, 1.08908254, 1.0550147,  1.02975162, 1.01078401, 0.99631446, 0.98537103, 0.97687792, 0.97037571, 0.96528757, 0.96132228, 0.95828854, 0.95592451, 0.95397714, 0.95252198, 0.95139334, 0.95050849, 0.94978578, 0.94930461])
            weights = weights ** n_weight
    elif dataset_name == "FashionMNIST":
        if imb_factor == 0.01:
            pretrain = "./imb_cll_pretrained/FashionMNIST/FashionMNIST_checkpoint_0799_-0.9020.pth.tar"
            weights = torch.tensor([1.37520534, 1.1097013,  1.04209213, 0.96593773, 0.9537242, 0.92060064, 0.91322734, 0.90291396, 0.90958861, 0.90700876])
            weights = weights ** n_weight
        elif imb_factor == 0.02:
            pretrain = "./imb_cll_pretrained/FashionMNIST/FashionMNIST_0.02_checkpoint_0799_-0.9137.pth.tar"
            weights = torch.tensor([1.37520534, 1.1097013,  1.04209213, 0.96593773, 0.9537242, 0.92060064, 0.91322734, 0.90291396, 0.90958861, 0.90700876])
            weights = weights ** n_weight
        elif imb_factor == 0.1:
            pretrain = "./imb_cll_pretrained/FashionMNIST/FashionMNIST_0.1_checkpoint_0799_-0.9278.pth.tar"
            weights = torch.tensor([1.37520534, 1.1097013,  1.04209213, 0.96593773, 0.9537242, 0.92060064, 0.91322734, 0.90291396, 0.90958861, 0.90700876])
            weights = weights ** n_weight
        elif imb_factor == 1:
            pretrain = "./balanced_cll_pretrained/FashionMNIST/FashionMNIST_checkpoint_0799_-0.9422.pth.tar"
            weights = torch.tensor([1.37520534, 1.1097013,  1.04209213, 0.96593773, 0.9537242, 0.92060064, 0.91322734, 0.90291396, 0.90958861, 0.90700876])
            weights = weights ** n_weight
    elif dataset_name == "MNIST":
        if imb_factor == 0.01:
            # pretrain = "./imb_cll_pretrained/MNIST/MNIST_checkpoint_0799_-0.8831.pth.tar"
            pretrain = "./imb_cll_pretrained/MNIST/MNIST_NoFlipCrop_checkpoint_0799_-0.8772.pth.tar"
            weights = torch.tensor([1.3762853, 1.11227572, 1.04531862, 0.96738018, 0.9514026, 0.91518053, 0.91251744, 0.90881726, 0.90619416, 0.90462819])
            weights = weights ** n_weight
        elif imb_factor == 0.02:
            # pretrain = "./imb_cll_pretrained/MNIST/MNIST_checkpoint_0799_-0.8831.pth.tar"
            pretrain = "./imb_cll_pretrained/MNIST/MNIST_0.02_checkpoint_0799_-0.8855.pth.tar"
            weights = torch.tensor([1.3762853, 1.11227572, 1.04531862, 0.96738018, 0.9514026, 0.91518053, 0.91251744, 0.90881726, 0.90619416, 0.90462819])
            weights = weights ** n_weight
        elif imb_factor == 0.1:
            # pretrain = "./imb_cll_pretrained/MNIST/MNIST_checkpoint_0799_-0.8831.pth.tar"
            pretrain = "./imb_cll_pretrained/MNIST/MNIST_0.1_checkpoint_0799_-0.8918.pth.tar"
            weights = torch.tensor([1.3762853, 1.11227572, 1.04531862, 0.96738018, 0.9514026, 0.91518053, 0.91251744, 0.90881726, 0.90619416, 0.90462819])
            weights = weights ** n_weight
        elif imb_factor == 1:
            pretrain = "./balanced_cll_pretrained/MNIST/MNIST_checkpoint_0799_-0.8980.pth.tar"
            weights = torch.tensor([1.3762853, 1.11227572, 1.04531862, 0.96738018, 0.9514026, 0.91518053, 0.91251744, 0.90881726, 0.90619416, 0.90462819])
            weights = weights ** n_weight
    elif dataset_name == "KMNIST":
        if imb_factor == 0.01:
            # pretrain = "./imb_cll_pretrained/KMNIST/KMNIST_checkpoint_0799_-0.8738.pth.tar"
            pretrain = "./imb_cll_pretrained/KMNIST/KMNIST_noflip_checkpoint_0799_-0.8990.pth.tar"
            weights = torch.tensor([1.37520534, 1.1097013,  1.04209213, 0.96593773, 0.9537242, 0.92060064, 0.91322734, 0.90291396, 0.90958861, 0.90700876])
            weights = weights ** n_weight
        elif imb_factor == 0.02:
            # pretrain = "./imb_cll_pretrained/KMNIST/KMNIST_checkpoint_0799_-0.8738.pth.tar"
            pretrain = "./imb_cll_pretrained/KMNIST/KMNIST_0.02_checkpoint_0799_-0.9035.pth.tar"
            weights = torch.tensor([1.37520534, 1.1097013,  1.04209213, 0.96593773, 0.9537242, 0.92060064, 0.91322734, 0.90291396, 0.90958861, 0.90700876])
            weights = weights ** n_weight
        elif imb_factor == 0.1:
            # pretrain = "./imb_cll_pretrained/KMNIST/KMNIST_checkpoint_0799_-0.8738.pth.tar"
            pretrain = "./imb_cll_pretrained/KMNIST/KMNIST_0.1_checkpoint_0799_-0.9222.pth.tar"
            weights = torch.tensor([1.37520534, 1.1097013,  1.04209213, 0.96593773, 0.9537242, 0.92060064, 0.91322734, 0.90291396, 0.90958861, 0.90700876])
            weights = weights ** n_weight
        elif imb_factor == 1:
            pretrain = "./balanced_cll_pretrained/KMNIST/KMNIST_checkpoint_0799_-0.9231.pth.tar"
            weights = torch.tensor([1.37520534, 1.1097013,  1.04209213, 0.96593773, 0.9537242, 0.92060064, 0.91322734, 0.90291396, 0.90958861, 0.90700876])
            weights = weights ** n_weight
    else:
        raise NotImplementedError

    return weights, pretrain

def adjust_learning_rate(epochs, epoch, learning_rate):
    """Sets the learning rate"""
    # total 200 epochs scheme
    if epochs == 200:
        epoch = epoch + 1
        if epoch >= 180:
            learning_rate = learning_rate * 0.01
        elif epoch >= 160:
            learning_rate = learning_rate * 0.1
        else:
            learning_rate = learning_rate
    # total 300 epochs scheme
    elif epochs == 300:
        epoch = epoch + 1
        if epoch > 280:
            learning_rate = learning_rate * 0.01
        elif epoch > 240:
            learning_rate = learning_rate * 0.1
        else:
            learning_rate = learning_rate
    elif epochs == 400:
        epoch = epoch + 1
        if epoch > 350:
            learning_rate = learning_rate * 0.01
        elif epoch > 300:
            learning_rate = learning_rate * 0.1
        else:
            learning_rate = learning_rate
    else:
        raise ValueError(
            "[Warning] Total epochs {} not supported !".format(epochs))
    return learning_rate

def _init_optimizer(self):
    if self.cfg.optimizer == 'sgd':
        print("=> Initialize optimizer {}".format(self.cfg.optimizer))
        optimizer = optim.SGD(self.model.parameters(),
                                self.cfg.learning_rate,
                                momentum=self.cfg.momentum,
                                weight_decay=self.cfg.weight_decay)
        return optimizer
    else:
        raise ValueError("[Warning] Selected Optimizer not supported !")
    

def compute_metrics_and_record(all_preds,
                                all_targets,
                                losses,
                                top1,
                                top5,
                                flag='Training'):
    """Responsible for computing metrics and prepare string for logger"""
    # if flag == 'Training':
    #     log = self.log_training
    # else:
    #     log = self.log_testing

    # if self.cfg.dataset == 'cifar100' or self.cfg.dataset == 'tiny200':
    #     all_preds = np.array(all_preds)
    #     all_targets = np.array(all_targets)
    #     many_acc, median_acc, low_acc = shot_acc(self.cfg,
    #                                                 all_preds,
    #                                                 all_targets,
    #                                                 self.train_dataset,
    #                                                 acc_per_cls=False)
    #     group_acc = np.array([many_acc, median_acc, low_acc])
    #     # Print Format
    #     group_acc_string = '%s Group Acc: %s' % (flag, (np.array2string(
    #         group_acc,
    #         separator=',',
    #         formatter={'float_kind': lambda x: "%.3f" % x})))
    #     print(group_acc_string)
    # else:
    #     group_acc = None
    #     group_acc_string = None

    group_acc = None
    group_acc_string = None

    # metrics (recall)
    cf = confusion_matrix(all_targets, all_preds).astype(float)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_acc = cls_hit / cls_cnt
    # overall epoch output
    epoch_output = (
        '{flag} Results: Prec@1 {top1.avg:.4f} Prec@5 {top5.avg:.4f} \
        Loss {loss.avg:.6f}'.format(flag=flag,
                                    top1=top1,
                                    top5=top5,
                                    loss=losses))
    # per class output
    cls_acc_string = '%s Class Recall: %s' % (flag, (np.array2string(
        cls_acc,
        separator=',',
        formatter={'float_kind': lambda x: "%.6f" % x})))
    print(epoch_output)
    print(cls_acc_string)

    # if eval with best model, just return
    # if self.cfg.best_model is not None:
    #     return cls_acc_string

    # log_and_tf(epoch_output,
    #                 cls_acc,
    #                 cls_acc_string,
    #                 losses,
    #                 top1,
    #                 top5,
    #                 log,
    #                 group_acc=group_acc,
    #                 group_acc_string=group_acc_string,
    #                 flag=flag)
    

def log_and_tf(self,
                epoch_output,
                cls_acc,
                cls_acc_string,
                losses,
                top1,
                top5,
                log,
                group_acc=None,
                group_acc_string=None,
                flag=None):
    """Responsible for recording logger and tensorboardX"""
    log.write(epoch_output + '\n')
    log.write(cls_acc_string + '\n')

    if group_acc_string is not None:
        log.write(group_acc_string + '\n')
    log.write('\n')
    log.flush()

    # TF
    if group_acc_string is not None:
        if flag == 'Training':
            self.tf_writer.add_scalars(
                'acc/train_' + 'group_acc',
                {str(i): x
                    for i, x in enumerate(group_acc)}, self.epoch)
        else:
            self.tf_writer.add_scalars(
                'acc/test_' + 'group_acc',
                {str(i): x
                    for i, x in enumerate(group_acc)}, self.epoch)

    else:
        if flag == 'Training':
            self.tf_writer.add_scalars(
                'acc/train_' + 'cls_recall',
                {str(i): x
                    for i, x in enumerate(cls_acc)}, self.epoch)
        else:
            self.tf_writer.add_scalars(
                'acc/test_' + 'cls_recall',
                {str(i): x
                    for i, x in enumerate(cls_acc)}, self.epoch)
    if flag == 'Trainig':
        self.tf_writer.add_scalar('loss/train', losses.avg, self.epoch)
        self.tf_writer.add_scalar('acc/train_top1', top1.avg, self.epoch)
        self.tf_writer.add_scalar('acc/train_top5', top5.avg, self.epoch)
        self.tf_writer.add_scalar('lr',
                                    self.optimizer.param_groups[-1]['lr'],
                                    self.epoch)
    else:
        self.tf_writer.add_scalar('loss/test_' + flag, losses.avg,
                                    self.epoch)
        self.tf_writer.add_scalar('acc/test_' + flag + '_top1', top1.avg,
                                    self.epoch)
        self.tf_writer.add_scalar('acc/test_' + flag + '_top5', top5.avg,
                                    self.epoch)
    
class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)