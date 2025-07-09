import torch
import random
import numpy as np

def euclidean_distance(vector1, vector2):
    # Flatten the 3D tensor to 1D using .view()
    tensor1 = vector1.view(-1)
    tensor2 = vector2.view(-1)
    distance = (torch.norm(tensor1 - tensor2)).item()
    return distance

def cosine_similarity(vector1, vector2):
    # Flatten the 3D tensor to 1D using .view()
    vector1 = vector1.view(-1)
    vector2 = vector2.view(-1)

    # Normalize the vectors to unit length
    vector1 = vector1 / torch.norm(vector1)
    vector2 = vector2 / torch.norm(vector2)

    # Calculate the dot product
    dot_product = torch.dot(vector1, vector2)

    # Calculate the cosine similarity (cosine of the angle between the vectors)
    cosine_similarity = dot_product.item()

    # Calculate the cosine distance (1 minus cosine similarity)
    cosine_distance = 1 - cosine_similarity

    return cosine_distance

def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    # Count the violent case when true label appearing in cl label
    # if (y[i] == ytrue[i] or y[i] == ytrue[j] or y[j] == ytrue[j] or y[j] == ytrue[i]):
    #     count_error += 1

    return mixed_x, y_a, y_b, lam

def icm_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda within the same class'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    mixed_x = x.clone()
    y_a, y_b = y.clone(), y.clone()

    for label in torch.unique(y):
        label_indices = torch.where(y == label)[0]
        if len(label_indices) > 1:  # Only mixup if there are at least 2 samples of the same class
            index = label_indices[torch.randperm(len(label_indices))]
            mixed_x[label_indices] = lam * x[label_indices] + (1 - lam) * x[index]
            y_b[label_indices] = y[index]

    return mixed_x, y_a, y_b, lam

def mixup_cl_data(x, y, ytrue, device, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    mixed_x = torch.zeros_like(x).to(device)
    y_a, y_b = torch.zeros_like(y).to(device), torch.zeros_like(y).to(device)  #to(device)
    for i in range(batch_size):
        while(True):
            j = random.randint(0, batch_size - 1)
            # if y[i] != ytrue[j] and y[j] != ytrue[i]: #Extra-Class Mixup Filter
            if y[i] == y[j]:
                mixed_x[i] = lam * x[i] + (1 - lam) * x[j]
                y_a[i], y_b[i] = y[i], y[j]
                break
    return mixed_x, y_a, y_b, lam

# This function to calculate the error under imbalanced CLL with Original Mixup
def mixup_cl_data_count_error (x, y, ytrue, device, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    count_error = 0
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    mixed_x = torch.zeros_like(x).to(device)
    y_a, y_b = torch.zeros_like(y).to(device), torch.zeros_like(y).to(device)  #to(device)
    for i in range(batch_size):
        j = random.randint(0, batch_size - 1)
        mixed_x[i] = lam * x[i] + (1 - lam) * x[j] #New Data
        # mixed_x[i] = x[i] # Soft-label
        y_a[i], y_b[i] = y[i], y[j]

        # Count the violent case when true label appearing in cl label
        if (y[i] == ytrue[i] or y[i] == ytrue[j] or y[j] == ytrue[j] or y[j] == ytrue[i]):
            count_error += 1
    return mixed_x, y_a, y_b, lam, count_error

# This function to calculate the error under imbalanced CLL with Mixup Intra Cluster
def intra_class_count_error(x, y, ytrue, k_cluster_label, device, dataset_name, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    count_error = 0

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    mixed_x = torch.zeros_like(x).to(device)
    y_a, y_b = torch.zeros_like(y).to(device), torch.zeros_like(y).to(device)  #to(device)

    # Precompute random indices that satisfy the condition
    matching_indices = (torch.tensor(k_cluster_label)[:, None] == torch.tensor(k_cluster_label)).clone().detach()
    for i in range(batch_size):
        matching_indices_i = torch.nonzero(matching_indices[i]).squeeze()
        if matching_indices_i.numel() >= 1:
            # j, k = torch.tensor(np.random.choice(matching_indices_i.clone().detach().cpu().numpy(), 2))
            j = torch.from_numpy(np.random.choice(matching_indices_i.cpu().numpy(), 1, replace=False)).clone().detach()

            # Move tensors to CPU if they are on CUDA
            mixed_x[i] = lam * x[i] + (1 - lam) * x[j] if dataset_name in ("CIFAR10", "PCLCIFAR10", "CIFAR20", "PCLCIFAR20", "MNIST", "FashionMNIST", "KMNIST") else x[i]
            # mixed_x[i] = x[i] if dataset_name in ("CIFAR10", "CIFAR20") else x[i]
            y_a[i], y_b[i] = y[i], y[j]

            # Count the violent case when true label appearing in cl label
            if (y[i] == ytrue[i] or y[i] == ytrue[j] or y[j] == ytrue[j] or y[j] == ytrue[i]):
                count_error += 1
    return mixed_x, y_a, y_b, lam, count_error


def aug_intra_class(x, y, ytrue, k_cluster_label, device, dataset_name, alpha):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    count_error = 0

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size()[0]
    mixed_x = torch.zeros_like(x).to(device)
    
    # Precompute random indices that satisfy the condition
    matching_indices = (torch.tensor(k_cluster_label[:, None]) == torch.tensor(k_cluster_label)).clone().detach()
    if dataset_name in ("CIFAR20", "PCLCIFAR20"):
        label_y = torch.zeros(512, 20).to(device)
    else:
        label_y = torch.zeros(512, 10).to(device)

    for i in range(batch_size):
        matching_indices_i = torch.nonzero(matching_indices[i]).squeeze()
        if matching_indices_i.numel() >= 2:
            # j, k = torch.tensor(np.random.choice(matching_indices_i.clone().detach().cpu().numpy(), 2, replace=False))
            if dataset_name in ("CIFAR20", "PCLCIFAR20"):
                indices = torch.from_numpy(np.random.choice(matching_indices_i.cpu().numpy(), 1, replace=False)).clone().detach()
                # Define the indices and initialize the lam values
                lam_values = [0] * 20
                y_values = [0] * 20
            else:
                indices = torch.from_numpy(np.random.choice(matching_indices_i.cpu().numpy(), 1, replace=False)).clone().detach()
                # Define the indices and initialize the lam values
                lam_values = [0] * 10
                y_values = [0] * 10
            indices = torch.cat((torch.tensor([i]), indices))
            # Calculate distances and lam values in a loop
            distances = [ lam if euclidean_distance(x[i], x[index]) != 0 else (1 - lam) for index in indices]  # Reverse Distance Weight 13/01/2024
            # distances = [ 1 if euclidean_distance(x[i], x[index]) != 0 else 1 for index in indices] # Hard-Label 13/01/2024
            # Calculate lam values
            for idx, distance in enumerate(distances):
                lam_values[idx] = distance
            lam_values = torch.tensor(lam_values)

            # Perform weighted calculations for mixed_x
            if dataset_name in ("CIFAR10", "PCLCIFAR10", "CIFAR20", "PCLCIFAR20", "MNIST", "FashionMNIST", "KMNIST"):
                # mixed_x[i] = lam * x[indices[0]] + (1 - lam) * x[indices[1]]  # New Data 13/01/2024
                mixed_x[i] = x[i]  #Soft-label and hard-label for ablation study 13/01/2024
            else:
                mixed_x[i] = x[i]
            
            for idx, index in enumerate(indices):
                y_values[idx] = y[index]
            y_values = torch.tensor(y_values)

            label_y[i] = recalculate_lambda_label_sharing(y_values, lam_values)

            # Count the violent case when true label appearing in cl label
            # if (y[i] == ytrue[i] or y[i] == ytrue[j] or y[j] == ytrue[j] or y[j] == ytrue[i]):
            #     count_error += 1

    return mixed_x, label_y

def aug_intra_class_three_images(x, y, ytrue, k_cluster_label, device, dataset_name, alpha):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    count_error = 0

    if alpha > 0:
        # Generate three random mixing coefficients from a Beta distribution
        s = np.random.dirichlet((alpha, alpha, alpha), 1)
        lam1 = s[0][0]
        lam2 = s[0][1]
        lam3 = s[0][2]

        # lam = np.random.beta(alpha, alpha)
        # lam = alpha
    else:
        lam1 = 1/3
        lam2 = 1/3
        lam3 = 1/3
        # lam = 0.0
    
    batch_size = x.size()[0]
    mixed_x = torch.zeros_like(x).to(device)
    
    # Precompute random indices that satisfy the condition
    matching_indices = (torch.tensor(k_cluster_label[:, None]) == torch.tensor(k_cluster_label)).clone().detach()
    if dataset_name in ("CIFAR20", "PCLCIFAR20"):
        label_y = torch.zeros(512, 20).to(device)
    else:
        label_y = torch.zeros(512, 10).to(device)

    for i in range(batch_size):
        matching_indices_i = torch.nonzero(matching_indices[i]).squeeze()
        if matching_indices_i.numel() >= 2:
            # j, k = torch.tensor(np.random.choice(matching_indices_i.clone().detach().cpu().numpy(), 2, replace=False))
            if dataset_name in ("CIFAR20", "PCLCIFAR20"):
                indices = torch.from_numpy(np.random.choice(matching_indices_i.cpu().numpy(), 2, replace=False)).clone().detach()
                # Define the indices and initialize the lam values
                lam_values = [0] * 20
                y_values = [0] * 20
            else:
                indices = torch.from_numpy(np.random.choice(matching_indices_i.cpu().numpy(), 2, replace=False)).clone().detach()
                # Define the indices and initialize the lam values
                lam_values = [0] * 10
                y_values = [0] * 10
            indices = torch.cat((torch.tensor([i]), indices))

            # # Calculate distances and lam values in a loop
            # distances = [euclidean_distance(x[i], x[index]) if euclidean_distance(x[i], x[index]) != 0 else 30 for index in indices]  #Reverse Distance Weight
            # # distances = [1 if euclidean_distance(x[i], x[index]) != 0 else 1 for index in indices]  # Hard-Label

            # # Calculate lam values with inverse distance weighting
            # for idx, distance in enumerate(distances):
            #     lam_values[idx] = (1 / distance) / sum(1 / dist for dist in distances) # Reserve Distance Weight 13/01/2024
            #     # lam_values[idx] = (1 / distance)  # Hard-Label 13/01/2024 (ablation study)
            # lam_values = torch.tensor(lam_values)

            # Perform weighted calculations for mixed_x
            if dataset_name in ("CIFAR10", "PCLCIFAR10", "CIFAR20", "PCLCIFAR20", "MNIST", "FashionMNIST", "KMNIST"):
                # Lambda is equal inverse distance weighting
                # mixed_x[i] = sum(lam * x[idx] for lam, idx in zip(lam_values, indices))

                # Lambda is equal the Beta distribution from alpha
                # mixed_x[i] = lam * x[indices[0]] + (1 - lam) * x[indices[1]]

                # Lambda is equal a fix value
                mixed_x[i] = lam1 * x[indices[0]] + lam2 * x[indices[1]] + lam3 * x[indices[2]] # New Data 13/01/2024
                # mixed_x[i] = x[i] # Soft-Label 13/01/2024
            else:
                mixed_x[i] = x[i]# Calculate distances and lam values in a loop

            # Method that proposed in ICM paper    
            # distances = [euclidean_distance(x[i], x[index]) if euclidean_distance(x[i], x[index]) != 0 else 40 for index in indices]  #Reverse Distance Weight
            # # distances = [1 if euclidean_distance(x[i], x[index]) != 0 else 1 for index in indices]  # Hard-Label

            # # Calculate lam values with inverse distance weighting
            # for idx, distance in enumerate(distances):
            #     lam_values[idx] = (1 / distance) / sum(1 / dist for dist in distances) # Reserve Distance Weight 13/01/2024
            #     # lam_values[idx] = (1 / distance)  # Hard-Label 13/01/2024 (ablation study)
            # lam_values = torch.tensor(lam_values)
            # Handle distances that are zero by setting lam_values to 1

            # Adjust after discussion with CLLab members
            distances = [euclidean_distance(x[i], x[index]) for index in indices]  # Calculate all distances

            for idx, distance in enumerate(distances):
                if distance == 0:
                    lam_values[idx] = 1

            # Calculate lam values for non-zero distances
            non_zero_distances = [dist for dist in distances if dist != 0]
            if non_zero_distances:
                for idx, distance in enumerate(distances):
                    if distance != 0:
                        lam_values[idx] = (distance) / sum(dist for dist in non_zero_distances)

            # Convert lam_values to a tensor
            lam_values = torch.tensor(lam_values)

            # Lambda X = Lambda Y which generated by Dirichlet distribution
            # for idx, _ in enumerate(indices):
            #     lam_values[idx] = s[0][idx]
            # lam_values = torch.tensor(lam_values)

            
            for idx, index in enumerate(indices):
                y_values[idx] = y[index]
            y_values = torch.tensor(y_values, dtype=y.dtype)

            label_y[i] = recalculate_lambda_label_sharing(y_values, lam_values)

            # Count the violent case when true label appears in cl label
            # if (y[i] == ytrue[j] or y[i] == ytrue[k] or y[j] == ytrue[i] or y[j] == ytrue[k] or y[k] == ytrue[i] or y[k] == ytrue[j]):
            #     count_error += 1
    return mixed_x, label_y

def recalculate_lambda_label_sharing(y_values, lam_values):
    final_y_values = []

    # Convert y_values and lam_values to NumPy arrays for better performance
    y_values = np.array(y_values)
    lam_values = np.array(lam_values)

    # Initialize an array to store the final_lam values
    final_lam = np.zeros_like(y_values, dtype=float)

    # Use NumPy indexing and aggregation to calculate final_lam
    for i in range(len(y_values)):
        final_lam[i] = np.sum(lam_values[y_values == i])

    for index in final_lam:
        if index == 0:
            new_y = 0
        else:
            new_y = 1
        final_y_values.append(new_y)

    final_lam = torch.tensor(final_lam)
    final_y_values = torch.tensor(final_y_values)
    
    return final_y_values * final_lam

# def aug_intra_class_three_images(x, y, ytrue, k_cluster_label, device, dataset_name, alpha=1.0):
#     '''Returns mixed inputs, pairs of targets, and lambda'''
#     count_error = 0

#     if alpha > 0:
#         # Generate three random mixing coefficients from a Beta distribution
#         lam1 = np.random.beta(alpha, alpha)
#         lam2 = np.random.beta(alpha, alpha)
#         # Calculate the fourth mixing coefficient to satisfy the sum constraint
#         lam3 = 1 - lam1 - lam2
#         lam3 = max(0, lam3)  # Ensure lam3 is non-negative
#         # Normalize the coefficients to ensure they sum up to 1
#         total_lam = lam1 + lam2 + lam3
#         lam1 /= total_lam
#         lam2 /= total_lam
#         lam3 /= total_lam
#     else:
#         lam1 = 1/3
#         lam2 = 1/3
#         lam3 = 1/3
    
#     batch_size = x.size()[0]
#     mixed_x = torch.zeros_like(x).to(device)
#     y_a, y_b, y_c = torch.zeros_like(y).to(device), torch.zeros_like(y).to(device), torch.zeros_like(y).to(device)  #to(device)

#     # Precompute random indices that satisfy the condition
#     matching_indices = (torch.tensor(k_cluster_label)[:, None] == torch.tensor(k_cluster_label)).clone().detach()
#     lambda_1 = []
#     lambda_2 = []
#     lambda_3 = []

#     for i in range(batch_size):
#         matching_indices_i = torch.nonzero(matching_indices[i]).squeeze()
#         if matching_indices_i.numel() >= 2:
#             # j, k = torch.tensor(np.random.choice(matching_indices_i.clone().detach().cpu().numpy(), 2))
#             j, k = torch.from_numpy(np.random.choice(matching_indices_i.cpu().numpy(), 2)).clone().detach()
    #         indices = [i, j, k]
    #         # Calculate distances and lam values in a loop
    #         distances = [euclidean_distance(x[i], x[index]) if euclidean_distance(x[i], x[index]) != 0 else 30 for index in indices]

    #         # Calculate lam values
    #         for idx, distance in enumerate(distances):
    #             lam_values = [0] * 3
    #             lam_values[idx] = (1 / distance) / sum(1 / dist for dist in distances)
    #             if idx == 0:
    #                 lambda_1.append(lam_values[0])
    #             if idx == 1:
    #                 lambda_2.append(lam_values[1])
    #             if idx == 2:
    #                 lambda_3.append(lam_values[2])
    #         # lam_values = torch.tensor(lam_values)
    #         # import pdb
    #         # pdb.set_trace()

    #         # Move tensors to CPU if they are on CUDA
    #         mixed_x[i] = lam_values[0] * x[i] + lam_values[1] * x[j] + lam_values[2] * x[k] if dataset_name in ("CIFAR10", "CIFAR20") else x[i]
    #         # mixed_x[i] = x[i] if dataset_name in ("CIFAR10", "CIFAR20") else x[i]
    #         y_a[i], y_b[i], y_c[i] = y[i], y[j], y[k]

    #         # Count the violent case when true label appears in cl label
    #         if (y[i] == ytrue[j] or y[i] == ytrue[k] or y[j] == ytrue[i] or y[j] == ytrue[k] or y[k] == ytrue[i] or y[k] == ytrue[j]):
    #             count_error += 1

    # lambda_1 = sum(lambda_1) / len(lambda_1)
    # lambda_2 = sum(lambda_2) / len(lambda_2)
    # lambda_3 = sum(lambda_3) / len(lambda_3)

    # return mixed_x, y_a, y_b, y_c, lambda_1, lambda_2, lambda_3, count_error

def aug_intra_class_four_images(x, y, ytrue, k_cluster_label, device, dataset_name, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    count_error = 0

    if alpha > 0:
        # Generate three random mixing coefficients from a Beta distribution
        lam1 = np.random.beta(1.0, 1.0)
        lam2 = np.random.beta(1.0, 1.0)
        lam3 = np.random.beta(1.0, 1.0)
        # Calculate the fourth mixing coefficient to satisfy the sum constraint
        lam4 = 1 - lam1 - lam2 - lam3
        lam4 = max(0, lam4)  # Ensure lam4 is non-negative
        
        # Normalize the coefficients to ensure they sum up to 1
        total_lam = lam1 + lam2 + lam3 + lam4
        lam1 /= total_lam
        lam2 /= total_lam
        lam3 /= total_lam
        lam4 /= total_lam
    else:
        lam1 = 1/4
        lam2 = 1/4
        lam3 = 1/4
        lam4 = 1/4
    
    batch_size = x.size()[0]
    mixed_x = torch.zeros_like(x).to(device)
    y_a, y_b, y_c, y_d = torch.zeros_like(y).to(device), torch.zeros_like(y).to(device), torch.zeros_like(y).to(device), torch.zeros_like(y).to(device)  #to(device)
    # Precompute random indices that satisfy the condition
    matching_indices = (torch.tensor(k_cluster_label)[:, None] == torch.tensor(k_cluster_label)).clone().detach()

    for i in range(batch_size):
        matching_indices_i = torch.nonzero(matching_indices[i]).squeeze()
        if matching_indices_i.numel() >= 2:
            j, k, l = torch.from_numpy(np.random.choice(matching_indices_i.cpu().numpy(), 3)).clone().detach()

            # Move tensors to CPU if they are on CUDA
            mixed_x[i] = lam1 * x[i] + lam2 * x[j] + lam3 * x[k] + lam4 * x[l] if dataset_name in ("CIFAR10", "CIFAR20") else x[i]
            y_a[i], y_b[i], y_c[i], y_d[i] = y[i], y[j], y[k], y[l]

            # Count the violent case when true label appears in cl label
            if (y[i] == ytrue[j] or y[i] == ytrue[k] or y[i] == ytrue[l] or y[j] == ytrue[i] or y[j] == ytrue[k] or y[j] == ytrue[l] 
                or y[k] == ytrue[i] or y[k] == ytrue[j] or y[k] == ytrue[l] or y[l] == ytrue[i] or y[l] == ytrue[j] or y[l] == ytrue[k]):
                count_error += 1

    return mixed_x, y_a, y_b, y_c, y_d, lam1, lam2, lam3, lam4, count_error

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    y_a = y_a.squeeze()
    y_b = y_b.squeeze()
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def get_k(n1, n2, f):
    k1 = pow(n1, f)
    k2 = pow(n2, f)
    return k1, k2


def get_lambda(x, k1, k2):
    lambda_lower = 0.0
    t_lower = 1.0
    lambda_upper = 1.0
    t_upper = 0.0
    lambda_middle = k1 / (k1 + k2)
    t_middle = 0.5
    if x < lambda_middle:
        lambda_target = ((-t_middle) *
                         (x - lambda_lower) / lambda_middle) + t_lower
    elif x > lambda_middle:
        lambda_target = ((x - lambda_upper) * (t_middle - t_upper) /
                         (lambda_middle - lambda_upper))
    else:
        raise ValueError("[-] Check Boundary Case !")
    return lambda_target

def mamix_intra_aug(x, y, k_cluster_label, mamix_ratio, cls_num_list, device, alpha=1.0):
    if alpha > 0:
        lam_x = np.random.beta(alpha, alpha)
    else:
        lam_x = 1

    cls_num_list = torch.tensor(cls_num_list)

    batch_size = x.size()[0]
    # get the index from random permutation for mix x
    # index = torch.randperm(batch_size)

    # check will store the pair chosen for mixup with each other [batch, 2]
    # check = []
    # for i, j in enumerate(index):
    #     check.append([cls_num_list[y[i]].item(), cls_num_list[y[j]].item()])
    # check = torch.tensor(check)

    check = []
    for i in range(batch_size):
        while(True):
            j = random.randint(0, batch_size - 1)                                                         
            if k_cluster_label[i] == k_cluster_label[j]:
                check.append([cls_num_list[y[i]].item(), cls_num_list[y[j]].item(), j])
                break
    check = torch.tensor(check)

    # Now, we are going to compute lam_y for every pair
    lam_y = list()
    new_index = list()
    for i in range(check.size()[0]):
        # temp1 = n_i; temp2 = n_j
        temp1 = check[i][0].item()
        temp2 = check[i][1].item()
        new_index.append(check[i][2].item())

        f = mamix_ratio
        k1, k2 = get_k(temp1, temp2, f)
        lam_t = get_lambda(lam_x, k1, k2)

        lam_y.append(lam_t)

    lam_y = torch.tensor(lam_y).to(device)

    mixed_x = (1 - lam_x) * x + lam_x * x[new_index, :]
    y_a, y_b = y, y[new_index]

    return mixed_x, y_a, y_b, lam_x, lam_y

def mamix_criterion(criterion, pred, y_a, y_b, lam_y, args):
    loss = torch.mul(criterion(pred, y_a), lam_y) + torch.mul(
        criterion(pred, y_b), (1 - lam_y))

    return loss.mean()