import torch
import random
import numpy as np

def euclidean_distance(vector1, vector2):
    # Flatten the 3D tensor to 1D using .view()
    tensor1 = vector1.view(-1)
    tensor2 = vector2.view(-1)
    distance = torch.norm(tensor1 - tensor2)
    return distance

def cosine_similarity(vector1, vector2):
    # Flatten the 3D tensor to 1D using .view()
    vector1 = vector1.view(-1)
    vector2 = vector2.view(-1)

    # Normalize the vectors to unit length
    vector1 = vector1 / torch.norm(vector1)
    vector2 = vector2 / torch.norm(vector2)

    # Calculate the dot product and cosine similarity
    dot_product = torch.dot(vector1, vector2)
    cosine_distance = 1 - dot_product  # Cosine distance

    return cosine_distance

def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

def mixup_cl_data(x, y, ytrue, device, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1
    
    batch_size = x.size(0)
    mixed_x = torch.zeros_like(x).to(device)
    y_a, y_b = torch.zeros_like(y).to(device), torch.zeros_like(y).to(device)
    
    for i in range(batch_size):
        while True:
            j = random.randint(0, batch_size - 1)
            if y[i] == y[j]:
                mixed_x[i] = lam * x[i] + (1 - lam) * x[j]
                y_a[i], y_b[i] = y[i], y[j]
                break
    return mixed_x, y_a, y_b, lam

def mixup_cl_data_count_error(x, y, ytrue, device, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    count_error = 0
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1
    
    batch_size = x.size(0)
    mixed_x = torch.zeros_like(x).to(device)
    y_a, y_b = torch.zeros_like(y).to(device), torch.zeros_like(y).to(device)
    
    for i in range(batch_size):
        j = random.randint(0, batch_size - 1)
        mixed_x[i] = lam * x[i] + (1 - lam) * x[j]
        y_a[i], y_b[i] = y[i], y[j]

        if (y[i] == ytrue[i] or y[i] == ytrue[j] or y[j] == ytrue[j] or y[j] == ytrue[i]):
            count_error += 1
    
    return mixed_x, y_a, y_b, lam, count_error

def intra_class_count_error(x, y, ytrue, k_cluster_label, device, dataset_name, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    count_error = 0

    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1
    
    batch_size = x.size(0)
    mixed_x = torch.zeros_like(x).to(device)
    y_a, y_b = torch.zeros_like(y).to(device), torch.zeros_like(y).to(device)
    
    k_cluster_tensor = torch.tensor(k_cluster_label, device=device)
    matching_indices = (k_cluster_tensor[:, None] == k_cluster_tensor).clone().detach()

    for i in range(batch_size):
        matching_indices_i = torch.nonzero(matching_indices[i]).squeeze()
        if matching_indices_i.numel() >= 1:
            j = matching_indices_i[torch.randint(0, matching_indices_i.size(0), (1,))]
            mixed_x[i] = lam * x[i] + (1 - lam) * x[j]
            y_a[i], y_b[i] = y[i], y[j]

            if (y[i] == ytrue[i] or y[i] == ytrue[j] or y[j] == ytrue[j] or y[j] == ytrue[i]):
                count_error += 1
    
    return mixed_x, y_a, y_b, lam, count_error

def aug_intra_class(x, y, ytrue, k_cluster_label, device, dataset_name, alpha):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    mixed_x = torch.zeros_like(x).to(device)

    k_cluster_tensor = torch.tensor(k_cluster_label, device=device)
    matching_indices = (k_cluster_tensor[:, None] == k_cluster_tensor).clone().detach()

    if dataset_name in ("CIFAR20", "PCLCIFAR20"):
        label_y = torch.zeros(512, 20, device=device)
    else:
        label_y = torch.zeros(512, 10, device=device)

    for i in range(batch_size):
        matching_indices_i = torch.nonzero(matching_indices[i]).squeeze()
        if matching_indices_i.numel() >= 2:
            indices = matching_indices_i[torch.randperm(matching_indices_i.size(0))[:1]]

            indices = torch.cat((torch.tensor([i], device=device), indices))
            distances = [lam if euclidean_distance(x[i], x[index]) != 0 else (1 - lam) for index in indices]

            lam_values = torch.tensor(distances, device=device)

            if dataset_name in ("CIFAR10", "PCLCIFAR10", "CIFAR20", "PCLCIFAR20", "MNIST", "FashionMNIST", "KMNIST"):
                mixed_x[i] = lam * x[indices[0]] + (1 - lam) * x[indices[1]]
            else:
                mixed_x[i] = x[i]
            
            y_values = torch.stack([y[index] for index in indices], dim=0)
            label_y[i] = recalculate_lambda_label_sharing(y_values, lam_values)
    
    return mixed_x, label_y

def aug_intra_class_three_images(x, y, ytrue, k_cluster_label, device, dataset_name, alpha):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    count_error = 0

    if alpha > 0:
        # Generate three random mixing coefficients from a Dirichlet distribution using PyTorch
        s = torch.distributions.Dirichlet(torch.tensor([alpha, alpha, alpha], device=device)).sample()
        lam1, lam2, lam3 = s[0], s[1], s[2]
    else:
        lam1, lam2, lam3 = 1/3, 1/3, 1/3
    
    batch_size = x.size(0)
    mixed_x = torch.zeros_like(x).to(device)
    
    # Precompute random indices that satisfy the condition and move to GPU
    k_cluster_tensor = torch.tensor(k_cluster_label, device=device)
    matching_indices = (k_cluster_tensor[:, None] == k_cluster_tensor).to(device)

    # Initialize label_y on GPU
    if dataset_name in ("CIFAR20", "PCLCIFAR20"):
        label_y = torch.zeros(512, 20, device=device)
    else:
        label_y = torch.zeros(512, 10, device=device)

    for i in range(batch_size):
        matching_indices_i = torch.nonzero(matching_indices[i]).squeeze()
        if matching_indices_i.numel() >= 2:
            # Select two random indices using PyTorch functions
            selected_indices = torch.randperm(matching_indices_i.size(0), device=device)[:2]
            j, k = matching_indices_i[selected_indices]

            indices = torch.cat((torch.tensor([i], device=device), torch.tensor([j], device=device), torch.tensor([k], device=device)))

            # Perform weighted calculations for mixed_x
            if dataset_name in ("CIFAR10", "PCLCIFAR10", "CIFAR20", "PCLCIFAR20", "MNIST", "FashionMNIST", "KMNIST"):
                mixed_x[i] = lam1 * x[indices[0]] + lam2 * x[indices[1]] + lam3 * x[indices[2]]
            else:
                mixed_x[i] = x[i]

            # Calculate distances and lam values
            distances = [euclidean_distance(x[i], x[index]) for index in indices]

            # Initialize lam_values for label adjustment
            lam_values = torch.zeros(len(indices), device=device)

            # Adjust lam values based on distances
            for idx, distance in enumerate(distances):
                if distance == 0:
                    lam_values[idx] = 1

            non_zero_distances = [dist for dist in distances if dist != 0]
            if non_zero_distances:
                for idx, distance in enumerate(distances):
                    if distance != 0:
                        lam_values[idx] = distance / sum(non_zero_distances)

            lam_values = lam_values.to(device)

            # Assign values for label_y
            y_values = torch.stack([y[index] for index in indices], dim=0).to(device)
            label_y[i] = recalculate_lambda_label_sharing(y_values, lam_values)

            # Count the violation case when true label appears in cluster label
            if (y[i] == ytrue[j] or y[i] == ytrue[k] or y[j] == ytrue[i] or y[j] == ytrue[k] or y[k] == ytrue[i] or y[k] == ytrue[j]):
                count_error += 1

    return mixed_x, label_y

def recalculate_lambda_label_sharing(y_values, lam_values):
    # y_values should be a tensor of shape [num_samples, num_classes], and lam_values should have matching shape
    num_classes = y_values.size(1)  # Number of classes (e.g., 10 for CIFAR-10)
    final_lam = torch.zeros(num_classes, dtype=torch.float32, device=lam_values.device)

    # Compute the weighted sum for each class
    for i in range(num_classes):
        class_mask = y_values[:, i]  # Mask for class i (should be 1 for the respective class)
        final_lam[i] = torch.sum(class_mask * lam_values)  # Weighted sum for class i

    # Create a binary final_y_values (1 if any weight is assigned, 0 otherwise)
    final_y_values = (final_lam > 0).float()

    return final_y_values * final_lam

def aug_intra_class_four_images(x, y, ytrue, k_cluster_label, device, dataset_name, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    count_error = 0

    if alpha > 0:
        # Generate three random mixing coefficients from a Beta distribution using PyTorch
        lam1 = torch.distributions.Beta(1.0, 1.0).sample().item()
        lam2 = torch.distributions.Beta(1.0, 1.0).sample().item()
        lam3 = torch.distributions.Beta(1.0, 1.0).sample().item()
        
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
        lam1 = lam2 = lam3 = lam4 = 1 / 4
    
    batch_size = x.size(0)
    mixed_x = torch.zeros_like(x).to(device)
    
    # Prepare tensors for labels
    y_a, y_b, y_c, y_d = (torch.zeros_like(y).to(device) for _ in range(4))

    # Ensure k_cluster_label is a tensor and move it to GPU
    k_cluster_label = torch.tensor(k_cluster_label, device=device)
    
    # Precompute matching indices on the GPU
    matching_indices = (k_cluster_label[:, None] == k_cluster_label).to(device)

    for i in range(batch_size):
        matching_indices_i = torch.nonzero(matching_indices[i]).squeeze()
        if matching_indices_i.numel() >= 3:
            # Randomly select 3 distinct indices from the matching set using PyTorch
            selected_indices = torch.randperm(matching_indices_i.size(0), device=device)[:3]
            j, k, l = matching_indices_i[selected_indices]

            # Perform the mixing based on dataset name
            if dataset_name in ("CIFAR10", "CIFAR20"):
                mixed_x[i] = lam1 * x[i] + lam2 * x[j] + lam3 * x[k] + lam4 * x[l]
            else:
                mixed_x[i] = x[i]

            # Assign the corresponding labels
            y_a[i], y_b[i], y_c[i], y_d[i] = y[i], y[j], y[k], y[l]

            # Check for violations
            if any([y[i] == ytrue[x] for x in [j, k, l]]) or \
               any([y[j] == ytrue[x] for x in [i, k, l]]) or \
               any([y[k] == ytrue[x] for x in [i, j, l]]) or \
               any([y[l] == ytrue[x] for x in [i, j, k]]):
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
    lambda_middle = k1 / (k1 + k2)
    t_middle = 0.5

    if x < lambda_middle:
        lambda_target = (-t_middle * (x - 0.0) / lambda_middle) + 1.0
    elif x > lambda_middle:
        lambda_target = ((x - 1.0) * (t_middle - 0.0) / (lambda_middle - 1.0))
    else:
        raise ValueError("[-] Check Boundary Case !")
    return lambda_target

def mamix_intra_aug(x, y, k_cluster_label, mamix_ratio, cls_num_list, device, alpha=1.0):
    if alpha > 0:
        lam_x = torch.distributions.Beta(alpha, alpha).sample().item()  # Replace np.random.beta with torch
    else:
        lam_x = 1

    cls_num_list = torch.tensor(cls_num_list, device=device)

    batch_size = x.size(0)

    check = []
    for i in range(batch_size):
        while True:
            j = random.randint(0, batch_size - 1)
            if k_cluster_label[i] == k_cluster_label[j]:
                check.append([cls_num_list[y[i]].item(), cls_num_list[y[j]].item(), j])
                break
    check = torch.tensor(check, device=device)

    # Compute lam_y for every pair
    lam_y = []
    new_index = []
    for i in range(check.size(0)):
        temp1 = check[i][0].item()  # n_i
        temp2 = check[i][1].item()  # n_j
        new_index.append(check[i][2].item())  # j index

        # Compute k1, k2 and lambda
        f = mamix_ratio
        k1, k2 = get_k(temp1, temp2, f)
        lam_t = get_lambda(lam_x, k1, k2)
        lam_y.append(lam_t)

    lam_y = torch.tensor(lam_y, device=device)

    # Perform mixup for x
    new_index = torch.tensor(new_index, device=device)
    mixed_x = (1 - lam_x) * x + lam_x * x[new_index, :]

    y_a, y_b = y, y[new_index]

    return mixed_x, y_a, y_b, lam_x, lam_y

def mamix_criterion(criterion, pred, y_a, y_b, lam_y, args):
    # Weighted criterion using lambda values
    loss = torch.mul(criterion(pred, y_a), lam_y) + torch.mul(criterion(pred, y_b), (1 - lam_y))
    return loss.mean()