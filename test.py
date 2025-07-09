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
#     y_a, y_b, y_c = torch.zeros_like(x).to(device), torch.zeros_like(y).to(device), torch.zeros_like(y).to(device)  #to(device)
#     # Precompute random indices that satisfy the condition
#     matching_indices = (torch.tensor(k_cluster_label)[:, None] == torch.tensor(k_cluster_label)).clone().detach()
#     lambda_y = []
#     label_y = []

#     for i in range(batch_size):
#         matching_indices_i = torch.nonzero(matching_indices[i]).squeeze()
#         if matching_indices_i.numel() >= 2:
#             # j, k = torch.tensor(np.random.choice(matching_indices_i.clone().detach().cpu().numpy(), 2))
#             j, k, l, n, m, q, r, u, v = torch.from_numpy(np.random.choice(matching_indices_i.cpu().numpy(), 9)).clone().detach()

#             # Define the indices and initialize the lam values
#             indices = [i, j, k, l, n, m, q, r, u, v]
#             lam_values = [0] * 10
#             y_values = [0] * 10

#             # Calculate distances and lam values in a loop
#             distances = [euclidean_distance(x[i], x[index]) if euclidean_distance(x[i], x[index]) != 0 else 10 for index in indices]

#             # Calculate lam values
#             for idx, distance in enumerate(distances):
#                 lam_values[idx] = (1 / distance) / sum(1 / dist for dist in distances)
#             lam_values = torch.tensor(lam_values)

#             # Now, lam_values contains the lam values for each index
#             # lam1, lam2, lam3, lam4, lam5, lam6, lam7, lam8, lam9, lam10 = lam_values

#             # # Move tensors to CPU if they are on CUDA
#             # mixed_x[i] = lam1 * x[i] + lam2 * x[j] + lam3 * x[k] if dataset_name in ("CIFAR10", "CIFAR20") else x[i]
#             # y_a[i], y_b[i], y_c[i] = y[i], y[j], y[k]

#             # Perform weighted calculations for mixed_x
#             if dataset_name in ("CIFAR10", "CIFAR20"):
#                 mixed_x[i] = x[i] #sum(lam * x[idx] for lam, idx in zip(lam_values, indices))
#             else:
#                 mixed_x[i] = x[i]

#             y_values = torch.tensor([y[index] for index in indices])

#             lam_values, y_values = recalculate_lambda_label_sharing(y_values, lam_values)

#             lambda_y.append(lam_values)
#             label_y.append(y_values)

#             # Count the violent case when true label appears in cl label
#             # if (y[i] == ytrue[j] or y[i] == ytrue[k] or y[j] == ytrue[i] or y[j] == ytrue[k] or y[k] == ytrue[i] or y[k] == ytrue[j]):
#             #     count_error += 1

# return mixed_x, label_y, lambda_y

# def recalculate_lambda_label_sharing(y_values, lam_values):
#     final_lam = []
#     final_y_values = []
#     for i in range(len(y_values)):
#         i_value = 0
#         for j, l in zip(y_values, lam_values):
#             if i == j:
#                 i_value = i_value + l
#         final_lam.append(i_value)

#     for y, index in zip(range(len(y_values)), final_lam):
#         if index == 0:
#             new_y = 0
#         else:
#             new_y = y
#         final_y_values.append(new_y)
#     final_lam = torch.tensor(final_lam)
#     final_y_values = torch.tensor(final_y_values)
    
#     return final_lam, final_y_values

# import numpy as np
# # idx = [1, 3, 6]
# # value_to_add = 9  # Replace 9 with the value you want to add
# # idx = [value_to_add] +  idx
# # print(idx)  # This will print: [1, 3, 6, 9]
# alpha = 1
# s = np.random.dirichlet((alpha, alpha, alpha, alpha), 1)
# print(s[0][0])
# print(s[0][1])
# print(s[0][2])
# print(s[0][3])

# import torch
# import numpy as np

# num_classes = 10
# eta = 0.2

# T = np.array(torch.full([num_classes, num_classes], (1-eta)/(num_classes-1)))
# for i in range(num_classes):
#     T[i][i] = eta
# for i in range(num_classes):
#     T[i] /= sum(T[i])
# T = torch.full([num_classes, num_classes], 1/(num_classes-1))
# for i in range(num_classes):
#     T[i][i] = 0

# # print(T)
# cls_num = 20
# img_max = 100
# imb_factor = 1/10
# img_num_per_cls = []

# for cls_idx in range(cls_num):
#     num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
#     img_num_per_cls.append(int(num))

# T_bias = img_num_per_cls.copy()
# for i in range(cls_num - 1):
# 	T_bias =  np.vstack((T_bias, img_num_per_cls))
# for i in range(cls_num):
# 	T_bias[i][i] = 0.0

# # Need to add dtype=float, otherwise gets all 0 T_bias
# T_bias = np.array(T_bias, dtype=float)
# for i in range(cls_num):
# 	T_bias[i, :] = T_bias[i, :] / np.sum(T_bias[i, :])

# print(img_num_per_cls)	
# print(T_bias)

# ord_label_1 = np.array([2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500])

# comp_label_1 = ord_label_1 @ T_bias
# print(comp_label_1)

# print("-------------------------------")
# print(sum(comp_label_1))

# import pdb
# pdb.set_trace()


# # T = torch.full([num_classes, num_classes], 1/(num_classes-1))
# # for i in range(num_classes):
# # 	T[i][i] = 0

# # print(T)


# ord_label_2 = np.array([5000, 2997, 1796, 1077, 645, 387, 232, 139, 83, 50])

# T = np.full([10, 10], 1/9)
# for i in range(10):
#     T[i][i] = 0

# print(T)

# comp_label_2 = ord_label_2 @ T
# print(comp_label_2)

# print("-------------------------------")
# print(sum(comp_label_2))

#-----------------------------------------------------------
# [[0.         0.26923077 0.19230769 0.15384615 0.11538462 0.07692308 0.07692308 0.03846154 0.03846154 0.03846154]
#  [0.34482759 0.         0.17241379 0.13793103 0.10344828 0.06896552 0.06896552 0.03448276 0.03448276 0.03448276]
#  [0.32258065 0.22580645 0.         0.12903226 0.09677419 0.06451613 0.06451613 0.03225806 0.03225806 0.03225806]
#  [0.3125     0.21875    0.15625    0.         0.09375    0.0625     0.0625     0.03125    0.03125    0.03125   ]
#  [0.3030303  0.21212121 0.15151515 0.12121212 0.         0.06060606 0.06060606 0.03030303 0.03030303 0.03030303]
#  [0.29411765 0.20588235 0.14705882 0.11764706 0.08823529 0.         0.05882353 0.02941176 0.02941176 0.02941176]
#  [0.29411765 0.20588235 0.14705882 0.11764706 0.08823529 0.05882353 0.         0.02941176 0.02941176 0.02941176]
#  [0.28571429 0.2        0.14285714 0.11428571 0.08571429 0.05714286 0.05714286 0.         0.02857143 0.02857143]
#  [0.28571429 0.2        0.14285714 0.11428571 0.08571429 0.05714286 0.05714286 0.02857143 0.         0.02857143]
#  [0.28571429 0.2        0.14285714 0.11428571 0.08571429 0.05714286 0.05714286 0.02857143 0.02857143 0.        ]]


# import torch

# # Assuming you have a tensor representing the vector
# x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

# # Calculate the mean of the vector
# mean_x = torch.tensor([1.7, 2.2, 1.7, 1.5, 3.1])

# N = 2

# variance_mic = torch.mean(torch.pow(x - mean_x, 2))

# import pdb
# pdb.set_trace()

# # Calculate the variance of the vector
# variance_x = torch.mean((x - mean_x) ** 2)

# print("Variance:", variance_x.item())

# import torch

# # Assuming you have mini-batch stochastic gradient and true gradient tensors
# # For example, let's create random tensors for illustration
# batch_size = 64
# sg = torch.randn((batch_size,))  # Mini-batch stochastic gradient
# true_gradient = torch.randn((batch_size,))  # True gradient

# # Calculate MSE
# mse = torch.mean((sg - true_gradient)**2)

# print("Mean Squared Error:", mse.item())


# Retry creating the visual representation
import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Define main categories and their connections
categories = ["LibCLL Toolkit", "Synthetic Datasets", "Real-World Datasets", "Models", "CL Assumptions", "CLL Algorithms"]
synthetic_datasets = ["MNIST", "KMNIST", "FMNIST", "Yeast", "Texture", "Dermatology", "Control", "CIFAR10", "CIFAR20", "MicroImageNet10", "MicroImageNet20"]
real_world_datasets = ["CLCIFAR10", "CLCIFAR20", "CLMicroImageNet10", "CLMicroImageNet20"]
models = ["ResNet18", "ResNet34", "DenseNet", "MLP", "Linear"]
cl_assumptions = ["Uniform", "Biased", "Noisy", "MCL"]
cll_algorithms = ["SCL-NL", "SCL-EXP", "URE-NN/GA", "DM", "MCL-MAE", "MCL-EXP", "MCL-LOG", "FWD", "URE-TNN/TGA", "CPE-I/F/T"]

# Add nodes and edges
G.add_node("LibCLL Toolkit")
for category in categories[1:]:
    G.add_node(category)
    G.add_edge("LibCLL Toolkit", category)

for dataset in synthetic_datasets:
    G.add_node(dataset)
    G.add_edge("Synthetic Datasets", dataset)

for dataset in real_world_datasets:
    G.add_node(dataset)
    G.add_edge("Real-World Datasets", dataset)

for model in models:
    G.add_node(model)
    G.add_edge("Models", model)

for assumption in cl_assumptions:
    G.add_node(assumption)
    G.add_edge("CL Assumptions", assumption)

for algorithm in cll_algorithms:
    G.add_node(algorithm)
    G.add_edge("CLL Algorithms", algorithm)

# Plot the graph
plt.figure(figsize=(14, 10))
pos = nx.spring_layout(G, seed=42)  # Adjust layout
nx.draw_networkx_nodes(G, pos, node_size=500, node_color="lightblue")
nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=15, edge_color="gray")
nx.draw_networkx_labels(G, pos, font_size=8, font_color="black")
plt.title("LibCLL Toolkit Structure", fontsize=16)
plt.axis("off")
plt.show()
