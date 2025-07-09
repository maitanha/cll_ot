import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10,6))

xpoints = np.array([0.1 * i for i in range(0, 11)])
ypoints = np.array([60.45,62.76,59.74,61.15,60.28,56.87,57.64,57.08,57.41,58.33,58.37])


# x_1points = np.array([0.1 * i for i in range(0, 11)])
# y_1points = np.array([41.04, 40.51, 40.92, 41.13, 41.45, 40.58, 40.21, 39.99, 38.67, 37.40, 34.80])

# x_2points = np.array([0.1 * i for i in range(0, 11)])
# y_2points = np.array([31.48, 34.17, 37.80, 38.86, 41.01, 40.74, 39.78, 39.58, 36.83, 33.43, 30.51])
# y_2points = np.array([66.47, 65.84, 66.47, 66.46, 65.74, 66.04, 65.45, 65.65, 65.65, 65.69, 66.60, 67.49, 65.12, 67.07, 65.79, 65.56, 65.31, 65.58, 65.65, 66.02])

x_1points = np.array([0.1 * i for i in range(1, 21)])
y_1points = np.array([77.89,77.72,77.61,77.65,77.17,77.18,76.95,76.97,76.94,77.06,76.46,76.82,76.67,76.79,76.61,76.34,76.47,76.63,76.58,76.26])

x_2points = np.array([0.1 * i for i in range(1, 21)])
y_2points = np.array([14.93,16.52,16.42,16.97,17.40,17.92,17.53,17.78,18.24,18.07,18.25,18.27,18.49,18.42,18.35,18.44,18.61,18.35,18.57,18.56])

# x_3points = np.array([0.1, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
# y_3points = np.array([77.68, 77.56, 77.95, 77.80, 77.65, 77.55, 78.03])

# plt.plot(xpoints, ypoints, label='Setup 2', color="red", marker='o', linewidth=2)
# plt.vlines(xpoints, 0, ypoints, linestyle="solid", color='gray', alpha=0.05)

# plt.plot(x_2points, y_2points, label='Setup 1', color="blue", marker='o', linewidth=2)
# plt.vlines(x_2points, 0, y_2points, linestyle="solid", color='gray', alpha=0.05)

plt.plot(x_1points, y_1points, label='Setup 2', color="red", marker='o', linewidth=2)
plt.vlines(x_1points, 0, y_1points, linestyle="solid", color='gray', alpha=0.05)

# plt.plot(x_3points, y_3points, label='Setup 2', color="red", marker='o', linewidth=2)
# plt.vlines(x_3points, 0, y_3points, linestyle="solid", color='gray', alpha=0.05)

min_margin = 1
max_margin = 1
plt.ylim(min(y_1points) - min_margin, max(y_1points) + max_margin)
# plt.ylim(min(y_2points) - min_margin, max(y_2points) + max_margin)
# plt.ylim(min(y_3points) - margin, max(y_3points) + margin)

# plt.xlim(min(xpoints), max(xpoints), margin)
# plt.grid(axis='y')
# plt.grid(axis='x')

# plt.xticks(xpoints)
plt.xticks(x_1points)
# plt.xticks(x_2points)
# plt.xticks(x_3points)

plt.legend()
plt.title('Relation between Alpha and performance in Cluster Mixup in MNIST')
plt.xlabel("Alpha in Dirichlet (alpha, alpha, alpha)")

# plt.title('Lambda in Cluster Mixup in KMNIST')
# plt.xlabel("Lambda")

plt.ylabel("Accuracy (%)")
plt.savefig('test.png')