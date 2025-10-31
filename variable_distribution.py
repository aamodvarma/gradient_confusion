import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

a = "/home/ajrv/Projects/research/davenport/gradient_confusion/data/poster-graphs/mnist_d5_w50_e10_lr0.001_c10-0.9.npz"

data = np.load(a, allow_pickle=True)
cosin_sims_numbered = data["cosin_sims_numbered"].item()
n = 1
main = cosin_sims_numbered[n]

plt.figure(figsize=(8, 5))
for num in range(10):
    if num == n:
        continue
    seven = cosin_sims_numbered[num]
    sub = np.abs(np.subtract(main, seven))
    sub = np.subtract(sub, np.min(sub)) / (np.max(sub) - np.min(sub))

    kde1 = gaussian_kde(sub)
    x_vals = np.linspace(-1, 1, 200)
    y1 = kde1(x_vals)
    plt.plot(x_vals, y1, label="Number " + str(num), linewidth=2)

plt.title("Density")
plt.xlabel("Value")
plt.ylabel("Density")
plt.grid(True)
plt.legend()
plt.show()
