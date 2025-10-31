import numpy as np


a = "/home/ajrv/Projects/research/davenport/gradient_confusion/data/poster-graphs/mnist_d5_w10_e10_lr0.001_c10-0.9.npz"

data = np.load(a, allow_pickle=True)
cosin_sims = data["cosin_sims"]
print(len(cosin_sims))
print(cosin_sims)
print(len(cosin_sims))

cosin_sims_numbered = data["cosin_sims_numbered"].item()
losses = data["losses"]

step_interval = len(losses) // len(cosin_sims_numbered[0])
steps_cosin = [i * step_interval for i in range(len(cosin_sims_numbered[0]))]

print(len(steps_cosin))
