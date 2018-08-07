import numpy as np

mps = np.load("mapping.npz")

mps = mps["arr_0"].tolist()

ks = []

for tds in mps:
	ks.append(tds)

mapping = {}

for k in ks:
	mapping[(mps[k][0], mps[k][1])] = k.split(",")