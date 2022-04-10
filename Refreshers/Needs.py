import numpy as np

ghgh = (1, 2, 3)
gh = np.shape(ghgh)

ee = np.random.RandomState(1).normal(
    loc=0.0, scale=0.07, size=1 + ghgh[1])

print(ee)
