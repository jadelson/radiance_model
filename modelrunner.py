from lee_semianalytical import RadModel
from matplotlib import pyplot as plt

import numpy as np

r = RadModel()

LAMBDA_MIN = 390.0
LAMBDA_MAX = 720.0
D_LAMBDA = 1.0
lambdas1 = np.arange(LAMBDA_MIN, LAMBDA_MAX + D_LAMBDA, D_LAMBDA)

lambdas2 = np.arange(LAMBDA_MIN, LAMBDA_MAX + 10, 20)

rs1 = r.spectrum(.2, 3, 100, lambdas1)
r.mean_turb = 9
rs2 = r.spectrum(.2, 3, 1, lambdas1)


plt.plot(lambdas1, rs1, 'b-')
plt.plot(lambdas2, rs2, 'r-')
plt.show()

