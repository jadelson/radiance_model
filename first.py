import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

LAMBDA_MIN = 390.0
LAMBDA_MAX = 720.0
D_LAMBDA = 10.0
Y = 2.2
bp400 = .1
a440 = 0.08



def fit_exponential(x, y, xnew):

	"""
	Spline interpolation and exponential fit extrapolation

	:param x:
	:param y:
	:param xnew:
	:return:
	"""

	def func(x, a, b):
		return a*x + b

	scale = 1
	x = x/scale
	xnew = xnew/scale
	xinterp = xnew[xnew >= np.min(x)]
	xinterp = xinterp[xinterp <= np.max(x)]
	xextrap1 = xnew[xnew < np.min(x)]
	xextrap2 = xnew[xnew > np.max(x)]
	popt, pcov = curve_fit(func, x, np.log(y))
	x = x * scale
	xinterp = xinterp * scale
	popt[1] = popt[1] / scale
	f2 = interp1d(x, y, kind='cubic')
	yextrap1 = np.exp(xextrap1*popt[0] + popt[1])
	yextrap2 = np.exp(xextrap2*popt[0] + popt[1])
	yinterp = f2(xinterp)
	yout = np.concatenate((yextrap1, yinterp, yextrap2))
	return yout

def fit_generic(x, y, xnew):
	f2 = interp1d(x, y, kind='cubic')
	plt.plot(x,y,'ko')
	xnew = np.arange(np.min(x), np.max(x) + D_LAMBDA, D_LAMBDA)
	yout = f2(xnew)
	return yout

# Select the relevant output wavelengths
lambdas = np.arange(LAMBDA_MIN, LAMBDA_MAX + D_LAMBDA, D_LAMBDA)
N = lambdas.size

# Load imperically found spectra
aw = np.genfromtxt('smithbaker_aw.csv', delimiter=',')
lambda_aw = np.genfromtxt('smithbaker_lambda.csv', delimiter=',')
PHYTO = np.genfromtxt('lee_aphyto.csv', delimiter=',')
lambda_phyto = PHYTO[:, 0]
aphyto = a440*PHYTO[:, 1]

SCATTERING_SPECTRUM = np.genfromtxt('lee_bbw.csv', delimiter=',')
lambda_bbw = SCATTERING_SPECTRUM[:,0]
bbw = SCATTERING_SPECTRUM[:,4]




# fit_exponential(llw, bbw)

BACKSCATTERING_SPECTRUM = {}


bbp = np.zeros(lambda_phyto.size)

for i in range(0, len(lambdas)):
	l = lambdas[i]
	BACKSCATTERING_SPECTRUM[l] = np.power(400.0/l, Y)
	bbp[i] = np.power(400.0/l, Y)*bp400
	# bbw[i] = BBW[l]


aw = fit_exponential(lambda_aw, aw, lambdas)
aphyto = fit_generic(lambda_phyto, aphyto, lambdas)



# aw = fit_exponential(lambda_phyto, aw, lambdas)

H = 10
step = 0.1

mean_turb = 4
turb = np.random.randn(10, 1) + 4
x = np.arange(0, -1 * H - step, -1 * step)
g0 = 0.084
g1 = 0.170
g2 = 1.0
alpha0 = 1.0
alpha1 = 1 / np.pi
theta_w = np.pi
D0 = 1.03
D1 = 2.4
D0_ = 1.04
D1_ = 5.4
rho = 0.3

a = aphyto + aw
bb = bbp

kappa = a + bb
u = bb / (a + bb)
rs_dp = (g0 + g1 * np.power(u, g2)) * u

rs = rs_dp * (1 - alpha0 * np.exp(-1 * (1 / np.cos(theta_w) + D0 * np.power(1 + D1 * u, 0.5)) * kappa * H)) + \
	 alpha1 * rho * np.exp(-1 * (1 / np.cos(theta_w) + D0_ * np.power(1 + D1_ * u, 0.5)) * kappa * H)

plt.plot(lambdas, rs)
plt.show()
