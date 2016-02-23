import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

LAMBDA_MIN = 200.0
LAMBDA_MAX = 800.0
D_LAMBDA = 10.0

def getaw():
	aw = {}
	with open('waterabsorptiondata.csv', 'rb') as csvfile:
		myreader = csv.reader(csvfile, delimiter=',')
		for row in myreader:
			aw[float(row[0])] = float(row[1])
	return aw

def getbbw():
	bbw = {}
	with open('bbw.csv', 'rb') as csvfile:
		myreader = csv.reader(csvfile, delimiter=',')
		for row in myreader:
			bbw[float(row[0])] = float(row[4])*10**-3
	return bbw


def fit_data(x, y, xnew):

	plt.plot(x, y, 'ko', label="Original Noised Data")
	plt.show()

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

	plt.figure()
	plt.plot(x, y, 'ko', label="Original Noised Data")
	# plt.plot(xextrap,  np.exp(xextrap*popt[0] + popt[1]), 'r-', label="Fitted Curve")
	plt.plot(xnew, yout, 'b--')
	plt.legend()
	plt.show()

	return yout


AW = getaw()
BBW =  getbbw()
bbw = []
llw = []
for l in np.sort(BBW.keys()):
	llw.append(l)
	bbw.append(BBW[l])

llw = np.asarray(llw)
bbw = np.asarray(bbw)


# fit_data(llw, bbw)

ABSORPTION_SPECTRUM = {640.0: {'a1': 0.0685, 'a0': 0.3331}, 390.0: {'a1': 0.0235, 'a0': 0.5813},
					   520.0: {'a1': 0.0981, 'a0': 0.6327}, 650.0: {'a1': 0.0713, 'a0': 0.3502},
					   400.0: {'a1': 0.0205, 'a0': 0.6843}, 530.0: {'a1': 0.0969, 'a0': 0.5681},
					   660.0: {'a1': 0.1128, 'a0': 0.561}, 410.0: {'a1': 0.0129, 'a0': 0.7782},
					   540.0: {'a1': 0.09, 'a0': 0.5046}, 670.0: {'a1': 0.1595, 'a0': 0.8435},
					   420.0: {'a1': 0.006, 'a0': 0.8637}, 550.0: {'a1': 0.0781, 'a0': 0.4262},
					   680.0: {'a1': 0.1388, 'a0': 0.7485}, 430.0: {'a1': 0.002, 'a0': 0.9603},
					   560.0: {'a1': 0.0659, 'a0': 0.3433}, 690.0: {'a1': 0.0812, 'a0': 0.389},
					   440.0: {'a1': 0.0, 'a0': 1.0}, 570.0: {'a1': 0.06, 'a0': 0.295},
					   700.0: {'a1': 0.0317, 'a0': 0.136}, 450.0: {'a1': 0.006, 'a0': 0.9634},
					   580.0: {'a1': 0.0581, 'a0': 0.2784}, 710.0: {'a1': 0.0128, 'a0': 0.0545},
					   460.0: {'a1': 0.0109, 'a0': 0.9311}, 590.0: {'a1': 0.054, 'a0': 0.2595},
					   720.0: {'a1': 0.005, 'a0': 0.025}, 470.0: {'a1': 0.0157, 'a0': 0.8697},
					   600.0: {'a1': 0.0495, 'a0': 0.2389}, 480.0: {'a1': 0.0152, 'a0': 0.789},
					   610.0: {'a1': 0.0578, 'a0': 0.2745}, 490.0: {'a1': 0.0256, 'a0': 0.7558},
					   620.0: {'a1': 0.0674, 'a0': 0.3197}, 500.0: {'a1': 0.0559, 'a0': 0.7333},
					   630.0: {'a1': 0.0718, 'a0': 0.3421}, 510.0: {'a1': 0.0865, 'a0': 0.6911}}
lambdas = np.sort(ABSORPTION_SPECTRUM.keys())
BACKSCATTERING_SPECTRUM = {}

Y = 2.2
bp400 = .1
a440 = 0.08
bbp = np.zeros(len(ABSORPTION_SPECTRUM))
aphyto = np.zeros(len(ABSORPTION_SPECTRUM))
aw = np.zeros(len(ABSORPTION_SPECTRUM))
# bbw = np.zeros(len(ABSORPTION_SPECTRUM))
for i in range(0,len(lambdas)):
	l = lambdas[i]
	BACKSCATTERING_SPECTRUM[l] = np.power(400.0/l, Y)
	aphyto[i] = a440*ABSORPTION_SPECTRUM[l]['a1']
	bbp[i] = np.power(400.0/l, Y)*bp400
	aw[i] = AW[l]
	# bbw[i] = BBW[l]


lambda_new = np.arange(LAMBDA_MIN, LAMBDA_MAX + D_LAMBDA, D_LAMBDA)


aphyto = fit_data(lambdas, aphyto, lambda_new)
aw = fit_data(lambdas, aw, lambda_new)
lambdas = lambda_new

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

# plt.plot(lambdas, rs)
# plt.show()
