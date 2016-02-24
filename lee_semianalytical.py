import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

# Default lambda Values
_lambda_min = 390.0
_lambda_max = 720.0
_d_lambda = 10.0
_lambdas = np.arange(_lambda_min, _lambda_max + _d_lambda, _d_lambda)


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
	xinterp = xnew[xnew >= np.min(x)]
	xinterp = xinterp[xinterp <= np.max(x)]
	xextrap1 = xnew[xnew < np.min(x)]
	xextrap2 = xnew[xnew > np.max(x)]
	popt, pcov = curve_fit(func, x, np.log(y))
	f2 = interp1d(x, y, kind='cubic')
	yextrap1 = np.exp(xextrap1*popt[0] + popt[1])
	yextrap2 = np.exp(xextrap2*popt[0] + popt[1])
	yinterp = f2(xinterp)
	yout = np.concatenate((yextrap1, yinterp, yextrap2))
	return yout


def fit_generic(x, y, xnew):
	f2 = interp1d(x, y, kind='cubic')
	plt.plot(x, y, 'ko')
	yout = f2(xnew)
	return yout


class RadModel:
	def __init__(self):
		# Parameters
		self.Y = 2.2
		self.g0 = 0.084
		self.g1 = 0.170
		self.g2 = 1.0
		self.alpha0 = 1.0
		self.alpha1 = 1 / np.pi
		self.theta_w = np.pi
		self.D0 = 1.03
		self.D1 = 2.4
		self.D0_ = 1.04
		self.D1_ = 5.4
		self.rho = 0.3

		# Load imperically found spectra
		self.awater = np.genfromtxt('smithbaker_aw.csv', delimiter=',')  # Absorption of pure water
		self.lambda_aw = np.genfromtxt('smithbaker_lambda.csv', delimiter=',')
		phyto_spectrum = np.genfromtxt('lee_aphyto.csv', delimiter=',')  # Phytoplankton reference spectrum
		self.lambda_phyto = phyto_spectrum[:, 0]
		self.aphyto = phyto_spectrum[:, 1]
		scattering_spectrum = np.genfromtxt('lee_bbw.csv', delimiter=',')  # Backscattering of pure water
		self.lambda_bbw = scattering_spectrum[:, 0]
		self.bbwater = scattering_spectrum[:, 4]

	def spectrum(self, turb, chla, H, lambdas=_lambdas):

		# Select the relevant output wavelengths
		n_ones = np.ones(lambdas.size)

		# Calculate reference optical parameters
		ap440 = 0.06 * np.power(chla, 0.65)  # From Lee 1998
		ag440 = 0.05  # Suggested in Lee 1999
		bp400 = turb * turb  # Totally made up needs

		# Interpolate and extroplate data
		aw = fit_exponential(self.lambda_aw, self.awater, lambdas)
		aphyto = fit_generic(self.lambda_phyto, self.aphyto, lambdas)
		bbw = fit_exponential(self.lambda_bbw, self.bbwater, lambdas)

		# Calculate inherant optical properties
		ap = ap440 * self.aphyto
		ag = ag440 * np.exp(-0.014*(lambdas - 440))
		bbp = np.power(400.0/lambdas, self.Y*n_ones) * bp400
		a = aphyto + aw + ag
		bb = bbp + bbw

		kappa = a + bb
		u = bb / (a + bb)
		rs_dp = (self.g0 + self.g1 * np.power(u, self.g2*n_ones)) * u

		rs = rs_dp * (1 - self.alpha0 * np.exp(-1 * (1 / np.cos(self.theta_w) +
			self.D0 * np.power(1 + self.D1 * u, 0.5*n_ones)) * kappa * H)) + \
			self.alpha1 * self.rho * np.exp(-1 * (1 / np.cos(self.theta_w) +
			self.D0_ * np.power(1 + self.D1_ * u, 0.5*n_ones)) * kappa * H)

		return rs
