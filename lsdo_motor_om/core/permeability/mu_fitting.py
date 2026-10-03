"""Fit the original silicon core iron B-H data without a CSDL dependency."""

from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit

DATA_FILE = Path(__file__).with_name('Magnetic_alloy_silicon_core_iron_C.tab')


def fit_dep_B(x, a, b, c):
    return (a*np.exp(b*x+c)+200)*x**1.4


def fit_dep_H(x, a, b=1., c=1., d=1.):
    # b, c, d were unused in the source; retain the four-coefficient interface.
    return a*np.tanh(x/300-.25)+.4


def fit_dep_H_old(x, coeff, order=None):
    return np.polyval(coeff, x)


@lru_cache(maxsize=8)
def _fit(path):
    data = np.genfromtxt(path, skip_header=1, delimiter='\t')
    h, b = data.T
    # The H fit has only one identifiable coefficient. The B fit similarly
    # identifies a*exp(c), so fit that amplitude with c fixed at one. These
    # are exactly the original function families, without singular covariance.
    basis = np.tanh(h[:15]/300-.25)
    amp_h = np.dot(basis, b[:15]-.4)/np.dot(basis, basis)
    def inverse(x, amplitude, exponent):
        return fit_dep_B(x, amplitude, exponent, 1.)
    amp_b, exponent = curve_fit(inverse, b, h, p0=[1., 1.], maxfev=10000)[0]
    return np.array([amp_h, 1., 1., 1.]), np.array([amp_b, exponent, 1.]), data


def permeability_fitting(file_name=None, test=False, order=10):
    """Return ``[B(H) coefficients, H(B) coefficients]``.

    ``file_name=None`` uses packaged data, independent of the working directory.
    ``order`` is retained for source compatibility; these fits are not polynomials.
    ``test=True`` additionally returns data and continuous plotting coordinates.
    """
    h, b, data = _fit(str(Path(file_name or DATA_FILE).resolve()))
    result = [h.copy(), b.copy()]
    if test:
        result.append(dict(H_data=data[:, 0].copy(), B_data=data[:, 1].copy(),
                           H_cont=np.linspace(0, data[:, 0].max(), 15000),
                           B_cont=np.linspace(0, data[:, 1].max(), 15000)))
    return result
