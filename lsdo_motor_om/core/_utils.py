"""Numerical helpers shared by the native OpenMDAO components."""

import numpy as np
import openmdao.api as om
from scipy.optimize import brentq

MOTOR_VARIABLE_NAMES = (
    'outer_stator_radius', 'pole_pitch', 'tooth_pitch', 'air_gap_depth', 'l_ef',
    'rotor_radius', 'turns_per_phase', 'Acu', 'tooth_width', 'height_yoke_stator',
    'slot_bottom_width', 'slot_height', 'slot_width_inner', 'Tau_y', 'L_j1', 'Kdp1',
    'bm', 'Am_r', 'phi_r', 'lambda_m', 'alpha_i', 'Kf', 'K_phi', 'K_theta', 'A_f2',
)


def scalar(value):
    return np.asarray(value).reshape(-1)[0]


def smooth_min(a, b, rho=20.0):
    """The original CSDL minimum (a stable KS smooth minimum)."""
    shift = np.minimum(np.real(a), np.real(b))
    return shift - np.log(np.exp(-rho*(a-shift)) + np.exp(-rho*(b-shift)))/rho


def sigmoid(x):
    """Overflow-safe logistic function that retains complex-step perturbations."""
    x = np.asarray(x)
    positive = np.real(x) >= 0
    z = np.exp(np.where(positive, -x, x))
    return np.where(positive, 1/(1+z), z/(1+z))


def bracketed_root(function, lower, upper, label):
    """Solve on a real bracket, with a useful error for infeasible inputs."""
    lower, upper = float(np.real(lower)), float(np.real(upper))
    if not np.isfinite([lower, upper]).all() or lower > upper:
        raise om.AnalysisError(f'{label}: invalid bracket [{lower}, {upper}].')
    lo, hi = function(lower), function(upper)
    if not np.isfinite([lo, hi]).all():
        raise om.AnalysisError(f'{label}: nonfinite residual at bracket endpoints.')
    if lo == 0:
        return lower
    if hi == 0:
        return upper
    if np.sign(lo) == np.sign(hi):
        raise om.AnalysisError(
            f'{label}: no root in [{lower:.6g}, {upper:.6g}]; '
            f'endpoint residuals are {lo:.6g}, {hi:.6g}.')
    return brentq(function, lower, upper, xtol=1e-13, rtol=1e-14, maxiter=200)


def lift_root(root, function, derivative):
    """Continue a real root for complex-step inputs using implicit differentiation."""
    for _ in range(3):
        root = root - function(root)/derivative(root)
    return root


class ScalarState(om.ImplicitComponent):
    """A scalar implicit equation with a local bracketed nonlinear solve.

    Subclasses define ``state_name``, ``residual(inputs, state)`` and
    ``solve_real(inputs)``. Residual partials, including the state Jacobian,
    are computed by complex step; OpenMDAO's DirectSolver handles linear solves.
    The nonlinear solve also supports complex step of the entire model.
    """

    def setup_state(self, state_name, dependencies, val=0.5, shape=1):
        self.state_name = state_name
        self.add_output(state_name, val=val, shape=shape)
        self.declare_partials(state_name, [state_name, *dependencies], method='cs')
        self.linear_solver = om.DirectSolver()

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals[self.state_name] = self.residual(inputs, outputs[self.state_name])

    def solve_nonlinear(self, inputs, outputs):
        real_inputs = {name: np.real(inputs[name]).copy() for name in inputs}
        root = np.asarray(self.solve_real(real_inputs)).reshape(outputs[self.state_name].shape)
        if inputs._under_complex_step:
            # Each node's equation is independent. Differentiate the residual
            # with respect to all states at once to obtain its diagonal.
            h = 1e-30
            jac = np.imag(self.residual(real_inputs, root.astype(complex)+1j*h))/h
            if np.any(np.abs(jac) < 1e-25):
                raise om.AnalysisError(f'{self.pathname}: singular state derivative.')
            root = root.astype(complex)
            for _ in range(3):
                root -= self.residual(inputs, root)/jac
        outputs[self.state_name] = root


def declare_motor_options(component, fitting=False):
    for name, default in [('pole_pairs', 6), ('phases', 3), ('num_slots', 36)]:
        component.options.declare(name, default=default, types=int, lower=1)
    component.options.declare('rated_current', default=123., types=(int, float), lower=1e-12)
    if fitting:
        component.options.declare('fit_coeff_dep_H', default=None, allow_none=True)
        component.options.declare('fit_coeff_dep_B', default=None, allow_none=True)


def fitting_options(options):
    from .permeability.mu_fitting import permeability_fitting
    h, b = options['fit_coeff_dep_H'], options['fit_coeff_dep_B']
    if h is None or b is None:
        default_h, default_b = permeability_fitting()
        h = default_h if h is None else h
        b = default_b if b is None else b
    if len(h) < 1 or len(b) != 3:
        raise ValueError('Permeability fits require at least one H coefficient and three B coefficients.')
    return np.asarray(h), np.asarray(b)


class MotorVariablesModel(om.ExplicitComponent):
    """Unpack the original 25-entry sizing vector, preserving its ordering."""

    def setup(self):
        self.add_input('motor_variables', shape=25)
        for i, name in enumerate(MOTOR_VARIABLE_NAMES):
            self.add_output(name)
            self.declare_partials(name, 'motor_variables', rows=[0], cols=[i], val=1.)

    def compute(self, inputs, outputs):
        for i, name in enumerate(MOTOR_VARIABLE_NAMES):
            outputs[name] = inputs['motor_variables'][i]
