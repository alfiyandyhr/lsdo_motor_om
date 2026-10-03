"""Voltage-limited maximum torque from the current-quartic discriminant."""

import numpy as np
import openmdao.api as om
from .._utils import ScalarState, bracketed_root, lift_root

QUARTIC_NAMES = ('A_quartic', 'B_quartic', 'C_quartic', 'D_quartic', 'E_quartic')
ELECTRICAL_NAMES = ('R_expanded', 'L_d_expanded', 'L_q_expanded', 'PsiF_expanded')


class ElectricalParameters(om.ExplicitComponent):
    """Broadcast scalar motor properties to independent operating nodes."""

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int, lower=1)

    def setup(self):
        n = self.options['num_nodes']
        for name, expanded in zip(('Rdc', 'L_d', 'L_q', 'PsiF'), ELECTRICAL_NAMES):
            self.add_input(name)
            self.add_output(expanded, shape=n)
            self.declare_partials(expanded, name, rows=np.arange(n), cols=np.zeros(n, int), val=1.)

    def compute(self, inputs, outputs):
        for name, expanded in zip(('Rdc', 'L_d', 'L_q', 'PsiF'), ELECTRICAL_NAMES):
            outputs[expanded] = inputs[name]


class TorqueLimitCoefficients(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('pole_pairs', default=6, types=int, lower=1)
        self.options.declare('V_lim', default=800., types=(int, float), lower=1e-12)
        self.options.declare('num_nodes', default=1, types=int, lower=1)

    def setup(self):
        n = self.options['num_nodes']
        for name in (*ELECTRICAL_NAMES, 'omega'):
            self.add_input(name, shape=n)
        for name in QUARTIC_NAMES:
            self.add_output(name, shape=n)
        self.declare_partials('*', '*', method='cs')

    def compute(self, x, outputs):
        p, v = self.options['pole_pairs'], self.options['V_lim']
        r, ld, lq, psi = (x[n] for n in ELECTRICAL_NAMES)
        w = x['omega']
        if np.any(np.real(w) <= 0) or np.any(np.real(ld-lq) == 0):
            raise om.AnalysisError('Torque limit requires positive omega and unequal d/q inductances.')
        den = 3*p*(ld-lq)
        a = den**2*((w*lq)**2+r**2)
        c1 = 12*p*w*r*(ld-lq)**2
        c2 = (3*p*psi)**2*(r**2+(w*lq)**2)-(v*den)**2
        d = -12*p*psi*(w**2*ld*lq+r**2)
        e = 4*((w*ld)**2+r**2)
        outputs['A_quartic'] = 256*a**2*e**3-128*a*e**2*c1**2+16*e*c1**4
        outputs['B_quartic'] = -256*a*e**2*c1*c2+144*a*d**2*e*c1+64*e*c1**3*c2-4*d**2*c1**3
        outputs['C_quartic'] = -128*a*e**2*c2**2+144*a*d**2*e*c2-27*a*d**4+96*c1**2*c2**2*e-12*d**2*c1**2*c2
        outputs['D_quartic'] = 64*e*c1*c2**3-12*d**2*c1*c2**2
        outputs['E_quartic'] = 16*e*c2**4-4*d**2*c2**3


class DiscreteCheck(om.ExplicitComponent):
    """Find a finite bracket beyond the largest stationary point of the quartic."""

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int, lower=1)

    def setup(self):
        n = self.options['num_nodes']
        for name in QUARTIC_NAMES:
            self.add_input(name, shape=n)
        self.add_output('lower_bracket', shape=n)
        self.add_output('upper_bracket', shape=n)
        # Bounds select a branch but do not enter the implicit residual.
        self.declare_partials('*', '*', method='cs')

    @staticmethod
    def evaluate_residual(x, a, b, c, d, e):
        return ((((a*x+b)*x+c)*x+d)*x+e)

    def compute(self, inputs, outputs):
        for i in range(self.options['num_nodes']):
            complex_coeff = np.array([inputs[name][i] for name in QUARTIC_NAMES])
            coeff = np.real(complex_coeff).copy()
            if not np.isfinite(coeff).all() or coeff[0] <= 0:
                raise om.AnalysisError(f'Torque limit node {i}: invalid quartic coefficients.')
            scale = np.max(np.abs(coeff))
            coeff /= scale
            roots = np.roots(np.polyder(coeff))
            real = roots.real[np.abs(roots.imag) < 1e-8*np.maximum(1, np.abs(roots.real))]
            lower = max(0., np.max(real))
            initial = np.polyval(coeff, lower)
            upper = lower+100.
            for _ in range(1000):
                if initial*np.polyval(coeff, upper) <= 0:
                    break
                upper += 100.
            else:
                raise om.AnalysisError(f'Torque limit node {i}: no positive torque root after 1000 bracket steps.')
            continued = lower
            if lower > 0:
                derivative = np.polyder(complex_coeff/scale)
                continued = lift_root(lower, lambda t: np.polyval(derivative, t),
                                      lambda t: np.polyval(np.polyder(derivative), t))
            outputs['lower_bracket'][i] = continued
            outputs['upper_bracket'][i] = continued+(upper-lower)


class MaxTorqueImplicitModel(ScalarState):
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int, lower=1)

    def setup(self):
        n = self.options['num_nodes']
        for name in (*QUARTIC_NAMES, 'lower_bracket', 'upper_bracket'):
            self.add_input(name, shape=n)
        self.setup_state('T_lim', QUARTIC_NAMES, val=100., shape=n)

    def residual(self, x, state):
        a, b, c, d, e = (x[name] for name in QUARTIC_NAMES)
        return ((((a/e*state+b/e)*state+c/e)*state+d/e)*state+1)/1e3

    def solve_real(self, inputs):
        result = []
        for i in range(self.options['num_nodes']):
            coeff = np.array([inputs[name][i] for name in QUARTIC_NAMES])
            coeff /= np.max(np.abs(coeff))
            result.append(bracketed_root(lambda t: np.polyval(coeff, t),
                                         inputs['lower_bracket'][i], inputs['upper_bracket'][i],
                                         f'Torque limit node {i}'))
        return result


class MaxTorqueModel(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int, lower=1)

    def setup(self):
        self.add_subsystem('implicit', MaxTorqueImplicitModel(num_nodes=self.options['num_nodes']),
                           promotes=['*'])


class TorqueLimitModel(om.Group):
    def initialize(self):
        self.options.declare('pole_pairs', default=6, types=int, lower=1)
        self.options.declare('V_lim', default=800., types=(int, float), lower=1e-12)
        self.options.declare('num_nodes', default=1, types=int, lower=1)
        self.options.declare('use_expanded', default=False, types=bool)

    def setup(self):
        n = self.options['num_nodes']
        if not self.options['use_expanded']:
            self.add_subsystem('electrical_parameters', ElectricalParameters(num_nodes=n), promotes=['*'])
        self.add_subsystem('coefficients', TorqueLimitCoefficients(
            pole_pairs=self.options['pole_pairs'], V_lim=self.options['V_lim'], num_nodes=n), promotes=['*'])
        self.add_subsystem('discrete_check', DiscreteCheck(num_nodes=n), promotes=['*'])
        self.add_subsystem('max_torque_model', MaxTorqueModel(num_nodes=n), promotes=['*'])
