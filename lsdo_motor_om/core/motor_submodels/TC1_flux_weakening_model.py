"""Voltage-boundary currents and brackets for flux weakening."""

import numpy as np
import openmdao.api as om
from .._utils import ScalarState, bracketed_root, lift_root, smooth_min
from .TC1_torque_limit_model import ELECTRICAL_NAMES


def fw_coefficients(t, w, r, ld, lq, psi, p, v):
    d = 3*p*(ld-lq)
    return np.array([
        d**2*(r**2+(w*ld)**2),
        18*p**2*psi*(ld-lq)*(r**2+w**2*ld*(2*ld-lq)),
        (3*p*psi)**2*(r**2+(w*(2*ld-lq))**2)-(3*p*v*(ld-lq))**2
        +6*p*w*(ld-lq)*(3*psi**2*p*w*ld+2*r*t*(ld-lq)),
        6*p*w*psi*(3*psi**2*p*w*(2*ld-lq)+4*r*t*(ld-lq))
        -18*(p*v)**2*psi*(ld-lq),
        (2*t*lq*w)**2+(3*psi**2*p*w)**2+(2*r*t)**2
        +12*psi**2*p*w*r*t-(3*p*v*psi)**2,
    ])


def fw_upper(w, r, ld, lq, psi, v):
    asymp = -psi/(ld-lq)
    arg = v**2*(w**2*ld**2+r**2)-(r*w*psi)**2
    if np.real(arg) < 0:
        raise om.AnalysisError('Flux weakening: voltage ellipse has no d-axis intercept.')
    voltage = (-w**2*psi*ld+np.sqrt(arg))/(r**2+(w*ld)**2)
    return asymp, voltage, smooth_min(asymp, voltage)


def fw_root(coeff, lower, upper):
    """Select the upper d-current intersection, as in the source's bracket."""
    real = np.real(coeff)
    scale = np.max(np.abs(real))
    if not scale:
        raise om.AnalysisError('Flux weakening: zero polynomial.')
    normalized = real/scale
    root = bracketed_root(lambda d: np.polyval(normalized, d), lower, upper, 'Flux weakening')
    return lift_root(root, lambda d: np.polyval(coeff/scale, d),
                     lambda d: np.polyval(np.polyder(coeff/scale), d))


def fw_currents(t, w, r, ld, lq, psi, p, v, lower):
    _, _, upper = fw_upper(w, r, ld, lq, psi, v)
    id_ = fw_root(fw_coefficients(t, w, r, ld, lq, psi, p, v), lower, upper)
    iq = t/(1.5*p*(psi+(ld-lq)*id_))
    return id_, iq


class FluxWeakeningCoefficients(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('pole_pairs', default=6, types=int, lower=1)
        self.options.declare('V_lim', default=800., types=(int, float), lower=1e-12)
        self.options.declare('num_nodes', default=1, types=int, lower=1)

    def setup(self):
        n = self.options['num_nodes']
        for name in (*ELECTRICAL_NAMES, 'T_em', 'omega'):
            self.add_input(name, shape=n)
        for name in ('a1', 'a2', 'a3', 'a4', 'a5', 'I_d_asymp', 'I_d_voltage_upper_lim', 'Id_upper_lim'):
            self.add_output(name, shape=n)
        self.add_output('I_d_upper_bracket_list', shape=(n, 2))
        self.declare_partials('*', '*', method='cs')

    def compute(self, x, outputs):
        p, v = self.options['pole_pairs'], self.options['V_lim']
        for i in range(self.options['num_nodes']):
            r, ld, lq, psi = (x[n][i] for n in ELECTRICAL_NAMES)
            coeff = fw_coefficients(x['T_em'][i], x['omega'][i], r, ld, lq, psi, p, v)
            for j in range(5):
                outputs[f'a{j+1}'][i] = coeff[j]
            asymp, voltage, upper = fw_upper(x['omega'][i], r, ld, lq, psi, v)
            outputs['I_d_asymp'][i] = asymp
            outputs['I_d_voltage_upper_lim'][i] = voltage
            outputs['Id_upper_lim'][i] = upper
            outputs['I_d_upper_bracket_list'][i] = [asymp, voltage]


class FluxWeakeningImplicitModel(ScalarState):
    def initialize(self):
        self.options.declare('pole_pairs', default=6, types=int, lower=1)
        self.options.declare('V_lim', default=800., types=(int, float), lower=1e-12)
        self.options.declare('num_nodes', default=1, types=int, lower=1)

    def setup(self):
        n = self.options['num_nodes']
        for name in ('a1', 'a2', 'a3', 'a4', 'a5', 'Id_fw_bracket', 'Id_upper_lim'):
            self.add_input(name, shape=n)
        self.setup_state('Id_fw', ['a1', 'a2', 'a3', 'a4', 'a5'], val=-100., shape=n)

    def residual(self, x, state):
        # Fixed scaling avoids a singular normalization when a5 crosses zero.
        return ((((x['a1']*state+x['a2'])*state+x['a3'])*state+x['a4'])*state+x['a5'])/1e6

    def solve_real(self, x):
        return [fw_root(np.array([x[f'a{j}'][i] for j in range(1, 6)]),
                        x['Id_fw_bracket'][i], x['Id_upper_lim'][i])
                for i in range(self.options['num_nodes'])]


class FluxWeakeningModel(om.Group):
    def initialize(self):
        self.options.declare('pole_pairs', default=6, types=int, lower=1)
        self.options.declare('V_lim', default=800., types=(int, float), lower=1e-12)
        self.options.declare('num_nodes', default=1, types=int, lower=1)

    def setup(self):
        opts = {n: self.options[n] for n in ('pole_pairs', 'V_lim', 'num_nodes')}
        n, p = opts['num_nodes'], opts['pole_pairs']
        self.add_subsystem('coefficients', FluxWeakeningCoefficients(**opts), promotes=['*'])
        self.add_subsystem('implicit', FluxWeakeningImplicitModel(**opts), promotes=['*'])
        self.add_subsystem('q_current', om.ExecComp(
            f'Iq_fw=T_em/({1.5*p}*(PsiF_expanded+(L_d_expanded-L_q_expanded)*Id_fw))',
            do_coloring=False,
            **{k: {'shape': n} for k in ('Iq_fw', 'T_em', 'PsiF_expanded', 'L_d_expanded',
                                       'L_q_expanded', 'Id_fw')}), promotes=['*'])


class FluxWeakeningBracketCoefficients(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('pole_pairs', default=6, types=int, lower=1)
        self.options.declare('V_lim', default=800., types=(int, float), lower=1e-12)
        self.options.declare('num_nodes', default=1, types=int, lower=1)

    def setup(self):
        n = self.options['num_nodes']
        for name in (*ELECTRICAL_NAMES, 'T_lim', 'omega'):
            self.add_input(name, shape=n)
        for name in ('a_bracket', 'c_bracket', 'd_bracket', 'e_bracket'):
            self.add_output(name, shape=n)
        self.declare_partials('*', '*', method='cs')

    def compute(self, x, outputs):
        p, v = self.options['pole_pairs'], self.options['V_lim']
        r, ld, lq, psi = (x[n] for n in ELECTRICAL_NAMES)
        w, t = x['omega'], x['T_lim']
        den = 3*p*(ld-lq)
        outputs['a_bracket'] = den**2*((w*lq)**2+r**2)
        outputs['c_bracket'] = (3*p*psi)**2*(r**2+(w*lq)**2)+12*p*w*r*t*(ld-lq)**2-(v*den)**2
        outputs['d_bracket'] = -12*p*psi*t*(r**2+w**2*ld*lq)
        outputs['e_bracket'] = 4*t**2*(r**2+(w*ld)**2)


class FluxWeakeningBracketImplicitModel(ScalarState):
    """Solve the stationary-current equation at the double-root torque limit.

    At T_lim the original current quartic has a repeated root and a singular
    state Jacobian. Its derivative cubic identifies the same root stably.
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int, lower=1)

    def setup(self):
        n = self.options['num_nodes']
        for name in ('a_bracket', 'c_bracket', 'd_bracket', 'e_bracket'):
            self.add_input(name, shape=n)
        self.setup_state('Iq_fw_bracket', ['a_bracket', 'c_bracket', 'd_bracket'], val=100., shape=n)

    def residual(self, x, q):
        return (4*x['a_bracket']*q**3+2*x['c_bracket']*q+x['d_bracket'])/1e3

    def solve_real(self, x):
        result = []
        for i in range(self.options['num_nodes']):
            roots = np.roots([4*x['a_bracket'][i], 0., 2*x['c_bracket'][i], x['d_bracket'][i]])
            valid = roots.real[(np.abs(roots.imag) < 1e-7) & (roots.real > 0)]
            if not len(valid):
                raise om.AnalysisError(f'Flux weakening bracket node {i}: no positive stationary current.')
            result.append(np.max(valid))
        return result


class FluxWeakeningBracketModel(om.Group):
    """Consume the original a/c/d/e bracket coefficients and expose Id/Iq bounds."""

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int, lower=1)
        self.options.declare('pole_pairs', default=6, types=int, lower=1)

    def setup(self):
        n, p = self.options['num_nodes'], self.options['pole_pairs']
        self.add_subsystem('implicit', FluxWeakeningBracketImplicitModel(num_nodes=n), promotes=['*'])
        self.add_subsystem('d_current', om.ExecComp(
            f'Id_fw_bracket=(2*T_lim/({3*p}*Iq_fw_bracket)-PsiF_expanded)/(L_d_expanded-L_q_expanded)',
            do_coloring=False,
            **{k: {'shape': n} for k in ('Id_fw_bracket', 'T_lim', 'Iq_fw_bracket', 'PsiF_expanded',
                                       'L_d_expanded', 'L_q_expanded')}), promotes=['*'])
