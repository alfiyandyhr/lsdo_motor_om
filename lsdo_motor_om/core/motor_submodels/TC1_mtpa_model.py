"""Maximum torque per ampere (MTPA) current control."""

import numpy as np
import openmdao.api as om
from .._utils import ScalarState, bracketed_root, lift_root


def mtpa_star(t):
    """Positive normalized q-current root; complex-step safe on the motoring branch."""
    real = float(np.real(t))
    if real < 0:
        raise om.AnalysisError('MTPA supports nonnegative motoring torque.')
    if real == 0:
        return t*0
    root = bracketed_root(lambda q: q**4+real*q-real**2, 0., max(1., np.sqrt(real)), 'MTPA')
    return lift_root(root, lambda q: q**4+t*q-t**2, lambda q: 4*q**3+t)


def mtpa_currents(torque, ld, lq, psi, p):
    base = -psi/(ld-lq)
    if np.real(base) <= 0:
        raise om.AnalysisError('MTPA requires PsiF > 0 and L_q > L_d.')
    tstar = torque/(1.5*p*psi*base)
    qstar = mtpa_star(tstar)
    iq = qstar*base
    # Equivalent to the torque equation, without cancellation/division at zero torque.
    id_ = base*(1-np.sqrt(1+4*qstar**2))/2
    return id_, iq


class MTPAImplicitModel(ScalarState):
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int, lower=1)

    def setup(self):
        n = self.options['num_nodes']
        self.add_input('T_em_star', shape=n)
        self.setup_state('Iq_MTPA_star', ['T_em_star'], val=.1, shape=n)

    def residual(self, x, state):
        # Same positive root as q**4+t*q-t**2=0, regular also at zero torque.
        return state*(1+np.sqrt(1+4*state**2))/2-x['T_em_star']

    def solve_real(self, x):
        return [mtpa_star(t) for t in x['T_em_star']]


class MTPAParameters(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('pole_pairs', default=6, types=int, lower=1)
        self.options.declare('num_nodes', default=1, types=int, lower=1)

    def setup(self):
        n = self.options['num_nodes']
        for name in ('T_em', 'L_d_expanded', 'L_q_expanded', 'PsiF_expanded'):
            self.add_input(name, shape=n)
        for name in ('I_base_expanded', 'T_em_star', 'MTPA_upper_bracket'):
            self.add_output(name, shape=n)
        self.declare_partials('*', '*', method='cs')

    def compute(self, x, outputs):
        base = -x['PsiF_expanded']/(x['L_d_expanded']-x['L_q_expanded'])
        if np.any(np.real(base) <= 0):
            raise om.AnalysisError('MTPA requires PsiF > 0 and L_q > L_d.')
        outputs['I_base_expanded'] = base
        outputs['T_em_star'] = x['T_em']/(1.5*self.options['pole_pairs']*x['PsiF_expanded']*base)
        outputs['MTPA_upper_bracket'] = 50*base


class MTPAModel(om.Group):
    def initialize(self):
        self.options.declare('pole_pairs', default=6, types=int, lower=1)
        self.options.declare('num_nodes', default=1, types=int, lower=1)

    def setup(self):
        n = self.options['num_nodes']
        self.add_subsystem('parameters', MTPAParameters(pole_pairs=self.options['pole_pairs'], num_nodes=n), promotes=['*'])
        self.add_subsystem('MTPAImplicitModel', MTPAImplicitModel(num_nodes=n), promotes=['*'])
        self.add_subsystem('dimensionalize', om.ExecComp(
            ['Iq_MTPA=Iq_MTPA_star*I_base_expanded',
             'Id_MTPA=I_base_expanded*(1-(1+4*Iq_MTPA_star**2)**0.5)/2'],
            do_coloring=False,
            **{k: {'shape': n} for k in ('Iq_MTPA', 'Id_MTPA', 'Iq_MTPA_star', 'I_base_expanded')}), promotes=['*'])
