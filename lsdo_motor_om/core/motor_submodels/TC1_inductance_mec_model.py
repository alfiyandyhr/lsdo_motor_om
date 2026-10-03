"""Nonlinear q-axis MEC and explicit d/q inductance equations."""

import numpy as np
import openmdao.api as om
from .._utils import ScalarState, bracketed_root, declare_motor_options, fitting_options, scalar
from ..permeability.mu_fitting import fit_dep_B

Q_INPUTS = ('alpha_i', 'pole_pitch', 'l_ef', 'K_theta', 'air_gap_depth', 'tooth_pitch',
            'tooth_width', 'slot_height', 'height_yoke_stator', 'Kaq', 'Kdp1',
            'turns_per_phase', 'L_j1')


def q_current(x, phi_aq, fit_b, p, m):
    baq = phi_aq/(x['alpha_i']*x['pole_pitch']*x['l_ef'])
    fsigma = 1.6*baq*x['K_theta']*x['air_gap_depth']/(np.pi*4e-7)
    bt = baq*x['tooth_pitch']/x['tooth_width']/.95
    ft = 2*fit_dep_B(bt, *fit_b)*x['slot_height']
    by = phi_aq/(2*x['l_ef']*x['height_yoke_stator'])
    fy = 2*fit_dep_B(by, *fit_b)*x['L_j1']
    return p*(fsigma+ft+fy)/(.9*m*x['Kaq']*x['Kdp1']*x['turns_per_phase'])


class InductanceQImplicitModel(ScalarState):
    def initialize(self):
        declare_motor_options(self, fitting=True)
        self.options.declare('rated_d_current', default=None, allow_none=True)

    def setup(self):
        _, self.fit_b = fitting_options(self.options)
        self.id_rated = self.options['rated_d_current']
        if self.id_rated is None:
            self.id_rated = self.options['rated_current']*np.sin(.6283)
        for name in (*Q_INPUTS, 'phi_air'):
            self.add_input(name)
        self.add_input('eps', val=1e-5)
        self.setup_state('phi_aq', Q_INPUTS, val=.001)

    def residual(self, inputs, state):
        iq = q_current(inputs, state, self.fit_b, self.options['pole_pairs'], self.options['phases'])
        return (self.id_rated**2+iq**2-self.options['rated_current']**2)/self.options['rated_current']**2

    def solve_real(self, inputs):
        return bracketed_root(lambda phi: self.residual(inputs, phi).item(),
                              scalar(inputs['eps']), scalar(inputs['phi_air']), 'Inductance MEC')


class InductanceOutputs(om.ExplicitComponent):
    def initialize(self):
        declare_motor_options(self, fitting=True)

    def setup(self):
        _, self.fit_b = fitting_options(self.options)
        names = (*Q_INPUTS, 'phi_aq', 'phi_air', 'F_total', 'F_delta', 'slot_bottom_width',
                 'slot_width_inner', 'Tau_y', 'Kf', 'K_sigma_air', 'lambda_n',
                 'lambda_leak_standard', 'Am_r', 'K_phi')
        for name in names:
            self.add_input(name)
        for name in ('I_q_temp', 'inductance_residual', 'Kad', 'f_a', 'L_d', 'L_q'):
            self.add_output(name)
        self.declare_partials('*', '*', method='cs')

    def compute(self, x, outputs):
        p, m, z = (self.options[n] for n in ('pole_pairs', 'phases', 'num_slots'))
        iw = self.options['rated_current']
        id_ = iw*np.sin(.6283)
        fi = 3000*p/60
        iq = q_current(x, x['phi_aq'], self.fit_b, p, m)
        outputs['I_q_temp'] = iq
        outputs['inductance_residual'] = id_**2+iq**2-iw**2
        kst = x['F_total']/x['F_delta']
        cx = 4*np.pi*fi*(np.pi*4e-7)*x['l_ef']*(x['Kdp1']*x['turns_per_phase'])**2/p
        lam_u = .0008/x['slot_bottom_width']+.0024/(x['slot_bottom_width']+x['slot_width_inner'])
        xs = 2*p*m*(lam_u+.45)*cx/(z*x['Kdp1']**2)
        xd1 = m*x['pole_pitch']*.1*cx/(x['air_gap_depth']*x['K_theta']*kst*(np.pi*x['Kdp1'])**2)
        xe = .47*cx*(x['l_ef']+.02-.64*x['Tau_y'])/(x['l_ef']*x['Kdp1']**2)
        x1 = xs+xd1+xe
        kad = 1/x['Kf']
        fad = .35*m*kad*x['Kdp1']*x['turns_per_phase']*id_/p
        fa = fad/(x['K_sigma_air']*.004*847138)
        eo = 4.44*fi*x['Kdp1']*x['turns_per_phase']*x['phi_air']*x['K_phi']
        bmn = x['lambda_n']*(1-fa)/(x['lambda_n']+1)
        phi_n = (bmn-(1-bmn)*x['lambda_leak_standard'])*x['Am_r']*1.1208
        ed = 4.44*fi*x['Kdp1']*x['turns_per_phase']*phi_n*x['K_phi']
        xad = np.sqrt((eo-ed)**2)/id_/np.sqrt(2)
        eaq = x['phi_aq']*eo/x['phi_air']
        outputs['Kad'] = kad
        outputs['f_a'] = fa
        outputs['L_d'] = (xad+x1)/(2*np.pi*fi)
        outputs['L_q'] = (eaq/iq+x1)/(2*np.pi*fi)


class InductanceModel(om.Group):
    def initialize(self):
        declare_motor_options(self, fitting=True)

    def setup(self):
        opts = {n: self.options[n] for n in ('pole_pairs', 'phases', 'num_slots',
                                            'rated_current', 'fit_coeff_dep_H', 'fit_coeff_dep_B')}
        self.add_subsystem('q_axis_factor', om.ExecComp('Kaq=.36/Kf'), promotes=['*'])
        self.add_subsystem('q_axis_implicit', InductanceQImplicitModel(**opts), promotes=['*'])
        self.add_subsystem('post_processing', InductanceOutputs(**opts), promotes=['*'])
