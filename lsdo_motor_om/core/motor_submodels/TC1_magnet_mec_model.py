"""Nonlinear permanent-magnet magnetic equivalent circuit (MEC)."""

import numpy as np
import openmdao.api as om
from .._utils import ScalarState, bracketed_root, fitting_options
from ..permeability.mu_fitting import fit_dep_B, fit_dep_H

MAGNET_INPUTS = ('tooth_pitch', 'tooth_width', 'slot_height', 'alpha_i',
                 'pole_pitch', 'l_ef', 'height_yoke_stator', 'L_j1', 'air_gap_depth',
                 'K_theta', 'A_f2', 'bm', 'phi_r', 'lambda_m')


def magnet_quantities(x, b_delta, fit_h, fit_b):
    bt = b_delta*x['tooth_pitch']/x['tooth_width']/.95
    ft = 2*fit_dep_B(bt, *fit_b)*x['slot_height']
    phi_air = x['alpha_i']*x['pole_pitch']*x['l_ef']*b_delta
    by = phi_air/(2*x['height_yoke_stator']*x['l_ef'])
    hy = fit_dep_B(by, *fit_b)
    fy = 2*x['L_j1']*hy
    fdelta = 1.6*b_delta*(.0001+x['K_theta']*x['air_gap_depth'])/(np.pi*4e-7)
    total = ft+fy+fdelta
    phi_f = fit_dep_H(total/.004, fit_h[0])*x['A_f2']
    phi_s = total*.336e-6
    phi_mag = phi_air+phi_f+phi_s
    lambda_leak = x['l_ef']*x['bm']*np.pi*4e-7/.0005
    fm = total+phi_mag/lambda_leak
    return dict(phi_air=phi_air, H_y=hy, F_delta=fdelta, F_total=total,
                phi_f=phi_f, phi_s=phi_s, phi_mag=phi_mag,
                residual=x['phi_r']-fm*x['lambda_m']-phi_mag)


class MagnetMECImplicitModel(ScalarState):
    """Solve the flux balance for air-gap density ``B_delta`` (tesla)."""

    def initialize(self):
        self.options.declare('fit_coeff_dep_H', default=None, allow_none=True)
        self.options.declare('fit_coeff_dep_B', default=None, allow_none=True)

    def setup(self):
        self.fit_h, self.fit_b = fitting_options(self.options)
        for name in MAGNET_INPUTS:
            self.add_input(name)
        self.setup_state('B_delta', MAGNET_INPUTS, val=.7)

    def residual(self, inputs, state):
        return magnet_quantities(inputs, state, self.fit_h, self.fit_b)['residual']

    def solve_real(self, inputs):
        return bracketed_root(lambda b: self.residual(inputs, b).item(),
                              1e-6, 1.2, 'Magnet MEC')


class MagnetMECOutputs(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('fit_coeff_dep_H', default=None, allow_none=True)
        self.options.declare('fit_coeff_dep_B', default=None, allow_none=True)

    def setup(self):
        self.fit_h, self.fit_b = fitting_options(self.options)
        for name in (*MAGNET_INPUTS, 'B_delta', 'Am_r'):
            self.add_input(name)
        for name in ('phi_air', 'H_y', 'F_delta', 'F_total', 'phi_f', 'phi_s', 'phi_mag',
                     'residual', 'K_sigma_air', 'lambda_n', 'lambda_leak_standard'):
            self.add_output(name)
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        values = magnet_quantities(inputs, inputs['B_delta'], self.fit_h, self.fit_b)
        for name, value in values.items():
            outputs[name] = value
        sigma = values['phi_mag']/values['phi_air']
        lam = 2*(values['phi_air']/values['F_total'])*.004/(1.05*np.pi*4e-7*inputs['Am_r'])
        outputs['K_sigma_air'] = sigma
        outputs['lambda_n'] = sigma*lam
        outputs['lambda_leak_standard'] = (sigma-1)*lam


class MagnetMECModel(om.Group):
    def initialize(self):
        self.options.declare('fit_coeff_dep_H', default=None, allow_none=True)
        self.options.declare('fit_coeff_dep_B', default=None, allow_none=True)

    def setup(self):
        fits = {name: self.options[name] for name in ('fit_coeff_dep_H', 'fit_coeff_dep_B')}
        self.add_subsystem('implicit', MagnetMECImplicitModel(**fits), promotes=['*'])
        self.add_subsystem('post_processing', MagnetMECOutputs(**fits), promotes=['*'])
