"""Minimize motor mass with the native OpenMDAO SLSQP driver and derivatives.

Run: FI_PROVIDER=tcp python examples/optimize_motor.py
"""

import openmdao.api as om
from lsdo_motor_om import TC1MotorModel
import os

# Uncomment when running with active VPN
os.environ["FI_PROVIDER"] = "tcp"

def build_problem():
    p = om.Problem(model=TC1MotorModel(), reports=False)
    p.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, maxiter=100, disp=False)
    p.model.add_design_var('D_i', lower=.25, upper=.45)
    p.model.add_design_var('L', lower=.10, upper=.35)
    p.model.add_objective('motor_mass', ref=100.)
    p.model.add_constraint('em_torque_constraint', lower=0., ref=1000.)
    p.model.add_constraint('current_amplitude', upper=123., ref=123.)
    p.model.add_constraint('efficiency', lower=.95)
    p.setup(force_alloc_complex=True)
    p['D_i'], p['L'] = .3723, .2755
    p['omega_rotor'], p['load_torque_rotor'] = [1500.], [400.]
    return p


def main():
    p = build_problem()
    result = p.run_driver()
    success = result.success if hasattr(result, 'success') else not result
    if not success:
        raise RuntimeError('Motor optimization did not converge.')
    for name in ('D_i', 'L', 'motor_mass', 'T_em', 'current_amplitude', 'efficiency'):
        print(f'{name}: {p[name]}')
    return p


if __name__ == '__main__':
    main()
