"""Sizing and analysis with two active and two inactive operating nodes.

Run: FI_PROVIDER=tcp python examples/basic.py
"""

import numpy as np
import openmdao.api as om
from lsdo_motor_om import TC1MotorModel
import os

# Uncomment when running with active VPN
os.environ["FI_PROVIDER"] = "tcp"


def build_problem():
    p = om.Problem(model=TC1MotorModel(num_nodes=4, num_active_nodes=2), reports=False)
    p.setup(force_alloc_complex=True)
    p.set_val('D_i', .3723)
    p.set_val('L', .2755)
    p.set_val('omega_rotor', [1500., 0., 1800., 0.])
    p.set_val('load_torque_rotor', [400., 0., 600., 0.])
    return p


def main():
    p = build_problem()
    p.run_model()
    print(f"Motor mass: {p['motor_mass'][0]:.6f} kg")
    print(f"Resistance: {p['Rdc'][0]:.8f} ohm")
    for name in ('T_em_full', 'input_power', 'efficiency'):
        print(f'{name}: {np.array2string(p[name], precision=6)}')
    print('Power derivatives with respect to diameter and length:')
    print(p.compute_totals(of=['input_power'], wrt=['D_i', 'L']))
    return p


if __name__ == '__main__':
    main()
