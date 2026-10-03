"""Evaluate the original efficiency-map variant at a grid of motor speeds/torques.

The input omega_mechanical is motor mechanical angular speed in rad/s.
Run: FI_PROVIDER=tcp python examples/efficiency_map.py
"""

import numpy as np
import openmdao.api as om
from lsdo_motor_om import TC1MotorSizingModel
from lsdo_motor_om.core._utils import MotorVariablesModel
from lsdo_motor_om.core.motor_submodels.TC1_motor_speed_model import MotorSpeedModel
from lsdo_motor_om.core.motor_submodels.TC1_magnet_mec_model import MagnetMECModel
from lsdo_motor_om.core.motor_submodels.TC1_inductance_mec_model import InductanceModel
from lsdo_motor_om.core.motor_submodels.TC1_torque_limit_model import TorqueLimitModel
from lsdo_motor_om.core.motor_submodels.TC1_flux_weakening_model import (
    FluxWeakeningBracketCoefficients, FluxWeakeningBracketModel,
)
from lsdo_motor_om.core.motor_submodels.TC1_efficiency_map_model import EfficiencyMapModel
import os

# Uncomment when running with active VPN
os.environ["FI_PROVIDER"] = "tcp"

class EfficiencyMapMotor(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', default=6, types=int, lower=1)

    def setup(self):
        n = self.options['num_nodes']
        self.add_subsystem('sizing', TC1MotorSizingModel(), promotes=['*'])
        self.add_subsystem('geometry', MotorVariablesModel(), promotes=['*'])
        self.add_subsystem('magnet', MagnetMECModel(), promotes=['*'])
        self.add_subsystem('inductance', InductanceModel(), promotes=['*'])
        self.add_subsystem('flux_linkage', om.ExecComp('PsiF=turns_per_phase*phi_air'), promotes=['*'])
        self.add_subsystem('speed', MotorSpeedModel(num_nodes=n), promotes=['*'])
        self.add_subsystem('torque_limit', TorqueLimitModel(num_nodes=n), promotes=['*'])
        self.add_subsystem('bracket_coefficients', FluxWeakeningBracketCoefficients(num_nodes=n), promotes=['*'])
        self.add_subsystem('bracket', FluxWeakeningBracketModel(num_nodes=n), promotes=['*'])
        self.add_subsystem('efficiency_map', EfficiencyMapModel(num_nodes=n), promotes=['*'])
        self.set_input_defaults('D_i', val=.3723)
        self.set_input_defaults('L', val=.2755)


def build_problem():
    speed, torque = np.meshgrid([400., 600., 800.], [100., 200.])
    p = om.Problem(model=EfficiencyMapMotor(num_nodes=speed.size), reports=False)
    p.setup(force_alloc_complex=True)
    p['omega_mechanical'] = speed.ravel()
    p['T_em'] = torque.ravel()
    return p, speed.shape


def main():
    p, shape = build_problem()
    p.run_model()
    print('Efficiency (rows: 100, 200 Nm; columns: mechanical speed 400, 600, 800 rad/s):')
    print(p['efficiency_active'].reshape(shape))
    print('Delivered load torque:')
    print(p['load_torque'].reshape(shape))
    return p


if __name__ == '__main__':
    main()
