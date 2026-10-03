"""TC1 motor analysis assembled entirely from native OpenMDAO systems."""

import numpy as np
import openmdao.api as om
from ._utils import MOTOR_VARIABLE_NAMES, MotorVariablesModel, declare_motor_options, smooth_min
from .motor_submodels.TC1_magnet_mec_model import MagnetMECModel
from .motor_submodels.TC1_inductance_mec_model import InductanceModel
from .motor_submodels.TC1_torque_limit_model import TorqueLimitModel, ElectricalParameters
from .motor_submodels.TC1_flux_weakening_model import (
    FluxWeakeningBracketCoefficients, FluxWeakeningBracketModel, FluxWeakeningModel,
)
from .motor_submodels.TC1_mtpa_model import MTPAModel
from .motor_submodels.TC1_motor_speed_model import MotorSpeedModel
from .motor_submodels.TC1_implicit_em_torque_model import EMTorqueModel
from .motor_submodels.TC1_post_processing_model import PostProcessingModel


class ParseActiveOperatingConditions(om.ExplicitComponent):
    """Gather nodes whose rotor RPM and rotor load torque are both nonzero.

    The active count is fixed at setup. The mask is piecewise constant and its
    derivatives are defined only while that mask stays unchanged.
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int, lower=1)
        self.options.declare('num_active_nodes', default=None, allow_none=True, types=int)

    def setup(self):
        n = self.options['num_nodes']
        a = self.options['num_active_nodes']
        self.active_count = n if a is None else a
        if not 0 < self.active_count <= n:
            raise ValueError('Parser requires 1 <= num_active_nodes <= num_nodes.')
        for name in ('omega_rotor', 'load_torque_rotor'):
            self.add_input(name, val=np.ones(n), units='rpm' if name == 'omega_rotor' else None)
        self.add_output('omega_rotor_active', shape=self.active_count, units='rpm')
        self.add_output('load_torque_rotor_active', shape=self.active_count)
        self.add_output('selection_indices', shape=(n, self.active_count))
        self.declare_partials('omega_rotor_active', 'omega_rotor')
        self.declare_partials('load_torque_rotor_active', 'load_torque_rotor')

    def selection(self, inputs):
        active = np.flatnonzero((np.real(inputs['omega_rotor']) != 0)
                               & (np.real(inputs['load_torque_rotor']) != 0))
        if len(active) != self.active_count:
            raise om.AnalysisError(
                f'Expected {self.active_count} active nodes, found {len(active)}. '
                'Set num_active_nodes to the count of nodes with nonzero RPM and torque.')
        return active

    def compute(self, inputs, outputs):
        active = self.selection(inputs)
        for name in ('omega_rotor', 'load_torque_rotor'):
            outputs[name+'_active'] = inputs[name][active]
        selection = np.zeros((self.options['num_nodes'], self.active_count))
        selection[active, np.arange(self.active_count)] = 1.
        outputs['selection_indices'] = selection

    def compute_partials(self, inputs, partials):
        active = self.selection(inputs)
        jac = np.zeros((self.active_count, self.options['num_nodes']))
        jac[np.arange(self.active_count), active] = 1.
        partials['omega_rotor_active', 'omega_rotor'] = jac
        partials['load_torque_rotor_active', 'load_torque_rotor'] = jac


class GearboxModel(om.ExplicitComponent):
    """Ideal gearbox: rotor RPM to motor mechanical rad/s, conserving power."""

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int, lower=1)
        self.options.declare('gear_ratio', default=4., types=(int, float), lower=1e-12)

    def setup(self):
        n = self.options['num_nodes']
        self.add_input('omega_rotor_active', shape=n, units='rpm')
        self.add_input('load_torque_rotor_active', shape=n)
        self.add_output('omega', shape=n, units='rad/s', desc='Alias of omega_mechanical')
        self.add_output('omega_mechanical', shape=n, units='rad/s')
        self.add_output('load_torque', shape=n)
        indices = np.arange(n)
        ratio = self.options['gear_ratio']
        for name in ('omega', 'omega_mechanical'):
            self.declare_partials(name, 'omega_rotor_active', rows=indices, cols=indices, val=ratio*2*np.pi/60)
        self.declare_partials('load_torque', 'load_torque_rotor_active', rows=indices, cols=indices, val=1/ratio)

    def compute(self, x, outputs):
        outputs['omega'] = x['omega_rotor_active']*self.options['gear_ratio']*2*np.pi/60
        outputs['omega_mechanical'] = outputs['omega']
        outputs['load_torque'] = x['load_torque_rotor_active']/self.options['gear_ratio']


class TorqueConstraints(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int, lower=1)

    def setup(self):
        n = self.options['num_nodes']
        self.add_input('T_em_max')
        for name in ('T_lim', 'load_torque', 'T_em'):
            self.add_input(name, shape=n)
        for name in ('T_upper_lim_curve', 'max_torque_constraint', 'em_torque_constraint'):
            self.add_output(name, shape=n)
        self.declare_partials('*', '*', method='cs')

    def compute(self, x, outputs):
        upper = smooth_min(x['T_lim'], x['T_em_max'])
        outputs['T_upper_lim_curve'] = upper
        outputs['max_torque_constraint'] = upper-x['load_torque']
        outputs['em_torque_constraint'] = upper-x['T_em']


class ExpandActiveOutputs(om.ExplicitComponent):
    output_mapping = {
        'input_power_active': 'input_power', 'efficiency_active': 'efficiency',
        'output_power': 'output_power_full', 'T_em': 'T_em_full',
        'current_amplitude': 'current_amplitude_full', 'load_torque': 'load_torque_full',
    }

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int, lower=1)
        self.options.declare('num_active_nodes', default=1, types=int, lower=1)

    def setup(self):
        n, a = self.options['num_nodes'], self.options['num_active_nodes']
        self.add_input('selection_indices', shape=(n, a))
        for active, full in self.output_mapping.items():
            self.add_input(active, shape=a)
            self.add_output(full, shape=n)
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        for active, full in self.output_mapping.items():
            outputs[full] = inputs['selection_indices'] @ inputs[active]


class IdleMotorOutputs(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int, lower=1)

    def setup(self):
        n = self.options['num_nodes']
        self.add_input('omega_rotor', val=np.zeros(n), units='rpm')
        self.add_input('load_torque_rotor', val=np.zeros(n))
        for name in ExpandActiveOutputs.output_mapping.values():
            self.add_output(name, val=np.zeros(n))

    def compute(self, x, outputs):
        if np.any((np.real(x['omega_rotor']) != 0) & (np.real(x['load_torque_rotor']) != 0)):
            raise om.AnalysisError('num_active_nodes=0 but a node has nonzero RPM and load torque.')
        for name in outputs:
            outputs[name] = 0.


class TC1MotorAnalysisModel(om.Group):
    """Analyze motor geometry for rotor RPM and rotor load-torque arrays.

    All original public option names and the 25-entry motor_variables layout are
    retained. ``model_test=True`` takes electromagnetic torque as an input and
    uses the source's alternate diagnostic loss equations.
    """

    motor_variable_names = MOTOR_VARIABLE_NAMES

    def initialize(self):
        declare_motor_options(self, fitting=True)
        self.options.declare('V_lim', default=800., types=(int, float), lower=1e-12)
        self.options.declare('num_nodes', default=1, types=int, lower=1)
        self.options.declare('num_active_nodes', default=None, allow_none=True, types=int)
        self.options.declare('model_test', default=False, types=bool)
        self.options.declare('gear_ratio', default=4., types=(int, float), lower=1e-12)

    def setup(self):
        n = self.options['num_nodes']
        a = self.options['num_active_nodes']
        a = n if a is None else a
        if not 0 <= a <= n:
            raise ValueError('num_active_nodes must be between zero and num_nodes.')
        fits = {name: self.options[name] for name in ('fit_coeff_dep_H', 'fit_coeff_dep_B')}
        motor = {name: self.options[name] for name in ('pole_pairs', 'phases', 'num_slots', 'rated_current')}
        self.add_subsystem('motor_variables_model', MotorVariablesModel(), promotes=['*'])
        self.add_subsystem('magnet_MEC_model', MagnetMECModel(**fits), promotes=['*'])
        self.add_subsystem('inductance_MEC_model', InductanceModel(**motor, **fits), promotes=['*'])
        self.add_subsystem('flux_linkage', om.ExecComp('PsiF=turns_per_phase*phi_air'), promotes=['*'])
        if a == 0:
            self.add_subsystem('idle', IdleMotorOutputs(num_nodes=n), promotes=['*'])
            return
        self.add_subsystem('parse_active_operating_conditions',
                           ParseActiveOperatingConditions(num_nodes=n, num_active_nodes=a), promotes=['*'])
        self.add_subsystem('electrical_parameters', ElectricalParameters(num_nodes=a), promotes=['*'])
        self.add_subsystem('gearbox', GearboxModel(num_nodes=a, gear_ratio=self.options['gear_ratio']), promotes=['*'])
        self.add_subsystem('motor_speed', MotorSpeedModel(pole_pairs=motor['pole_pairs'], num_nodes=a),
                           promotes=['*'])
        control = dict(pole_pairs=motor['pole_pairs'], V_lim=self.options['V_lim'], num_nodes=a)
        self.add_subsystem('torque_limit_model', TorqueLimitModel(**control, use_expanded=True), promotes=['*'])
        self.add_subsystem('flux_weakening_bracket_coefficients', FluxWeakeningBracketCoefficients(**control), promotes=['*'])
        self.add_subsystem('flux_weakening_bracket_method',
                           FluxWeakeningBracketModel(pole_pairs=motor['pole_pairs'], num_nodes=a), promotes=['*'])
        performance = dict(**control, rated_current=motor['rated_current'], phases=motor['phases'])
        if self.options['model_test']:
            self.add_subsystem('flux_weakening_model', FluxWeakeningModel(**control), promotes=['*'])
            self.add_subsystem('mtpa_model', MTPAModel(pole_pairs=motor['pole_pairs'], num_nodes=a), promotes=['*'])
            self.add_subsystem('post_processing', PostProcessingModel(**performance, loss_model='model_test'), promotes=['*'])
        else:
            self.add_subsystem('implicit_em_torque_model', EMTorqueModel(**performance), promotes=['*'])
        self.add_subsystem('torque_constraints', TorqueConstraints(num_nodes=a), promotes=['*'])
        self.add_subsystem('expand_active_outputs', ExpandActiveOutputs(num_nodes=n, num_active_nodes=a), promotes=['*'])
