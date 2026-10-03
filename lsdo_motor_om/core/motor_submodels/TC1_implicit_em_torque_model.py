"""Solve electromagnetic torque or delivered load torque including losses."""

import numpy as np
import openmdao.api as om
from scipy.optimize import minimize_scalar
from .._utils import ScalarState, bracketed_root
from .TC1_torque_limit_model import ELECTRICAL_NAMES
from .TC1_flux_weakening_model import FluxWeakeningModel
from .TC1_mtpa_model import MTPAModel
from .TC1_motor_speed_model import SPEED_UNITS
from .TC1_post_processing_model import (
    PostProcessingModel, declare_performance_options, node_performance,
)


class EMTorqueImplicitModel(ScalarState):
    """Native implicit state for the original ``load_torque=eta*T_em`` equation.

    The equivalent positive-power residual ``load+loss/omega_mechanical-T_em`` avoids the
    spurious zero-load solution in efficiency-map mode. Input-load mode selects
    the first feasible positive torque root, and raises AnalysisError when the
    requested load cannot be delivered within the voltage torque limit.
    """

    def initialize(self):
        declare_performance_options(self)
        self.options.declare('mode', default='input_load', values=['input_load', 'efficiency_map'])

    def setup(self):
        n = self.options['num_nodes']
        self.state_name = 'T_em' if self.options['mode'] == 'input_load' else 'load_torque'
        known = 'load_torque' if self.state_name == 'T_em' else 'T_em'
        equation_names = [*ELECTRICAL_NAMES, *SPEED_UNITS, known, 'motor_variables',
                          'I_q_temp', 'B_delta', 'D_i']
        for name in (*ELECTRICAL_NAMES, *SPEED_UNITS, known, 'Id_fw_bracket', 'T_lim'):
            self.add_input(name, shape=n, units=SPEED_UNITS.get(name))
        self.add_input('T_lower_lim', val=np.zeros(n))
        for name in ('I_q_temp', 'B_delta', 'D_i'):
            self.add_input(name)
        self.add_input('motor_variables', shape=25)
        self.setup_state(self.state_name, equation_names, val=15., shape=n)

    def residual(self, x, state):
        result = []
        for i in range(self.options['num_nodes']):
            torque = state[i] if self.state_name == 'T_em' else x['T_em'][i]
            load = x['load_torque'][i] if self.state_name == 'T_em' else state[i]
            values = node_performance(x, i, torque, load, self.options)
            result.append(load+values['P_loss']/x['omega_mechanical'][i]-torque)
        return np.asarray(result)

    def solve_real(self, x):
        result = []
        for i in range(self.options['num_nodes']):
            speed = x['omega_mechanical'][i]
            if speed <= 0:
                raise om.AnalysisError(f'EM torque node {i}: omega_mechanical must be positive.')
            upper = x['T_lim'][i]*(1-1e-9)
            if self.state_name == 'load_torque':
                torque = x['T_em'][i]
                if not 0 < torque < upper:
                    raise om.AnalysisError(f'Efficiency map node {i}: T_em must lie strictly inside (0, T_lim).')
                values = node_performance(x, i, torque, 0., self.options)
                load = (torque-values['P_loss']/speed)/1.01
                if load <= 0:
                    raise om.AnalysisError(f'Efficiency map node {i}: torque cannot cover no-load losses.')
                result.append(load)
                continue
            load = x['load_torque'][i]
            lower = max(load, x['T_lower_lim'][i], 1e-8)
            if load <= 0 or lower >= upper:
                raise om.AnalysisError(f'EM torque node {i}: load must be positive and below T_lim.')
            def equation(t):
                values = node_performance(x, i, t, load, self.options)
                return load+values['P_loss']/speed-t
            # Loss feedback can create two roots; the source endpoint bracket
            # misses both when its upper endpoint has a positive residual.
            grid = np.geomspace(lower, upper, 100)
            left, fleft = grid[0], equation(grid[0])
            for right in grid[1:]:
                fright = equation(right)
                if fleft*fright <= 0:
                    result.append(bracketed_root(equation, left, right, f'EM torque node {i}'))
                    break
                left, fleft = right, fright
            else:
                minimum = minimize_scalar(equation, bounds=(lower, upper), method='bounded')
                if minimum.success and minimum.fun <= 0:
                    result.append(bracketed_root(equation, lower, minimum.x, f'EM torque node {i}'))
                else:
                    raise om.AnalysisError(
                        f'EM torque node {i}: requested load {load:.6g} cannot be delivered '
                        f'with the original loss equations below T_lim={upper:.6g}.')
        return result


class EMTorqueModel(om.Group):
    """Torque state plus separately inspectable flux weakening, MTPA, and losses."""

    def initialize(self):
        declare_performance_options(self)
        self.options.declare('mode', default='input_load', values=['input_load', 'efficiency_map'])

    def setup(self):
        opts = {name: self.options[name] for name in ('pole_pairs', 'phases', 'rated_current',
                'V_lim', 'num_nodes', 'motor_variable_names', 'loss_model')}
        self.add_subsystem('implicit', EMTorqueImplicitModel(**opts, mode=self.options['mode']), promotes=['*'])
        control = {n: self.options[n] for n in ('pole_pairs', 'V_lim', 'num_nodes')}
        self.add_subsystem('flux_weakening_model', FluxWeakeningModel(**control), promotes=['*'])
        self.add_subsystem('mtpa_model', MTPAModel(pole_pairs=opts['pole_pairs'], num_nodes=opts['num_nodes']), promotes=['*'])
        self.add_subsystem('post_processing', PostProcessingModel(**opts), promotes=['*'])
