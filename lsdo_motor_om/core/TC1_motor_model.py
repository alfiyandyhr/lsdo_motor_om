"""Convenience group connecting the original sizing and analysis models."""

import openmdao.api as om
from ._utils import declare_motor_options
from .TC1_motor_sizing_model import TC1MotorSizingModel
from .TC1_motor_analysis_model import TC1MotorAnalysisModel


class TC1MotorModel(om.Group):
    def initialize(self):
        declare_motor_options(self, fitting=True)
        self.options.declare('V_lim', default=800., types=(int, float), lower=1e-12)
        self.options.declare('num_nodes', default=1, types=int, lower=1)
        self.options.declare('num_active_nodes', default=None, allow_none=True, types=int)
        self.options.declare('gear_ratio', default=4., types=(int, float), lower=1e-12)
        self.options.declare('model_test', default=False, types=bool)
        self.options.declare('fitting_order', default=1, values=[0, 1, 2, 3, 4])

    def setup(self):
        motor = {n: self.options[n] for n in ('pole_pairs', 'phases', 'num_slots', 'rated_current')}
        self.add_subsystem('sizing', TC1MotorSizingModel(**motor, fitting_order=self.options['fitting_order']),
                           promotes=['*'])
        analysis = {n: self.options[n] for n in ('fit_coeff_dep_H', 'fit_coeff_dep_B',
                    'V_lim', 'num_nodes', 'num_active_nodes', 'gear_ratio', 'model_test')}
        self.add_subsystem('analysis', TC1MotorAnalysisModel(**motor, **analysis), promotes=['*'])
        self.set_input_defaults('D_i', val=.182)
