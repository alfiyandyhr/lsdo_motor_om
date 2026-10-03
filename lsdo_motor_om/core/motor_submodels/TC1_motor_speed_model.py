"""Separate mechanical rotation from electrical rotation and frequency."""

import numpy as np
import openmdao.api as om

SPEED_UNITS = {
    'omega_mechanical': 'rad/s',
    'omega_electrical': 'rad/s',
    'electrical_frequency': 'Hz',
}


class MotorSpeedModel(om.ExplicitComponent):
    """Convert motor mechanical rad/s to electrical rad/s and electrical Hz.

    Use this component when assembling standalone control/performance groups.
    The complete motor analysis supplies these quantities automatically.
    """

    def initialize(self):
        self.options.declare('pole_pairs', default=6, types=int, lower=1)
        self.options.declare('num_nodes', default=1, types=int, lower=1)

    def setup(self):
        n, p = self.options['num_nodes'], self.options['pole_pairs']
        self.add_input('omega_mechanical', shape=n, units='rad/s')
        indices = np.arange(n)
        for name, factor in [('omega_electrical', p), ('electrical_frequency', p/(2*np.pi))]:
            self.add_output(name, shape=n, units=SPEED_UNITS[name])
            self.declare_partials(name, 'omega_mechanical', rows=indices, cols=indices, val=factor)

    def compute(self, inputs, outputs):
        outputs['omega_electrical'] = self.options['pole_pairs']*inputs['omega_mechanical']
        outputs['electrical_frequency'] = outputs['omega_electrical']/(2*np.pi)
