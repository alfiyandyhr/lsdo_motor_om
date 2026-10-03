"""Current blending and the original TC1 power-loss equations.

The source's standalone PostProcessingModel referenced undefined ``op_voltage``.
This implementation uses the working EMTorqueImplicitModel's voltage blend.
"""

import numpy as np
import openmdao.api as om
from .._utils import MOTOR_VARIABLE_NAMES, sigmoid, scalar
from .TC1_torque_limit_model import ELECTRICAL_NAMES
from .TC1_mtpa_model import mtpa_currents
from .TC1_flux_weakening_model import fw_currents

PERFORMANCE_OUTPUTS = (
    'I_d', 'I_q', 'current_amplitude', 'voltage_amplitude', 'U_MTPA',
    'P_copper', 'P_eddy', 'P_eddy_s', 'P_h', 'P_stress', 'P_wo', 'P_loss',
    'output_power', 'input_power_active', 'efficiency_active',
)


def declare_performance_options(component):
    component.options.declare('pole_pairs', default=6, types=int, lower=1)
    component.options.declare('phases', default=3, types=int, lower=1)
    component.options.declare('rated_current', default=123., types=(int, float), lower=1e-12)
    component.options.declare('V_lim', default=800., types=(int, float), lower=1e-12)
    component.options.declare('num_nodes', default=1, types=int, lower=1)
    component.options.declare('motor_variable_names', default=MOTOR_VARIABLE_NAMES)
    component.options.declare('loss_model', default='input_load',
                              values=['input_load', 'efficiency_map', 'model_test'])


def operating_performance(torque, load, w, r, ld, lq, psi, geometry, diameter,
                          bdelta, iq_rated, lower, p, m, rated_current, voltage,
                          loss_model='input_load', currents=None):
    """Evaluate one node; also used inside the electromagnetic-torque residual."""
    if currents is None:
        id_mtpa, iq_mtpa = mtpa_currents(torque, ld, lq, psi, p)
        id_fw, iq_fw = fw_currents(torque, w, r, ld, lq, psi, p, voltage, lower)
    else:
        id_mtpa, iq_mtpa, id_fw, iq_fw = currents
    ud = r*id_mtpa-w*lq*iq_mtpa
    uq = w*ld*id_mtpa+r*iq_mtpa+w*psi
    u_mtpa = np.sqrt(ud**2+uq**2)
    fi = 5000*p/60
    ud_rated = -r*rated_current*np.sin(.6283)-2*np.pi*fi*lq*iq_rated
    uq_rated = r*rated_current*np.sin(.6283)+2*np.pi*fi*(psi-ld*iq_rated)
    u_rated = np.sqrt(ud_rated**2+uq_rated**2)
    if loss_model == 'model_test':
        weight = sigmoid(u_rated-voltage)
    else:
        k = 1. if loss_model == 'efficiency_map' else .5
        weight = sigmoid(k*(u_mtpa-voltage))
    iq = weight*iq_fw+(1-weight)*iq_mtpa
    id_ = (torque/(1.5*p*iq)-psi)/(ld-lq)
    current2 = iq**2+id_**2
    # Preserve the original main-path speed factors. See docs/migration.md.
    speed_factor = 1. if loss_model == 'model_test' else 2*np.pi/60
    p0 = load*w*speed_factor
    frequency = w*p/60
    pcopper = m*r*current2
    lef, d1, d2, bm, acu = (geometry[i] for i in (4, 0, 5, 16, 7))
    vs = np.pi*lef*(d1-diameter)**2/4-36*lef*acu
    vr = np.pi*lef*(d2-.3*d2)**2/4
    vt = 2*p*lef*bm*.004
    ke = (.00055*np.pi)**2*2e6/60
    magnetic_volume = vs+vr if loss_model == 'input_load' else vs
    if loss_model == 'model_test':
        magnetic_volume = np.pi*lef*(d1-diameter)**2-36*lef*acu
        ke *= 10
    peddy = ke*magnetic_volume*(bdelta*frequency)**2
    peddy_s = ke*vt*(bdelta*frequency)**2 if loss_model == 'input_load' else 0.*peddy
    ph = 100*magnetic_volume*frequency*bdelta**2
    pstress = .01*p0
    pwo = 4*np.pi*.003*1.225*(2*np.pi*frequency)**2*lef*d2**4
    pm = 100. if loss_model == 'input_load' else 0.
    loss = pcopper+peddy+peddy_s+ph+pstress+pwo+pm
    return dict(I_d=id_, I_q=iq, current_amplitude=np.sqrt(current2),
                voltage_amplitude=u_rated, U_MTPA=u_mtpa, P_copper=pcopper,
                P_eddy=peddy, P_eddy_s=peddy_s, P_h=ph, P_stress=pstress,
                P_wo=pwo, P_loss=loss, output_power=p0,
                input_power_active=p0+loss, efficiency_active=p0/(p0+loss))


def node_performance(x, index, torque, load, options, currents=None):
    r, ld, lq, psi = (x[n][index] for n in ELECTRICAL_NAMES)
    return operating_performance(
        torque, load, x['omega'][index], r, ld, lq, psi, x['motor_variables'],
        scalar(x['D_i']), scalar(x['B_delta']), scalar(x['I_q_temp']),
        x['Id_fw_bracket'][index], options['pole_pairs'], options['phases'],
        options['rated_current'], options['V_lim'], options['loss_model'], currents)


class PostProcessingModel(om.ExplicitComponent):
    """Losses and efficiencies for known load and electromagnetic torque."""

    def initialize(self):
        declare_performance_options(self)

    def setup(self):
        n = self.options['num_nodes']
        for name in (*ELECTRICAL_NAMES, 'omega', 'T_em', 'load_torque',
                     'Id_fw_bracket', 'Id_fw', 'Iq_fw', 'Id_MTPA', 'Iq_MTPA'):
            self.add_input(name, shape=n)
        for name in ('D_i', 'B_delta', 'I_q_temp'):
            self.add_input(name)
        self.add_input('motor_variables', shape=25)
        for name in PERFORMANCE_OUTPUTS:
            self.add_output(name, shape=n)
        self.declare_partials('*', '*', method='cs')

    def compute(self, x, outputs):
        for i in range(self.options['num_nodes']):
            currents = tuple(x[n][i] for n in ('Id_MTPA', 'Iq_MTPA', 'Id_fw', 'Iq_fw'))
            values = node_performance(x, i, x['T_em'][i], x['load_torque'][i], self.options, currents)
            for name, value in values.items():
                outputs[name][i] = value
