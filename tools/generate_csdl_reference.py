"""Optional maintainer tool: capture reference outputs from the untouched CSDL source.

This captures legacy equations, including their inconsistent speed factors.
Tests use these outputs for unchanged geometry/magnetics and currents evaluated
at the same electrical speed; legacy operating power/loss results are historical.
Requires the original CSDL dependencies. The installed port and its tests do not.
Run from the root: python tools/generate_csdl_reference.py
"""
import contextlib
import io
import json
from pathlib import Path
import sys
import warnings
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'lsdo_motor'))
import csdl
from python_csdl_backend import Simulator
from lsdo_motor.core.TC1_motor_sizing_model import TC1MotorSizingModel
from lsdo_motor.core.TC1_motor_analysis_model import TC1MotorAnalysisModel
from lsdo_motor.core.permeability.mu_fitting import permeability_fitting
from lsdo_motor.core.motor_submodels.TC1_implicit_em_torque_model import EMTorqueImplicitModel
from lsdo_motor.core.motor_submodels.TC1_efficiency_map_model import LoadTorqueImplicitModel

NAMES = ['Rdc', 'motor_mass', 'T_em_max', 'motor_variables', 'B_delta', 'phi_air',
         'H_y', 'phi_f', 'phi_s', 'phi_mag', 'F_total', 'F_delta', 'K_sigma_air',
         'lambda_n', 'lambda_leak_standard', 'L_d', 'L_q', 'I_q_temp', 'PsiF',
         'omega', 'load_torque', 'T_lim', 'Iq_fw_bracket', 'Id_fw_bracket', 'T_em',
         'current_amplitude', 'output_power', 'input_power_active', 'efficiency_active',
         'input_power', 'max_torque_constraint']


def main():
    warnings.filterwarnings('ignore')
    fits = permeability_fitting(str(ROOT/'lsdo_motor/lsdo_motor/core/permeability/Magnetic_alloy_silicon_core_iron_C.tab'))
    cases = []
    for diameter, length, rpm, torque in [(.182, .086, 10000., 40.), (.3723, .2755, 1500., 400.)]:
        class Motor(csdl.Model):
            def define(self):
                self.add(TC1MotorSizingModel(pole_pairs=6, phases=3, num_slots=36, rated_current=123), 'sizing')
                self.add(TC1MotorAnalysisModel(pole_pairs=6, phases=3, num_slots=36, rated_current=123,
                         V_lim=800, fit_coeff_dep_H=fits[0], fit_coeff_dep_B=fits[1], num_nodes=1,
                         num_active_nodes=1), 'analysis')
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            sim = Simulator(Motor())
            sim['D_i'], sim['L'] = diameter, length
            sim['omega_rotor'] = np.array([rpm])
            sim['load_torque_rotor'] = np.array([torque])
            sim.run()
        outputs = {name: sim[name].tolist() for name in NAMES}
        residual = np.array(outputs['load_torque'])-np.array(outputs['efficiency_active'])*np.array(outputs['T_em'])
        if np.max(np.abs(residual)) > 1e-7:
            raise RuntimeError(f'CSDL case failed convergence: {residual}; last log: {log.getvalue()[-1000:]}')
        cases.append(dict(inputs=dict(D_i=diameter, L=length, omega_rotor=[rpm], load_torque_rotor=[torque]),
                          outputs=outputs))
        print(f'Captured CSDL D_i={diameter}, RPM={rpm}: T_em={outputs["T_em"]}, eta={outputs["efficiency_active"]}')
    submodels = []
    base = cases[1]
    for kind, cls in [('input_load', EMTorqueImplicitModel), ('efficiency_map', LoadTorqueImplicitModel)]:
        options = dict(pole_pairs=6, phases=3, rated_current=123, V_lim=800, num_nodes=1,
                       motor_variable_names=TC1MotorAnalysisModel().motor_variable_names)
        with contextlib.redirect_stdout(io.StringIO()):
            sim = Simulator(cls(**options))
            vals = dict(T_em=np.array([100.]), load_torque=np.array([90.]),
                        omega=np.array(base['outputs']['omega']), D_i=base['inputs']['D_i'])
            for n in ('motor_variables', 'B_delta', 'I_q_temp', 'Id_fw_bracket'):
                vals[n] = np.array(base['outputs'][n])
            for scalar_name, vector_name in [('Rdc', 'R_expanded'), ('L_d', 'L_d_expanded'),
                                            ('L_q', 'L_q_expanded'), ('PsiF', 'PsiF_expanded')]:
                vals[vector_name] = np.array(base['outputs'][scalar_name])
            for n, value in vals.items():
                sim[n] = value
            sim.run()
        names = ['Id_fw', 'Iq_fw', 'Iq_MTPA_star', 'Iq_MTPA', 'current_amplitude',
                 'voltage_amplitude', 'output_power', 'input_power_active', 'efficiency_active']
        submodels.append(dict(loss_model=kind, inputs={n: np.asarray(v).tolist() for n, v in vals.items()},
                              outputs={n: sim[n].tolist() for n in names}))
    class DiagnosticMotor(csdl.Model):
        def define(self):
            self.add(TC1MotorSizingModel(pole_pairs=6, phases=3, num_slots=36, rated_current=123), 'sizing')
            self.add(TC1MotorAnalysisModel(pole_pairs=6, phases=3, num_slots=36, rated_current=123,
                     V_lim=800, fit_coeff_dep_H=fits[0], fit_coeff_dep_B=fits[1], num_nodes=1,
                     num_active_nodes=1, model_test=True), 'analysis')
    diagnostic_inputs = dict(D_i=.182, L=.086, omega_rotor=[1000.], load_torque_rotor=[40.], T_em=[15.])
    with contextlib.redirect_stdout(io.StringIO()):
        sim = Simulator(DiagnosticMotor())
        for name, value in diagnostic_inputs.items():
            sim[name] = np.asarray(value) if isinstance(value, list) else value
        sim.run()
    diagnostic_names = ['T_em', 'Id_fw', 'Iq_fw', 'Iq_MTPA', 'current_amplitude',
                        'voltage_amplitude', 'output_power', 'input_power', 'efficiency']
    diagnostic = dict(inputs=diagnostic_inputs, outputs={n: sim[n].tolist() for n in diagnostic_names})
    if not all(np.isfinite(v).all() for v in diagnostic['outputs'].values()):
        raise RuntimeError('CSDL diagnostic case has nonfinite outputs.')
    data = dict(source='Original local CSDL working tree; see tools/generate_csdl_reference.py',
                fit_coeff_dep_H=fits[0].tolist(), fit_coeff_dep_B=fits[1].tolist(), cases=cases,
                submodels=submodels, diagnostic=diagnostic)
    (ROOT/'tests/fixtures/csdl_reference.json').write_text(json.dumps(data, indent=2)+'\n')

if __name__ == '__main__':
    main()
