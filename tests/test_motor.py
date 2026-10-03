"""Legacy geometry parity, SI speed conventions, controls, and derivatives."""

import importlib
import json
from pathlib import Path
import sys
import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_totals, assert_check_partials

from lsdo_motor_om import TC1MotorModel, TC1MotorSizingModel, ParseActiveOperatingConditions
from lsdo_motor_om.core._utils import MOTOR_VARIABLE_NAMES
from lsdo_motor_om.core.TC1_motor_analysis_model import GearboxModel
from lsdo_motor_om.core.motor_submodels.TC1_motor_speed_model import MotorSpeedModel, SPEED_UNITS
from lsdo_motor_om.core.motor_submodels.TC1_mtpa_model import MTPAModel
from lsdo_motor_om.core.motor_submodels.TC1_torque_limit_model import TorqueLimitModel
from lsdo_motor_om.core.motor_submodels.TC1_flux_weakening_model import (
    FluxWeakeningBracketCoefficients, FluxWeakeningBracketModel, FluxWeakeningModel,
)
from lsdo_motor_om.core.motor_submodels.TC1_implicit_em_torque_model import EMTorqueModel
from lsdo_motor_om.core.motor_submodels.TC1_efficiency_map_model import EfficiencyMapModel
from lsdo_motor_om.core.motor_submodels.TC1_post_processing_model import PostProcessingModel
from lsdo_motor_om.core.permeability.mu_fitting import permeability_fitting, fit_dep_B

REFERENCE = json.loads((Path(__file__).parent/'fixtures/csdl_reference.json').read_text())
UNCHANGED_REFERENCE_OUTPUTS = (
    'Rdc', 'motor_mass', 'T_em_max', 'motor_variables', 'B_delta', 'phi_air',
    'H_y', 'phi_f', 'phi_s', 'phi_mag', 'F_total', 'F_delta', 'K_sigma_air',
    'lambda_n', 'lambda_leak_standard', 'L_d', 'L_q', 'I_q_temp', 'PsiF',
    'omega', 'load_torque',
)


def problem(model):
    p = om.Problem(model=model if isinstance(model, om.Group) else None, reports=False)
    if not isinstance(model, om.Group):
        p.model.add_subsystem('component', model, promotes=['*'])
    p.setup(force_alloc_complex=True)
    return p


def reference_problem(index):
    p = problem(TC1MotorModel(fit_coeff_dep_H=REFERENCE['fit_coeff_dep_H'],
                             fit_coeff_dep_B=REFERENCE['fit_coeff_dep_B']))
    for name, value in REFERENCE['cases'][index]['inputs'].items():
        p.set_val(name, value)
    p.run_model()
    return p


def assert_loss_units(p, kind):
    """Independent loss evaluation from mechanical rad/s and electrical Hz."""
    geometry = p['motor_variables']
    lef, d1, d2, bm, acu = (geometry[i] for i in (4, 0, 5, 16, 7))
    vs = np.pi*lef*(d1-p['D_i'])**2/4-36*lef*acu
    vr = np.pi*lef*(d2-.3*d2)**2/4
    volume = vs+vr if kind == 'input_load' else vs
    ke = (.00055*np.pi)**2*2e6/60
    if kind == 'model_test':
        volume = np.pi*lef*(d1-p['D_i'])**2-36*lef*acu
        ke *= 10
    # Known pole count in these fixtures; frequency is cycles per second.
    frequency = 6*p['omega_mechanical']/(2*np.pi)
    np.testing.assert_allclose(p['electrical_frequency'], frequency, rtol=1e-12)
    np.testing.assert_allclose(p['P_h'], 100*volume*frequency*p['B_delta']**2, rtol=1e-12)
    np.testing.assert_allclose(p['P_eddy'], ke*volume*(p['B_delta']*frequency)**2, rtol=1e-12)
    magnet_eddy = ke*2*6*lef*bm*.004*(p['B_delta']*frequency)**2 if kind == 'input_load' else 0.
    np.testing.assert_allclose(p['P_eddy_s'], magnet_eddy, rtol=1e-12)
    np.testing.assert_allclose(p['P_wo'], 4*np.pi*.003*1.225*p['omega_mechanical']**2*lef*d2**4,
                               rtol=1e-12)
    np.testing.assert_allclose(p['output_power'], p['load_torque']*p['omega_mechanical'], rtol=1e-12)
    np.testing.assert_allclose(p['P_stress'], .01*p['output_power'], rtol=1e-12)
    loss = sum(p[name] for name in ('P_copper', 'P_eddy', 'P_eddy_s', 'P_h', 'P_stress', 'P_wo'))
    if kind == 'input_load':
        loss += 100.
    np.testing.assert_allclose(p['P_loss'], loss, rtol=1e-12)
    np.testing.assert_allclose(p['input_power_active'], p['output_power']+loss, rtol=1e-12)
    np.testing.assert_allclose(p['efficiency_active'], p['output_power']/p['input_power_active'], rtol=1e-12)


@pytest.mark.parametrize('gear_ratio', [1., 4., 7.5])
@pytest.mark.parametrize('pole_pairs', [1, 6])
def test_speed_units_and_ideal_gearbox_power(gear_ratio, pole_pairs):
    group = om.Group()
    group.add_subsystem('gearbox', GearboxModel(num_nodes=2, gear_ratio=gear_ratio), promotes=['*'])
    group.add_subsystem('speed', MotorSpeedModel(num_nodes=2, pole_pairs=pole_pairs), promotes=['*'])
    p = problem(group)
    p['omega_rotor_active'] = [60., 1500.]
    p['load_torque_rotor_active'] = [20., 400.]
    p.run_model()
    mechanical = np.array([2*np.pi, 50*np.pi])*gear_ratio
    np.testing.assert_allclose(p['omega_mechanical'], mechanical)
    np.testing.assert_array_equal(p['omega'], p['omega_mechanical'])
    np.testing.assert_allclose(p['omega_electrical'], pole_pairs*mechanical)
    np.testing.assert_allclose(p['electrical_frequency'], pole_pairs*np.array([1., 25.])*gear_ratio)
    np.testing.assert_allclose(p['load_torque']*p['omega_mechanical'], [40*np.pi, 20000*np.pi])
    result = p.check_partials(method='cs', out_stream=None)
    assert_check_partials(result, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('index', [0, 1])
def test_csdl_geometry_and_magnetic_parity_with_corrected_analysis(index):
    p = reference_problem(index)
    for name, expected in REFERENCE['cases'][index]['outputs'].items():
        # The old fixture's speed-dependent operating results contain unit bugs.
        if name not in UNCHANGED_REFERENCE_OUTPUTS:
            continue
        np.testing.assert_allclose(p.get_val(name), expected, rtol=1e-7, atol=2e-9,
                                   err_msg=name)
    np.testing.assert_allclose(p['load_torque'], p['T_em']*p['efficiency_active'], atol=1e-10)
    np.testing.assert_allclose(p['input_power'], p['output_power']+p['P_loss'], atol=1e-10)
    expected_power = np.asarray(REFERENCE['cases'][index]['inputs']['load_torque_rotor'])*np.asarray(
        REFERENCE['cases'][index]['inputs']['omega_rotor'])*2*np.pi/60
    np.testing.assert_allclose(p['output_power'], expected_power, rtol=1e-12)
    np.testing.assert_allclose(p['input_power_active'], p['T_em']*p['omega_mechanical'], rtol=1e-12)
    assert_loss_units(p, 'input_load')
    w = 6*p['omega_mechanical']
    r, ld, lq, psi = (p[name] for name in ('Rdc', 'L_d', 'L_q', 'PsiF'))
    ud = r*p['Id_fw']-w*lq*p['Iq_fw']
    uq = r*p['Iq_fw']+w*(ld*p['Id_fw']+psi)
    np.testing.assert_allclose(np.sqrt(ud**2+uq**2), 800., rtol=1e-10)
    ud = r*p['Id_MTPA']-w*lq*p['Iq_MTPA']
    uq = r*p['Iq_MTPA']+w*(ld*p['Id_MTPA']+psi)
    np.testing.assert_allclose(p['U_MTPA'], np.sqrt(ud**2+uq**2), rtol=1e-12)
    totals = p.compute_totals(of=['output_power'], wrt=['omega_rotor', 'load_torque_rotor'])
    np.testing.assert_allclose(totals['output_power', 'omega_rotor'].ravel(),
                               p['load_torque_rotor']*2*np.pi/60, rtol=1e-12)
    np.testing.assert_allclose(totals['output_power', 'load_torque_rotor'].ravel(),
                               p['omega_rotor']*2*np.pi/60, rtol=1e-12)


@pytest.mark.parametrize('kind', ['input_load', 'efficiency_map'])
def test_csdl_currents_at_same_electrical_speed_and_corrected_losses(kind):
    case = next(c for c in REFERENCE['submodels'] if c['loss_model'] == kind)
    group = om.Group()
    group.add_subsystem('speed', MotorSpeedModel(), promotes=['*'])
    group.add_subsystem('fw', FluxWeakeningModel(), promotes=['*'])
    group.add_subsystem('mtpa', MTPAModel(), promotes=['*'])
    group.add_subsystem('losses', PostProcessingModel(loss_model=kind), promotes=['*'])
    p = problem(group)
    for name, value in case['inputs'].items():
        # The original dq equations treated omega as electrical rad/s.
        if name == 'omega':
            p['omega_mechanical'] = np.asarray(value)/6
        else:
            p.set_val(name, value)
    p.run_model()
    for name, expected in case['outputs'].items():
        if name in ('output_power', 'input_power_active', 'efficiency_active'):
            continue
        np.testing.assert_allclose(p[name], expected, rtol=1e-7, atol=1e-8, err_msg=name)
    assert_loss_units(p, kind)


def test_diagnostic_branch():
    p = problem(TC1MotorModel(model_test=True, fit_coeff_dep_H=REFERENCE['fit_coeff_dep_H'],
                             fit_coeff_dep_B=REFERENCE['fit_coeff_dep_B']))
    for name, value in REFERENCE['diagnostic']['inputs'].items():
        p[name] = value
    p.run_model()
    for name, expected in REFERENCE['diagnostic']['outputs'].items():
        if name in ('Id_fw', 'Iq_fw', 'input_power', 'efficiency'):
            continue
        np.testing.assert_allclose(p[name], expected, rtol=1e-7, atol=1e-8, err_msg=name)
    np.testing.assert_allclose(p['output_power'], p['load_torque']*p['omega_mechanical'])
    assert_loss_units(p, 'model_test')


@pytest.mark.parametrize('mode', ['fwd', 'rev'])
def test_derivative_modes(mode):
    p = om.Problem(model=TC1MotorModel(), reports=False)
    p.setup(mode=mode, force_alloc_complex=True)
    p['omega_rotor'], p['load_torque_rotor'] = [10000.], [40.]
    p.run_model()
    result = p.check_totals(of=['input_power', 'efficiency'], wrt=['D_i', 'L'],
                            method='cs', out_stream=None)
    assert_check_totals(result, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize('method', ['cs', 'fd'])
def test_motor_total_derivatives(method):
    p = reference_problem(1)
    kwargs = dict(method=method, out_stream=None)
    if method == 'fd':
        kwargs.update(form='central', step=1e-6)
    result = p.check_totals(of=['motor_mass', 'L_d', 'L_q', 'T_lim', 'T_em', 'input_power', 'efficiency'],
                            wrt=['D_i', 'L', 'omega_rotor', 'load_torque_rotor'], **kwargs)
    assert_check_totals(result, atol=1e-5 if method == 'fd' else 1e-7,
                        rtol=1e-5 if method == 'fd' else 1e-7)


def test_vector_active_nodes_and_derivatives():
    p = problem(TC1MotorModel(num_nodes=4, num_active_nodes=2))
    p['omega_rotor'] = [10000., 0., 12000., 7000.]
    p['load_torque_rotor'] = [40., 40., 60., 0.]
    p.run_model()
    np.testing.assert_array_equal(p['selection_indices'], [[1, 0], [0, 0], [0, 1], [0, 0]])
    for name in ['input_power', 'efficiency', 'output_power_full', 'T_em_full', 'current_amplitude_full']:
        assert p[name].shape == (4,)
        np.testing.assert_array_equal(p[name][[1, 3]], 0.)
    totals = p.compute_totals(of=['input_power'], wrt=['load_torque_rotor'])
    jac = totals['input_power', 'load_torque_rotor']
    assert jac[0, 0] > 0 and jac[2, 2] > 0
    np.testing.assert_allclose(jac-np.diag(np.diag(jac)), 0., atol=1e-10)
    result = p.check_totals(of=['input_power', 'efficiency'], wrt=['omega_rotor'],
                            method='cs', out_stream=None)
    assert_check_totals(result, atol=1e-8, rtol=1e-7)


def test_active_count_mismatch_is_clear():
    p = problem(ParseActiveOperatingConditions(num_nodes=3, num_active_nodes=2))
    p['omega_rotor'] = [1000., 0., 0.]
    p['load_torque_rotor'] = [40., 0., 0.]
    with pytest.raises(om.AnalysisError, match='Expected 2 active nodes, found 1'):
        p.run_model()


def test_all_inactive_nodes():
    p = problem(TC1MotorModel(num_nodes=3, num_active_nodes=0))
    p.run_model()
    for name in ['input_power', 'efficiency', 'T_em_full']:
        np.testing.assert_array_equal(p[name], np.zeros(3))
    assert p['motor_mass'][0] > 0


def test_positive_torque_root_and_power_balance():
    p = problem(TC1MotorModel())
    p['omega_rotor'] = [1000.]
    p['load_torque_rotor'] = [40.]
    p.run_model()
    assert 10 < p['T_em'][0] < 20
    np.testing.assert_allclose(p['efficiency_active']*p['T_em'], p['load_torque'], atol=1e-10)


def test_infeasible_load_raises_instead_of_returning_unconverged_state():
    p = problem(TC1MotorModel())
    p['omega_rotor'] = [1000.]
    p['load_torque_rotor'] = [8000.]
    with pytest.raises(om.AnalysisError, match='cannot be delivered'):
        p.run_model()


@pytest.mark.parametrize('cls', [EfficiencyMapModel, EMTorqueModel])
@pytest.mark.parametrize('method', ['cs', 'fd'])
def test_efficiency_map_nonzero_load_and_derivatives(cls, method):
    ref = reference_problem(1)
    kwargs = dict(num_nodes=1)
    if cls is EMTorqueModel:
        kwargs['mode'] = 'efficiency_map'
    p = problem(cls(**kwargs))
    for name in ['T_lim', *SPEED_UNITS, 'motor_variables', 'I_q_temp', 'B_delta', 'D_i',
                 'Id_fw_bracket', 'R_expanded', 'L_d_expanded', 'L_q_expanded', 'PsiF_expanded']:
        p[name] = ref[name]
    p['T_em'] = ref['T_em']
    p.run_model()
    assert p['load_torque'][0] > 0
    np.testing.assert_allclose(p['load_torque'], p['efficiency_active']*p['T_em'], atol=1e-10)
    if cls is EMTorqueModel:
        np.testing.assert_allclose(p['load_torque'], ref['load_torque'], atol=1e-9)
    kwargs = dict(method=method, out_stream=None)
    if method == 'fd':
        kwargs.update(form='central', step=1e-6)
    result = p.check_totals(of=['load_torque', 'input_power_active', 'efficiency_active'],
                            wrt=['T_em', 'R_expanded', *SPEED_UNITS], **kwargs)
    assert_check_totals(result, atol=1e-5 if method == 'fd' else 1e-8,
                        rtol=1e-5 if method == 'fd' else 1e-7)


@pytest.mark.parametrize('pole_pairs', [1, 3, 6])
def test_flux_weakening_voltage_boundary_and_mtpa_torque(pole_pairs):
    group = om.Group()
    group.add_subsystem('speed', MotorSpeedModel(pole_pairs=pole_pairs), promotes=['*'])
    group.add_subsystem('limit', TorqueLimitModel(V_lim=1000., pole_pairs=pole_pairs), promotes=['*'])
    group.add_subsystem('bracket_coefficients', FluxWeakeningBracketCoefficients(V_lim=1000., pole_pairs=pole_pairs), promotes=['*'])
    group.add_subsystem('bracket', FluxWeakeningBracketModel(pole_pairs=pole_pairs), promotes=['*'])
    group.add_subsystem('fw', FluxWeakeningModel(V_lim=1000., pole_pairs=pole_pairs), promotes=['*'])
    group.add_subsystem('mtpa', MTPAModel(pole_pairs=pole_pairs), promotes=['*'])
    p = problem(group)
    values = dict(Rdc=.0313, L_d=.0011, L_q=.0022, PsiF=.5494, omega_mechanical=[2200./pole_pairs], T_em=[1000.*pole_pairs/6])
    for name, value in values.items():
        p[name] = value
    p.run_model()
    for id_name, iq_name in [('Id_fw', 'Iq_fw'), ('Id_MTPA', 'Iq_MTPA')]:
        id_, iq = p[id_name], p[iq_name]
        torque = 1.5*pole_pairs*iq*(.5494+(.0011-.0022)*id_)
        np.testing.assert_allclose(torque, p['T_em'], rtol=1e-10)
    ud = .0313*p['Id_fw']-2200*.0022*p['Iq_fw']
    uq = 2200*.0011*p['Id_fw']+.0313*p['Iq_fw']+2200*.5494
    np.testing.assert_allclose(np.sqrt(ud**2+uq**2), 1000., rtol=1e-10)
    assert p['Id_fw'][0] < 0
    result = p.check_totals(of=['T_lim', 'Id_fw', 'Iq_fw', 'Iq_MTPA'],
                            wrt=['omega_mechanical', 'T_em', 'Rdc'], method='cs', out_stream=None)
    assert_check_totals(result, atol=1e-7, rtol=1e-7)


def test_mtpa_zero_torque_has_regular_derivatives():
    p = problem(MTPAModel(num_nodes=2))
    p['T_em'] = [0., 50.]
    p['L_d_expanded'] = [.0011, .0011]
    p['L_q_expanded'] = [.0022, .0022]
    p['PsiF_expanded'] = [.5494, .5494]
    p.run_model()
    assert p['Iq_MTPA'][0] == 0
    totals = p.compute_totals(of=['Iq_MTPA'], wrt=['T_em'])
    np.testing.assert_allclose(totals['Iq_MTPA', 'T_em'][0, 0], 1/(9*.5494), rtol=1e-10)


def test_sizing_parameter_vector_and_geometry_derivatives():
    p = problem(TC1MotorSizingModel())
    p.run_model()
    assert len(MOTOR_VARIABLE_NAMES) == len(p['motor_variables']) == 25
    before = p['motor_mass'].copy()
    p['L'] = .12
    p.run_model()
    assert p['motor_mass'][0] > before[0]
    result = p.check_totals(of=['Rdc', 'motor_mass', 'T_em_max', 'motor_variables'],
                            wrt=['D_i', 'L', 'N_p'], method='fd', form='central',
                            step=1e-6, out_stream=None)
    assert_check_totals(result, atol=1e-6, rtol=1e-5)


def test_permeability_data_and_path_independence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    h, b, data = permeability_fitting(test=True)
    assert h.shape == (4,) and b.shape == (3,)
    assert len(data['B_data']) == 51
    original = REFERENCE['fit_coeff_dep_B']
    np.testing.assert_allclose(fit_dep_B(data['B_data'], *b),
                               fit_dep_B(data['B_data'], *original), rtol=1e-5, atol=.01)


def test_runtime_modules_use_native_openmdao():
    for filename in (Path(__file__).parents[1]/'lsdo_motor_om/core/motor_submodels').glob('*.py'):
        module = importlib.import_module('lsdo_motor_om.core.motor_submodels.'+filename.stem)
        for name, cls in vars(module).items():
            if name.endswith('Model') and isinstance(cls, type) and cls.__module__ == module.__name__:
                assert issubclass(cls, (om.Group, om.ExplicitComponent, om.ImplicitComponent))
    assert 'csdl' not in sys.modules
    assert 'python_csdl_backend' not in sys.modules
