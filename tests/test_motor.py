"""Numerical parity, conservation, control branches, and optimization derivatives."""

import importlib
import json
from pathlib import Path
import sys
import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_totals

from lsdo_motor_om import TC1MotorModel, TC1MotorSizingModel, ParseActiveOperatingConditions
from lsdo_motor_om.core._utils import MOTOR_VARIABLE_NAMES
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


@pytest.mark.parametrize('index', [0, 1])
def test_csdl_numerical_parity(index):
    p = reference_problem(index)
    for name, expected in REFERENCE['cases'][index]['outputs'].items():
        # The CSDL Newton solve is ill-conditioned at the repeated-current root.
        tolerance = 2e-5 if name in ('Id_fw_bracket', 'Iq_fw_bracket') else 1e-7
        np.testing.assert_allclose(p.get_val(name), expected, rtol=tolerance, atol=2e-9,
                                   err_msg=name)
    np.testing.assert_allclose(p['load_torque'], p['T_em']*p['efficiency_active'], atol=1e-10)
    np.testing.assert_allclose(p['input_power'], p['output_power']+p['P_loss'], atol=1e-10)


@pytest.mark.parametrize('kind', ['input_load', 'efficiency_map'])
def test_csdl_current_and_loss_submodels(kind):
    case = next(c for c in REFERENCE['submodels'] if c['loss_model'] == kind)
    group = om.Group()
    group.add_subsystem('fw', FluxWeakeningModel(), promotes=['*'])
    group.add_subsystem('mtpa', MTPAModel(), promotes=['*'])
    group.add_subsystem('losses', PostProcessingModel(loss_model=kind), promotes=['*'])
    p = problem(group)
    for name, value in case['inputs'].items():
        p.set_val(name, value)
    p.run_model()
    for name, expected in case['outputs'].items():
        np.testing.assert_allclose(p[name], expected, rtol=1e-7, atol=1e-8, err_msg=name)


def test_csdl_diagnostic_branch():
    p = problem(TC1MotorModel(model_test=True, fit_coeff_dep_H=REFERENCE['fit_coeff_dep_H'],
                             fit_coeff_dep_B=REFERENCE['fit_coeff_dep_B']))
    for name, value in REFERENCE['diagnostic']['inputs'].items():
        p[name] = value
    p.run_model()
    for name, expected in REFERENCE['diagnostic']['outputs'].items():
        # The unused voltage-boundary branch lies close to the current
        # asymptote here, amplifying the CSDL d-current solve's tolerance.
        tolerance = 5e-5 if name in ('Id_fw', 'Iq_fw') else 1e-7
        np.testing.assert_allclose(p[name], expected, rtol=tolerance, atol=1e-8, err_msg=name)
    np.testing.assert_allclose(p['output_power'], p['load_torque']*p['omega'])


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


def test_first_root_when_original_endpoint_bracket_misses_it():
    p = problem(TC1MotorModel())
    p['omega_rotor'] = [1000.]
    p['load_torque_rotor'] = [40.]
    p.run_model()
    assert 10 < p['T_em'][0] < 20
    np.testing.assert_allclose(p['efficiency_active']*p['T_em'], p['load_torque'], atol=1e-10)


def test_infeasible_load_raises_instead_of_returning_unconverged_state():
    p = problem(TC1MotorModel())
    p['omega_rotor'] = [1000.]
    p['load_torque_rotor'] = [4000.]
    with pytest.raises(om.AnalysisError, match='cannot be delivered'):
        p.run_model()


@pytest.mark.parametrize('cls', [EfficiencyMapModel, EMTorqueModel])
def test_efficiency_map_nonzero_load_and_derivatives(cls):
    ref = reference_problem(1)
    kwargs = dict(num_nodes=1)
    if cls is EMTorqueModel:
        kwargs['mode'] = 'efficiency_map'
    p = problem(cls(**kwargs))
    for name in ['T_lim', 'omega', 'motor_variables', 'I_q_temp', 'B_delta', 'D_i',
                 'Id_fw_bracket', 'R_expanded', 'L_d_expanded', 'L_q_expanded', 'PsiF_expanded']:
        p[name] = ref[name]
    p['T_em'] = ref['T_em']
    p.run_model()
    assert p['load_torque'][0] > 0
    np.testing.assert_allclose(p['load_torque'], p['efficiency_active']*p['T_em'], atol=1e-10)
    if cls is EMTorqueModel:
        np.testing.assert_allclose(p['load_torque'], ref['load_torque'], atol=1e-9)
    result = p.check_totals(of=['load_torque', 'input_power_active', 'efficiency_active'],
                            wrt=['T_em', 'R_expanded'], method='cs', out_stream=None)
    assert_check_totals(result, atol=1e-8, rtol=1e-7)


def test_flux_weakening_voltage_boundary_and_mtpa_torque():
    group = om.Group()
    group.add_subsystem('limit', TorqueLimitModel(V_lim=1000.), promotes=['*'])
    group.add_subsystem('bracket_coefficients', FluxWeakeningBracketCoefficients(V_lim=1000.), promotes=['*'])
    group.add_subsystem('bracket', FluxWeakeningBracketModel(), promotes=['*'])
    group.add_subsystem('fw', FluxWeakeningModel(V_lim=1000.), promotes=['*'])
    group.add_subsystem('mtpa', MTPAModel(), promotes=['*'])
    p = problem(group)
    values = dict(Rdc=.0313, L_d=.0011, L_q=.0022, PsiF=.5494, omega=[2200.], T_em=[1000.])
    for name, value in values.items():
        p[name] = value
    p.run_model()
    for id_name, iq_name in [('Id_fw', 'Iq_fw'), ('Id_MTPA', 'Iq_MTPA')]:
        id_, iq = p[id_name], p[iq_name]
        torque = 9*iq*(.5494+(.0011-.0022)*id_)
        np.testing.assert_allclose(torque, 1000., rtol=1e-10)
    ud = .0313*p['Id_fw']-2200*.0022*p['Iq_fw']
    uq = 2200*.0011*p['Id_fw']+.0313*p['Iq_fw']+2200*.5494
    np.testing.assert_allclose(np.sqrt(ud**2+uq**2), 1000., rtol=1e-10)
    assert p['Id_fw'][0] < 0
    result = p.check_totals(of=['T_lim', 'Id_fw', 'Iq_fw', 'Iq_MTPA'],
                            wrt=['omega', 'T_em', 'Rdc'], method='cs', out_stream=None)
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
