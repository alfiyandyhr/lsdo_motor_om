# lsdo_motor_om

A native OpenMDAO migration of the LSDO Motor TC1 permanent-magnet
synchronous motor models. The original CSDL-based `lsdo_motor` package was
developed by LSDOLab and is available in the
[LSDOLab/lsdo_motor repository](https://github.com/LSDOlab/lsdo_motor/).
This repository contains the migrated implementation, tests, examples, and packaging.
The runtime depends only on OpenMDAO, NumPy, and SciPy.

## Install and run

Install the package in editable mode from this repository:

```sh
python -m pip install -e .
```

For tests and examples in the OpenMDAO development environment:

```sh
conda activate openmdao
python -m pip install -e '.[test]'
python examples/basic.py
python examples/efficiency_map.py
python examples/optimize_motor.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

The examples run serially. Disabling pytest plugin autoload avoids the unrelated
Dash/Werkzeug plugin conflict present in the supplied environment.

## Use the complete motor

```python
import openmdao.api as om
from lsdo_motor_om import TC1MotorModel

problem = om.Problem(model=TC1MotorModel(
    pole_pairs=6, phases=3, num_slots=36, rated_current=123,
    V_lim=800, num_nodes=1, num_active_nodes=1,
), reports=False)
problem.setup(force_alloc_complex=True)
problem.set_val('D_i', 0.3723)               # stator bore diameter, m
problem.set_val('L', 0.2755)                # stack length, m
problem.set_val('omega_rotor', [1500.])     # rotor RPM before the gearbox
problem.set_val('load_torque_rotor', [400.]) # rotor load torque, Nm
problem.run_model()

print(problem['motor_mass'])   # approximately [334.34260786] kg
print(problem['input_power'])  # approximately [6867.662186] W
print(problem['efficiency'])   # approximately [0.95807512]
print(problem.compute_totals(of=['input_power'], wrt=['D_i', 'L']))
```

The sizing and analysis groups can also be added separately and promoted with
`promotes=['*']`. Set the shared diameter default on the parent group using
`model.set_input_defaults('D_i', val=...)` when composing them manually.
Their original class names, option names, submodel filenames, and 25-entry
`motor_variables` ordering are retained.

## Structure

```text
lsdo_motor_om/
  __init__.py
  core/
    TC1_motor_sizing_model.py
    TC1_motor_analysis_model.py
    TC1_motor_model.py                    convenience sizing + analysis group
    motor_submodels/
      TC1_magnet_mec_model.py
      TC1_inductance_mec_model.py
      TC1_torque_limit_model.py
      TC1_flux_weakening_model.py
      TC1_mtpa_model.py
      TC1_implicit_em_torque_model.py
      TC1_efficiency_map_model.py
      TC1_post_processing_model.py
    permeability/
      mu_fitting.py
      Magnetic_alloy_silicon_core_iron_C.tab
examples/                                analysis, efficiency map, optimization
tests/                                   CSDL parity and derivative regression tests
docs/migration.md                        model mapping, conventions, and repairs
tools/generate_csdl_reference.py          optional CSDL reference capture
```

## Inputs and outputs

Sizing inputs are `D_i`, `L`, and `N_p` (parallel conductors, default 2). Motor
properties include `motor_mass`, `Rdc`, `T_em_max`, `motor_variables`, `B_delta`,
`phi_air`, `L_d`, `L_q`, and `PsiF`. Permeability coefficients are optional;
the default fit reads the packaged data independently of the working directory.

| Output | Shape | Meaning |
|---|---|---|
| `input_power`, `efficiency` | `num_nodes` | Power/efficiency scattered back to all nodes |
| `T_em_full`, `current_amplitude_full` | `num_nodes` | Electromagnetic torque/current at all nodes |
| `output_power_full`, `load_torque_full` | `num_nodes` | Output power and gearbox-adjusted load torque |
| `T_em`, `current_amplitude`, `output_power` | `num_active_nodes` | Active-node results |
| `input_power_active`, `efficiency_active` | `num_active_nodes` | Active-node power and efficiency |
| `T_lim`, `T_upper_lim_curve` | `num_active_nodes` | Voltage torque limit, then smooth structural/voltage minimum |
| `max_torque_constraint` | `num_active_nodes` | Original torque margin: upper limit minus load torque |
| `em_torque_constraint` | `num_active_nodes` | Upper limit minus electromagnetic torque |

Nodes are active when both rotor RPM and rotor torque are nonzero. Declare their
count with `num_active_nodes`; it must match at runtime. Inactive full-array
outputs are zero. `num_active_nodes=0` produces zero full-array operating outputs
while still calculating motor properties; active-array outputs are absent.
The active mask must remain fixed during a derivative calculation or optimization.

`EMTorqueModel(mode='efficiency_map')` reverses the main torque solve while using
the main loss equations. `EfficiencyMapModel` preserves the separate original
efficiency-map loss variant. `TC1MotorModel(model_test=True)` accepts active-node
`T_em` inputs and preserves the alternate diagnostic equations.

The original model contains inconsistent speed factors and empirical constants.
This migration preserves its working numerical equations; consult
[the migration notes](docs/migration.md) before treating the power/speed convention
as a physically consistent SI formulation.

## Validation

Tests compare against saved outputs from the original local CSDL working tree at
two converged operating points, both original current/loss variants, and the
alternate diagnostic branch.
They also check power balance, torque/efficiency equality, voltage-boundary
currents, inactive-node routing, infeasible loads, and OpenMDAO total derivatives
using complex step and central finite differences. Ordinary tests need no CSDL.
The optional reference-capture script requires the original CSDL dependencies.

The original MIT license is retained in `LICENSE.txt`.
