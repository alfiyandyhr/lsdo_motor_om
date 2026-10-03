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
python -m pip install -e '.[test,plot]'
python examples/basic.py
python examples/efficiency_map.py
python examples/optimize_motor.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

The examples run serially. Disabling pytest plugin autoload avoids the unrelated
Dash/Werkzeug plugin conflict present in the supplied environment.

### Plot an efficiency map

Install the optional plotting dependency and run the example:

```sh
python -m pip install -e '.[plot]'
python examples/efficiency_map.py
```

This opens a filled contour plot and saves `efficiency_map.png`. The x axis is
motor mechanical speed in RPM, the y axis is delivered shaft torque in N m,
and the color and contour labels show efficiency in percent. The example uses
the original efficiency-map loss variant, with `D_i=0.3723 m`, `L=0.2755 m`,
six pole pairs, an 800 V voltage limit, and a 123 A current limit. Gray regions
are outside the feasible or sampled positive-power motoring region; zero speed,
zero torque, and regeneration are excluded.

For a saved plot without a display, or different sampling ranges:

```sh
python examples/efficiency_map.py --no-show --output /tmp/efficiency_map.png
python examples/efficiency_map.py --rpm-max 3000 --torque-max 900 --rpm-points 101 --torque-points 101
python examples/efficiency_map.py --torque-axis em
```

The torque range controls the sampled electromagnetic torque `T_em`; the default
plot uses the model's resulting shaft torque `load_torque`. `--torque-axis em`
plots against `T_em` directly. The original motor loss equations remain unchanged.

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
print(problem['output_power']) # approximately [62831.853072] W
print(problem['input_power'])  # approximately [75196.634347] W
print(problem['efficiency'])   # approximately [0.83556736]
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
tests/                                   speed units, physics, legacy geometry, derivatives
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
| `omega_mechanical`, `omega` | `num_active_nodes` | Motor mechanical speed (rad/s); `omega` is an alias |
| `omega_electrical`, `electrical_frequency` | `num_active_nodes` | Motor electrical speed (rad/s) and frequency (Hz) |
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

The analysis uses distinct speed quantities throughout:

```text
omega_mechanical = omega_rotor * gear_ratio * 2*pi/60
omega_electrical = pole_pairs * omega_mechanical
electrical_frequency = omega_electrical / (2*pi)
output_power = load_torque * omega_mechanical
```

Rotor inputs remain RPM. Shaft power, windage, and the implicit torque loss
balance use mechanical rad/s. All dq voltage, voltage-limit, and flux-weakening
equations use electrical rad/s. Iron and magnet losses use electrical Hz.
The complete analysis supplies these conversions automatically, with OpenMDAO
units metadata and derivatives. `omega` remains an output alias for motor
mechanical speed.

When composing standalone submodels, add `MotorSpeedModel(pole_pairs=...,
num_nodes=...)` with `promotes=['*']` and supply `omega_mechanical` in rad/s.
Import it with `from lsdo_motor_om import MotorSpeedModel`. Torque-limit and
flux-weakening components now take `omega_electrical`; performance and implicit
torque components take all three explicit speed quantities. Their ambiguous
legacy `omega` input has been removed. Use the same pole count throughout the
assembly; see [the efficiency-map example](examples/efficiency_map.py).

These corrections change operating results and optimization outcomes relative
to the original CSDL package. The empirical loss coefficients and the separate
main, efficiency-map, and diagnostic loss variants remain; see
[the migration notes](docs/migration.md) for the equations and remaining model
assumptions. `voltage_amplitude` remains the inherited 5000 motor RPM
rated-voltage diagnostic; `U_MTPA` reports voltage at the operating speed.

## Validation

Tests retain original CSDL references for geometry, magnetic properties, and
currents evaluated at the same electrical speed. Legacy power and loss outputs
are historical and are not correctness targets. Physics checks cover ideal
gearbox power conservation, shaft power and its exact RPM/torque derivatives,
electrical frequency, iron/magnet/windage losses in all three variants, and dq
voltage-boundary currents across multiple pole counts. Tests also check torque
and efficiency balance, inactive-node routing, infeasible loads, and OpenMDAO
derivatives using complex step and central finite differences in both solve
modes. Ordinary tests need no CSDL. The optional reference-capture script
requires the original CSDL dependencies.

The original MIT license is retained in `LICENSE.txt`.
