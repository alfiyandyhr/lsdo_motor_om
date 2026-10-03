# Migration notes

## Model mapping

| Original CSDL system | Native OpenMDAO system |
|---|---|
| `TC1MotorSizingModel` | Group of explicit geometry and torque/mass components |
| `MagnetMECImplicitModel` | ImplicitComponent with a bracketed flux-balance solve |
| `MagnetMECModel` | Group exposing the flux state and circuit quantities |
| `InductanceQImplicitModel` | ImplicitComponent for rated-current flux balance |
| `InductanceModel` | Group of q-axis solve and explicit d/q inductance equations |
| `TorqueLimitModel`, `MaxTorqueModel` | Quartic coefficients, bracket selection, implicit torque state |
| `FluxWeakeningBracketModel` | Implicit stationary-current solve plus explicit d-current bound |
| `FluxWeakeningModel` | Quartic coefficients, implicit d-current, explicit q-current |
| `MTPAModel` | Normalized torque, implicit q-current, dimensional currents |
| `EMTorqueModel` | Implicit torque state, control groups, explicit power/loss component |
| `EfficiencyMapModel` | Same assembly with a load-torque state and the original map loss variant |
| `PostProcessingModel` | Working explicit current blend and loss calculation |
| Speed convention correction | `MotorSpeedModel` with explicit mechanical/electrical speeds and frequency |
| `ParseActiveOperatingConditions` | ExplicitComponent with exact gather Jacobians |

The implementation imports no CSDL or CSDL backend and does not translate a CSDL
graph at runtime. Implicit state residuals and their Jacobians are exposed to
OpenMDAO. Local nonlinear solves use SciPy's bracketed root solver; each implicit
component uses an OpenMDAO DirectSolver for its residual linear system. Residual
partials are computed by complex step. Local solves continue the real root under
complex perturbations, so whole-model complex-step derivative checks also work.
This follows OpenMDAO's native
[ImplicitComponent interface](https://openmdao.org/newdocs/versions/latest/features/core_features/working_with_components/implicit_component.html).

## Preserved equations and conventions

`D_i` is a diameter despite several original comments calling it a radius.
`outer_stator_radius` and `rotor_radius` retain their original names and formulas,
which also behave as diameters. The packed geometry vector has the exact original
ordering. A four-to-one gearbox is the default; `gear_ratio` can override it.

## Corrected speed conventions

The inherited CSDL implementation converted rotor RPM to motor mechanical
rad/s in the gearbox, then multiplied by `2*pi/60` again in shaft power. This
reduced shaft power to about 10.47% of its correct value. It also used mechanical
speed directly in electrical voltage equations and used `/60` to turn mechanical
rad/s into an iron-loss frequency. These speed factors are now corrected in all
three loss variants and both directions of the implicit torque solve.

| Quantity | Equation | Uses |
|---|---|---|
| `omega_rotor` | User-supplied rotor RPM | Active-node gather and gearbox |
| `omega_mechanical` | `omega_rotor * gear_ratio * 2*pi/60` | Shaft power, windage, torque loss balance |
| `omega_electrical` | `pole_pairs * omega_mechanical` | dq voltages, torque limit, flux weakening and brackets |
| `electrical_frequency` | `omega_electrical / (2*pi)` | Stator/rotor iron and magnet eddy losses |
| `omega` | Alias of `omega_mechanical` | Compatibility output of the complete analysis |

The ideal gearbox sets `load_torque = load_torque_rotor / gear_ratio`, so
`output_power = load_torque * omega_mechanical` equals rotor torque times rotor
angular speed regardless of gear ratio. Windage retains its empirical quadratic
speed dependence, now using mechanical rad/s.

The steady dq voltage equations are
`u_d = R*i_d - omega_electrical*L_q*i_q` and
`u_q = R*i_q + omega_electrical*(L_d*i_d + PsiF)`. Electrical angular speed is
pole count times mechanical angular speed, consistent with the
[MathWorks PMSM equations](https://www.mathworks.com/help/mcb/ref/pmsmhdl.html).
All voltage polynomials, current brackets, and control blending use that same
electrical speed.

The torque residual is `load_torque + P_loss/omega_mechanical - T_em = 0`.
The reverse efficiency-map solve uses
`load_torque = (T_em - P_loss_at_zero_load/omega_mechanical)/1.01`, where 1.01
accounts for the existing 1% shaft-power stress loss. Residual derivatives
include separate mechanical speed, electrical speed, and electrical frequency
dependencies; OpenMDAO propagates their conversion derivatives back to rotor
RPM. The implicit solve and explicit post-processing share the loss evaluator.

`MotorSpeedModel` has exact diagonal conversion Jacobians and units metadata.
The complete motor inserts it after the gearbox. Standalone assemblies should
add it with the same `pole_pairs` and `num_nodes` as their control groups, and
set `omega_mechanical` in rad/s. The former standalone `omega` input is replaced
by `omega_electrical` in voltage/control components, and by all three speed
quantities in performance/torque components. See
[the efficiency-map example](../examples/efficiency_map.py).

For the README point (`D_i=.3723`, `L=.2755`, rotor RPM 1500, rotor torque
400 Nm, gear ratio 4, 6 pole pairs), the corrected mechanical speed is
628.318531 rad/s, electrical speed is 3769.911184 rad/s, and frequency is 600 Hz.
Shaft power is 62831.853072 W, input power is 75196.634347 W, and efficiency
is 0.83556736. The original values were about 6579.736267 W shaft power,
6867.662186 W input power, and 0.95807512 efficiency. The changed voltage and
loss calculations also affect current, torque limits, and optimized geometry.

## Preserved empirical loss variants

The main loss model uses rotor and stator iron volumes, magnet eddy losses, a
constant 100 W loss, and smoothing coefficient 0.5. The separate efficiency-map
file uses stator iron volume, omits magnet eddy and constant losses, and uses
smoothing coefficient 1. The `model_test=True` diagnostic branch retains its
alternate volume, eddy coefficient, and rated-voltage blend.
`voltage_amplitude` still reports the inherited rated-voltage diagnostic at
5000 motor mechanical RPM; `U_MTPA` uses the operating electrical speed.
The original fixed 36-slot constants in sizing/loss expressions and placeholder
torque/mass coefficients for orders 3 and 4 remain as supplied.

Supported operating solves are positive-speed motoring cases with `PsiF > 0`
and `L_q > L_d`. Regeneration and equal-inductance surface-PM cases were not
supported by the original positive torque brackets and are not new capabilities
of this migration. Zero rotor RPM or zero load is routed as an inactive node.

## Numerical repairs

1. The source's torque bracket loop constructed a KeyError without raising it
   after 1000 iterations. The port bounds the search and raises AnalysisError.
2. The electromagnetic torque equation can have two positive roots. A broad
   endpoint bracket can miss both; the port locates the first sign change and
   returns the first feasible torque root. A bounded minimum search handles
   narrow feasible intervals. Infeasible loads raise AnalysisError.
3. At the maximum voltage torque, the current quartic has a repeated root.
   Solving its stationary-point cubic finds the same current with a regular
   Jacobian. The original Newton iteration stopped at a slightly displaced
   current; current correctness is checked using the dq voltage boundary.
4. The MTPA positive root is represented by
   `q*(1+sqrt(1+4*q**2))/2 - T_star = 0`, equivalent to the original quartic
   on the motoring branch, with a regular derivative at zero torque.
5. A stable logistic expression avoids exponential overflow in current blending.
   The KS smooth minimum retains the source's smoothing parameter of 20.
6. Efficiency-map solving uses the equivalent positive-power loss residual,
   avoiding the zero-load root introduced by multiplying the torque equality
   by a zero efficiency. Torque too small to cover no-load losses raises an error.
7. The original standalone PostProcessingModel referenced undefined
   `op_voltage`. The port uses the working electromagnetic-torque model's blend
   and expanded electrical inputs. It is now a usable explicit component.
8. Permeability fitting identifies only the active H-fit amplitude and the
   identifiable exponential B-fit amplitude/slope. This preserves the original
   functional families without their singular covariance warning. The public
   four- and three-coefficient interfaces remain available; explicit coefficient
   arrays can be passed for exact legacy-fit reproduction.
9. Full-node output routing is provided for efficiency, torque, current, power,
   and load, extending the original input-power-only routing. The active-node
   names remain available. Invalid active counts fail with a clear error.

## Validation and reference capture

The saved CSDL fixture uses the existing local source, including its preexisting
working-tree edits. It remains a historical record of the inherited equations;
its speed-dependent power, losses, torque limits, and solved torque are no
longer regression targets. Tests retain geometry and magnetic-property parity
at both original operating points, and current parity after interpreting the
legacy dq speed as electrical rad/s. The source checkout is not modified.

Independent checks verify shaft power and its analytic RPM/torque derivatives,
gearbox power conservation at multiple ratios, electrical frequency, iron and
windage loss expressions in all variants, and voltage-boundary currents at
multiple pole counts. The main and reverse torque solves are checked for power
balance and with both complex-step and central finite-difference derivatives,
including speed and geometry dependencies. Both forward and reverse OpenMDAO
linear modes are exercised. All three examples, including motor optimization,
run with the corrected conventions. The implementation has been exercised with
OpenMDAO 3.45.1, NumPy 2.5.3, and SciPy 1.18.1 in the `openmdao` conda environment.

To regenerate the historical CSDL fixture when the original dependencies are installed:

```sh
conda activate openmdao
FI_PROVIDER=tcp python tools/generate_csdl_reference.py
```

Reference capture rejects unconverged end-to-end CSDL states. Tests read saved
JSON and do not import CSDL. Both forward and reverse OpenMDAO derivatives can
be requested through `problem.setup(mode='fwd')` or `mode='rev'`; for derivative
checks involving complex perturbations, also pass `force_alloc_complex=True`.
