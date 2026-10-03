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

In the main analysis path, `omega = omega_rotor * gear_ratio * 2*pi/60` and
`load_torque = load_torque_rotor / gear_ratio`. The original torque residual then
uses `output_power = load_torque * omega * 2*pi/60`, while its iron-loss frequency
is `omega*pole_pairs/60`. Its voltage equations also use `omega` directly.
These conventions are inconsistent if `omega` is interpreted as a single SI
angular speed. They are deliberately preserved to match the supplied CSDL code.
No units metadata is attached that could imply OpenMDAO resolves this inconsistency.
A future physical-units correction should be a separate, independently validated
change with new reference results.

The main loss model uses rotor and stator iron volumes, magnet eddy losses, a
constant 100 W loss, and smoothing coefficient 0.5. The separate efficiency-map
file uses stator iron volume, omits magnet eddy and constant losses, and uses
smoothing coefficient 1. The `model_test=True` diagnostic branch retains its
alternate volume, eddy coefficient, power factor, and rated-voltage blend.
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
   current; parity tests allow 2e-5 relative error for these two bracket currents.
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

The saved fixture uses the existing local source, including its preexisting
working-tree edits. The port does not modify that checkout. Full-model reference
cases are `(D_i=.182, L=.086, rotor RPM=10000, rotor torque=40)` and
`(D_i=.3723, L=.2755, rotor RPM=1500, rotor torque=400)`, with 6 pole pairs,
3 phases, 36 slots, 123 A rated current, and 800 V voltage limit. Additional
fixtures compare explicit current/loss evaluations at known torque for both
main and efficiency-map variants, and the alternate diagnostic branch. The
diagnostic case's unused flux-weakening branch lies close to a current asymptote;
its reference-current tolerance is 5e-5, while the blended current and power
outputs are checked at 1e-7. The implementation has been exercised with
OpenMDAO 3.45.1, NumPy 2.5.3, and SciPy 1.18.1 in the `openmdao` conda environment.

To regenerate fixtures when the original dependencies are installed:

```sh
conda activate openmdao
FI_PROVIDER=tcp python tools/generate_csdl_reference.py
```

Reference capture rejects unconverged end-to-end CSDL states. Tests read saved
JSON and do not import CSDL. Both forward and reverse OpenMDAO derivatives can
be requested through `problem.setup(mode='fwd')` or `mode='rev'`; for derivative
checks involving complex perturbations, also pass `force_alloc_complex=True`.
