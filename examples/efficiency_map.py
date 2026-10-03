"""Plot the TC1 efficiency-map loss variant against motor RPM and torque.

Install plotting support: python -m pip install -e '.[plot]'
Run interactively: python examples/efficiency_map.py
Save without a GUI: python examples/efficiency_map.py --no-show --output map.png

The default vertical axis is delivered shaft torque. Use --torque-axis em for
an electromagnetic-torque axis. RPM is motor mechanical speed, with no gearbox.
Only positive-power motoring points inside the voltage, structural torque, and
123 A rated-current limits are plotted. Zero speed/torque and regeneration are
outside this example's model domain.
"""

import argparse
from dataclasses import dataclass
import os
from pathlib import Path

# The OpenMDAO environment may import MPI even for these serial examples.
os.environ.setdefault('FI_PROVIDER', 'tcp')

import numpy as np
import openmdao.api as om

from lsdo_motor_om import TC1MotorSizingModel
from lsdo_motor_om.core._utils import MotorVariablesModel
from lsdo_motor_om.core.motor_submodels.TC1_motor_speed_model import (
    MotorSpeedModel, SPEED_UNITS,
)
from lsdo_motor_om.core.motor_submodels.TC1_magnet_mec_model import MagnetMECModel
from lsdo_motor_om.core.motor_submodels.TC1_inductance_mec_model import InductanceModel
from lsdo_motor_om.core.motor_submodels.TC1_torque_limit_model import (
    TorqueLimitModel, ELECTRICAL_NAMES,
)
from lsdo_motor_om.core.motor_submodels.TC1_flux_weakening_model import (
    FluxWeakeningBracketCoefficients, FluxWeakeningBracketModel,
)
from lsdo_motor_om.core.motor_submodels.TC1_efficiency_map_model import EfficiencyMapModel
from lsdo_motor_om.core.motor_submodels.TC1_post_processing_model import node_performance


RATED_CURRENT = 123.


class EfficiencyMapMotor(om.Group):
    """Original efficiency-map assembly, optionally stopping at operating limits."""

    def initialize(self):
        self.options.declare('num_nodes', default=6, types=int, lower=1)
        self.options.declare('limits_only', default=False, types=bool)

    def setup(self):
        n = self.options['num_nodes']
        self.add_subsystem('sizing', TC1MotorSizingModel(), promotes=['*'])
        self.add_subsystem('geometry', MotorVariablesModel(), promotes=['*'])
        self.add_subsystem('magnet', MagnetMECModel(), promotes=['*'])
        self.add_subsystem('inductance', InductanceModel(), promotes=['*'])
        self.add_subsystem('flux_linkage', om.ExecComp('PsiF=turns_per_phase*phi_air'), promotes=['*'])
        self.add_subsystem('speed', MotorSpeedModel(num_nodes=n), promotes=['*'])
        self.add_subsystem('torque_limit', TorqueLimitModel(num_nodes=n), promotes=['*'])
        self.add_subsystem('bracket_coefficients', FluxWeakeningBracketCoefficients(num_nodes=n), promotes=['*'])
        self.add_subsystem('bracket', FluxWeakeningBracketModel(num_nodes=n), promotes=['*'])
        if not self.options['limits_only']:
            self.add_subsystem('efficiency_map', EfficiencyMapModel(num_nodes=n), promotes=['*'])
        self.set_input_defaults('D_i', val=.3723)
        self.set_input_defaults('L', val=.2755)


def build_problem(rpm, torque, *, limits_only=False):
    """Build paired/broadcast operating points, converting mechanical RPM to rad/s."""
    rpm, torque = np.broadcast_arrays(rpm, torque)
    p = om.Problem(model=EfficiencyMapMotor(
        num_nodes=rpm.size, limits_only=limits_only), reports=False)
    p.setup()
    p.set_val('omega_mechanical', rpm.ravel()*2*np.pi/60, units='rad/s')
    if not limits_only:
        p.set_val('T_em', torque.ravel())
    return p, rpm.shape


@dataclass
class EfficiencyMap:
    rpm: np.ndarray
    electromagnetic_torque: np.ndarray
    shaft_torque: np.ndarray
    efficiency: np.ma.MaskedArray
    current: np.ndarray


def evaluate_map(rpm=None, torque=None):
    """Evaluate a rectangular RPM/T_em grid; mask unsupported operating points.

    Screening uses the model's own losses with load=0. Its stress loss is
    0.01*load*omega, so delivered torque is (T_em-loss_at_zero_load/omega)/1.01.
    Feasible points are then evaluated through the original EfficiencyMapModel.
    This prevents one infeasible point from aborting the whole vectorized solve.
    """
    rpm = np.asarray(np.linspace(100., 4000., 81) if rpm is None else rpm, dtype=float)
    torque = np.asarray(np.linspace(5., 1000., 81) if torque is None else torque, dtype=float)
    for name, axis in [('RPM', rpm), ('torque', torque)]:
        if (axis.ndim != 1 or axis.size < 2 or not np.isfinite(axis).all()
                or np.any(axis <= 0) or np.any(np.diff(axis) <= 0)):
            raise ValueError(f'{name} must contain at least two finite, positive, increasing values.')

    speed_grid, torque_grid = np.meshgrid(rpm, torque)
    efficiency = np.full(speed_grid.shape, np.nan)
    current = np.full(speed_grid.shape, np.nan)
    # Finite placeholders keep contour coordinates valid in masked regions.
    shaft_torque = torque_grid.copy()
    valid = np.zeros(speed_grid.shape, dtype=bool)

    limits, _ = build_problem(rpm, 0., limits_only=True)
    try:
        limits.run_model()
        names = (*ELECTRICAL_NAMES, *SPEED_UNITS, 'motor_variables', 'D_i',
                 'B_delta', 'I_q_temp', 'Id_fw_bracket')
        values = {name: limits.get_val(name).copy() for name in names}
        upper = np.minimum(limits.get_val('T_lim')*(1-1e-8),
                           limits.get_val('T_em_max'))
        options = EfficiencyMapModel().options
        for j in range(rpm.size):
            for i in np.flatnonzero(torque < upper[j]):
                performance = node_performance(values, j, torque[i], 0., options)
                load = (torque[i]-performance['P_loss']/values['omega_mechanical'][j])/1.01
                current[i, j] = performance['current_amplitude']
                shaft_torque[i, j] = load
                valid[i, j] = (np.isfinite(load) and load > 0
                               and np.isfinite(current[i, j])
                               and current[i, j] <= RATED_CURRENT)
    finally:
        limits.cleanup()

    if not valid.any():
        raise ValueError('No feasible motoring points in this grid; adjust the RPM/torque ranges.')

    p, _ = build_problem(speed_grid[valid], torque_grid[valid])
    try:
        p.run_model()
        efficiency[valid] = p.get_val('efficiency_active')
        shaft_torque[valid] = p.get_val('load_torque')
        current[valid] = p.get_val('current_amplitude')
    finally:
        p.cleanup()
    if not np.isfinite(efficiency[valid]).all() or np.any(
            (efficiency[valid] <= 0) | (efficiency[valid] > 1)):
        raise RuntimeError('The motor model returned invalid motoring efficiencies.')
    return EfficiencyMap(speed_grid, torque_grid, shaft_torque,
                         np.ma.array(efficiency, mask=~valid), current)


def plot_map(data, *, torque_axis='shaft'):
    """Create a filled efficiency (%) contour map; return its figure and axes."""
    import matplotlib.pyplot as plt

    if torque_axis not in ('shaft', 'em'):
        raise ValueError("torque_axis must be 'shaft' or 'em'.")
    mask = np.ma.getmaskarray(data.efficiency)
    # Contours need at least one complete cell. Do not bridge infeasible points.
    cells = ~(mask[:-1, :-1] | mask[1:, :-1] | mask[:-1, 1:] | mask[1:, 1:])
    if not cells.any():
        raise ValueError('No complete feasible contour cells; use a finer grid or adjust the ranges.')
    torque = data.shaft_torque if torque_axis == 'shaft' else data.electromagnetic_torque
    percent = 100*data.efficiency
    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
    ax.set_facecolor('#eeeeee')
    filled = ax.contourf(data.rpm, torque, percent, levels=np.linspace(70, 100, 61),
                         cmap='viridis', extend='min', corner_mask=False)
    levels = [level for level in (50, 70, 80, 90, 95, 97, 98)
              if percent.min() < level < percent.max()]
    if levels:
        lines = ax.contour(data.rpm, torque, percent, levels=levels,
                           colors='black', linewidths=.6, corner_mask=False)
        ax.clabel(lines, fmt='%g%%', fontsize=8)
    fig.colorbar(filled, ax=ax, label='Efficiency (%)', ticks=np.arange(70, 101, 5))
    ax.set(xlabel='Motor speed (RPM)',
           ylabel=('Shaft torque (N m)' if torque_axis == 'shaft'
                   else 'Electromagnetic torque (N m)'),
           title='TC1 motor efficiency map',
           xlim=(0, data.rpm.max()),
           ylim=(0, data.electromagnetic_torque.max()))
    ax.text(.98, .97, 'Gray: outside feasible or sampled motoring region\n'
            f'Current limit: {RATED_CURRENT:g} A; voltage limit: 800 V',
            transform=ax.transAxes, fontsize=9, ha='right', va='top',
            bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
    return fig, ax


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rpm-min', type=float, default=100.)
    parser.add_argument('--rpm-max', type=float, default=4000.)
    parser.add_argument('--rpm-points', type=int, default=81)
    parser.add_argument('--torque-min', type=float, default=5., help='Minimum sampled electromagnetic torque (N m).')
    parser.add_argument('--torque-max', type=float, default=1000., help='Maximum sampled electromagnetic torque (N m).')
    parser.add_argument('--torque-points', type=int, default=81)
    parser.add_argument('--torque-axis', choices=('shaft', 'em'), default='shaft')
    parser.add_argument('--output', type=Path, default=Path('efficiency_map.png'))
    parser.add_argument('--no-show', action='store_true', help='Save the plot without opening a window.')
    args = parser.parse_args(argv)
    if (not np.isfinite([args.rpm_min, args.rpm_max, args.torque_min, args.torque_max]).all()
            or not 0 < args.rpm_min < args.rpm_max
            or not 0 < args.torque_min < args.torque_max
            or args.rpm_points < 2 or args.torque_points < 2):
        parser.error('Ranges must be finite, positive, increasing, with at least two points per axis.')
    try:
        import matplotlib
        if args.no_show:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Install plotting support with: python -m pip install -e '.[plot]'") from exc

    data = evaluate_map(np.linspace(args.rpm_min, args.rpm_max, args.rpm_points),
                        np.linspace(args.torque_min, args.torque_max, args.torque_points))
    fig, _ = plot_map(data, torque_axis=args.torque_axis)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f'Saved efficiency map: {args.output.resolve()}')
    print(f'Feasible points: {data.efficiency.count()}/{data.efficiency.size}; '
          f'peak efficiency: {100*data.efficiency.max():.2f}%')
    if not args.no_show:
        plt.show()
    plt.close(fig)
    return data


if __name__ == '__main__':
    main()
