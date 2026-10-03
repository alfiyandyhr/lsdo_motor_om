"""RPM conversion, feasible-region masking, and shaft-torque plotting."""

import numpy as np
import pytest

from examples.efficiency_map import (
    RATED_CURRENT, build_problem, evaluate_map, plot_map,
)


def test_map_uses_mechanical_rpm_and_loss_adjusted_shaft_torque():
    data = evaluate_map([500., 1000., 1500.], [100., 300., 500.])
    assert data.efficiency.count() == 9
    np.testing.assert_allclose(data.shaft_torque,
                               data.electromagnetic_torque*data.efficiency)
    p, shape = build_problem(data.rpm, data.electromagnetic_torque)
    try:
        p.run_model()
        np.testing.assert_allclose(p['omega_mechanical'].reshape(shape),
                                   data.rpm*2*np.pi/60)
        np.testing.assert_allclose(p['omega_electrical'], 6*p['omega_mechanical'])
        np.testing.assert_allclose(data.efficiency, p['efficiency_active'].reshape(shape))
        np.testing.assert_allclose(p['input_power_active'],
                                   p['output_power']+p['P_loss'])
    finally:
        p.cleanup()


def test_map_masks_no_load_losses_voltage_and_rated_current_limits():
    data = evaluate_map([500., 1500., 5000.], [.01, 100., 10000.])
    mask = np.ma.getmaskarray(data.efficiency)
    assert mask[0].all()  # Torque cannot cover losses.
    assert mask[2].all()  # Above structural/voltage torque limits.
    assert mask[1, 2]     # Flux-weakening current exceeds the rating.
    assert not mask[1, :2].any()
    assert data.current[1, 2] > RATED_CURRENT
    assert np.all(data.current[~mask] <= RATED_CURRENT)
    assert np.all((data.efficiency.compressed() > 0)
                  & (data.efficiency.compressed() <= 1))


@pytest.mark.parametrize('rpm, torque', [
    ([0., 1000.], [100., 200.]),
    ([1000., 500.], [100., 200.]),
    ([500., np.nan], [100., 200.]),
    ([500., 1000.], [100.]),
])
def test_map_rejects_invalid_axes(rpm, torque):
    with pytest.raises(ValueError, match='finite, positive, increasing'):
        evaluate_map(rpm, torque)


def test_map_rejects_grid_with_no_feasible_points():
    with pytest.raises(ValueError, match='No feasible motoring points'):
        evaluate_map([500., 1000.], [.001, .002])


def test_map_plots_shaft_and_electromagnetic_torque(tmp_path):
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = evaluate_map([500., 1000., 1500.], [100., 300., 500.])
    for axis, label in [('shaft', 'Shaft torque (N m)'),
                        ('em', 'Electromagnetic torque (N m)')]:
        fig, ax = plot_map(data, torque_axis=axis)
        try:
            assert ax.get_xlabel() == 'Motor speed (RPM)'
            assert ax.get_ylabel() == label
            path = tmp_path/f'{axis}.png'
            fig.savefig(path)
            assert path.stat().st_size > 1000
        finally:
            plt.close(fig)

    data = evaluate_map([500., 1500., 5000.], [.01, 100., 10000.])
    with pytest.raises(ValueError, match='No complete feasible contour cells'):
        plot_map(data)
