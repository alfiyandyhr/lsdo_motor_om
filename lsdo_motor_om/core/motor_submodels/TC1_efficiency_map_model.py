"""Original efficiency-map variant (stator-only iron losses, smoothing k=1)."""

from .TC1_implicit_em_torque_model import EMTorqueImplicitModel, EMTorqueModel


class LoadTorqueImplicitModel(EMTorqueImplicitModel):
    def __init__(self, **kwargs):
        kwargs['mode'] = 'efficiency_map'
        kwargs['loss_model'] = 'efficiency_map'
        super().__init__(**kwargs)


class EfficiencyMapModel(EMTorqueModel):
    """Given T_em, solve nonzero load torque and expose efficiency and input power."""

    def __init__(self, **kwargs):
        kwargs['mode'] = 'efficiency_map'
        kwargs['loss_model'] = 'efficiency_map'
        super().__init__(**kwargs)
