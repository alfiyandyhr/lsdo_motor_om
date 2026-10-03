"""LSDO Motor TC1 models implemented in pure OpenMDAO."""

__version__ = '0.1.0'

from .core.TC1_motor_sizing_model import TC1MotorSizingModel, TorqueMassModel
from .core.TC1_motor_analysis_model import TC1MotorAnalysisModel, ParseActiveOperatingConditions
from .core.TC1_motor_model import TC1MotorModel
from .core.permeability.mu_fitting import permeability_fitting

__all__ = ['TC1MotorSizingModel', 'TC1MotorAnalysisModel', 'TC1MotorModel',
           'TorqueMassModel', 'ParseActiveOperatingConditions', 'permeability_fitting']
