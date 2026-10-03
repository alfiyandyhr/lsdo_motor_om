"""TC1 motor geometry, winding resistance, mass, and structural torque."""

import numpy as np
import openmdao.api as om
from ._utils import declare_motor_options, scalar


class TorqueMassModel(om.ExplicitComponent):
    """Original torque-versus-mass polynomial (linear by default)."""

    fitting_coeff = {
        0: [1.], 1: [26.0489, -112.1432], 2: [.4840, 3.3169, 60.8142],
        3: [1., 1., 1., 1.], 4: [1., 1., 1., 1., 1.],
    }

    def initialize(self):
        self.options.declare('fitting_order', default=1, values=list(self.fitting_coeff))

    def setup(self):
        self.add_input('motor_mass')
        self.add_output('T_em_max')
        self.add_output('torque_fitting_array', shape=self.options['fitting_order']+1)
        self.declare_partials('*', 'motor_mass', method='cs')

    def compute(self, inputs, outputs):
        coeff = self.fitting_coeff[self.options['fitting_order']]
        outputs['torque_fitting_array'] = [c*scalar(inputs['motor_mass'])**i
                                          for i, c in zip(range(len(coeff)-1, -1, -1), coeff)]
        outputs['T_em_max'] = np.sum(outputs['torque_fitting_array'])


class MotorGeometryModel(om.ExplicitComponent):
    """Explicit sizing equations, with the source's 25-entry geometry vector."""

    def initialize(self):
        declare_motor_options(self)

    def setup(self):
        self.add_input('D_i', val=.182)
        self.add_input('L', val=.086)
        self.add_input('N_p', val=2.)
        self.add_output('Rdc')
        self.add_output('motor_mass')
        self.add_output('motor_variables', shape=25)
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        m, p, z = (self.options[n] for n in ('phases', 'pole_pairs', 'num_slots'))
        iw = self.options['rated_current']
        d, length, np_ = (scalar(inputs[n]) for n in ('D_i', 'L', 'N_p'))
        if np.real(d) <= 0 or np.real(length) <= 0 or np.real(np_) <= 0:
            raise om.AnalysisError('D_i, L, and N_p must be positive.')
        mu0 = 4e-7*np.pi
        q = z/(2*m*p)
        outer = 1.25*d
        pole = np.pi*d/(2*p)
        tooth = np.pi*d/z
        gap = .4*30000*pole/(.9e6*.85)
        lef = length+2*gap
        rotor = d-2*gap
        shaft = .3*rotor
        conductors_phase = np.pi*d*30000/(m*iw)
        conductors_slot = m*conductors_phase/z
        turns = conductors_phase/2
        acu = iw/(5*np_)*1e-6
        tooth_width = tooth*.85/(.95*1.7)
        hys = pole*.7*.85/(2*.95*1.35)
        bsb = .15*(360/z)*np.pi*d/360
        hslot = (outer-d)/2-hys
        hk, hos = .0008, .0012
        # The original sizing equations use 36 in both slot-width expressions.
        bs1 = np.pi*(d+2*(hos+hk))/36-tooth_width
        bs2 = np.pi*(d+2*hslot)/36-tooth_width
        tauy = np.pi*(d+hslot)/(2*p)
        lj1 = np.pi*(outer-hys)/(4*p)
        slot_area = (bs1+bs2)*(hslot-hk-hos)/2
        alpha = 360*p/z
        kdp1 = np.sin(q*alpha/2)/(q*np.sin(alpha/2))
        hm = .004
        bm = (rotor-.002)*np.pi*(.78*360/(2*p))/360
        hc = (1+(75-20)*(-.12)/100)*907000
        amr = bm*lef
        mass_magnet = 2*p*bm*hm*lef*7.6e3
        phir = 1.2*amr
        lambda_m = phir/(2*hc*hm)
        alpha_p1 = bm/pole
        alpha_i = alpha_p1+4/(pole/gap+6/(1-alpha_p1))
        kf = 4*np.sin(alpha_i*np.pi/2)/np.pi
        kphi = 8.5*np.sin(alpha_i*np.pi/2)/(np.pi**2*alpha_i)
        ktheta = tooth*(4.4*gap+.75*bsb)/(tooth*(4.4*gap+.75*bsb)-bsb**2)
        af2 = hm*lef
        coil = lef+.02+2*pole
        outputs['Rdc'] = 2*.0217e-6*turns*coil/(acu*np_)
        mass_cu = 1.05*coil*conductors_slot*z*acu*8.9e3
        deficit_slot = slot_area*lef*7.8e3*z
        deficit_mag = 2*p*bm*hm*lef*7.8e3
        outputs['motor_mass'] = (mass_cu-deficit_slot+mass_magnet-deficit_mag
                                +np.pi*lef*7.8e3*((outer/2)**2-(shaft/2)**2))
        values = [outer, pole, tooth, gap, lef, rotor, turns, acu, tooth_width,
                  hys, bsb, hslot, bs1, tauy, lj1, kdp1, bm, amr, phir,
                  lambda_m, alpha_i, kf, kphi, ktheta, af2]
        outputs['motor_variables'] = values


class TC1MotorSizingModel(om.Group):
    """Size a motor from stator bore diameter ``D_i`` and stack length ``L`` (m)."""

    def initialize(self):
        declare_motor_options(self)
        self.options.declare('fitting_order', default=1, values=list(TorqueMassModel.fitting_coeff))

    def setup(self):
        names = ('pole_pairs', 'phases', 'num_slots', 'rated_current')
        self.add_subsystem('geometry', MotorGeometryModel(**{n: self.options[n] for n in names}),
                           promotes_inputs=['*'],
                           promotes_outputs=['Rdc', 'motor_mass', 'motor_variables'])
        self.add_subsystem('max_torque_model', TorqueMassModel(fitting_order=self.options['fitting_order']),
                           promotes=['*'])
