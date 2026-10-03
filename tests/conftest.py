import os

# OpenMDAO may import mpi4py even for a serial problem in this environment.
os.environ.setdefault('FI_PROVIDER', 'tcp')
os.environ.setdefault('OPENMDAO_REPORTS', '0')
