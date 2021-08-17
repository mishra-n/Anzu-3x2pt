import matplotlib.pyplot as plt
import numpy as np
import os
import time

import camb
from camb import model, initialpower
print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))

import anzu
import scipy.integrate as integrate
import scipy.interpolate as interp
import scipy.misc as misc
import scipy.stats as stats
import scipy.special as special

from anzu.emu_funcs import LPTEmulator
import twopoint.twopoint as twopoint

import os
os.environ['PATH'] = '/global/common/sw/cray/sles15/x86_64/texlive/live/gcc/8.2.0/tiozj27/bin/x86_64-linux/:{}'.format(os.environ['PATH'])

from matplotlib import rc
rc('font',**{'size':'20','family':'serif','serif':['CMU serif']})
rc('mathtext', **{'fontset':'cm'})
rc('text', usetex=True)
rc('legend',**{'fontsize':'13'})


import warnings
warnings.filterwarnings('ignore')

import time

from astropy.io import fits
