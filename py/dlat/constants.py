"""
set constant values used in DLA finder
"""

from scipy.constants import speed_of_light

# set redshift window for quasars
zmin_qso = 2.0
zmax_qso = 4.25

# set the wave window for DLA search
# rest-frame angstroms
search_minlam = 912.
search_maxlam = 1216.

# set the wave windows for SNR computation
redsnr_min = 1420
redsnr_max = 1480
bluesnr_min = 1040
bluesnr_max = 1205

# set the log10 column density search range
# for DLAs, units of (cm^-2)
nhimin = 20.1
nhimax = 22.6

# set delta chi2 threshold for detection
detection = 0.01

# set constants for DLA profile and BAL masking
c = speed_of_light/1000. # m/s -> km/s
Lya_line = float(1215.67) ## Lya wavelength [A]
Lyb_line = float(1025.72) ## Lyb wavelength [A]
Lyinf = float(911.75) # lyman limit [A]
oscillator_strength_Lya = 0.41641
oscillator_strength_Lyb = 0.079142
gamma_Lya = 6.2648e08  # s^-1 damping constant
gamma_Lyb = 1.6725e8  # s^-1 damping constant
gastemp = 5 * 1e4  # K


# constants for masking broad absorption lines 
# line centers identical to those defined in igmhub/picca
bal_lines={
    "CIV" : 1549.,
    "SiIV2" : 1403.,
    "SiIV1" : 1394.,
    "NV" : 1240.81,
    "Lya" : 1216.1,
    "CIII(1175)" : 1175.,
    "PV2" : 1128.,
    "PV1" : 1117.,
    "SIV2" : 1074.,
    "SIV1" : 1062.,
    "OIV" : 1031.,
    "OVI" : 1037.,
    "OI" : 1039.,
    "Lyb" : 1025.7,
    "Ly3" : 972.5,
    "CIII(977)" : 977.0,
    "NIII" : 989.9,
    "Ly4" : 949.7
    }

Lyman_series = dict()

# optical depth parameters from Kamble et al. (2020)
# used by QSO-HIZv1.1, N>2 are neglected
# arxiv 1904.01110
Lyman_series['kamble20'] =  {
    'Lya'     : { 'line':Lya_line, 'A':0.00554, 'B':3.182 },
    }

# optical depth parameters from Turner et al. (2024)
# arxiv 2405.06743
Lyman_series['turner24'] = {
    'Lya'     : { 'line':Lya_line, 'A':0.00246, 'B':3.62 },
    'Lyb'     : { 'line':Lyb_line, 'A':0.00246*(Lyb_line/Lya_line)/5.2615,   'B':3.62 },
    }
