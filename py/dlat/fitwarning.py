"""
Mask bit definitions for dla fit warnings
"""

class DLAFLAG(object):
    ZBOUNDARY = 2**0 # refined DLA solve relaxed to z window boundary
    NHIBOUNDARY = 2**1 # refined DLA solve relaxed to nhi window boundary
    POTENTIAL_BAL = 2**2 # DLA solution overlaps with Lya or NV BAL, potential false positive
    BAD_ZFIT = 2**3 # bad parabola fit to chi2(refined z) surface
    BAD_NHIFIT = 2**4 # bad parabola fit to chi2(refined nhi) surface
