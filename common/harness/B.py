# Tune point B: set 013.03 of tune v3 run tune2. It changes 20 parameters of the TICLv6 defaults of ticlv6-modifier
# 6d6471d043b (the p2 parameters are the defaults there), so it needs that code.
def _tv3Set(obj, leaf, value, kind):
    if not hasattr(obj, leaf):
        setattr(obj, leaf, dict(float=cms.float, double=cms.double, int=cms.int32, uint=cms.uint32, bool=cms.bool)[kind](value))
        return
    old = getattr(obj, leaf)
    if isinstance(old, (cms.vdouble, cms.vfloat)):
        setattr(obj, leaf, type(old)(*([value] * len(old))))
    else:
        setattr(obj, leaf, type(old)(value))

_tv3Set(process.ticlCLUEstering, "deltac", 1.0721262539531558, "float")
_tv3Set(process.ticlCLUEstering, "rhoc", 0.25735168812970755, "float")
_tv3Set(process.ticlCLUEstering, "outlierDistance", 2.4900035277562167, "float")
_tv3Set(process.ticlCLUEstering, "seedingDistance", 2.0768484566921432, "float")
_tv3Set(process.ticlCLUEstering, "layerScale", 6.340798824384827, "float")
_tv3Set(process.ticlCLUEstering, "rhocEtaExponent", 3.018023257896621, "float")
_tv3Set(process.ticlCLUEstering, "rhocPivotRadius", 0.5485607856056766, "float")
_tv3Set(process.ticlTracksterLinks.linkingPSet, "maxLongitudinalDistance", 80.58417737153393, "float")
_tv3Set(process.ticlTracksterLinks.linkingPSet, "maxBackwardDistance", 18.35276603442385, "float")
_tv3Set(process.ticlTracksterLinks.linkingPSet, "emTransverseRadius0", 1.827717230531924, "float")
_tv3Set(process.ticlTracksterLinks.linkingPSet, "emTransverseSlope", 0.011794695508979148, "float")
_tv3Set(process.ticlTracksterLinks.linkingPSet, "transverseRadius0", 2.683344736115218, "float")
_tv3Set(process.ticlTracksterLinks.linkingPSet, "transverseSlope", 0.01649422975450933, "float")
_tv3Set(process.ticlTracksterLinks.linkingPSet, "minEmittedEnergy", 1.3477372146498938, "float")
_tv3Set(process.ticlTracksterLinks.linkingPSet, "minEmittedPt", 0.817877494041699, "float")
_tv3Set(process.ticlTracksterLinks.linkingPSet, "minEmittedEnergyAnyPt", 0.5779264476640262, "float")
_tv3Set(process.ticlTracksterLinks.linkingPSet, "mergeFragments", False, "bool")
_tv3Set(process.ticlTracksterLinks.linkingPSet, "mergeDropped", False, "bool")
_tv3Set(process.ticlEGammaSuperClusters, "minScore", 0.6239226981185718, "double")
_tv3Set(process.ticlEGammaSuperClusters, "minScoreHighEt", 0.7708272767847733, "double")
