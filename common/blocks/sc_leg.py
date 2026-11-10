# Training leg of the TICL e/gamma supercluster models for t7 (rt7): no supercluster score cut, no type score and no
# photon retype (as the basep2 and basepx legs of weekend/egamma); the rt7 models; the e/gamma truth probe on e/gamma
# and TenTau files (probe_v6x); the t7 parameters last.
process.ticlEGammaSuperClusters.scoreModel = ""
process.ticlEGammaSuperClusters.typeScoreModel = ""
process.ticlCandidate.photonMinScore = 0.
exec(open("/cms-hlt-nfs/user/fpantale/ticlv6tune/rt7/blocks/models_rt7.py").read())
if "/eg/" in process.source.fileNames[0] or "/tt/" in process.source.fileNames[0]:
    from EGStudy.Probe.customiseEGProbe import customise as _egProbe
    _egProbe(process, v6=True)
