"""Study customise, not for integration: the e/gamma truth probe on a RECO job with the truth validation.

customise(process, v6) adds egProbe (EGStudyTruthProbe) to truthBranchValidationPath. For TICLv6 it also adds
ticlCandidateAllShare, a copy of ticlCandidate that applies the neutral share model at every pT, so the probe
stores the energy of each neutral candidate with and without the share (x[11] of the candidate rows).
"""

import FWCore.ParameterSet.Config as cms

_REC_HITS = cms.VInputTag(cms.InputTag("HGCalRecHit", "HGCEERecHits"), cms.InputTag("HGCalRecHit", "HGCHEFRecHits"),
                          cms.InputTag("HGCalRecHit", "HGCHEBRecHits"))


def customise(process, v6=True, minObjectPt=2.0):
    alt = ""
    if v6:
        process.ticlCandidateAllShare = process.ticlCandidate.clone(neutralShareMinPt=-1.0)
        process.truthBranchValidationPath += process.ticlCandidateAllShare
        alt = "ticlCandidateAllShare"
    process.egProbe = cms.EDAnalyzer("EGStudyTruthProbe",
        layerClusters=cms.InputTag("hgcalMergeLayerClusters"),
        graph=cms.InputTag("truthLogicalGraphProducer"),
        hitIndex=cms.InputTag("truthLogicalGraphHitIndexProducer"),
        genParticles=cms.InputTag("genParticles"),
        tracks=cms.InputTag("generalTracks"),
        recHits=_REC_HITS,
        layerTiles=cms.InputTag("ticlLayerTileProducer"),
        hitMap=cms.InputTag("recHitMapProducer", "hgcalRecHitMap"),
        hits=cms.InputTag("recHitMapProducer", "RefProdVectorHGCRecHitCollection"),
        superClusters=cms.VInputTag(["ticlEGammaSuperClusters", "ticlElectronSuperClusters"] if v6
                                    else ["particleFlowSuperClusterHGCal"]),
        photons=cms.InputTag("photonsHGC"),
        electrons=cms.InputTag("ecalDrivenGsfElectronsHGC"),
        candidates=cms.InputTag("ticlCandidate"),
        candidatesAlt=cms.InputTag(alt),
        eGammaTracksters=cms.InputTag("ticlEGammaTracksters" if v6 else ""),
        interpretations=cms.string("ticlTracksterInterpretations"),
        minObjectPt=cms.double(minObjectPt))
    process.truthBranchValidationPath += process.egProbe
    return process
