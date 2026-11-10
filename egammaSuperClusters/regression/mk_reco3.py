#!/usr/bin/env python3
"""Write the RECO job config of one leg of study 3 from the RECO templates of the earlier studies.
usage: mk_reco3.py <v5|v6> <step2.root> <out dir> <threads> <streams> [--main KEY=VALUE ...]
         [--clone NAME:KEY=VALUE;KEY=VALUE ...] [--dump LABEL ...] [--timing N] [--events N]
--main sets parameters of ticlEGammaSuperClusters (the supercluster that the photons use).
--clone adds ticlEGammaSuperClusters<NAME>, a clone of the main module with the parameters given.
--dump adds EGammaSCFeatureDumper on the supercluster module LABEL (output dump.root, tree <LABEL>/sc).
"""
import argparse
import os

TEMPL = {"v5": "/data/fpantale/vbf/v3dev/cfg/recoE200_v5_a/cfg.py",
         "v6": "/data/fpantale/vbf/v3dev/cfg/recoE200_v6fix_a/cfg.py"}
p = argparse.ArgumentParser()
p.add_argument("leg")
p.add_argument("fin")
p.add_argument("out")
p.add_argument("nthr", type=int)
p.add_argument("nstr", type=int)
p.add_argument("--main", nargs="*", default=[])
p.add_argument("--clone", nargs="*", default=[])
p.add_argument("--dump", nargs="*", default=[])
p.add_argument("--timing", type=int, default=0)
p.add_argument("--events", type=int, default=-1)
p.add_argument("--no-ele", action="store_true")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
src = open(TEMPL[a.leg]).read()
head = src[:src.index("# Job block")]
b = f'''# Job block: study 3 (TICL e/gamma supercluster composition and regression), leg {a.leg}.
process.source.fileNames = cms.untracked.vstring("file:{a.fin}")
process.maxEvents.input = {a.events}
process.options.numberOfThreads = {a.nthr}
process.options.numberOfStreams = {a.nstr}
process.options.wantSummary = True
process.schedule.remove(process.AODSIMoutput_step)
del process.AODSIMoutput_step, process.AODSIMoutput
process.schedule.remove(process.DQMoutput_step)
del process.DQMoutput_step, process.DQMoutput
process.abOut = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string("file:{a.out}/ab.root"),
    outputCommands = cms.untracked.vstring("drop *",
        "keep recoGenParticles_genParticles_*_*",
        "keep recoVertexs_offlinePrimaryVertices_*_*",
        "keep recoSuperClusters_particleFlowSuperClusterHGCal_*_*",
        "keep recoCaloClusters_particleFlowSuperClusterHGCal_*_*",
        "keep *_ticlEGammaSuperClusters*_*_*",
        "keep *_ticlEGammaTracksters_*_*",
        "keep recoPhotons_photonsHGC_*_*",
        "keep recoGsfElectrons_ecalDrivenGsfElectronsHGC_*_*"))
process.abOut_step = cms.EndPath(process.abOut)
process.schedule.append(process.abOut_step)
'''
for kv in a.main:
    k, v = kv.split("=", 1)
    b += f"process.ticlEGammaSuperClusters.{k} = {v}\n"
if a.clone:
    b += "process.sc3Task = cms.Task()\n"
    for spec in a.clone:
        name, pars = spec.split(":", 1)
        args = ", ".join(x.strip() for x in pars.split(";") if x.strip())
        b += f"process.ticlEGammaSuperClusters{name} = process.ticlEGammaSuperClusters.clone({args})\n"
        b += f"process.sc3Task.add(process.ticlEGammaSuperClusters{name})\n"
    b += "process.abOut_step.associate(process.sc3Task)\n"
if a.dump:
    b += f'process.TFileService = cms.Service("TFileService", fileName = cms.string("{a.out}/dump.root"))\n'
    b += "process.dumpSeq = cms.Sequence()\n"
    for lab in a.dump:
        b += (f"process.dump{lab} = cms.EDAnalyzer(\"EGammaSCFeatureDumper\", superClusters = cms.InputTag(\"{lab}\"),"
              " eGammaTracksters = cms.InputTag(\"ticlEGammaTracksters\"), genParticles = cms.InputTag(\"genParticles\"),"
              " hitMap = cms.InputTag(\"recHitMapProducer\", \"hgcalRecHitMap\"),"
              " hits = cms.InputTag(\"recHitMapProducer\", \"RefProdVectorHGCRecHitCollection\"))\n")
        b += f"process.dumpSeq += process.dump{lab}\n"
    b += "process.dump_step = cms.EndPath(process.dumpSeq)\nprocess.schedule.append(process.dump_step)\n"
if a.timing:
    b += f'''# Timing: FastTimerService JSON summary.
process.load("HLTrigger.Timer.FastTimerService_cfi")
process.FastTimerService.writeJSONSummary = True
process.FastTimerService.jsonFileName = "{a.out}/resources.json"
process.maxEvents.input = {a.timing}
'''
if a.no_ele:
    b = b.replace(''',
        "keep recoGsfElectrons_ecalDrivenGsfElectronsHGC_*_*"))''', "))")
open(os.path.join(a.out, "cfg.py"), "w").write(head + b)
print(f"written {a.out}/cfg.py")
