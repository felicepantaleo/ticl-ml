"""The tune job: one copy of the HGCAL chain (2D CLUEstering to the final tracksters) per parameter set, on the cached
input. Each copy has its trackster associator and TracksterTuneProbe, in the TFileService directory p<i>Probe.
usage: python3 make_tune_cfg.py <job cfg.py of the sample> <closure.json> <particles.json> <cache files (comma separated)>
       <output cfg> <ntuple file> [threads]; the output cfg runs the job cfg and build() when cmsRun loads it.
particles.json: a list of {"<module>.<parameter>[.<parameter>...]": value or [value, kind]}; the empty dict is the
default chain."""
import json
import runpy
import sys

import FWCore.ParameterSet.Config as cms

CACHED = {"generalTracks", "mtdSoA", "muons1stStep", "ticlSeedingGlobal", "HGCalRecHit", "offlineBeamSpot",
          "offlinePrimaryVertices", "offlinePrimaryVerticesWithBS", "truthLogicalGraphProducer",
          "truthLogicalGraphHitIndexProducer", "truthBranchTargets", "genParticles"}
ALL_STAGES = ["ticlTrackstersCLUEstering", "ticlTracksterLinks", "ticlTracksterInterpretations"]
RECHITS = [cms.InputTag("HGCalRecHit", "HGCEERecHits"), cms.InputTag("HGCalRecHit", "HGCHEFRecHits"),
           cms.InputTag("HGCalRecHit", "HGCHEBRecHits")]


def retarget(pset, labels, prefix):
    """Points every input tag, and every string equal to a module label, of the chain at the copy of the chain."""
    for name in pset.parameterNames_():
        v = getattr(pset, name)
        if isinstance(v, cms.InputTag):
            if v.getModuleLabel() in labels:
                v.setModuleLabel(prefix + v.getModuleLabel())
        elif isinstance(v, cms.VInputTag):
            new = []
            for t in v:
                t = cms.InputTag(t) if isinstance(t, str) else t
                if t.getModuleLabel() in labels:
                    t = cms.InputTag(prefix + t.getModuleLabel(), t.getProductInstanceLabel(), t.getProcessName())
                new.append(t)
            setattr(pset, name, cms.VInputTag(*new))
        elif isinstance(v, cms.string):
            if v.value() in labels:
                setattr(pset, name, cms.string(prefix + v.value()))
        elif isinstance(v, cms.vstring):
            if any(s in labels for s in v):
                setattr(pset, name, cms.vstring(*[prefix + s if s in labels else s for s in v]))
        elif isinstance(v, cms.PSet):
            retarget(v, labels, prefix)
        elif isinstance(v, cms.VPSet):
            for p in v:
                retarget(p, labels, prefix)


KINDS = {"float": cms.float, "int": cms.int32, "uint": cms.uint32, "bool": cms.bool}


def setParameter(module, path, value):
    """value: a number, or [number, kind] (kind of space.py) to also create a parameter left at its plugin default."""
    value, kind = value if isinstance(value, list) and len(value) == 2 and isinstance(value[1], str) else (value, None)
    *parents, leaf = path
    obj = module
    for name in parents:
        obj = getattr(obj, name)
    if not hasattr(obj, leaf):
        setattr(obj, leaf, KINDS[kind](value))
        return
    old = getattr(obj, leaf)
    if isinstance(old, (cms.vdouble, cms.vfloat)):
        setattr(obj, leaf, type(old)(*([value] * len(old))))
    else:
        setattr(obj, leaf, type(old)(value))


def build(process, closure, particles, cache, ntuple, threads):
    chain = json.load(open(closure))["chain"]
    sets = json.load(open(particles))
    labels = set(chain)
    # The stages the probe reads: the trackster collections of the chain, in reconstruction order.
    STAGES = [s for s in ALL_STAGES if s in labels]

    process.setName_("TUNE")
    process.source.fileNames = cms.untracked.vstring(*["file:" + f for f in cache.split(",")])
    process.source.inputCommands = cms.untracked.vstring("keep *")
    process.source.dropDescendantsOfDroppedBranches = cms.untracked.bool(False)
    process.maxEvents.input = -1
    process.options.numberOfThreads = threads
    process.options.numberOfStreams = threads
    for name in list(process.endpaths_()) + list(process.paths_()) + list(process.outputModules_()):
        delattr(process, name)
    for name in ("DQMStore", "dqmSaver"):
        if hasattr(process, name):
            delattr(process, name)
    process.TFileService = cms.Service("TFileService", fileName=cms.string(ntuple))

    # The EventSetup modules of the job: some are in the tasks of the paths removed above.
    task = cms.Task(*[getattr(process, n) for d in (process.es_producers_(), process.es_sources_()) for n in d])
    mods = dict(process.producers_())

    def reads(m):
        out = []

        def walk(x):
            if isinstance(x, cms.InputTag):
                out.append(x.getModuleLabel())
            elif isinstance(x, cms.VInputTag):
                out.extend(t if isinstance(t, str) else t.getModuleLabel() for t in x)
            elif isinstance(x, cms.PSet) or hasattr(x, "parameterNames_"):
                for n in x.parameterNames_():
                    walk(getattr(x, n))
            elif isinstance(x, cms.VPSet):
                for p in x:
                    walk(p)
        walk(m)
        return {t.split(":")[0] for t in out if t}

    probes = []
    for i, params in enumerate(sets):
        prefix = f"p{i}"
        for n in chain:
            m = getattr(process, n).clone()
            retarget(m, labels, prefix)
            setattr(process, prefix + n, m)
            task.add(m)
        for key, value in params.items():
            module, *path = key.split(".")
            setParameter(getattr(process, prefix + module), path, value)
        assoc = process.truthBranchTracksterAssociators.clone(
            recoCollections=cms.VInputTag(*[prefix + s for s in STAGES]),
            layerClusters=cms.InputTag(prefix + "hgcalMergeLayerClusters"))
        # The denominator is HGCAL only: the barrel PF rechits weight no cell of it.
        assoc.pfRecHits = cms.VInputTag()
        setattr(process, prefix + "TracksterAssociators", assoc)
        task.add(assoc)
        probe = cms.EDAnalyzer("TracksterTuneProbe",
                               graph=cms.InputTag("truthLogicalGraphProducer"),
                               hitIndex=cms.InputTag("truthLogicalGraphHitIndexProducer"),
                               targets=cms.InputTag("truthBranchTargets", "truthToRecoTargetsReconstructableFinalState"),
                               hgcalRecHits=cms.VInputTag(*RECHITS),
                               associator=cms.string(prefix + "TracksterAssociators"),
                               tracksters=cms.vstring(*[prefix + s for s in STAGES]),
                               minAbsEta=cms.double(1.5), maxAbsEta=cms.double(3.0), minPt=cms.double(1.0),
                               jetR=cms.double(0.4), minJetPt=cms.double(20.0), minJetAbsEta=cms.double(1.7),
                               maxJetAbsEta=cms.double(2.8))
        setattr(process, prefix + "Probe", probe)
        probes.append(probe)
    # Producers outside the copies that the copies need and that the cache does not hold: run once, shared.
    copies = {f"p{i}{n}" for i in range(len(sets)) for n in chain} | {f"p{i}TracksterAssociators" for i in range(len(sets))}
    shared, todo = set(), set().union(*(reads(getattr(process, n)) for n in copies))
    while todo:
        n = todo.pop()
        if n in copies or n in CACHED or n in shared or n not in mods:
            continue
        shared.add(n)
        todo |= reads(mods[n])
    for n in sorted(shared):
        task.add(getattr(process, n))
    print("shared producers:", sorted(shared))
    seq = probes[0]
    for p in probes[1:]:
        seq = seq + p
    process.tunePath = cms.EndPath(seq, task)
    process.schedule = cms.Schedule(process.tunePath)
    print(f"{len(sets)} parameter sets, {len(chain)} modules each")
    return process


def main():
    cfg, closure, particles, cache, out, ntuple = sys.argv[1:7]
    threads = int(sys.argv[7]) if len(sys.argv) > 7 else 8
    here = __file__.rsplit("/", 1)[0]
    with open(out, "w") as f:
        f.write(f"""import runpy, sys
sys.path.insert(0, {here!r})
from make_tune_cfg import build
process = build(runpy.run_path({cfg!r})["process"], {closure!r}, {particles!r}, {cache!r}, {ntuple!r}, {threads})
""")


if __name__ == "__main__":
    main()
