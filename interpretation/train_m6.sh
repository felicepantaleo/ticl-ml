#!/bin/bash
# M6d-recipe training of the retrain TAG (H9 scripts; outputs under <RUNDIR>/<TAG>) in felice-1gpu-nvl
# (1 H100 NVL) inside cmssw-el8 --nv on /cms-hlt-nfs, on the finished dump folders listed in CHAINDIRS (lines "<sample> <folder>").
# PART=hyp: tables and pairs (TenTau + e/gamma), the four hypothesis seeds on the GPU, their ensemble and the species
# model. PART=share: share tables (TenTau, QCD, high-pT QCD) and the share model. Markers <TAG>/train/<part>.done or
# <part>.failed; both parts done: <TAG>/train/train.done.
X=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
W=${RUNDIR:?RUNDIR}
S=$X/interpretation/hypothesis
TAG=${TAG:?TAG}
D=$W/$TAG
M=$D/dumps
T=$D/train/tables
O=$D/models/interp
part=${PART:?PART=hyp or PART=share}
fail() { echo "$1" > $D/train/$part.failed; exit 1; }
step() { echo "$(date '+%F %T') $*"; }
rm -f $D/train/$part.done $D/train/$part.failed $D/train/train.done
case "$(stat -f -c %T /cms-hlt-nfs)" in nfs*) ;; *) echo "cms-hlt-nfs is not NFS"; exit 1 ;; esac
cd ${TRAIN_ENV:-/shared/fpantale/retrain_cs/CMSSW_20_1_0_pre4}/src && source /cvmfs/cms.cern.ch/cmsset_default.sh && eval "$(SCRAM_IGNORE_HOOKS=1 scram runtime -sh)" || fail cmsenv
mkdir -p $T/tt $T/eg $T/all $O $D/train/share_tables
cd $M
[ -s "${CHAINDIRS:-}" ] || fail "no chain list in CHAINDIRS"
export CHAINDIRS
for s in tt eg qcd qcdhi; do step "$s chains in the list: $(awk -v s=$s '$1 == s' $CHAINDIRS | wc -l)"; done
if [ $part = share ]; then
  step "share tables"
  M6ROOT=$M SHAREOUT=$D/train/share_tables python3 $X/interpretation/share/prepare_share_m6.py > $D/train/share_tables.log 2>&1 || fail "share tables"
  step "share model"
  SHARETABLES=$D/train/share_tables OMP_NUM_THREADS=16 python3 $X/interpretation/share/train_share_m6.py $O/neutralShare_mlp_m6.onnx > $O/share.log 2>&1 || fail share
else
  step "tables"
  ( DUMPROOT=$M python3 $S/prepare_c2.py tt $T/tt > $T/tt/prepare.log 2>&1 && \
    NPROC=8 python3 $S/prepare_pairs.py $M/tt $T/tt > $T/tt/pairs.log 2>&1 ) || fail "tables tt" &
  ( DUMPROOT=$M python3 $S/prepare_c2.py eg $T/eg > $T/eg/prepare.log 2>&1 && \
    NPROC=4 python3 $S/prepare_pairs.py $M/eg $T/eg > $T/eg/pairs.log 2>&1 ) || fail "tables eg" &
  wait
  [ -f $D/train/$part.failed ] && exit 1
  step "combine and conflict pairs"
  python3 $S/combine_tables.py $T/all $T/tt $T/eg > $T/all/combine.log 2>&1 || fail combine
  NPROC=12 python3 $S/prepare_cpairs.py $T/all $M/tt $M/eg > $T/all/cpairs.log 2>&1 || fail cpairs
  # hypothesis seeds one after the other on the GPU; a seed that fails on the GPU is retried with the rows in host memory
  seed() {
    local s=$1 tag=fc_both03g_s$1
    for dd in cuda cpu; do
      env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 DATADEV=$dd SEED=$s PLATT=global PAIRSEL=all PAIRW=0.3 \
        PAIRFILE=pairs_both python3 $S/train_d2.py $T/all $tag > $T/all/arb_$tag.$dd.log 2>&1 && return 0
    done
    return 1
  }
  step "hypothesis seeds"
  seed 1 && seed 2 && seed 3 && seed 4 || fail "hypothesis seeds"
  step "ensemble and species model"
  python3 $S/ens_export.py $T/all $O/hypothesis_mlp_m6.onnx 0 2=0.5 fc_both03g_s1 fc_both03g_s2 fc_both03g_s3 \
    fc_both03g_s4 > $O/ensemble.log 2>&1 || fail ensemble
  CUDA_VISIBLE_DEVICES=0 python3 $X/interpretation/species/train_species_m6.py $T/all $O/neutralSpecies_mlp_m6.onnx > $O/species.log 2>&1 \
    || fail species
fi
step "done $part"
echo "$part done $(date '+%F %T')" > $D/train/$part.done
[ -f $D/train/hyp.done ] && [ -f $D/train/share.done ] && echo "models $(ls $O/*.onnx | xargs -n1 basename | paste -sd' ')" > $D/train/train.done
exit 0
