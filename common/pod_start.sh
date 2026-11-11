#!/bin/bash
# The steps of a retrain in a GPU pod of the NGT cluster, from a kubectl exec shell. The pod must hold a Kerberos ticket
# for /cms-hlt-nfs (a renewer from a keytab). Each step runs detached inside cmssw-el8; logs in <RUNDIR>/<TAG>/pod/<pod>/.
#   area:         unpacks <RUNDIR>/area.tgz (the CMSSW area of the run) into <SCRATCH>/area, ProjectRename, plugin check
#   pools <STEP>: one start_pool.sh STEP=probe or dumps per lane of LANES ("<first cpu>:<workers>:<cpus per worker> ...")
#   mix:          showerShapePID/train_mix.sh (GPU)
#   train:        interpretation/start_train.sh (GPU)
# usage: RUNDIR=<run folder> TAG=<tag> SCRATCH=<pod local dir> [LANES=...] [COPYSLOTS=4] pod_start.sh area | pools <step> | mix | train
X=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
W=${RUNDIR:?RUNDIR}
T=${TAG:?TAG}
S=${SCRATCH:?SCRATCH}
L=$W/$T/pod/$(hostname -s)
export KRB5CCNAME=${KRB5CCNAME:-FILE:/tmp/krb5cc_$(id -u)}
klist -s || { echo "no Kerberos ticket"; exit 1; }
case "$(stat -f -c %T /cms-hlt-nfs 2>/dev/null)" in nfs*) ;; *) echo "cms-hlt-nfs is not NFS"; exit 1 ;; esac
[ -s $W/$T/point.env ] || { echo "no $W/$T/point.env"; exit 1; }
mkdir -p $S $L
EL8="/cvmfs/cms.cern.ch/common/cmssw-el8 -B /shared -B /cms-hlt-nfs -B /scratch"
area() { ls -d $S/area/CMSSW_* 2>/dev/null | head -1; }
cd /scratch
case "$1" in
  area)
    rm -rf $S/area && mkdir -p $S/area && tar -C $S/area -xzf $W/area.tgz || { echo "AREA_FAIL untar"; exit 1; }
    A=$(area)
    cat > $S/area_check.sh <<EOF
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd $A/src && scram b ProjectRename > $L/rename.log 2>&1 && eval "\$(SCRAM_IGNORE_HOOKS=1 scram runtime -sh)" &&
  edmPluginHelp -p HypothesisTruthDumper > /dev/null && edmPluginHelp -p StageTruthProbe > /dev/null && echo AREA_OK $A
EOF
    $EL8 --command-to-run bash $S/area_check.sh 2>&1 | tail -1 | tee $L/area.check ;;
  pools)
    A=$(area); [ -n "$A" ] || { echo "no area in $S/area"; exit 1; }
    mkdir -p $S/work_$2
    for lane in ${LANES:?LANES}; do
      IFS=: read -r lo nw per <<< "$lane"
      setsid nohup $EL8 --command-to-run env RUNDIR=$W TAG=$T STEP=$2 AREA=$A WORK=$S/work_$2 COPYSLOTS=${COPYSLOTS:-4} \
        bash $X/common/start_pool.sh $lo $nw $per > $L/pool_$2_$lo.log 2>&1 < /dev/null &
      echo "pool $2 at $lo: PID $!"; sleep 20
    done ;;
  mix) setsid nohup $EL8 --nv --command-to-run env RUNDIR=$W TAG=$T bash $X/showerShapePID/train_mix.sh > $L/mix.log 2>&1 < /dev/null &
       echo "mix: PID $!" ;;
  train) setsid nohup $EL8 --nv --command-to-run env RUNDIR=$W TAG=$T bash $X/interpretation/start_train.sh > $L/train.log 2>&1 < /dev/null &
       echo "train: PID $!" ;;
  *) echo "usage: RUNDIR=... TAG=... SCRATCH=... pod_start.sh area | pools <probe|dumps> | mix | train"; exit 2 ;;
esac
