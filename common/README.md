# Common scripts of a retrain

These are the scripts of run rt3p2, without changes. `job3.sh` and `rest.sh` are the versions that ran rt3p2. Later
versions changed only the number of streams and of slots.

## One run

A run is one tune point (`POINT`) and one tag (`TAG`). Run rt3p2 is `POINT=p2 TAG=rt3p2`.

| script | what it does |
|---|---|
| `setup_tag.sh` | makes the blocks (`make_blocks.sh`), the folders of `<TAG>` and the queues: probe (96), dumps (1414), supercluster legs (148). No queue holds a validation chain. |
| `make_blocks.sh` | writes `blocks/{probe,m6,m6s,sc_leg}_<POINT>.py`, `models_<TAG>.py` and `val_<POINT>rt.py` from `blocks/templates` and `harness/<POINT>.py` |
| `start_pool.sh` | `STEP=probe` or `STEP=dumps`: one pool of `pool3.sh` workers on the queue of the step. `dumps` waits for the mixture PIDs and installs them in the CMSSW area. |
| `pool3.sh` | workers that claim queue lines with an atomic `mkdir`; a `STOP` file next to the queue stops new claims |
| `job3.sh` | one chain: copies the DIGI to local disk, writes the RECO job (`cfg/reco_v6.py`, the chain's DIGI, the truth preset of TenTau, the block), runs 50 events, writes `done` or `failed` |
| `install_models.sh` | copies the interpretation models of a run into the CMSSW area under the names of `blocks/models_<TAG>.py` |
| `rest.sh` | after the interpretation training: installs the models, runs the 148 supercluster legs, trains the score and type models, writes `blocks/models_sc_<TAG>.py`, runs the validation leg |
| `ticl_leg.sh` | one RECO leg of a chain (supercluster legs and validation): the leg's area, template and parameter block from `legs.conf` |

## Blocks

python appended to the RECO job:

| block | step |
|---|---|
| `blocks/probe.py`, `probe_p2.py` | probe chains of the mixture PID |
| `blocks/m6s.py`, `m6s_p2.py` | dumps of the hypothesis, species and share rows (one pass) |
| `blocks/m6.py` | the same without the share rows (not used in rt3p2) |
| `blocks/sc_leg.py`, `sc_leg_p2.py` | supercluster legs |
| `blocks/models_rt3p2.py`, `models_sc_rt3p2.py` | the models of run rt3p2 in the job |

The `_p2` blocks are the templates with the p2 parameters (`harness/p2.py`, which uses `harness/make_tune_cfg.py`).

## Configuration templates

- `cfg/reco_v6.py`: cmsDriver step3 `RAW2DIGI,RECO`, era Phase2C26I13M9, geometry ExtendedRun4D127,
  `--procModifiers ticl_v6`.
- `cfg/digi_pu200.py`: a PU200 DIGI template of the same release series (from the VBF production), for reference.
  The DIGI files of the chains come from the productions c1, c2 and c3.
