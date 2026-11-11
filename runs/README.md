# Retrain runs

One line per run of the full retrain (`common/setup_tag.sh`, then the steps of the top README).

| run | tune point | code of the RECO jobs | started | where | models |
|---|---|---|---|---|---|
| rt3p2 | p2 (`common/harness/p2.py`, tune3 set it000s2) | `ticlv6-modifier` a6d5063e719 + `cmssw/patches`, CMSSW_20_1_X_2026-11-02-2300 | 2026-11-06 | patatrack-bg-01 (T4 GPUs) | the deployed v2 and v3 files in `models/` |
| rt4B | B (`common/harness/B.py`, tune v3 set 013.03) | `ticlv6-modifier` 6d6471d043b + `cmssw/patches`, CMSSW_20_1_X_2026-11-08-0000, CLUEstering 2.14.2 | 2026-11-11 | patatrack-bg-01 and the NGT pod felice-train-1gpu-nvl (H100 NVL) | in production |

Differences of rt4B from rt3p2:

- The probes and the dumps run with the interpretation models of the code (the rt3p2 models, which are the defaults of
  6d6471d043b). In rt3p2 they were M6d.
- The jobs run with half as many streams as threads (`common/job3.sh`).
- The supercluster score cuts of the validation leg keep the fake count of score_v2 at the cuts of B (0.624 below
  10 GeV, 0.771 above), `SC_REF` of `common/rest.sh`.
