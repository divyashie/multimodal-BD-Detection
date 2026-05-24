# Outline: Methods paper

1. Introduction
- Argument overview: rising use of cross-corpus multimodal mental-health ML and the risk of construct mismatch.
- Brief statement of contributions.

2. Background: validation conventions in multimodal mental-health ML
- Describe common checks: label shuffling, noise injection, train/test splits, calibration (ECE).
- Survey of prior work and limitations (brief references).

3. Worked example: the multimodal-BD-Detection pipeline
- Describe the legacy pipeline (point to `legacy/`).
- Explain the GMM-based alignment and fusion model.

4. Standard validation results
- Present the original validation numbers (accuracy, ECE, shuffle test).

5. Diagnostic experiments demonstrating the failure
- Out-of-population validation.
- Construct-alignment audit: vocabulary overlap, label-space mapping diagnostics.
- Label-provenance disclosure: what labels mean and where they came from.

6. Discussion: a reporting checklist
- Practical checklist authors should follow when fusing public corpora for mental-health claims.

7. Limitations

8. Conclusion

Key references (verify before final draft):
- Schmidt et al. 2018 (WESAD)
- Ouzar et al. 2025 (CALYPSO)
- Langholm et al. 2023 (mindLAMP)
- Wang et al. 2014 (StudentLife)
- Chancellor & De Choudhury 2020
