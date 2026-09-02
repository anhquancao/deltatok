# pair_sampling — which frame pairs and gaps the tokenizer trains on

Timestep selection, gap range (`max_gap`), and the 1ecb17c switch to consecutive windows.

| Date | File | Stage | Question | Verdict |
|---|---|---|---|---|
| 2026-05-31 | [augmentation_parity_plan](2026-05-31_impl_augmentation_parity.md) | solution | Random per-pair stride + horizontal flip, as upstream DeltaTok | `reverse_seq` arm ran 2026-08-04 |
| 2026-08-02 | [timestep_sampling_1ecb17c](2026-08-02_analysis_timestep_sampling_1ecb17c.html) | analysis | What an item contains before and after 1ecb17c | Post-change eval is easier (all gaps one slot); eval loss is not comparable across the boundary |
| 2026-08-05 | [randint_vs_consec_convergence_slides](2026-08-05_report_randint_vs_consec_convergence_slides.html) | results + findings | Random-interval vs consecutive frames | Random-interval converges ~3× faster |
| 2026-08-10 | [maxgap_sweep_report](2026-08-10_report_maxgap_sweep.html) | results | Loss, pointmap AR and z-spread vs `max_gap` | maxgap9 adopted for all later arms |

**Open.** None.
