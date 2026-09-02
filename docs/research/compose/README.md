# compose — additive composition of delta tokens

Train `d13 + d35 ≈ d15` so a span is composed in latent space instead of chained through the decoder. The Q1 cycle
(plain vs compose at equal budget) lives in `../tc_width/2026-08-26_task_compose_convergence_sigreg_tc.md`.

| Date | File | Stage | Question | Verdict |
|---|---|---|---|---|
| 2026-08-12 | [additive_composition_plan](2026-08-12_impl_additive_composition.md) | solution | Compose a span as plain addition of hop deltas | Landed 2026-08-13 as the compose arm |
| 2026-08-18 | [token_composition_impact](2026-08-18_report_token_composition_impact.html) | results | What composition does to the code | Deck |
| 2026-08-25 | [compose_vs_plain_slides](2026-08-25_report_compose_vs_plain_slides.html) | results + findings | Compose vs plain at equal budget, ep 99 | Compose loses single-step recon +3% but wins LossRecon_Comp 3.4×; plain regresses on composability after ep 11 |

**Open.** Composed ≠ autoregressive: compose is worse on LossRecon_AR while owning LossRecon_Comp. A short-schedule
plain tokenizer may buy most of the composability for free; untested.
