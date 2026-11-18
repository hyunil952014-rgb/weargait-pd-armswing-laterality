# Amendment 002 — Conditional validity of wrist arm-swing laterality (post-result exploratory)

Written: 7 October 2026, before these analyses were run. Sections 7–9 were appended later on the dates given.

> Translated from the Korean amendment; internal workflow annotations were omitted. The original file is available from the authors on request.

## 0. Information already known when this amendment was written (transparency)

- The locked primary result (SP 65.7%, 44/67, from the run on the initially downloaded files) and all results of amendment 001.
- During manuscript review on 7 October 2026, the following had been checked:
  - Side-specific agreement in the primary cohort: clinically left-affected 22/40 (55%), right-affected 22/27 (81%).
  - In controls during walking with turns (SPmT), the right peak velocity was lower in 67.5% (56/83); mean log(R/L) about −0.12.
  - Applying a control-based correction to the SP primary cohort gave agreement of 71.6–73.1%.
  - Agreement by DBS status: none 32/40, DBS 6/15, not recorded 6/12.
- Analysis A as applied to SP and analysis D are therefore analyses after partial inspection of results. The task-matched analysis of A (SPmT, TUG) and analyses B and C had not been run.

## 1. Primary analysis unchanged

Populations, measures and criteria for H1 (SP agreement, success criterion), H2 (SP vs HP), the task model and V2 stability are not changed. The primary result is reported first in the manuscript.

## 2. Analysis A — normative side bias and control-referenced correction

- Reference value θ: median control log(R/L) peak velocity. **Patient data are not used.** Participant bootstrap 95% CI (5000 resamples, seed 20261007).
- A1 presence of bias: exact binomial test of the proportion of controls with a lower right side (vs 0.5); CI of θ.
- A2 task-matched correction (main check): in SPmT and TUG, patient ratios are corrected with the control θ of the same task (corrected ratio = log(R/L) − θ). Agreement and side-specific agreement before and after correction are reported. The CI of the difference is obtained by a bootstrap resampling patients and controls separately. Patient population rules are those of the locked primary analysis (same-day examination, ties excluded, bilaterally eligible).
- A3 application to SP (supporting): controls have no SP task, so the SPmT θ is applied to the SP primary cohort; the task mismatch is stated as a limitation. (Translator's note: straight self-paced and fast-walking control files were later found on Synapse and analysed on 26 October 2026; see `docs/chronology.md`.)
- A4 reference-free measure: AUC of the signed log(R/L) for the clinical side (R = 1) with bootstrap CI (SP, SPmT, TUG).
- A5 turning direction: in each SPmT file, left and right turns are counted from the sign of the integrated lumbar angular velocity in Turn intervals, and the association of net turning direction with the control ratio is assessed with Spearman correlation.
- Interpretation rule: if A2 narrows the gap between side-specific agreement after correction in at least one of SPmT and TUG, and the overall change is in the same direction even if its CI includes 0, this is described as "evidence consistent with a bias". If the direction is opposite, the bias interpretation is withdrawn.

## 3. Analysis B — magnitude-dependent interpretability

- B1 accuracy–coverage curve: agreement among the remaining participants when labels are withheld in ascending order of corrected |ratio|. No specific cut-off is claimed; only withholding 0%, 25% and 50% is tabulated.
- B2 logistic model: agreement ~ |sensor ratio| + |clinical R−L difference|, comparing which magnitude predicts agreement.

## 4. Analysis C — six-month direction change and baseline magnitude

- Among V2 participants with SP at both sessions, session-1 |log(R/L)| is compared between those whose direction changed and those whose direction persisted (Mann–Whitney). Persistence is also compared between session-1 |ratio| < 0.10 and ≥ 0.10 (Fisher).

## 5. Analysis D — DBS effect modification (hypothesis-generating)

- Participants in the primary cohort with recorded DBS status: agreement ~ DBS + |clinical difference| + disease duration, logistic (Firth noted if needed). Participants without recorded DBS status are excluded and their number reported.
- In the manuscript this is covered by one paragraph and a supplementary table only, not in the title or conclusions.

## 6. Multiplicity

All p values from A–D are descriptive. Interpretation prioritizes confidence intervals and consistency across tasks.

## 7. Additional check after running (7 October 2026, after the results were seen)

- Because controls were on average 7 years older and included more women than patients, the association of control log(R/L) with age and sex was checked: age Spearman ρ = 0.18 (p = 0.11), sex Mann–Whitney p = 0.30. Among controls aged 75 years or younger (n = 42), the median was −0.154 and 76% had a lower right side.
- Labelled in the manuscript as a check added after the results were seen.

## 8. Amendment 003 — cross-site validation (7 October 2026, written after the v11 results and before running)

- Aim: to check whether the control reference transfers to patients at another site (internal–external validation).
- Sites: NLS = JHOC (patients), WPD/WHC = VA-Seattle (patients and controls), HC = site with controls only.
- Method: the reference is always the median of controls from a **different site** than the patients: (1) HC controls → all patients, (2) VA controls → JHOC patients, (3) HC + NLS controls → VA patients. Tasks: SP (SPmT reference), SPmT, TUG. For SP, a pooled leave-site-out estimate in which each site uses controls from other sites is also reported.
- Interpretation rule: if agreement rises in the same direction in most combinations when the reference is changed, this is described as "reproduced across sites"; differences in the size of site-specific bias are reported as observed.
- Script: `analysis/05_analysis/exploratory_v6/run_cross_site_v7.py` (seed 20261008)

## 9. Addition during manuscript review (21 October 2026, after all results were seen)

- Balanced accuracy (mean of left-affected and right-affected agreement) is added. Reason: the primary cohort was imbalanced (40 left-affected, 27 right-affected in the initial run), so an improvement in overall agreement could reflect case mix.
- The CI of the change uses the same approach as A2 (resampling patients and controls separately, re-estimating the reference, 5000 resamples, seed 20261008 = 20261007 + 1).
- Results (initial run): walking with turns 70.7 → 77.0% (+0.0 to +11.5 points), TUG 72.4 → 77.6% (−3.8 to +9.8), standard walking 68.2 → 72.1% (−6.2 to +12.0); same direction in all three tasks, but each single-task CI includes 0.
- Labelled in the manuscript as added after the results were seen. On the same day, the Mann–Whitney comparison already run in analysis C (pre-specified in section 4) was reported, and the origin of the 0.10 boundary (the lowest edge of the descriptive bands of amendment 001, fixed in section 4 before running) was stated.

Note: the numbers in sections 0 and 9 come from the run on the initially downloaded files (n = 67). The reported analyses were rerun after the data-coverage correction of amendment 004 (n = 73).
