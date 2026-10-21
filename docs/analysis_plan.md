# Analysis plan

Written before data collection. Any deviation is listed in the paper.

## Questions and hypotheses

Main question: does a lightweight retrieval-agreement signal help people trust a document
assistant's answers appropriately, relying on them when they are right and checking or
overriding them when they are wrong (Lee & See, 2004)?

- **H1, confidence resolution.** Confidence separates correct from incorrect final answers
  better in the badge conditions than in baseline (confidence by condition interaction).
- **H2, appropriate reliance.** Whether a participant accepts the assistant's answer depends
  more on whether it is correct in the badge conditions than in baseline.
- **H3, selective verification.** Opening the documents is more likely on Low and Medium items
  than High items in the badge conditions, with a weaker or absent pattern in baseline.
- **H4, exploratory.** In the rewording condition, clicking a suggested rewording is followed by
  more correct final answers than not clicking, on comparable Low and Medium items.
- **RQ5, qualitative.** How do participants understand, use or ignore the signal?

A null or negative result is reported as a result.

## Design

Between-subjects, three conditions: baseline (answer and passages), badge, badge plus up to
three suggested rewordings on Low and Medium items. Balanced random assignment by the server.
Target 30 participants (10 per condition), minimum 24. Each does one practice item, 12 scored
items and 2 attention checks in a random order, within a fixed time budget, with no feedback.

The assistant's answers are computed once, saved, and replayed identically. Two raters grade
each saved answer against an answer key before the study, blind to the badge. The item set is
chosen from a pool of at least 30 candidate questions written before running the system, fixed
and hashed before the pilot, and mirrors how the badge behaves in real use while keeping some
misleading cases:

| Badge | Items | Assistant wrong | Assistant right |
|---|---|---|---|
| Low | 4 | 3 | 1 |
| Medium | 4 | 2 | 2 |
| High | 4 | 1 | 3 |

Before the pilot, check how well the badge predicts correctness on these 12 items (for example
AUROC of the badge score against `ai_correct`; below about 0.7, change the items or cutoffs, not
afterwards). Report the achieved composition and the full candidate pool.

## Measures (columns in `per_question.csv`)

Reliance: `accepted`, with `overreliance` and `underreliance`. Final correctness:
`final_correct`. Calibration: `stated_confidence` against `final_correct`. Verification:
`consulted_doc`, `doc_seconds`. Rewording: `reformulations_shown`, `n_reformulation_clicks`,
`final_via_reformulation`. Time: `time_total_s`. Trust and usefulness: `survey.csv`.

## Exclusions (fixed in advance)

Exclude a participant who fails both attention checks or reaches fewer than 70% of the scored
items. Report every exclusion and the reason as a flow count: recruited, assigned, excluded,
analysed. Practice items are excluded from all analyses. Unreached items are missing, not
imputed.

## Models

Mixed-effects logistic regression with random intercepts for participant and item (R
`lme4::glmer`; if a model does not converge, drop the item effect and use cluster-robust
errors). Centre confidence.

- H1: `final_correct ~ confidence * condition + (1|participant) + (1|item)`; test the
  interaction, baseline against each badge condition.
- H2: `accepted ~ ai_correct * condition + (1|participant) + (1|item)`.
- H3: `consulted_doc ~ bucket * condition + (1|participant) + (1|item)`, using the logged
  `bucket` for every condition even where it was hidden.
- H4: descriptive, within the rewording condition.

Correct for the three primary hypotheses with Holm; label everything else exploratory. Always
show the 2 by 2 reliance table per condition, calibration plots by condition, mean confidence on
correct versus incorrect answers, and effect sizes (odds ratios) with 95% confidence intervals.
With about 10 participants per condition, power for between-condition effects is low, so lead
with effect sizes and intervals and call the findings exploratory unless the evidence is strong.
Secondary: time per item, viewer seconds, and questionnaire scores by condition.

## Questionnaire scoring

Trust items `trust_1` to `trust_5` (reverse-score `trust_3`), mean of the five; `effort_1`
alone; badge items for the badge conditions (reverse-score `badge_3`); `manip_check` verifies
participants knew whether an indicator was shown.

## Interviews

A 5 to 10 minute semi-structured interview after each session: how they decided what to trust;
when and why they opened the documents; times they disagreed with the assistant; what the
indicator meant and whether it changed what they did (baseline: whether anything told them how
much to trust an answer); use of the rewordings; effect of the time limit; what they thought the
study was about (asked before the full debrief is discussed); what they would change. Two
coders independently open-code the first four interviews, agree a codebook, code the rest, and
build themes (reflexive thematic analysis, Braun & Clarke, 2006). Report themes with short
quotes, labelled by condition and by what the participant did, next to the quantitative
pattern they explain or complicate.

## References (check each before citing)

Lee & See (2004), Human Factors 46(1); Zhang, Liao & Bellamy (2020), FAT\*; Yin, Wortman Vaughan
& Wallach (2019), CHI; Buçinca, Malaya & Gajos (2021), CSCW; Braun & Clarke (2006), Qualitative
Research in Psychology 3(2).
