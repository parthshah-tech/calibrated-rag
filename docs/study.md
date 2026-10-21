# Running the user study

Study mode turns the app into a closed experiment on whether a retrieval-agreement badge helps
people trust a document assistant's answers appropriately. The design and hypotheses are in
[`analysis_plan.md`](analysis_plan.md); hosting is in [`deploy.md`](deploy.md). This repository
contains no study data and no questions.

## What a participant does

1. Opens `/study?code=P017` (or `/study`, which generates a code), reads the information sheet
   and ticks consent.
2. Reads the instructions. Badge conditions get one extra sentence about the indicator.
3. Does a practice question, then the scored questions in an order that is random per
   participant. For each: press **Ask the assistant**, read the answer and passages (plus, by
   condition, the badge and suggested rewordings), optionally open the documents, then **accept
   the assistant's answer or type their own**, and rate from 1 to 7 how confident they are that
   their final answer is correct. No feedback is ever shown.
4. Fills in a short questionnaire and reads a debrief.

The condition (baseline, badge, badge plus rewordings) is assigned by the server with balanced
random assignment and stored, so a refresh cannot change it. It is enforced in the server's
responses: a baseline participant's responses never contain the badge or rewordings. A page
reload resumes at the first unfinished question. A countdown (`STUDY_TIME_BUDGET_MIN`, default
35) starts when they begin the practice question.

## One-time preparation

1. **Documents.** Ingest the 3 to 5 study documents with the normal interface (without
   `STUDY_MODE`).
2. **Questions.** Write `study/questions.json` (gitignored: it holds the answers). See
   `study/questions.example.json`.

   ```json
   {"id": "q01", "question": "...", "answer": "the answer key",
    "source_file": "doc.pdf", "gold_phrase": "a short phrase from the answering passage",
    "answerable": true, "coverage": "well-covered",
    "attention_check": false, "practice": false}
   ```

   Exactly one question should have `"practice": true`. Unanswerable questions use
   `"answerable": false` with empty `source_file` and `gold_phrase`.
3. **Cutoffs.** Run `uv run python -m eval.eval_harness --corpus ./eval/corpus --qa <your
   questions in that format>`, then `uv run python -m eval.fit_cutoffs`, and put the printed
   `CONFIDENCE_LOW` and `CONFIDENCE_HIGH` in `.env`. Cutoffs fit on other datasets do not
   transfer.
4. **Freeze.** With the full install and an LLM key:

   ```bash
   uv run python -m backend.study.freeze
   ```

   This runs every question and every suggested rewording once and saves the results to
   `study/frozen.json`, so the study server needs no models and no key and every participant sees
   identical text. It also writes `study/ai_answers_to_grade.csv` (blind: no badge, no condition)
   and `study/frozen_report.csv` (the badge and automatic correctness per item, for choosing
   items), and prints how the scored items fall across the badge levels. Freezing fails if any
   answer could not be written. Run it again after any change to questions, documents, settings
   or cutoffs; the server refuses a frozen file made from a different question set.
5. **Grade the assistant's answers.** Two raters fill in `grader1` and `grader2` in
   `ai_answers_to_grade.csv`, resolve disagreements into `final` (1 for correct, 0 for
   incorrect), and save it as `ai_answers_graded.csv`. Do this before the pilot.
6. **Check the item set** against `analysis_plan.md` (composition, and that the badge is
   informative on these items) before the pilot, never after.
7. Optional: edit the participant wording in `study/texts.json` and the questionnaire in
   `study/survey.json` (see `backend/study/texts.py` and `survey.py` for the keys).

## Running sessions

Start the server as described in `deploy.md` and give each participant a link containing their
code. Codes must be 1 to 32 letters, digits, `-` or `_` and contain no personal information. Run
a pilot with two or three people first, then **stop the server and delete `STUDY_DB`** so pilot
data does not mix in. Conduct the interview straight after each session.

## Getting the data

Either download from the running server (needs `STUDY_ADMIN_TOKEN`):

```
/study/admin/per_question.csv  /study/admin/events.csv  /study/admin/survey.csv
/study/admin/ai_answers.csv    /study/admin/human_answers.csv  /study/admin/db
```

(pass `?token=...` or an `X-Admin-Token` header), or export from the database file:

```bash
uv run python -m backend.study.export --db study.sqlite --out data/study_export \
    --salt <STUDY_SALT> --ai-grades ai_answers_graded.csv --human-grades human_answers_graded.csv
```

Participant codes become salted hashes unless `--keep-codes` is given.

**Grading the participants' own answers.** The export writes `grading/human_answers.csv`:
every answer a participant typed instead of accepting, shuffled, with the answer key and no
condition or participant. Raters fill in `final` (1 or 0) blind to condition and save it as
`human_answers_graded.csv`. Re-run the export with both graded files to get
`ai_correct`, `human_correct`, `final_correct` and the reliance columns.

## Data dictionary: `per_question.csv`

One row per participant and question, including questions never reached.

| Column | Meaning |
|---|---|
| `participant_id`, `condition`, `config_hash` | Hashed id, assigned condition, frozen-config fingerprint. |
| `position`, `question_id`, `practice`, `attention_check`, `coverage`, `answerable` | The item and where it came in this participant's order (0 is the practice item). |
| `unreached` | The participant never saw the item (time ran out). |
| `n_asks`, `reformulations_shown`, `n_reformulation_clicks`, `final_via_reformulation` | Interaction with the assistant. |
| `bucket`, `badge_score` | The badge level and score of the first answer, recorded for every condition. |
| `shown_badge`, `final_bucket` | Whether the final answer showed the badge, and its level. |
| `ai_answer_id`, `ai_refused` | Which saved assistant answer ended up final, and whether it said the documents don't answer. |
| `ai_correct`, `ai_correct_source` | Whether the assistant's answer was correct and whether that came from real grading (`graded`) or the automatic stand-in (`auto`). |
| `accepted`, `final_answer_text` | Whether the participant accepted the assistant's answer, and what they typed if not. |
| `human_correct`, `final_correct` | Graded correctness of the typed answer; of the final answer (the assistant's, if accepted). Empty until graded. |
| `appropriate_reliance`, `overreliance`, `underreliance` | Accepted a right answer or rejected a wrong one; accepted a wrong one; rejected a right one. |
| `stated_confidence` | The 1 to 7 rating of their final answer. |
| `consulted_doc`, `doc_seconds` | Whether and how long the document viewer was open. |
| `time_to_confidence_s`, `time_total_s`, `submitted` | Seconds from showing the question to the rating and to submission; whether it was finished. |

The automatic `ai_correct` only checks whether the right passage was retrieved and whether the
model refused, not whether the answer text is right. Use real grades for analysis.

Other files: `events.csv` (every event with its JSON payload), `survey.csv` (one row per
participant, columns as in the questionnaire; `reverse` items are scored 8 minus the answer).

## Limits

- One server process and a SQLite file; fine for a lab study, not for a public deployment.
- `/study/config`, `/study/info` and the document endpoints have no authentication, so serve on
  a network you control or behind your tunnel. The admin downloads need the token.
- No participant-typed text is sent to any outside service in replay mode.
