# Live voluntary-use study

This flow supports two server-assigned conditions, baseline and confidence bar. Participants
can complete the document task without consulting the assistant. See the [shared app guide](public-app.md)
to host this study, the regular upload app, and its administration page together. The commands below
run the study as a standalone process.

## Participant flow

1. Information and explicit consent. Opening the page creates a browser cookie, with no
   participant record until consent.
2. Instructions and one unscored practice item. Documents and the assistant are available.
3. Main task, with a server-timed budget. The task brief, questions, answer fields and confidence
   ratings are on the same page as the optional assistant. Drafts save after a brief pause in typing.
4. Shared answer judgments, if evaluation examples are configured. Each participant sees the
   same prepared answers. Only the confidence condition receives their scores.
5. Questionnaire and optional written responses.
6. Participant-controlled debrief.

Reloading restores the assigned condition, phase, transcript, saved draft and remaining time.
The cookie identifies this browser session. Switching browsers or clearing cookies creates a
new session. Stop study marks the session withdrawn. Exported records retain that flag so the
team can exclude the session and handle any requested deletion.

## Prepare the materials

### Edit in the browser

Open `/study/admin` on the live study server and enter its `STUDY_ADMIN_TOKEN`.
Edit the title, task brief, practice and task questions, grading notes and participant
information. Select one or more TXT, Markdown, text PDFs or DOCX files with **Add documents**,
check the extracted text, and remove any previous documents you are replacing. Add or remove
task questions in the same editor. The study draft has no fixed count for documents or questions.
The individual file size limit still applies. The saved draft stays beside the
configured study database, and the editor can download it as `study-draft.json`.

Saving a draft leaves the current study running with its existing materials. To run
the new draft, use its downloaded file as `STUDY_TASK` on a separate pilot instance.
Use a new `STUDY_DB` and `DATA_DIR`, and an unused local port for the pilot. A change
to the task, questions or documents clears the previous shared answers. The team can
prepare and grade new ones before enabling that stage. Questionnaire items can still
be edited in the downloaded file.

### Prepare a pilot or final run

`study/live.example.json` contains fictional review material, a task with six decisions, a
practice item, rubrics, consent text and a provisional project-specific questionnaire. The
questionnaire requires the qualitative lead's review. It is not a validated SUS scale.

Copy your downloaded draft or the example to the gitignored `study/live.json` and review
it with the team. `title` and `task` control the main brief, `practice` defines the practice question, and
`items` defines the main questions in display order. The page adapts to the number of questions.
`documents` sets the collection shared by both conditions. Give each entry a unique `.txt` or
`.md` name and the full text. Extract and check the text of PDFs or DOCX files before inclusion.
Update the consent (`info`), `instructions`, `survey` and `debrief` for the chosen study. Keep the
answer keys in the manifest for blind grading. Public responses exclude these keys.

Prepare the common answer-evaluation examples using the actual model. Set `DATA_DIR` to a
dedicated study directory in `.env`. When changing the document collection, use a fresh data
directory so retrieval and answer generation use the new documents. Then run:

```bash
uv run python -m backend.study.prepare_live \
  --task study/live.json \
  --queries study/live-evaluation.example.json \
  --out study/live-prepared.json
```

Replace the example queries with the team's choices. The preparation command records the real
answers, source passages and continuous agreement scores once. Two raters should grade the
examples before research sessions and record the adjudicated `ai_correct` values in the prepared
manifest. The example corpus may produce mostly correct answers. Select the final task and item
composition through the team's pilot and planned method.

The bar displays the raw 0–1 agreement score. Bucket cutoffs are unused in this flow. Whether the
score predicts generated-answer correctness on the final corpus still needs evaluation.

## Run locally

Configure `.env` with these values, along with the LLM credentials:

```dotenv
STUDY_MODE=1
STUDY_FLOW=live
STUDY_TASK=study/live-prepared.json
DATA_DIR=./data/live-corpus
STUDY_DB=./data/live-study.sqlite
STUDY_TIME_BUDGET_MIN=25
STUDY_SALT=<stable random secret>
STUDY_ADMIN_TOKEN=<different random secret>
```

```bash
uv run uvicorn backend.main:app --env-file .env --host 127.0.0.1 --port 8012 --workers 1
```

Open `http://localhost:8012/study`. The server assigns balanced conditions. Participants cannot
choose a condition or supply model settings. Baseline responses omit the system score both
during live chat and during the shared-answer block. The score is retained in protected records.

Use a separate database for the pilot and real collection. Set `preview` to false in the final
manifest after the materials have been reviewed. The server checks the material and configuration
fingerprint against existing participants and rejects a changed configuration with the same DB.
Run one worker because pipeline operations and phase transitions use a process-local lock.

## Export

Open `/study/admin`, enter the administrator token, and select **Download participant data**.
The ZIP includes the current run's stored responses. The API also accepts the administrator
token in a header for scripted downloads. Keep that token out of participant links.

```bash
curl -H "X-Admin-Token: $STUDY_ADMIN_TOKEN" \
  http://localhost:8012/study/admin/export.zip -o live-study-export.zip
```

The shell variable must already contain the configured token. `/study/admin/db` provides a
consistent SQLite backup with the same header. The archive includes:

- `sessions.csv` with condition, completion/withdrawal status, task duration, query and view counts.
  Its `participant_code` matches the on-screen code for linking interview notes and removal requests
  to the salted `participant_id` used throughout the export. Keep this mapping with the research team.
- `task_answers.csv` with final text, self-rated confidence and links to blind grading IDs.
- `human_answers_blind.csv` and `ai_answers_blind.csv` with grader columns and no group or score.
- `answer_links.csv` linking generated answers and recorded scores to participants and phases.
- `judgments.csv` and `survey.csv` for the final two blocks.
- `events.jsonl` with pseudonymised, timestamped records and `materials.json` with the frozen content.

Correctness and appropriate reliance require grading and analysis. The application does not
infer them from source overlap. Keep practice separate from the main task, include zero-query
task completions, and exclude withdrawn and preview sessions from the research analysis.

Viewing durations reflect the interval between opening and closing a source. Switching away
from the browser tab closes the viewer and attempts to record its closing time. A disconnected
browser may leave an unmatched open event. A logged open records access to a source, and cannot
establish that it was read. Free-text responses can contain personal information even though
participant IDs are pseudonyms. Follow the team's reviewed consent and data-handling procedure.
