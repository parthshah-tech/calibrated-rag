# Hosting the study

In replay mode the study server needs only a small set of packages (`requirements-study.txt`): no
models, no index, no API key. It reads `study/frozen.json`, writes a SQLite log, and serves one
page. That makes it cheap to host, but the log must survive, so choose where it runs with that in
mind.

## Recommended: a machine you control, for scheduled sessions

Run it on a team laptop or desktop and give participants a link. Your data stays on disk, you
control backups, and nothing depends on a hosting plan.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-study.txt
export STUDY_MODE=1
export STUDY_QUESTIONS=study/questions.json
export STUDY_FROZEN=study/frozen.json
export STUDY_DB=./data/study.sqlite
export STUDY_ADMIN_TOKEN=<a long random string>
export STUDY_SALT=<another secret, used to hash participant codes>
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

- **Same room or network:** participants open `http://<your-ip>:8000/study?code=P017`.
- **Remote participants:** put a tunnel in front of port 8000 (for example Cloudflare's quick
  tunnels with `cloudflared tunnel --url http://localhost:8000`; check its current
  documentation) and send the HTTPS address it prints plus `/study?code=P017`. Keep the machine
  awake and online during sessions.

Copy `study/frozen.json` and `study/questions.json` to the machine; both are gitignored. Open
`/study/config` and confirm the `config_hash` matches the one recorded when you froze.

## Before each session day

1. Open the link yourself with a test code, then use a different test code for the pilot.
2. After the pilot, stop the server and delete `STUDY_DB`; the real run starts empty.
3. Keep `study/frozen.json` and the questions file unchanged for the whole study.

## Backups

Download the database after every session day:

```
curl -o study-$(date +%F).sqlite "http://localhost:8000/study/admin/db?token=$STUDY_ADMIN_TOKEN"
```

The admin endpoints answer 404 when `STUDY_ADMIN_TOKEN` is unset and 403 for a wrong token.

## Avoid: free hosting tiers without a persistent disk

Many free tiers spin an app down when it is idle and give it a disk that is wiped on restart, so
a SQLite log would disappear between participants. Free web services on Render behaved this
way when this guide was written. If you use any hosted service, confirm it provides a
persistent disk, mount the database path on it, and still download backups. A small VPS with
the commands above works the same as a laptop; put HTTPS in front of it.

## Fine-tuning during the study

You may clear the database (stop the server, delete `STUDY_DB`) and adjust wording in
`study/texts.json` or `study/survey.json` before the real run. Do not change the questions,
documents, model or cutoffs afterwards: re-freezing changes `config_hash` and invalidates
comparisons.
