# Host the shared app

This setup serves document chat at `/`, the participant study at `/study`, and administration
at `/study/admin`. Use one local server and, when needed, one Microsoft Dev Tunnel.

## Start locally

Install Python 3.12 and `uv`, then run the commands from the repository root.

```bash
uv sync --group dev --extra models
cp .env.example .env
```

In PowerShell, use `Copy-Item .env.example .env` for the second command. Set the following in
`.env`. Replace the example values with your own before starting. Add an `LLM_API_KEY` locally
to enable answer generation.

```dotenv
STUDY_MODE=0
STUDY_TASK=study/live.example.json
STUDY_DB=./data/study/study.sqlite
DATA_DIR=./data/study/corpus
STUDY_TIME_BUDGET_MIN=25
STUDY_SALT=
PUBLIC_RAG_DATA_DIR=./data/rag
PUBLIC_ADMIN_USERNAME=
PUBLIC_ADMIN_PASSWORD=
PUBLIC_ORIGIN=
```

Use distinct random values for the study salt and administrator password. To generate one locally,
run `uv run python -c "import secrets; print(secrets.token_urlsafe(32))"`. Keep the salt stable
for an existing study. The included study is a fictional preview. See the [study guide](live-study.md)
for preparing, checking, and grading a manifest for data collection.

```bash
uv run uvicorn backend.public_app:create_public_app --factory --env-file .env --host 127.0.0.1 --port 8024 --workers 1 --forwarded-allow-ips 127.0.0.1
```

Open <http://127.0.0.1:8024/>. Use `/study` to try the participant flow, and `/study/admin` to
sign in with the username and password in your local configuration. The browser keeps the study
condition across reloads. Use a new browser session to start another participant.

The administration page publishes an edited version for new participants. Consented sessions stay
with their original version, and the download selector lets you export each version separately.
Changing the task, practice, questions, or documents clears the shared evaluation answers and
puts the new version in preview. A title or instructions edit keeps the existing answer bank.
To start a new collection with different graded judgment items, use its reviewed manifest and a
new study database path. Back up the previous database and version directory privately for exports.

## Open access with Microsoft Dev Tunnels

Install and sign in to the [Dev Tunnels CLI][install] on the host. Create a persistent tunnel
and add the same port used above. Replace `TUNNEL_ID` with the ID returned by the create command.

```text
devtunnel user login
devtunnel create
devtunnel port create TUNNEL_ID -p 8024 --protocol http
devtunnel host TUNNEL_ID
```

Copy the HTTPS forwarding origin shown by `devtunnel host` into `PUBLIC_ORIGIN` in `.env`
(for example, `https://your-tunnel-host.example`, with no route suffix). Restart the local app
so its same-origin checks use the tunnel address. Keep the local app and tunnel host running
whenever the links need to work. Reuse the same tunnel ID when restarting.

The tunnel requires provider sign-in by default. If participants need to join from a link alone,
enable anonymous access for the forwarded port. The study's administrator sign-in still protects
`/study/admin` and its API endpoints.

```text
devtunnel access create TUNNEL_ID --port 8024 --anonymous
```

Before sharing the study link, open `/study/admin` in a fresh private window and verify that the
browser challenges for a username and password. Share the `/study` URL with participants and
keep exports in the research team's private storage. Use a single app worker so version changes
and in-flight sessions share the same process state.

[install]: https://learn.microsoft.com/en-us/azure/developer/dev-tunnels/get-started
