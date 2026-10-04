import json
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from backend.config import Settings
from backend.llm import CachedLLM, FakeLLM, LLMError, OpenAICompatClient, build_llm


class Stub(BaseHTTPRequestHandler):
    script: list = []  # (status, headers, body) consumed in order
    seen: list = []

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        Stub.seen.append(
            {
                "path": self.path,
                "auth": self.headers.get("Authorization"),
                "ua": self.headers.get("User-Agent"),
                "body": json.loads(self.rfile.read(n)),
            }
        )
        status, headers, body = Stub.script.pop(0)
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    Stub.script, Stub.seen = [], []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


def ok(text="hello [1]", finish="stop"):
    return (
        200,
        {},
        {"model": "m-x", "choices": [{"message": {"content": text}, "finish_reason": finish}]},
    )


def client(url, **kw):
    return OpenAICompatClient(url, "sk-secret", "my-model", sleep=lambda s: None, **kw)


def test_request_shape_and_parsing(server):
    Stub.script = [ok("answer [1]")]
    r = client(server, reasoning_effort="low").complete(
        [{"role": "user", "content": "hi"}], max_tokens=77
    )
    req = Stub.seen[0]
    assert r.text == "answer [1]" and r.model == "m-x" and not r.cached
    assert req["path"] == "/v1/chat/completions" and req["auth"] == "Bearer sk-secret"
    assert req["ua"] == "calibrated-rag/0.1"
    assert req["body"]["model"] == "my-model" and req["body"]["max_tokens"] == 77
    assert req["body"]["temperature"] == 0 and req["body"]["reasoning_effort"] == "low"


def test_reasoning_effort_is_omitted_when_unset(server):
    Stub.script = [ok()]
    client(server).complete([{"role": "user", "content": "hi"}])
    assert "reasoning_effort" not in Stub.seen[0]["body"]


def test_retries_rate_limits_then_succeeds(server):
    Stub.script = [(429, {"Retry-After": "0"}, {"error": "slow down"}), (503, {}, {}), ok("fine")]
    assert client(server).complete([{"role": "user", "content": "x"}]).text == "fine"
    assert len(Stub.seen) == 3


def test_gives_up_after_retries_and_hides_the_key(server):
    Stub.script = [(429, {}, {"error": "x"})] * 4
    with pytest.raises(LLMError) as e:
        client(server, retries=3).complete([{"role": "user", "content": "x"}])
    assert "429" in str(e.value) and "sk-secret" not in str(e.value)


def test_auth_errors_are_not_retried(server):
    Stub.script = [(401, {}, {"error": "bad key"})]
    with pytest.raises(LLMError, match="401"):
        client(server).complete([{"role": "user", "content": "x"}])
    assert len(Stub.seen) == 1


def test_empty_text_after_the_retry_explains_the_reasoning_budget(server):
    Stub.script = [ok("", finish="length"), ok("", finish="length")]
    with pytest.raises(LLMError, match="reasoning"):
        client(server).complete([{"role": "user", "content": "x"}], max_tokens=100)
    assert [r["body"]["max_tokens"] for r in Stub.seen] == [100, 300]


def test_budget_exhausted_by_reasoning_is_retried_with_triple_the_budget(server):
    Stub.script = [ok("", finish="length"), ok("rewritten query")]
    r = client(server).complete([{"role": "user", "content": "x"}], max_tokens=300)
    assert r.text == "rewritten query"
    assert [q["body"]["max_tokens"] for q in Stub.seen] == [300, 900]


def test_retry_budget_is_capped(server):
    Stub.script = [ok("", finish="length"), ok("fine")]
    client(server).complete([{"role": "user", "content": "x"}], max_tokens=2000)
    assert Stub.seen[1]["body"]["max_tokens"] == 3000


def test_empty_text_that_is_not_a_length_cutoff_is_not_retried(server):
    Stub.script = [ok("", finish="stop")]
    with pytest.raises(LLMError, match="no text"):
        client(server).complete([{"role": "user", "content": "x"}])
    assert len(Stub.seen) == 1


def test_unreachable_server_raises_llm_error():
    c = OpenAICompatClient(
        "http://127.0.0.1:9/v1", "k", "m", retries=1, timeout=1, sleep=lambda s: None
    )
    with pytest.raises(LLMError, match="unreachable"):
        c.complete([{"role": "user", "content": "x"}])


def test_cache_hits_survive_a_new_instance_and_skip_the_model(tmp_path):
    inner = FakeLLM(["one", "two"])
    msgs = [{"role": "user", "content": "q"}]
    a = CachedLLM(inner, tmp_path)
    first = a.complete(msgs)
    again = a.complete(msgs)
    assert (first.text, first.cached, again.text, again.cached) == ("one", False, "one", True)
    assert len(inner.calls) == 1 and (a.hits, a.misses) == (1, 1)
    b = CachedLLM(FakeLLM(["other"]), tmp_path)  # new process, same directory
    assert b.complete(msgs).cached and b.complete(msgs).text == "one"
    assert not list(tmp_path.glob("*.tmp"))


def test_cache_key_depends_on_prompt_and_limits(tmp_path):
    c = CachedLLM(FakeLLM(["a", "b", "c"]), tmp_path)
    c.complete([{"role": "user", "content": "q1"}])
    c.complete([{"role": "user", "content": "q2"}])
    c.complete([{"role": "user", "content": "q1"}], max_tokens=50)
    assert c.misses == 3 and c.hits == 0


def test_sampled_output_is_never_cached(tmp_path):
    inner = FakeLLM(["a", "b"])
    c = CachedLLM(inner, tmp_path)
    c.complete([{"role": "user", "content": "q"}], temperature=0.7)
    c.complete([{"role": "user", "content": "q"}], temperature=0.7)
    assert len(inner.calls) == 2 and not list(tmp_path.glob("*.json"))


def test_failures_are_not_cached(tmp_path):
    c = CachedLLM(FakeLLM(error=LLMError("boom")), tmp_path)
    with pytest.raises(LLMError):
        c.complete([{"role": "user", "content": "q"}])
    assert not list(tmp_path.glob("*.json"))


def test_build_llm_needs_a_key_and_settings_never_print_it(tmp_path):
    assert build_llm(Settings(data_dir=str(tmp_path))) is None
    s = Settings(data_dir=str(tmp_path), llm_api_key="sk-secret")
    assert isinstance(build_llm(s), CachedLLM) and "sk-secret" not in repr(s)


def test_settings_accept_either_key_name(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "g-key")
    assert Settings.from_env().llm_api_key == "g-key"
    monkeypatch.setenv("LLM_API_KEY", "l-key")
    assert Settings.from_env().llm_api_key == "l-key"
    monkeypatch.setenv("LLM_REASONING_EFFORT", "low")
    assert Settings.from_env().llm_reasoning_effort == "low"


def test_urllib_error_is_the_only_network_type_we_depend_on():
    assert issubclass(urllib.error.HTTPError, urllib.error.URLError)
