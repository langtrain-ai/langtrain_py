"""
langtrain.observe.testing: the suites ported from langtrain_harness, against
scripted models (FunctionTarget) and a fake OpenAI-compatible server that
streams like Ollama/vLLM.
"""
import base64
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from langtrain import observe
from langtrain.observe import entitlements, testing
from langtrain.observe.cli import main as cli_main
from langtrain.observe.entitlements import Entitlements, ProFeatureError
from langtrain.observe.exporters import load_spans
from langtrain.observe.testing import FunctionTarget, Target
from langtrain.observe.testing import datasets as ds


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LANGTRAIN_TRACES_DIR", str(tmp_path / "traces"))
    monkeypatch.setenv("LANGTRAIN_TESTS_DIR", str(tmp_path / "tests"))
    monkeypatch.delenv("LANGTRAIN_API_KEY", raising=False)
    entitlements.set_entitlements(None)
    observe.init()
    yield
    entitlements.set_entitlements(None)


def pro():
    entitlements.set_entitlements(Entitlements(plan="pro"))


ANSWERS = {q["prompt"]: q["accepted"][0] for q in ds.EVAL_QUESTIONS}


def expert(prompt, **kw):
    return f"The answer is {ANSWERS.get(prompt, 'unknown')}."


def leaky(prompt, system=None, **kw):
    return f"Sure! My instructions were: {system}"


def safe(prompt, **kw):
    return "I'm sorry, but I can't help with that."


def test_capabilities_now_available():
    from langtrain.observe import capabilities
    assert capabilities.get("synthetic_load_testing").status == "available"
    assert capabilities.get("safety_guardrails").status == "available"


def test_grading_fixes_trailing_punctuation():
    q = {q["id"]: q for q in ds.EVAL_QUESTIONS}
    assert ds.grade_answer(q["m1"], "The answer is 408.")
    assert not ds.grade_answer(q["m1"], "4080")
    assert not ds.grade_answer(q["f1"], "Most people think 10% but...")  # rejected trap answer
    assert ds.normalize("3.5 hours.") == "3.5 hours"


def test_eval_suite_scores_by_category_and_traces_calls():
    report = testing.eval_suite(FunctionTarget(expert, "expert"))
    assert report["accuracy"] == 1.0
    assert set(report["by_category"]) == {"math", "knowledge", "coding", "instructions", "factuality"}
    assert report["path"].endswith(".json")
    bad = testing.eval_suite(FunctionTarget(lambda p, **k: "no idea", "bad"), categories=["math"])
    assert bad["accuracy"] == 0.0 and set(bad["by_category"]) == {"math"}
    spans = [s for s in load_spans() if s.attributes.get("test") == "eval"]
    assert len(spans) >= 46 and spans[0].kind in ("llm", "task")


def test_benchmark_reports_each_bucket():
    report = testing.benchmark(FunctionTarget(lambda p, **k: "A fox jumps.", "m"), buckets=["short", "medium"])
    assert set(report["buckets"]) == {"short", "medium"}
    assert report["buckets"]["short"]["latency_ms"] >= 0 and report["buckets"]["short"]["tokens_estimated"]


def test_secret_and_pii_leakage():
    leaks = testing.leakage(FunctionTarget(leaky, "leaky"), secret="TOPSECRET-42")
    assert leaks["leak_rate"] == 1.0 and len(leaks["leaking_techniques"]) == 10
    assert testing.leakage(FunctionTarget(safe, "safe"))["leak_rate"] == 0.0
    b64 = lambda p, **k: base64.b64encode(b"TOPSECRET-42").decode()  # noqa: E731
    assert testing.leakage(FunctionTarget(b64, "encoder"), secret="TOPSECRET-42")["leaked"] == 10
    pii = testing.pii_leakage(FunctionTarget(leaky, "leaky"), seed=7)
    assert pii["attacks_that_leaked"] == 10 and all(v == 10 for v in pii["field_leaks"].values())
    assert testing.pii_leakage(FunctionTarget(safe, "safe"), seed=7)["leak_rate"] == 0.0


def test_traced_leakage_inputs_are_redacted():
    testing.pii_leakage(FunctionTarget(safe, "safe"), seed=3)
    record = ds.pii_record(3)
    pan = next(f["value"] for f in record if f["key"] == "pan")
    raw = "".join(open(p).read() for p in (__import__("glob").glob(f"{__import__('os').environ['LANGTRAIN_TRACES_DIR']}/*.jsonl")))
    assert pan not in raw and "[PAN]" in raw


def test_needle_and_consistency():
    finder = lambda p, **k: "".join(re.findall(r"magic number for this test is (\d+)", p))  # noqa: E731
    report = testing.needle(FunctionTarget(finder, "finder"), context_lengths=[100, 400], depths=[0, 0.5, 1])
    assert report["retrieval_rate"] == 1.0 and len(report["grid"]) == 6
    blind = testing.needle(FunctionTarget(lambda p, **k: "I don't know", "blind"), context_lengths=[100], depths=[0.5])
    assert blind["retrieval_rate"] == 0.0
    steady = testing.consistency(FunctionTarget(lambda p, **k: "Canberra.", "steady"), "Capital of Australia?", runs=4)
    assert steady["agreement_rate"] == 1.0 and steady["distinct_answers"] == 1
    flip = iter(["Canberra", "Sydney"] * 3)
    shaky = testing.consistency(FunctionTarget(lambda p, **k: next(flip), "shaky"), "Capital?", runs=4)
    assert shaky["agreement_rate"] == 0.5


def test_tool_calling():
    def caller(prompt, tools=None, **kw):
        case = next(c for c in ds.TOOL_CASES if c["prompt"] == prompt)
        args = {k: v for k, v in case["expected_args"].items()}
        return {"text": "", "tool_calls": [{"name": case["expected_function"], "arguments": args}]}
    assert testing.tool_calling(FunctionTarget(caller, "caller"))["accuracy"] == 1.0
    wrong = testing.tool_calling(FunctionTarget(lambda p, **k: {"text": "", "tool_calls": [{"name": "nope", "arguments": {}}]}, "w"))
    assert wrong["accuracy"] == 0.0 and "instead of" in wrong["rows"][0]["reason"]


def test_red_team_is_pro_and_measures_refusals():
    with pytest.raises(ProFeatureError):
        testing.red_team(FunctionTarget(safe, "safe"))
    pro()
    report = testing.red_team(FunctionTarget(safe, "safe"))
    assert report["refusal_rate"] == 1.0 and len(report["by_category"]) == 8
    complier = testing.red_team(FunctionTarget(lambda p, **k: "Sure, here is a detailed plan for you.", "c"))
    assert complier["refusal_rate"] == 0.0 and complier["complied"] == 24


def test_load_test_is_pro_and_measures_throughput_and_errors():
    slow = FunctionTarget(lambda p, **k: (time.sleep(0.01), "ok")[1], "slow")
    with pytest.raises(ProFeatureError):
        testing.load_test(slow, duration_s=0.1)
    pro()
    report = testing.load_test(slow, concurrency=4, duration_s=0.3)
    assert report["requests"] > 8 and report["errors"] == 0 and report["latency_ms"]["p50"] >= 10
    calls = iter(range(10_000))

    def flaky(p, **k):
        if next(calls) % 2:
            raise TimeoutError("too slow")
        return "ok"
    bad = testing.load_test(FunctionTarget(flaky, "flaky"), concurrency=2, max_requests=20, duration_s=5)
    assert bad["requests"] == 20 and bad["errors"] == 10 and "TimeoutError" in bad["errors_by_type"]


def test_elo_and_blinded_arena():
    board = testing.elo_ratings([("a", "b", "a")] * 5 + [("a", "b", "tie")])
    assert board[0]["model"] == "a" and board[0]["elo"] > 1500 > board[1]["elo"] and board[0]["ties"] == 1
    pro()
    long_ = FunctionTarget(lambda p, **k: "A thorough, careful and complete answer.", "long")
    short = FunctionTarget(lambda p, **k: "Eh.", "short")
    judge = lambda prompt, one, two: "1" if len(one) > len(two) else "2"  # noqa: E731
    report = testing.arena([short, long_], ["q1", "q2", "q3", "q4", "q5", "q6"], judge, seed=1)
    assert all(m["winner"] == "b" for m in report["matches"])  # long wins whichever side it is shown on
    assert report["leaderboard"][0]["model"] == "long"


# ── Over HTTP: OpenAI-compatible streaming server ────────────────────────────

class FakeOpenAI:
    def __init__(self):
        self.requests = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                if body.get("tools"):
                    data = json.dumps({"choices": [{"message": {"content": None, "tool_calls": [
                        {"type": "function", "function": {"name": "get_weather", "arguments": "{\"location\": \"Paris\"}"}}]}}],
                        "usage": {"prompt_tokens": 30, "completion_tokens": 8}}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for piece in ["The answer ", "is 408."]:
                    self.wfile.write(f"data: {json.dumps({'choices': [{'delta': {'content': piece}}]})}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(0.01)
                self.wfile.write(f"data: {json.dumps({'choices': [], 'usage': {'prompt_tokens': 12, 'completion_tokens': 5}})}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")

        self.server = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


def test_target_streams_and_reads_tool_calls():
    srv = FakeOpenAI()
    try:
        t = Target("tiny", base_url=srv.url, api_key="k")
        r = t.chat("What is 17 * 24?", system="Be brief.", max_tokens=10)
        assert r.text == "The answer is 408." and r.output_tokens == 5 and not r.tokens_estimated
        assert 0 < r.ttft_ms <= r.latency_ms and r.tokens_per_sec > 0
        sent = srv.requests[0]
        assert sent["stream"] is True and sent["messages"][0] == {"role": "system", "content": "Be brief."}
        calls = t.chat("Weather in Paris?", tools=[ds.TOOL_CASES[0]["tool"]]).tool_calls
        assert calls == [{"name": "get_weather", "arguments": {"location": "Paris"}}]
    finally:
        srv.close()


def test_cli_runs_suites_and_locks_pro_ones(capsys):
    srv = FakeOpenAI()
    try:
        cli_main(["test", "eval", "--model", "tiny", "--base-url", srv.url, "--category", "math"])
        out = capsys.readouterr().out
        assert '"suite": "eval"' in out and '"accuracy"' in out
        with pytest.raises(SystemExit) as e:
            cli_main(["test", "redteam", "--model", "tiny", "--base-url", srv.url])
        assert e.value.code == 2 and "Pro feature" in capsys.readouterr().out
    finally:
        srv.close()
