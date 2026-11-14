import json

import dspy
from dspy.experimental import Noul
from tests.clients.test_lm_engine_execution import DualResponse, native_transport


def test_pdf_predict_uses_openrouter_decisions(monkeypatch):
    wire = native_transport(monkeypatch, [DualResponse(status=200, body=json.dumps({
        "model": "google/gemini-decision-1-20261111",
        "answers": {"complete": {"type": "noul", "noul": 0.9, "confidence": 0.9}},
        "usage": {"input_tokens": 100, "output_tokens": 0, "cost": 0.00001},
    }).encode())])

    class Check(dspy.Signature):
        document: dspy.File = dspy.InputField()
        complete: Noul = dspy.OutputField(desc="Is the application complete?")

    lm = dspy.LM("openrouter-decisions/google/gemini-decision-1-20261111", api_key="test")
    predict = dspy.Predict(Check)
    predict.set_lm(lm)
    result = predict(document=dspy.File.from_bytes(b"%PDF-1.4\nfixture", mime_type="application/pdf"))
    assert result.complete.value is True
    assert result.complete.probability == 0.9
    assert wire.requests[0].url == "https://openrouter.ai/api/alpha/decisions"
    payload = json.loads(wire.requests[0].body)
    assert any(isinstance(part, dict) and part.get("type") == "file" for part in payload["state"])


def test_pdf_can_reach_openrouter_reflection_helper(monkeypatch):
    wire = native_transport(monkeypatch, [DualResponse(status=200, body=json.dumps({
        "id": "fixture", "object": "chat.completion", "model": "openai/gpt-6-luna",
        "choices": [{"index": 0, "message": {"role": "assistant", "content":
            "<answer>RELEASED, CLEAR</answer>"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
    }).encode())])

    class Read(dspy.Signature):
        document: dspy.File = dspy.InputField()
        answer: str = dspy.OutputField()

    helper = dspy.LM("openrouter/openai/gpt-6-luna", engine="lm15", api_key="test")
    helper.kwargs["provider"] = {"zdr": True}
    predict = dspy.Predict(Read)
    predict.set_lm(helper)
    result = predict(document=dspy.File.from_bytes(b"%PDF-1.4\nfixture", mime_type="application/pdf"))
    assert result.answer == "RELEASED, CLEAR"
    assert wire.requests[0].url == "https://openrouter.ai/api/v1/chat/completions"
    body = json.loads(wire.requests[0].body)
    assert body["provider"] == {"zdr": True}
    files = [part for message in body["messages"] if isinstance(message["content"], list)
             for part in message["content"] if part["type"] == "file"]
    assert len(files) == 1
    assert files[0]["file"]["file_data"].startswith("data:application/pdf;base64,JVBER")
