import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parents[1] / "training/environments/oolong"


class Backend:
    def __init__(self, **kw):
        self.calls = []

    async def start(self, *a, **kw):
        pass

    async def load_context(self, payload):
        self.context = payload

    async def update_history(self, messages):
        self.history = messages

    async def execute(self, code):
        self.calls.append(code)
        return SimpleNamespace(stdout="done", stderr="", final_answer="gold")

    async def stop(self):
        pass


class Response:
    def raise_for_status(self):
        pass

    def json(self):
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "x",
                                "function": {
                                    "name": "repl",
                                    "arguments": "{\"code\": \"answer['content']='gold'; answer['ready']=True\"}",
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }


class Client:
    def __init__(self, **kw):
        self.payloads = []

    async def post(self, path, json):
        self.payloads.append(json)
        return Response()

    async def aclose(self):
        pass


def test_native_eval_keeps_context_out_of_prompt_and_scores(tmp_path):
    import sys
    import types

    package = types.ModuleType("oolong")
    package.__path__ = [str(ROOT / "oolong")]
    env = types.ModuleType("oolong.scoring")
    env.synth_score = lambda row, answer: 1.0 if answer == "gold" else 0.0
    sys.modules["oolong"] = package
    sys.modules["oolong.scoring"] = env
    sys.path.insert(0, str(ROOT))
    from oolong.native_eval import evaluate

    clients = []

    def cf(**kw):
        c = Client(**kw)
        clients.append(c)
        return c

    rows = [
        {
            "id": "a",
            "context": "PRIVATE CONTEXT",
            "root_prompt": "Question?",
            "answer": "['gold']",
            "answer_type": "ANSWER_TYPE.STRING",
        }
    ]
    result = asyncio.run(
        evaluate(
            rows,
            base_url="http://localhost/v1",
            model="m",
            run_dir=tmp_path,
            backend_factory=Backend,
            client_factory=cf,
        )
    )
    assert result[0]["final_answer"] == "gold" and result[0]["score"] == 1
    assert "PRIVATE CONTEXT" not in json.dumps(clients[0].payloads)
    saved = json.loads((tmp_path / "task-000.json").read_text())
    assert saved["events"][0]["actions"][0]["stdout"] == "done"
    assert json.loads((tmp_path / "summary.json").read_text())["count"] == 1
