import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

from dashboard import writing
from dashboard.server import make_handler


def test_write_requires_origin_token_and_current_version(tmp_path, monkeypatch):
    docs = tmp_path / "docs/research"
    docs.mkdir(parents=True)
    (docs / "test.md").write_text("original")
    monkeypatch.setattr(writing, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(writing, "BACKUP_ROOT", tmp_path / "backups")
    monkeypatch.setattr(writing.data, "WRITING_DOCS", {"test": "test.md"})
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(method, path, body=None, headers=None):
        conn = HTTPConnection(*server.server_address)
        conn.request(method, path, body=body, headers=headers or {})
        response = conn.getresponse()
        result = (response.status, response.read())
        conn.close()
        return result

    try:
        status, body = request("GET", "/api/writing/test")
        assert status == 200
        doc = json.loads(body)
        body = json.dumps({"markdown": "edited", "version": doc["version"]})
        status, _ = request("POST", "/api/writing/test", body, {"Content-Type": "application/json"})
        assert status == 403
        headers = {
            "Content-Type": "application/json",
            "Origin": f"http://127.0.0.1:{server.server_port}",
            "X-Write-Token": doc["write_token"],
        }
        assert request("POST", "/api/writing/test", body, headers)[0] == 200
        assert (docs / "test.md").read_text() == "edited"
        assert request("POST", "/api/writing/test", body, headers)[0] == 409
        headers["Origin"] = "https://untrusted.example"
        assert request("POST", "/api/writing/test", body, headers)[0] == 403
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
