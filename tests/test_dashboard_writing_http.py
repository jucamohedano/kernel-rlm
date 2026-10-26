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


def test_static_assets_have_content_type(tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)

    try:
        thread.start()
        for path, content_type in {
            "/": "text/html; charset=utf-8",
            "/app.js": "text/javascript; charset=utf-8",
            "/styles.css": "text/css; charset=utf-8",
            "/fonts/InterVariable.woff2": "font/woff2",
        }.items():
            conn = HTTPConnection("127.0.0.1", server.server_port)
            conn.request("GET", path)
            response = conn.getresponse()
            assert response.status == 200
            assert response.getheader("Content-Type") == content_type
            response.read()
            conn.close()
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_literature_assets_are_allowlisted_and_do_not_expose_manifest(tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for name, content_type, signature in (
            ("research-map.svg", "image/svg+xml; charset=utf-8", b"<svg"),
            ("train-evaluate.svg", "image/svg+xml; charset=utf-8", b"<svg"),
            ("rlm-figure-2.png", "image/png", b"\x89PNG"),
            ("kernelbench-figure-1.png", "image/png", b"\x89PNG"),
        ):
            conn = HTTPConnection(*server.server_address)
            conn.request("GET", f"/literature/{name}")
            response = conn.getresponse()
            assert response.status == 200
            assert response.getheader("Content-Type") == content_type
            assert signature in response.read()
            conn.close()
        for path in (
            "/literature/sources.json",
            "/literature/unknown.svg",
            "/literature/harness-context.svg",
            "/literature/../server.py",
            "/literature/%2e%2e/server.py",
            "/literature/%72esearch-map.svg",
            "/literature/research-map.svg/extra",
            "/literature//research-map.svg",
        ):
            conn = HTTPConnection(*server.server_address)
            conn.request("GET", path)
            response = conn.getresponse()
            response.read()
            assert response.status == 404
            conn.close()
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
