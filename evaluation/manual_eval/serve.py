#!/usr/bin/env python3
"""Static server for the scoring / live-compare pages, with an /api reverse proxy.

Serves _serve/ and forwards anything under /api/ to the HTX-3D backend on
127.0.0.1:8000. Proxying (rather than calling the backend directly from the
browser) keeps everything same-origin, so no CORS change and no backend restart
is needed.

    python3 serve.py [port]
"""

import http.server
import shutil
import socketserver
import sys
import urllib.error
import urllib.request
from pathlib import Path

BACKEND = "http://127.0.0.1:8000"
ROOT = Path(__file__).resolve().parent / "_serve"
TIMEOUT = 900  # generation can be slow; don't cut the poll off


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def log_message(self, fmt, *args):
        if "/api/" in (self.path or ""):
            super().log_message(fmt, *args)

    def do_GET(self):
        if self.path.startswith("/api/"):
            return self._proxy("GET")
        return super().do_GET()

    def do_HEAD(self):
        if self.path.startswith("/api/"):
            return self._proxy("HEAD")
        return super().do_HEAD()

    def do_POST(self):
        if self.path.startswith("/api/"):
            return self._proxy("POST")
        self.send_error(405, "POST only supported under /api/")

    def do_DELETE(self):
        if self.path.startswith("/api/"):
            return self._proxy("DELETE")
        self.send_error(405)

    def _proxy(self, method: str):
        body = None
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            body = self.rfile.read(length)

        req = urllib.request.Request(BACKEND + self.path, data=body, method=method)
        for h in ("Content-Type", "Accept", "Range"):
            v = self.headers.get(h)
            if v:
                req.add_header(h, v)

        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                self.send_response(r.status)
                for k, v in r.headers.items():
                    if k.lower() in ("transfer-encoding", "connection", "content-encoding"):
                        continue
                    self.send_header(k, v)
                self.end_headers()
                if method != "HEAD":
                    shutil.copyfileobj(r, self.wfile)
        except urllib.error.HTTPError as e:
            payload = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", e.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except Exception as e:                     # backend down / timeout
            msg = f'{{"detail":"proxy error: {type(e).__name__}: {e}"}}'.encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8899
    with Server(("0.0.0.0", port), Handler) as httpd:
        print(f"serving {ROOT} on :{port}  (/api -> {BACKEND})", flush=True)
        httpd.serve_forever()
