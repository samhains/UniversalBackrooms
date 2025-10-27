#!/usr/bin/env python3
"""
Simple HTTP server to serve the backrooms viewer and provide Supabase config.
Loads credentials from .env so they don't need to be hardcoded in HTML.

Usage:
    python viewer_server.py
    # Then open http://localhost:8000
"""

import os
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib import error, request

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Fallback: simple .env parser
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip().strip("'\""))

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_KEY", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_PROXY_ENABLED = bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY)


class BackroomsHandler(SimpleHTTPRequestHandler):
    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _proxy_backrooms(self):
        if not SUPABASE_PROXY_ENABLED:
            self._send_json({"error": "supabase proxy disabled"}, status=404)
            return

        target_url = f"{SUPABASE_URL}{self.path}"
        req = request.Request(target_url, method="GET")
        req.add_header("Accept", "application/json")
        req.add_header("apikey", SUPABASE_SERVICE_ROLE_KEY)
        req.add_header("Authorization", f"Bearer {SUPABASE_SERVICE_ROLE_KEY}")

        # Forward optional headers commonly used for pagination/counts.
        for header_name in ("Range", "Prefer"):
            header_value = self.headers.get(header_name)
            if header_value:
                req.add_header(header_name, header_value)

        try:
            with request.urlopen(req, timeout=30) as resp:
                data = resp.read()
                status = getattr(resp, "status", resp.getcode())
                self.send_response(status)
                content_type = resp.headers.get("Content-Type", "application/json")
                self.send_header("Content-Type", content_type)
                content_range = resp.headers.get("Content-Range")
                if content_range:
                    self.send_header("Content-Range", content_range)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        except error.HTTPError as exc:
            body = exc.read()
            status = exc.code
            if not body:
                body = json.dumps({"error": exc.reason}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            body = json.dumps({"error": str(exc)}).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/config":
            # Serve Supabase config as JSON
            if SUPABASE_PROXY_ENABLED:
                config = {
                    "supabaseUrl": "",
                    "supabaseKey": "",
                    "proxyEnabled": True,
                }
            else:
                config = {
                    "supabaseUrl": SUPABASE_URL,
                    "supabaseKey": SUPABASE_ANON_KEY,
                    "proxyEnabled": False,
                }
            self._send_json(config)
        elif SUPABASE_PROXY_ENABLED and self.path.startswith("/rest/v1/backrooms"):
            self._proxy_backrooms()
        elif self.path.startswith("/t/"):
            # Pretty URLs for transcripts: /t/filename
            # Serve t.html for all /t/* paths
            self.path = "/t.html"
            super().do_GET()
        else:
            # Serve static files
            super().do_GET()


def main():
    port = int(os.getenv("PORT", "8000"))
    server = HTTPServer(("", port), BackroomsHandler)
    print(f"🌀 Backrooms Viewer running at http://localhost:{port}")
    print(f"📂 Serving from: {Path.cwd()}")
    print(f"Press Ctrl+C to stop")
    server.serve_forever()


if __name__ == "__main__":
    main()
