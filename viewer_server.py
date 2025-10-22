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


class BackroomsHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/config":
            # Serve Supabase config as JSON
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            config = {
                "supabaseUrl": os.getenv("SUPABASE_URL", ""),
                "supabaseKey": os.getenv("SUPABASE_KEY", "")
            }
            self.wfile.write(json.dumps(config).encode())
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
