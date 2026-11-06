# Dev server: like `python3 -m http.server`, but tells the browser never to
# cache — otherwise Chrome keeps running stale ES modules after code changes.
import http.server, sys

class NoCache(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8741
http.server.ThreadingHTTPServer(('', port), NoCache).serve_forever()
