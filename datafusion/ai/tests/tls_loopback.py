# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

import http.server
import pathlib
import ssl
import subprocess
import sys
import tempfile


class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        assert self.path == "/v1/chat/completions"
        self.rfile.read(int(self.headers["Content-Length"]))
        body = b'{"choices":[{"message":{"content":"trusted"}}]}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


with tempfile.TemporaryDirectory() as directory:
    path = pathlib.Path(directory)
    (path / "leaf.ext").write_text("subjectAltName=IP:127.0.0.1\n")
    common = ["-newkey", "rsa:2048", "-nodes"]
    for args in [
        ["req", "-x509", *common, "-subj", "/CN=Test CA", "-days", "1",
         "-addext", "basicConstraints=critical,CA:TRUE", "-keyout", "ca.key", "-out", "ca.pem"],
        ["req", *common, "-subj", "/CN=localhost", "-keyout", "leaf.key", "-out", "leaf.csr"],
        ["x509", "-req", "-in", "leaf.csr", "-CA", "ca.pem", "-CAkey", "ca.key",
         "-CAcreateserial", "-extfile", "leaf.ext", "-days", "1", "-out", "leaf.pem"],
    ]:
        subprocess.run(["openssl", *args], cwd=path, check=True, capture_output=True, timeout=5)

    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls.load_cert_chain(path / "leaf.pem", path / "leaf.key")
    with http.server.HTTPServer(("127.0.0.1", 0), Handler) as server:
        server.timeout = 8
        server.socket = tls.wrap_socket(server.socket, server_side=True)
        pem = (path / "ca.pem").read_bytes()
        print(f"{server.server_port} {len(pem)}", flush=True)
        sys.stdout.buffer.write(pem)
        sys.stdout.buffer.flush()
        for _ in range(2):
            server.handle_request()
