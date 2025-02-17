import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

class MyRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b"Hello, World!")

class MyServer:
    def __init__(self, port):
        self.server = HTTPServer(('localhost', port), MyRequestHandler)

    def run(self):
        print(f'Starting server on port {self.server.server_port}...')
        self.server.serve_forever()

# 创建一个服务器实例
server = MyServer(8080)

# 在单独的线程中运行服务器
server_thread = threading.Thread(target=server.run)
server_thread.start()

# 在此处添加其他任务，它们将在服务器运行时并行执行
print("Server is running in the background...")