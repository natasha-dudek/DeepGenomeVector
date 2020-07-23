import socket
import numpy as np
from genome_embeddings import util

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.connect(('localhost', 15555))
idx = np.arange(4)
#server.send(request.encode('utf8'))
#response = server.recv(255).decode('utf8')
util.send_msg(server, idx.tobytes())
# server.send(idx.tobytes())
#response = server.recv(255)
response = util.recv_msg(server)
print(len(response))
out = np.frombuffer(response, np.float32)
print(out)