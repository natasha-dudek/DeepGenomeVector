import socket
import numpy as np
from genome_embeddings import util

def _get_data(idx):
	server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	server.connect(('localhost', 15555))
	#server.send(request.encode('utf8'))
	#response = server.recv(255).decode('utf8')
	util.send_msg(server, idx.tobytes())
	# server.send(idx.tobytes())
	#response = server.recv(255)
	response = util.recv_msg(server)
	print(len(response))
	out = np.frombuffer(response, np.float32)
	n_rows = len(idx)
	n_cols = int(len(out) / n_rows)
	out = np.reshape(out, (n_rows, n_cols))
	return out


def get_train_data(idx):
	return _get_data(idx)
	
def get_test_data(idx):
	# todo: figure out this number
	offset = 100
	idx = idx + offset
	return _get_data(idx)
	
if __name__ == '__main__':
	# example usage
	idx = np.array([1, 4, 5])
	print('train data', get_train_data(idx).shape)
	print('test data', get_test_data(idx).shape)
	