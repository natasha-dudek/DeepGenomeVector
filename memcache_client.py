import socket
import time

import numpy as np
import torch

from genome_embeddings import util

def is_up():
	try:
		get_train_data_size()
	except ConnectionRefusedError:
		return False
	return True
	
def wait_till_up(ping_time_s=5):
	while not is_up():
		time.sleep(ping_time_s)


class CachedDataset(torch.utils.data.Dataset):
	def __getitem__(self, key):
		print(key)
		return get_train_data(np.array([key]))
	
	def __len__(self):
		l, f = get_train_data_size()
		return l

def _get_data_raw(idx):
	server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	server.connect(('localhost', 15555))
	#server.send(request.encode('utf8'))
	#response = server.recv(255).decode('utf8')
	util.send_msg(server, idx.tobytes())
	# server.send(idx.tobytes())
	#response = server.recv(255)
	response = util.recv_msg(server)
	out = np.frombuffer(response, np.float32)
	return out


def _get_data(idx):
	out = _get_data_raw(idx)
	n_rows = len(idx)
	n_cols = int(len(out) / n_rows)
	out = np.reshape(out, (n_rows, n_cols))
	return out


def get_train_data(idx):
	return _get_data(idx)

def get_train_data_size():
	out = _get_data_raw(np.array([])).astype(np.int)
	return tuple(out)
	
#def get_test_data():
#	# todo: figure out this number
#	offset = 100
#	idx = idx + offset
#	return _get_data(idx)
	
if __name__ == '__main__':
	# example usage
#	idx = np.array([1])
#	
#	t0 = time.time()
#	get_train_data(idx)
#	print('took', time.time() - t0)
	
#	print('train data', get_train_data(idx).shape)
#	print('test data', get_test_data(idx).shape)
	wait_till_up()