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
	def __init__(self, idx):
		self.idx = idx
#		self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#		self.server.connect(('localhost', 15555))
	
	def __getitem__(self, key):
		train_data = get_train_data(np.array([self.idx[key]]))
		_, num_cols = train_data.shape
		num_features = int(num_cols/2)
		x = train_data[0, :num_features]
		y = train_data[0, num_features:]
		return(torch.Tensor(x), torch.Tensor(y))
	
	def __len__(self):
		return len(self.idx)

def _get_data_raw(idx, server=None):
	
	while True:
		try:
			if server is None:
				server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
				server.settimeout(1000000)
				server.connect(('localhost', 15555))
			break
		except TimeoutError:
			pass
			
					
	#server.send(request.encode('utf8'))
	#response = server.recv(255).decode('utf8')
	#import ipdb; ipdb.set_trace()
	util.send_msg(server, idx.tobytes())
	# server.send(idx.tobytes())
	#response = server.recv(255)
	response = util.recv_msg(server)
	out = np.frombuffer(response, np.float32)
	return out


def _get_data(idx, server=None):
	out = _get_data_raw(idx, server)
	n_rows = len(idx)
	n_cols = int(len(out) / n_rows)
	out = np.reshape(out, (n_rows, n_cols))
	return out


def get_train_data(idx, server=None):
	return _get_data(idx, server)

def get_train_data_size(server=None):
	out = _get_data_raw(np.array([]), server).astype(np.int)
	return tuple(out)
	
#def get_test_data():
#	# todo: figure out this number
#	offset = 100
#	idx = idx + offset
#	return _get_data(idx)
	
if __name__ == '__main__':
	# example usage
	idx = np.array([1])
#	
#	t0 = time.time()
#	print(get_train_data(idx))
#	print('took', time.time() - t0)
#	print(get_train_data_size())
#	print('train data', get_train_data(idx).shape)
#	print('test data', get_test_data(idx).shape)
#	wait_till_up()

	ds = CachedDataset([1])
	print(10)
	print(ds[0])
	time.sleep(1)
	print(11)
	print(ds[0])
	#import ipdb; ipdb.set_trace()
	time.sleep(1)
	print(12)
	ds[0]
	time.sleep(1)
	print(13)
