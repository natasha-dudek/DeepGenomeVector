import asyncio, socket

import torch
import numpy as np
import pandas as pd
from genome_embeddings import util

DATA_FP = '/Users/natasha/Desktop/mcgill_postdoc/ncbi_genomes/genome_embeddings/data/'


# Keep data in numpy on the server side
train_data = torch.load(DATA_FP+"corrupted_train_07-17-20.pt").numpy()
# test_data = torch.load(DATA_FP+"corrupted_test_07-17-20.pt").numpy()
# data = np.concatenate([train_data, test_data], 0)
data = train_data
print(data.shape)
#df, cluster_names = util.load_data(DATA_FP, "kegg")
## To make predictions on (ROC + AUC)
#num_features = int(train_data.shape[1]/2)
#tensor_test_data = torch.tensor([i.numpy() for i in test_data]).float()
#corrupt_test_data = tensor_test_data[:,:num_features]
#target = tensor_test_data[:,num_features:].detach().numpy()
#
#print("loading genome_to_tax")
#genome_to_tax = np.load(DATA_FP+'genome_to_tax.npy',allow_pickle='TRUE').item()
#genome_idx_train = torch.load(DATA_FP+"genome_idx_train_07-17-20.pt")
#genome_idx_test = torch.load(DATA_FP+"genome_idx_test_07-17-20.pt")
#df_train_data = pd.DataFrame(train_data.numpy()) 


async def handle_client(reader, writer):
    # request should be a list of indices that client wants
#    request = (await reader.read(255)).decode('utf8')
    # request = await reader.read(255)
    request = await util.recv_msg_async(reader)
    idx = np.frombuffer(request, np.int)
    if len(idx) > 0:
        response = data[idx].tobytes()
    else: # means send the shape
        response = np.array(data.shape, dtype=np.float32).tobytes()
#    response = str(request) + 'bar'
#    writer.write(response.encode('utf8'))
    util.send_msg_asyncio(writer, response)
    # writer.write(response)
    await writer.drain()
    writer.close()

loop = asyncio.get_event_loop()
loop.create_task(asyncio.start_server(handle_client, 'localhost', 15555))
print('running')
loop.run_forever()

