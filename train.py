#CUDA_VISIBLE_DEVICES=0 /remote-home/mzhong/anaconda3/bin/python train.py --subgraph_size 10 --batch_size 200 
import argparse, os
import math
import torch
import random
import numpy as np
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

from utils_mp import Subgraph, preprocess
from subgcon import SugbCon
from model import Encoder, Scorer, Pool

    
def get_parser():
    parser = argparse.ArgumentParser(description='Description: Script to run our model.')
    parser.add_argument('--dataset',help='Cora, Citeseer or Pubmed. Default=Cora', default='Cora')
    parser.add_argument('--batch_size', type=int, help='batch size', default=500)
    parser.add_argument('--subgraph_size', type=int, help='subgraph size', default=20)
    parser.add_argument('--n_order', type=int, help='order of neighbor nodes', default=10)
    parser.add_argument('--hidden_size', type=int, help='hidden size', default=1024)
    return parser
    
if __name__ == '__main__':
    parser = get_parser()
    try:
        args = parser.parse_args()
    except:
        exit()
    print (args)
    
    # Loading data
    data = Planetoid(root='./dataset/' + args.dataset, name=args.dataset)
    num_classes = data.num_classes
    data = data[0]
    num_node = data.x.size(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setting up the subgraph extractor
    ppr_path = './subgraph/' + args.dataset
    subgraph = Subgraph(data.x, data.edge_index, ppr_path, args.subgraph_size, args.n_order)
    subgraph.build()
    
    # Setting up the model and optimizer
    model = SugbCon(
        hidden_channels=args.hidden_size, encoder=Encoder(data.num_features, args.hidden_size),
        pool=Pool(in_channels=args.hidden_size),
        scorer=Scorer(args.hidden_size)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        
    def train(epoch):
        # Model training
        model.train()
        optimizer.zero_grad()
        sample_idx = random.sample(range(data.x.size(0)), args.batch_size)
        batch, index = subgraph.search(sample_idx)
        z, summary = model(batch.x.cuda(), batch.edge_index.cuda(), batch.batch.cuda(), index.cuda())
        
        loss = model.loss(z, summary)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    
    def get_all_node_emb(model, mask):
        # Obtain central node embs from subgraphs 
        node_list = np.arange(0, num_node, 1)[mask]
        list_size = node_list.size
        z = torch.Tensor(list_size, args.hidden_size).cuda() 
        group_nb = math.ceil(list_size/args.batch_size)
        for i in range(group_nb):
            maxx = min(list_size, (i + 1) * args.batch_size)
            minn = i * args.batch_size 
            batch, index = subgraph.search(node_list[minn:maxx])
            node, _ = model(batch.x.cuda(), batch.edge_index.cuda(), batch.batch.cuda(), index.cuda())
            z[minn:maxx] = node
        return z
    
    
    def test(model):
        # Model testing
        model.eval()
        with torch.no_grad():
            train_z = get_all_node_emb(model, data.train_mask)
            val_z = get_all_node_emb(model, data.val_mask)
            test_z = get_all_node_emb(model, data.test_mask)
        
        train_y = data.y[data.train_mask]
        val_y = data.y[data.val_mask]
        test_y = data.y[data.test_mask]
        val_acc, test_acc = model.test(train_z, train_y, val_z, val_y, test_z, test_y)
        print('val_acc = {} test_acc = {}'.format(val_acc, test_acc))
        return val_acc, test_acc

    
    print('Start training !!!')
    best_acc_from_val = 0
    best_val_acc = 0
    best_ts_acc = 0
    max_val = 0
    stop_cnt = 0
    patience = 20
    
    for epoch in range(10000):
        loss = train(epoch)
        print('epoch = {}, loss = {}'.format(epoch, loss))
        val_acc, test_acc = test(model) 
        best_val_acc = max(best_val_acc, val_acc)
        best_ts_acc = max(best_ts_acc, test_acc)
        if val_acc >= max_val:
            max_val = val_acc
            best_acc_from_val = test_acc
            stop_cnt = 0
        else:
            stop_cnt += 1
        print('best_val_acc = {}, best_test_acc = {}'.format(best_val_acc, best_ts_acc))
        if stop_cnt >= patience:
            break
    print('best_acc_from_val = {}'.format(best_acc_from_val))

