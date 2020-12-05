import random
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.inits import reset
from sklearn.linear_model import LogisticRegression


EPS = 1e-15

class SugbCon(torch.nn.Module):

    def __init__(self, hidden_channels, encoder, pool, scorer):
        super(SugbCon, self).__init__()
        self.encoder = encoder
        self.hidden_channels = hidden_channels
        self.pool = pool
        self.scorer = scorer
        self.marginloss = nn.MarginRankingLoss(0.5)
        self.sigmoid = nn.Sigmoid()
        self.reset_parameters()
        
    def reset_parameters(self):
        reset(self.scorer)
        reset(self.encoder)
        reset(self.pool)
        
    def forward(self, x, edge_index, batch=None, index=None):
        r""" Return node and subgraph representations of each node before and after being shuffled """
        hidden = self.encoder(x, edge_index)
        if index is None:
            return hidden
        
        z = hidden[index]
        summary = self.pool(hidden, edge_index, batch)
        return z, summary
    
    
    def loss(self, hidden1, summary1):
        r"""Computes the margin objective."""

        shuf_index = torch.randperm(summary1.size(0))

        hidden2 = hidden1[shuf_index]
        summary2 = summary1[shuf_index]
        
        logits_aa = torch.sigmoid(torch.sum(hidden1 * summary1, dim = -1))
        logits_bb = torch.sigmoid(torch.sum(hidden2 * summary2, dim = -1))
        logits_ab = torch.sigmoid(torch.sum(hidden1 * summary2, dim = -1))
        logits_ba = torch.sigmoid(torch.sum(hidden2 * summary1, dim = -1))
        
        TotalLoss = 0.0
        ones = torch.ones(logits_aa.size(0)).cuda(logits_aa.device)
        TotalLoss += self.marginloss(logits_aa, logits_ba, ones)
        TotalLoss += self.marginloss(logits_bb, logits_ab, ones)
        
        return TotalLoss


    def test(self, train_z, train_y, val_z, val_y, test_z, test_y, solver='lbfgs',
             multi_class='auto', *args, **kwargs):
        r"""Evaluates latent space quality via a logistic regression downstream task."""
        clf = LogisticRegression(solver=solver, multi_class=multi_class, *args,
                                 **kwargs).fit(train_z.detach().cpu().numpy(),
                                               train_y.detach().cpu().numpy())
        val_acc = clf.score(val_z.detach().cpu().numpy(), val_y.detach().cpu().numpy())
        test_acc = clf.score(test_z.detach().cpu().numpy(), test_y.detach().cpu().numpy())
        return val_acc, test_acc