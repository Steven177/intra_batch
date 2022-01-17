import torch
import torch.nn.functional as F
from torch import nn
from .utils import *

class FinetuningNetwork(nn.Module):
    def __init__(self, embed_dim, dev, finetuning_params, num_layers):
        super(FinetuningNetwork, self).__init__()
        
        num_classes = finetuning_params['classifier']['num_classes']
        layers = [FinetuningLayer(dev, finetuning_params, embed_dim) for _ in range(num_layers)]
        self.layers = Sequential(*layers)
        self.batchnorm = nn.BatchNorm1d(embed_dim)
        self.linear = nn.Linear(embed_dim, num_classes)
        """"
        layers[-2].bias.requires_grad_(False)
        layers[-2].apply(weights_init_kaiming)
        layers[-1].apply(weights_init_classifier)
        """
        
    def forward(self, feats):
        x, out = list(), list()
        
        for layer in self.layers:
            feats = layer(feats)
            out.append(feats)
        
        # See forward() of GnnReID for this
        feats = [feats[-1]]
        feats = [self.batchnorm(feats[-1])]
        x = [self.linear(feats[-1])]

        # Gnn also normalizes feats
        feats = [F.normalize(f, p=2, dim=1) for f in feats]
        return x, feats

class FinetuningLayer(torch.nn.Module):
    def __init__(self, dev, params: dict = None, embed_dim: int = 2048, d_hid=None):
        super(FinetuningLayer, self).__init__()
        self.dev = dev
        self.params = params
        self.num_classes = params['classifier']['num_classes']
        d_hid = 4 * embed_dim if d_hid is None else d_hid

        self.prenorm = params['finetuning']['prenorm']
        self.res1 = params['finetuning']['res1']
        self.res2 = params['finetuning']['res2']
        self.mlp = params['finetuning']['mlp']
        self.linear1 = nn.Linear(embed_dim, d_hid) if params['finetuning']['mlp'] else None
        self.linear2 = nn.Linear(d_hid, embed_dim) if params['finetuning']['mlp'] else None
        self.norm1 = LayerNorm(embed_dim) if params['finetuning']['norm1'] else None
        self.norm2 = LayerNorm(embed_dim) if params['finetuning']['norm2'] else None
        self.dropout = nn.Dropout(params['finetuning']['dropout_mlp'])
        self.dropout1 = nn.Dropout(params['finetuning']['dropout_1'])
        self.dropout2 = nn.Dropout(params['finetuning']['dropout_2'])

        self.act = F.relu

    def forward(self, feats):
        if self.prenorm:
            # Layer 1
            if self.norm1:
                feats2 = self.norm1(feats)
            else:
                feats2 = feats
            if self.mlp:
                feats2 = self.linear2(self.dropout(self.act(self.linear1(feats2))))
            
            feats2 = self.dropout1(feats2)
            if self.res1:
                feats = feats + feats2
            else:
                feats = feats2

            # Layer 2
            if self.norm2:
                feats2 = self.norm2(feats)
            else:
                feats2 = feats
            if self.mlp:
                feats2 = self.linear2(self.dropout(self.act(self.linear1(feats2))))
            
            feats2 = self.dropout2(feats2)
            if self.res2:
                feats = feats + feats2
            else:
                feats = feats2
        else:
            # Layer 1
            if self.mlp:
                feats2 = self.linear2(self.dropout(self.act(self.linear1(feats))))
            else:
                feats2 = feats
            feats2 = self.dropout1(feats2)
            if self.res1:
                feats = feats + feats2
            else:
                feats = feats2
            
            if self.norm1:
                feats = self.norm1(feats)
            # Layer 2
            if self.mlp:
                feats2 = self.linear2(self.dropout(self.act(self.linear1(feats))))
            else:
                feats2 = feats
            feats2 = self.dropout2(feats2)
            if self.res2:
                feats = feats + feats2
            else:
                feats = feats2
            if self.norm2:
                feats = self.norm2(feats)
        return feats
        