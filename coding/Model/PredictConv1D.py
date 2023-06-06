import torch
from torch import nn
import math
from collections import OrderedDict

#Reference
#https://www.kdnuggets.com/2020/07/pytorch-lstm-text-generation-tutorial.html

class NextWordConv1D(nn.Module):
    def __init__( self, wordsize,numberOfCNN,embeddingDim,seqlen,batchsize):
        super(NextWordConv1D, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.kernelSize = 3
        self.embedding_dim = embeddingDim
        self.seqlen=seqlen
        self.batchsize=batchsize
        self.finallen=self.seqlen
        self.n_vocab = wordsize
        layer_list = list()
        for i in range(numberOfCNN):
            self.finallen= math.floor(((self.finallen-(self.kernelSize-1))/self.kernelSize)+1)
            assert self.finallen>0,f"CNN too deep for seqlen:{self.seqlen}"
            layer_list.append((f'Conv1D{i}_1',nn.Conv1d(self.embedding_dim,self.embedding_dim,kernel_size=self.kernelSize,padding=1)))
            layer_list.append((f'Conv1D{i}_2',nn.Conv1d(self.embedding_dim,self.embedding_dim,kernel_size=self.kernelSize,padding=1)))
            layer_list.append((f'MaxPool{i}',nn.MaxPool1d(self.kernelSize)))
        print(f'CNN: Conv1D Block:{numberOfCNN}, finalDim={self.finallen}',flush=True)
        layer_list.append(('ReLU',nn.ReLU()))
        layer_list.append(('Flatten',nn.Flatten()))
        layer_list.append(('Linear',nn.Linear(self.embedding_dim*self.finallen, self.n_vocab)))
        layer_list.append(('SoftMax',nn.Softmax(dim=1)))
        self.embedding=nn.Embedding(num_embeddings=self.n_vocab,embedding_dim=self.embedding_dim).to(self.device)
        layerDict = OrderedDict(layer_list)
        self.CNN=torch.nn.Sequential(layerDict).to(self.device)

    def forward(self, x):
        x=x.to(self.device)
        x=self.embedding(x)
        x=x.reshape(self.batchsize,self.seqlen,-1)
        x=x.permute(0,2,1)
        out=self.CNN(x)
        return out

