import torch
from torch import nn
import numpy as np
from collections import OrderedDict
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.poisson import Poisson
#Reference
#https://www.kdnuggets.com/2020/07/pytorch-lstm-text-generation-tutorial.html

class StrangeModel(nn.Module):
    def __init__( self, wordsize,embeddingDim,seqlen,batchsize,num_layers,hidden_size,bidirection=False,softmax=False):
        super(StrangeModel, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.lstm_size = hidden_size
        self.embedding_dim = embeddingDim
        self.num_layers = num_layers
        self.seqlen=seqlen
        self.batchsize=batchsize
        self.n_vocab = wordsize
        self.kernelSize=2
        self.mergeNumOfConvBlock=0
        self.bidirection=1 if (bidirection==False) else 2
        self.softmax=softmax
        self.embeddingSpace = nn.Embedding(
                num_embeddings=self.n_vocab,
                embedding_dim=self.embedding_dim,
            ).to(self.device)
        self.RNN1 = nn.LSTM(
                input_size=self.embedding_dim,
                hidden_size=self.lstm_size,
                num_layers=self.num_layers,
                batch_first=True,
                bidirectional=bidirection
            ).to(self.device)

        self.predict=nn.Conv1d(self.bidirection*self.lstm_size,
                                                       3,
                                                       kernel_size=2,
                                                       padding=0).to(self.device)
        self.predict2=nn.Conv1d(self.bidirection*self.lstm_size,
                                                       3,
                                                       kernel_size=2,
                                                       padding=0).to(self.device)
        layer_list=list()
        for i in range(self.seqlen):
            layer_list.append((f'DrStrangConv1D_{i}',nn.Conv1d(self.bidirection*self.lstm_size,
                                                               self.bidirection*self.lstm_size,
                                                               kernel_size=2)))
        layer_list=OrderedDict(layer_list)
        self.DrStrangeLayer1 = nn.Sequential(layer_list).to(self.device)
        self.DrStrangeLayer2 = nn.Sequential(layer_list).to(self.device)
        self.DrStrangeLayer3 = nn.Sequential(layer_list).to(self.device)
        self.DrStrangeLayer4 = nn.Sequential(layer_list).to(self.device)
        self.DrStrangeLayer5 = nn.Sequential(layer_list).to(self.device)


    def forward(self, sequence,state):
        sequence=torch.tensor(sequence).to(self.device)
        lstminput = self.embeddingSpace(sequence).reshape(self.batchsize,self.seqlen,self.embedding_dim) # Embedding Layer
        o,state = self.RNN1(lstminput,state)
        o=o.permute(0,2,1)
        o = torch.cat([o,o[:,:,0:1]],dim=-1)
        out = self.predict(o) # o.shape = N*L*1
        return out,o,state

    def approximate(self,input):
        approximate1 = self.DrStrangeLayer1(input)
        approximate2 = self.DrStrangeLayer2(input)
        approximate3 = self.DrStrangeLayer3(input)
        approximate4 = self.DrStrangeLayer4(input)
        approximate5 = self.DrStrangeLayer5(input)
        approximate = torch.concat([approximate1,approximate2,approximate3,approximate4,approximate5],dim=-1)
        out = torch.cat([approximate,approximate[:,:,0:1]],dim=-1)
        out = self.predict2(out)
        return out,approximate


    def forwardFuture(self,sequence,state):
        block=sequence.shape[1]//self.seqlen
        output=[]
        for i in range(0,block):
            futureBlock=torch.tensor(sequence[:,i*self.seqlen:(i*self.seqlen)+self.seqlen]).to(self.device)
            lstminput = self.embeddingSpace(futureBlock).reshape(self.batchsize,self.seqlen,self.embedding_dim) # Embedding Layer
            o,state = self.RNN1(lstminput,state)
            o = o.permute(0,2,1)
            o = o.detach()
            output.append(o)
        return output

    def validation(self,x,state):
        x=torch.tensor(x).to(self.device)
        lstminput1 = self.embeddingSpace(x).reshape(self.batchsize,self.seqlen,self.embedding_dim) # Embedding Layer
        o1,state1 = self.RNN1(lstminput1,state)
        o1 = o1.permute(0,2,1) # N,H,L
        o1 = torch.cat([o1,o1[:,:,0:1]],dim=-1)
        a1= self.DrStrangeLayer1(o1) # o.shape = N*L*1
        a2 = self.DrStrangeLayer2(o1) # o.shape = N*L*1
        a3 = self.DrStrangeLayer3(o1) # o.shape = N*L*1
        a4 = self.DrStrangeLayer4(o1) # o.shape = N*L*1
        a5 = self.DrStrangeLayer5(o1) # o.shape = N*L*1
        o1 = torch.concat([a1,a2,a3,a4,a5],dim=-1)
        o1 = torch.cat([o1,o1[:,:,0:1]],dim=-1)
        o1 = self.predict(o1)
        return o1,state1



    def init_state(self,double=False):
        batchsize=self.batchsize*2  if(double) else self.batchsize
        return (torch.zeros(self.bidirection*self.num_layers,batchsize, self.lstm_size).to(self.device),
                torch.zeros(self.bidirection*self.num_layers,batchsize, self.lstm_size).to(self.device))

