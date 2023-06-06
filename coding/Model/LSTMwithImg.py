import torch
from torch import nn
import numpy as np
from collections import OrderedDict
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.poisson import Poisson
#Reference
#https://www.kdnuggets.com/2020/07/pytorch-lstm-text-generation-tutorial.html

class LSTMwithImg(nn.Module):
    def __init__( self, wordsize,embeddingDim,seqlen,batchsize,num_layers,hidden_size,bidirection=False,additionalloss=0):
        super(LSTMwithImg, self).__init__()
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
        self.bidirection=1 if (bidirection==False) else 2
        self.additionalloss=additionalloss
        self.embeddingSpace = nn.Embedding(
                num_embeddings=self.n_vocab,
                embedding_dim=self.embedding_dim,
            ).to(self.device)
        #self.embeddingSpace2=torch.nn.Embedding(num_embeddings=self.n_vocab,embedding_dim=self.embedding_dim).to(self.device)
        self.predict = torch.nn.Linear(self.bidirection*self.lstm_size,3).to(self.device)
        self.RNN = nn.LSTM(
                input_size=self.embedding_dim,
                hidden_size=self.lstm_size,
                num_layers=self.num_layers,
                batch_first=True,
                bidirectional=True if(self.bidirection==2) else False
            ).to(self.device)
        if(self.additionalloss==1):
            self.generator =  torch.nn.Sequential(
                torch.nn.ConvTranspose2d(16,512,3,stride=1),        #   (4-1)*1+(3-1)+1 = 3+2+1 = 6
                torch.nn.ReLU(), #(6,5),
                torch.nn.ConvTranspose2d(512,256,2,stride=2),        #  (6-1)*2+(2-1)+1 = 10+1+1 = 12
                torch.nn.ReLU(), #(6,5)
                torch.nn.ConvTranspose2d(256,64,2,stride=2),        #   (12-1)*2+(2-1)+1 = 22+1+1 = 24
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(64,1,2,stride=2),        #  (24-1)*2+(2-1)+1 = 46+1+1 = 48
            ).to(self.device)


    def forward(self, x, state,nidx,pidx):
        x=torch.tensor(x).to(self.device)
        h_state=state[0].to(self.device)
        c_state=state[1].to(self.device)
        dream=None
        btwGroupSimilarity=None
        lstminput = self.embeddingSpace(x).reshape(self.batchsize,self.seqlen,self.embedding_dim) # Embedding Layer
        o,state = self.RNN(lstminput,(h_state,c_state))
        cstate=state[1].view(self.num_layers, self.bidirection , self.batchsize, self.lstm_size)
        o = self.predict(o) # o.sha= N*L*1
        o=o.permute(0,2,1)
        if(self.additionalloss==1):
            cstate = cstate[-1,0,:,:].reshape(self.batchsize,16,4,4)
            dream=self.generator(cstate)
        if(self.additionalloss==2):
            minlen =len(pidx) if (len(nidx)>len(pidx)) else len(nidx)
            posCstate= cstate[pidx,:]
            negCstate= cstate[nidx,:]
            btwgroup = torch.triu(torch.mm(negCstate[:minlen,:],posCstate[:minlen,:].T),diagonal=1)
            btwgridx,btwgcidx=torch.triu_indices(btwgroup.shape[0],btwgroup.shape[1],offset=1)
            btwgroup=btwgroup[btwgridx,btwgcidx]
            btwGroupSimilarity=torch.mean(btwgroup)
        return o,dream,btwGroupSimilarity,state


    def init_state(self,double=False):
        batchsize=self.batchsize*2  if(double) else self.batchsize
        return (torch.zeros(self.bidirection*self.num_layers,batchsize, self.lstm_size),
                torch.zeros(self.bidirection*self.num_layers,batchsize, self.lstm_size))

