import torch
from torch import nn
import numpy as np
from collections import OrderedDict
#Reference
#https://www.kdnuggets.com/2020/07/pytorch-lstm-text-generation-tutorial.html

class PredictLSTMNormal(nn.Module):
    def __init__( self, wordsize,embeddingDim,seqlen,maxMerge,batchsize,num_layers,hidden_size,outdim=1,bidirection=False):
        super(PredictLSTMNormal, self).__init__()
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
        self.maxMerge=maxMerge
        self.predict_kernelSize=2
        self.outDim=outdim
        self.embeddingSpace=[]
        self.embeddingSpace2=[]
        self.numOfConvBlock=0
        self.prediction_numOfConvBlock=0
        self.competition_numOfConvBlock=0
        self.competition=None
        self.bidirection= 1 if(bidirection==False) else 2
        assert self.maxMerge<self.seqlen , "max merge size can't > seqlen"
        ## calculate the conv block size of Merge Embedding
        nextlayer=self.maxMerge
        while nextlayer!=1:
            nextlayer= np.floor(((nextlayer-(self.kernelSize-1)-1)/1)+1)
            if(nextlayer>=1):
                self.numOfConvBlock+=1
            if(nextlayer==1):
                break
        print(f'Embedding Space 2 Conv1D block size:{self.numOfConvBlock}')
        ## calculate the conv block size of prediction
        nextlayer=self.seqlen
        while nextlayer!=self.outDim:
            nextlayer = np.floor(((nextlayer-(self.predict_kernelSize-1)-1)/1)+1)
            if(nextlayer>=self.outDim):
                self.prediction_numOfConvBlock+=1
            if(nextlayer==self.outDim):
                break
        print(f'Predict Conv1D Block size:{self.prediction_numOfConvBlock}')
        # Preparing Embedding Space 1 and 2
        for i in range(1):
            self.embeddingSpace.append(
                nn.Embedding(
                    num_embeddings=self.n_vocab,
                    embedding_dim=self.embedding_dim,
                ).to(self.device)
            )

        # Create Conv1D prediction Block
        #layer_list=list()
        #for i in range(self.prediction_numOfConvBlock):
        #    layer_list.append((f'Predict_Conv1D_{i}_1',nn.Conv1d(self.lstm_size,
        #                                                self.lstm_size,
        #                                                kernel_size=self.predict_kernelSize)))
            #layer_list.append((f'Conv1D_{i}_2',nn.Conv1d(self.embedding_dim,self.embedding_dim,kernel_size=self.kernelSize,padding=1,dilation=2)))
        #    layer_list.append((f'Predict_ReLU_{i}',nn.ReLU()))
        #layer_list.append(('Predict_flatten',nn.Flatten()))
        #layer_list.append(('Predict_linear',nn.Linear(self.lstm_size,1)))
        #layerDict = OrderedDict(layer_list)
        #self.predict = torch.nn.Sequential(layerDict).to(self.device)
        self.predict = torch.nn.Linear(self.bidirection*self.lstm_size,1).to(self.device)
        self.flatten = torch.nn.Flatten()
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=bidirection
        ).to(self.device)
        #self.predictAndCompetition=nn.Linear(self.batchsize,1,bias=False).to(self.device)


    def forward(self, x, h_state,c_state,switch=0):
        x=torch.tensor(x).to(self.device)
        h_state=h_state.to(self.device)
        c_state=c_state.to(self.device)
        lstminput = self.embeddingSpace[0](x).reshape(self.batchsize,self.seqlen,self.embedding_dim,1) # Embedding Layer
        lstminput=lstminput.reshape(self.batchsize,self.seqlen,self.embedding_dim)
        out, (h, c) = self.lstm(lstminput, (h_state,c_state) ) # o.shape = N*L*Hout ,don't squeeze for future use in batchsize
        #out, (_, _) = self.lstm2(out) # o.shape = N*L*Hout ,don't squeeze for future use in batchsize
        lasthiddendlayer = h.reshape(self.num_layers,self.bidirection,self.batchsize,-1)
        lasthiddendlayer = lasthiddendlayer[self.num_layers-1]
        out = torch.permute(lasthiddendlayer,(1,0,2)) # (batchsiz, num_layer, h_out) , layer as channel ?
        out = self.flatten(out)
        out = self.predict(out) # o.shape = N*L*1
        #out = torch.permute(c,(1,2,0)) # (batchsiz, num_layer, h_out) , layer as channel ?
        #out = self.predictConv1D(out)
        #out = torch.concat([out,pos],dim=1)
        #out = self.predictlinear(out)
        return out


    def init_state(self):
        return (torch.zeros(self.bidirection*self.num_layers, self.batchsize, self.lstm_size),
                torch.zeros(self.bidirection*self.num_layers, self.batchsize, self.lstm_size))

