import torch
from torch import nn
import numpy as np
from collections import OrderedDict

#Reference
#https://www.kdnuggets.com/2020/07/pytorch-lstm-text-generation-tutorial.html

class PredictLSTM(nn.Module):
    def __init__( self, wordsize,embeddingDim,seqlen,maxMerge,batchsize,num_layers,hidden_size,outdim=1,selection=False):
        super(PredictLSTM, self).__init__()
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
        self.selection=selection
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
        ## calculate the conv block size of Competition
        if(self.selection):
            nextlayer=self.seqlen
            while nextlayer!=self.outDim:
                nextlayer = np.floor(((nextlayer-(self.predict_kernelSize-1)-1)/2)+1)
                if(nextlayer>=self.outDim):
                    self.competition_numOfConvBlock+=1
                if(nextlayer==self.outDim):
                    break
            print(f'Compleiton Conv1D Block size:{self.competition_numOfConvBlock}')
        ## calculate the conv block size of prediction
        nextlayer=self.lstm_size
        while nextlayer!=self.outDim:
            nextlayer = np.floor(((nextlayer-(self.predict_kernelSize-1)-1)/2)+1)
            if(nextlayer>=self.outDim):
                self.prediction_numOfConvBlock+=1
            if(nextlayer==self.outDim):
                break
        print(f'Predict Conv1D Block size:{self.prediction_numOfConvBlock}')
        # Preparing Embedding Space 1 and 2
        for i in range(self.batchsize):
            self.embeddingSpace.append(
                nn.Embedding(
                    num_embeddings=self.n_vocab,
                    embedding_dim=self.embedding_dim,
                ).to(self.device)
            )
            layer_list=list()
            for j in range(self.numOfConvBlock):
                layer_list.append((f'Conv1D_{j}_1',nn.Conv1d(self.embedding_dim,self.embedding_dim,kernel_size=self.kernelSize)))
                #layer_list.append((f'Conv1D_{i}_2',nn.Conv1d(self.embedding_dim,self.embedding_dim,kernel_size=self.kernelSize,padding=1,dilation=2)))
                layer_list.append((f'Tanh_{j}',nn.Tanh()))
            layerDict = OrderedDict(layer_list)
            self.embeddingSpace2.append(torch.nn.Sequential(layerDict).to(self.device))
        if(self.selection):
            # Create Conv1D Competition Block
            layer_list=list()
            previous_channel=self.embedding_dim
            for i in range(self.competition_numOfConvBlock):
                if(previous_channel<0):
                    previous_channel=1
                layer_list.append((f'Competition_Conv1D_{i}_1',nn.Conv1d(previous_channel,
                                                            int(np.ceil(previous_channel/2)),
                                                            kernel_size=self.predict_kernelSize,stride=2)))
                #layer_list.append((f'Conv1D_{i}_2',nn.Conv1d(self.embedding_dim,self.embedding_dim,kernel_size=self.kernelSize,padding=1,dilation=2)))
                layer_list.append((f'Competition_Tanh_{i}',nn.Tanh()))
                previous_channel=int(np.ceil(previous_channel/2))
            layer_list.append(('Competition_Flatten',nn.Flatten()))
            layerDict = OrderedDict(layer_list)
            self.competitionConv1D = torch.nn.Sequential(layerDict).to(self.device)
            self.competitionlayer = nn.Sequential(OrderedDict([
                                    ('linear', nn.Linear(previous_channel+1,1)),
                                    ('softmax', nn.Softmax(dim=0))
                                ])).to(self.device)

        # Create Conv1D prediction Block
        layer_list=list()
        previous_channel=self.num_layers
        for i in range(self.prediction_numOfConvBlock):
            if(previous_channel<0):
                previous_channel=1
            layer_list.append((f'Predict_Conv1D_{i}_1',nn.Conv1d(previous_channel,
                                                        int(np.ceil(previous_channel/2)),
                                                        kernel_size=self.predict_kernelSize,stride=2)))
            #layer_list.append((f'Conv1D_{i}_2',nn.Conv1d(self.embedding_dim,self.embedding_dim,kernel_size=self.kernelSize,padding=1,dilation=2)))
            layer_list.append((f'Predict_Tanh_{i}',nn.Tanh()))
            previous_channel=int(np.ceil(previous_channel/2))
        layer_list.append(('Predict_Flatten',nn.Flatten()))
        layerDict = OrderedDict(layer_list)
        self.predictConv1D = torch.nn.Sequential(layerDict).to(self.device)
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            batch_first=True,
        ).to(self.device)
        #self.predictAndCompetition=nn.Linear(self.batchsize,1,bias=False).to(self.device)


    def forward(self, x, h_state,c_state,pos):
        x=x.to(self.device)
        h_state=h_state.to(self.device)
        c_state=c_state.to(self.device)
        lstminput=torch.zeros((self.batchsize,self.seqlen,self.embedding_dim,1)).to(self.device)
        for i in range(self.batchsize):
            # For each Token Strategy
            # Temp Tensor to hold input to Conv1D , Max Merge Token == 4 ,if no merge (ie. 1 token), the max Batch Size of Conv1D is == Seqlen
            tmp=torch.zeros((self.seqlen,self.embedding_dim,self.maxMerge),dtype=torch.float).to(self.device)
            if(i==1):
                token=torch.randint(2,3,(self.seqlen,1))
            if(i==2):
                token=torch.randint(3,4,(self.seqlen,1))
            if(i==3):
                token=torch.randint(1,self.maxMerge,(self.seqlen,1))
            if(i==4):
                token=torch.randint(2,self.maxMerge,(self.seqlen,1))
            currentidx=self.seqlen
            j=self.seqlen-1
            # Merge By Con1D in backward direction => LSTM
            if(i>0):
                embed = self.embeddingSpace[i](x) # Embedding Layer
                embed=embed.permute(1,0) # (embeddingDim,seqlen)
                while(currentidx>0):
                    start=currentidx-token[j]
                    if(start<0):
                        start=0
                    mergeToken = embed[:, start:currentidx] # Get # embedded vectors accroding to Strategy to merge
                    if(currentidx-start<4): # Padding Zero if merge token < 4
                        remain=torch.zeros((self.embedding_dim,4-(currentidx-start))).to(self.device)
                        mergeToken=torch.concat([mergeToken,remain],dim=1)
                    tmp[j,:,:]=mergeToken
                    currentidx-=token[j]
                    j-=1
                embed=self.embeddingSpace2[i](tmp)
            else:
                embed = self.embeddingSpace[i](x) # Embedding Layer
            lstminput[i]=embed
        lstminput=lstminput.reshape(self.batchsize,self.seqlen,self.embedding_dim)
        if(self.selection):
            pos = torch.full((self.batchsize,1),pos).to(self.device)
            competition = torch.swapaxes(lstminput,1,2) # (batchsiz, embedding, seqlen) , layer as channel ?
            competition = self.competitionConv1D(competition) # (batchsize,1,1)
            competition = torch.concat([competition,pos],dim=1)
            competition = self.competitionlayer(competition)

        o, (h, c) = self.lstm(lstminput)
        out = torch.swapaxes(c,0,1) # (batchsiz, num_layer, h_out) , layer as channel ?
        out = self.predictConv1D(out)
        if(self.selection):
            out = torch.nn.functional.linear(out,weight=competition.T)
            self.competition=competition.T
        return out,(h,c)


    def init_state(self, batchSize):
        return (torch.zeros(self.num_layers, batchSize, self.lstm_size),
                torch.zeros(self.num_layers, batchSize, self.lstm_size))

