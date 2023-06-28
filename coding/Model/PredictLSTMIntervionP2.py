import torch
from torch import nn
import numpy as np
from collections import OrderedDict
from torch.distributions.poisson import Poisson
from scipy import ndimage
import math
#try:
#     set_start_method('spawn')
#except RuntimeError:
#    pass
#Reference
#https://www.kdnuggets.com/2020/07/pytorch-lstm-text-generation-tutorial.html
def prepareMerge(x,token,permuteidx,seqlen,maxMerge):
    tmp=np.zeros((seqlen,maxMerge),dtype=np.int)
    currentidx=seqlen
    j=seqlen-1
    individualmergeidx=[]
    while(currentidx>0):
        start=currentidx-token[j]
        if(start<0):
            start=0
        mergeToken = x[start:currentidx] # Get # embedded vectors accroding to Strategy to merge
        maxidx = np.max(permuteidx[start:currentidx])
        individualmergeidx.append(list(range(start,currentidx)))
        if(currentidx-start<maxMerge): # Padding Zero if merge token < 4
            remain=np.zeros((maxMerge-(currentidx-start)))
            mergeToken=np.concatenate((mergeToken,remain),axis=-1)
        tmp[maxidx,:]=mergeToken
        currentidx-=token[j].item()
        j-=token[j].item()
    return tmp,individualmergeidx

class PredictLSTMIntervionP(nn.Module):
    def __init__( self, wordsize,embeddingDim,seqlen,outseqlen,minMerge,maxMerge,batchsize,GRU,num_layers,hidden_size,bidirection=False,softmax=False,numofconv1d=1,groupRelu=1,convpredict=False,kernelsize=2,mergeRate=2):
        super(PredictLSTMIntervionP, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.lstm_size = hidden_size
        self.embedding_dim = embeddingDim
        self.num_layers = num_layers
        self.seqlen=seqlen
        self.outseqlen= seqlen if(outseqlen==0 or outseqlen>seqlen) else outseqlen
        self.batchsize=batchsize
        self.n_vocab = wordsize
        self.kernelSize=2
        self.maxMerge=maxMerge
        self.minMerge=minMerge
        self.predict_kernelSize=kernelsize
        self.mergeNumOfConvBlock=0
        self.numOfConv1DBlock=numofconv1d
        self.mergeRate=mergeRate
        self.bidirection=1 if (bidirection==False) else 2
        self.poisson = Poisson(torch.tensor([self.mergeRate],dtype=torch.float))
        self.softmax=softmax
        self.convpredict=convpredict
        self.groupRelu = groupRelu
        self.GRU =GRU
        assert self.maxMerge<=self.seqlen , "max merge size can't > seqlen"
        #assert self.mergeRate<self.maxMerge/2 , "merge rate can't > (max marge size)/2"
        ## calculate the conv block size of Merge Embedding
        nextlayer=self.maxMerge
        while nextlayer!=1:
            nextlayer= np.floor(((nextlayer-(self.kernelSize-1)-1)/1)+1)
            if(nextlayer>=1):
                self.mergeNumOfConvBlock+=1
            if(nextlayer==1):
                break
        print(f'Embedding Space 2 Conv1D block size:{self.mergeNumOfConvBlock}')
        ## calculate the conv block size of prediction
        #nextlayer=self.seqlen
        #while nextlayer!=self.outDim:
        #    nextlayer = np.floor(((nextlayer-(self.predict_kernelSize-1)-1)/1)+1)
        #    if(nextlayer>=self.outDim):
        #        self.prediction_numOfConvBlock+=1
        #    if(nextlayer==self.outDim):
        #        break
        #print(f'Predict Conv1D Block size:{self.prediction_numOfConvBlock}')
        # Preparing Embedding Space 1 and 2
        self.embeddingSpace = nn.Embedding(
                num_embeddings=self.n_vocab,
                embedding_dim=self.embedding_dim,
            ).to(self.device)
        #self.embeddingSpace2=torch.nn.Embedding(num_embeddings=self.n_vocab,embedding_dim=self.embedding_dim).to(self.device)
        layer_list=list()
        for j in range(self.mergeNumOfConvBlock):
            layer_list.append((f'Embed_Conv1D_{j}',nn.Conv1d(self.embedding_dim,self.embedding_dim,kernel_size=self.kernelSize)))
        layer_list.append((f'Embed_Conv1D_{j}_ReLU',nn.ReLU()))
        layerDict = OrderedDict(layer_list)
        self.mergeConv1D = torch.nn.Sequential(layerDict).to(self.device)

        # Create Conv1D prediction Block
        if(self.numOfConv1DBlock>0):
            layer_list=list()
            layer_list.append((f'Conv1d_0',nn.Conv1d(self.bidirection*self.lstm_size,
                                                               self.bidirection*self.lstm_size,
                                                               kernel_size=self.predict_kernelSize,
                                                               padding=0)))
            if(self.groupRelu==2):
                layer_list.append((f'Conv1D_ReLU',nn.ReLU()))
            for i in range(1,self.numOfConv1DBlock):
                layer_list.append((f'Conv1d_{i}',nn.Conv1d(self.bidirection*self.lstm_size,
                                                                   self.bidirection*self.lstm_size,
                                                                   kernel_size=self.predict_kernelSize,
                                                                   padding='same')))
                if(self.groupRelu==2):
                    layer_list.append((f'Conv1D_ReLU',nn.ReLU()))
            if(self.groupRelu==1):
                layer_list.append((f'Conv1D_ReLU',nn.ReLU()))
            layerDict = OrderedDict(layer_list)
            self.conv1d = torch.nn.Sequential(layerDict).to(self.device)
        if(self.convpredict):
            self.predict=nn.Conv1d(self.bidirection*self.lstm_size,
                                                           3,
                                                           kernel_size=self.predict_kernelSize,
                                                           padding=0).to(self.device)

        else:
            self.predict = torch.nn.Linear(self.bidirection*self.lstm_size,3).to(self.device) if (self.softmax==True) else torch.nn.Linear(self.bidirection*self.lstm_size,1).to(self.device)
        if(self.GRU==False):
            self.RNN = nn.LSTM(
                    input_size=self.embedding_dim,
                    hidden_size=self.lstm_size,
                    num_layers=self.num_layers,
                    batch_first=True,
                    bidirectional=bidirection
                ).to(self.device)
        else:
            self.RNN = nn.GRU(
                    input_size=self.embedding_dim,
                    hidden_size=self.lstm_size,
                    num_layers=self.num_layers,
                    batch_first=True,
                    bidirectional=bidirection
                ).to(self.device)

    def forward(self, x, state,switch=0,permuteidx=None,onlyMerge=None,poslist=None,consecutive=False):
        x = x[:,permuteidx]
        poslist=poslist[:,permuteidx]
        if(self.GRU==False):
            h_state=state[0].to(self.device)
            c_state=state[1].to(self.device)
        else:
            h_state=state.to(self.device)
        lstminput=torch.zeros((self.batchsize,self.seqlen,self.embedding_dim,1),requires_grad=True).to(self.device)
        mergeidx=[]
        if(switch==0):
            x=torch.tensor(x).to(self.device)
            lstminput = self.embeddingSpace(x).reshape(self.batchsize,self.seqlen,self.embedding_dim) # Embedding Layer
        else:
            # For each Token Strategy
            # Temp Tensor to hold input to Conv1D , Max Merge Token == 4 ,if no merge (ie. 1 token), the max Batch Size of Conv1D is == Seqlen
            if(len(onlyMerge)==0):
                token=self.poisson.sample([self.seqlen]).type(torch.LongTensor)# avoid zero
                token=torch.where(token>self.maxMerge,self.maxMerge,token)
                token=torch.where((token==0)|(token==1),self.minMerge,token)
                x=torch.tensor(x).to(self.device)
                embed = self.embeddingSpace(x) # Embedding Layer -> Batch * seqlen * embeddingDim
                embed=embed.permute(0,2,1) # (embeddingDim,seqlen) -> Batch * embeddingDim * seqlen
                tmp=torch.zeros((self.batchsize,self.seqlen,self.embedding_dim,self.maxMerge),dtype=torch.float,requires_grad=True).to(self.device)
                currentidx=self.seqlen
                j=self.seqlen-1
                while(currentidx>0):
                    start=currentidx-token[j]
                    if(start<0):
                        start=0
                    mergeToken = embed[:,:, start:currentidx] # Get # embedded vectors accroding to Strategy to merge
                    maxidx = np.max(permuteidx[start:currentidx])
                    mergeidx.append(list(range(start,currentidx)))
                    if(currentidx-start<self.maxMerge): # Padding Zero if merge token < 4
                        remain=torch.zeros((self.batchsize,self.embedding_dim,self.maxMerge-(currentidx-start))).to(self.device)
                        mergeToken=torch.cat([mergeToken,remain],dim=-1)
                    tmp[:,maxidx,:,:]=mergeToken
                    currentidx-=token[j].item()
                    j-=1
            else:
                token=torch.ones((self.batchsize,self.seqlen),dtype=torch.long)
                tmp = torch.zeros_like(token)
                if(consecutive and len(onlyMerge)>1):
                    ridx,cidx=np.where(poslist==onlyMerge[0])
                    for l in range(1,len(onlyMerge)):
                        cidx = np.where(cidx+1>self.seqlen-1,cidx,cidx+1)
                        tmp[ridx,cidx]=torch.tensor(np.where(poslist[ridx,cidx]==onlyMerge[l],l,0))
                        ridx,cidx=torch.where(tmp==l)
                    token[ridx,cidx]=len(onlyMerge)
                else:
                    ridx,cidx=np.where(np.isin(poslist,onlyMerge))
                    token[ridx,cidx]=0
                    token[ridx,np.where(cidx+1>self.seqlen-1,self.seqlen-1,cidx+1)]=0
                    remain=self.poisson.sample([(token.numel()-token.nonzero().size(0))]).type(torch.LongTensor)# avoid zero
                    remain=torch.where(remain>self.maxMerge,self.maxMerge,remain)
                    remain=torch.where((remain==1)|(remain==0),self.minMerge,remain)
                    ridx,cidx=torch.where(token==0)
                    token[ridx,cidx]=remain.reshape(-1)
                tokenidx=torch.zeros((self.batchsize,self.seqlen,self.maxMerge),dtype=torch.int)
                result =[  prepareMerge(x[i],token[i],permuteidx,self.seqlen,self.maxMerge) for i in range(self.batchsize) ]
                for i in range(self.batchsize):
                    tokenidx[i]=torch.tensor(result[i][0])
                    mergeidx.append(result[i][1])
                tmp = tokenidx.to(self.device)
                tmp = self.embeddingSpace(tmp)
                tmp = tmp.permute(0,1,3,2) # (embeddingDim,seqlen) -> Batch * embeddingDim * seqlen
                b,s,m = torch.where(tokenidx==0)
                tmp[b,s,m,:]=0
            # Merge By Con1D in backward direction => LSTM
            for i in range(self.batchsize):
                lstminput[i,:,:,:]=self.mergeConv1D(tmp[i,:,:,:])
        lstminput=lstminput.reshape(self.batchsize,self.seqlen,self.embedding_dim)
        o,state = self.RNN(lstminput,(h_state,c_state)) if (self.GRU==False) else self.RNN(lstminput,h_state)
        if(self.convpredict==True):
            o = o.permute(0,2,1) # N,H,L
            if(self.numOfConv1DBlock>0):
                rotate=self.kernelSize-1
                o = torch.cat([o,o[:,:,0:rotate]],dim=-1)
                o = self.conv1d(o)
            if(self.seqlen==self.outseqlen):
                rotate=self.predict_kernelSize-1
                o = torch.cat([o,o[:,:,0:rotate]],dim=-1)
        o = self.predict(o) # o.shape = N*L*1
        if(self.convpredict==False and self.softmax==True):
            o=o.permute(0,2,1)
        return o,state,mergeidx

    def testRotation(self, x, state,switch=0,permuteidx=None,onlyMerge=None,poslist=None,consecutive=False,degree=0):
        x = x[:,permuteidx]
        poslist=poslist[:,permuteidx]
        w=self.embeddingSpace.weight.cpu().to_numpy()
        self.embeddingSpace.weight=torch.from_numpy(ndimage.rotate(w, degree, reshape=False)).to(self.device)
        if(self.GRU==False):
            h_state=state[0].to(self.device)
            c_state=state[1].to(self.device)
        else:
            h_state=state.to(self.device)
        lstminput=torch.zeros((self.batchsize,self.seqlen,self.embedding_dim,1),requires_grad=True).to(self.device)
        mergeidx=[]
        if(switch==0):
            x=torch.tensor(x).to(self.device)
            lstminput = self.embeddingSpace(x).reshape(self.batchsize,self.seqlen,self.embedding_dim) # Embedding Layer
        else:
            # For each Token Strategy
            # Temp Tensor to hold input to Conv1D , Max Merge Token == 4 ,if no merge (ie. 1 token), the max Batch Size of Conv1D is == Seqlen
            if(len(onlyMerge)==0):
                token=self.poisson.sample([self.seqlen]).type(torch.LongTensor)# avoid zero
                token=torch.where(token>self.maxMerge,self.maxMerge,token)
                token=torch.where((token==0)|(token==1),self.minMerge,token)
                x=torch.tensor(x).to(self.device)
                embed = self.embeddingSpace(x) # Embedding Layer -> Batch * seqlen * embeddingDim
                embed=embed.permute(0,2,1) # (embeddingDim,seqlen) -> Batch * embeddingDim * seqlen
                tmp=torch.zeros((self.batchsize,self.seqlen,self.embedding_dim,self.maxMerge),dtype=torch.float,requires_grad=True).to(self.device)
                currentidx=self.seqlen
                j=self.seqlen-1
                while(currentidx>0):
                    start=currentidx-token[j]
                    if(start<0):
                        start=0
                    mergeToken = embed[:,:, start:currentidx] # Get # embedded vectors accroding to Strategy to merge
                    maxidx = np.max(permuteidx[start:currentidx])
                    mergeidx.append(list(range(start,currentidx)))
                    if(currentidx-start<self.maxMerge): # Padding Zero if merge token < 4
                        remain=torch.zeros((self.batchsize,self.embedding_dim,self.maxMerge-(currentidx-start))).to(self.device)
                        mergeToken=torch.cat([mergeToken,remain],dim=-1)
                    tmp[:,maxidx,:,:]=mergeToken
                    currentidx-=token[j].item()
                    j-=1
            else:
                token=torch.ones((self.batchsize,self.seqlen),dtype=torch.long)
                tmp = torch.zeros_like(token)
                if(consecutive and len(onlyMerge)>1):
                    ridx,cidx=np.where(poslist==onlyMerge[0])
                    for l in range(1,len(onlyMerge)):
                        cidx = np.where(cidx+1>self.seqlen-1,cidx,cidx+1)
                        tmp[ridx,cidx]=torch.tensor(np.where(poslist[ridx,cidx]==onlyMerge[l],l,0))
                        ridx,cidx=torch.where(tmp==l)
                    token[ridx,cidx]=len(onlyMerge)
                else:
                    ridx,cidx=np.where(np.isin(poslist,onlyMerge))
                    token[ridx,cidx]=0
                    token[ridx,np.where(cidx+1>self.seqlen-1,self.seqlen-1,cidx+1)]=0
                    remain=self.poisson.sample([(token.numel()-token.nonzero().size(0))]).type(torch.LongTensor)# avoid zero
                    remain=torch.where(remain>self.maxMerge,self.maxMerge,remain)
                    remain=torch.where((remain==1)|(remain==0),self.minMerge,remain)
                    ridx,cidx=torch.where(token==0)
                    token[ridx,cidx]=remain.reshape(-1)
                tokenidx=torch.zeros((self.batchsize,self.seqlen,self.maxMerge),dtype=torch.int)
                result =[  prepareMerge(x[i],token[i],permuteidx,self.seqlen,self.maxMerge) for i in range(self.batchsize) ]
                for i in range(self.batchsize):
                    tokenidx[i]=torch.tensor(result[i][0])
                    mergeidx.append(result[i][1])
                tmp = tokenidx.to(self.device)
                tmp = self.embeddingSpace(tmp)
                tmp = tmp.permute(0,1,3,2) # (embeddingDim,seqlen) -> Batch * embeddingDim * seqlen
                b,s,m = torch.where(tokenidx==0)
                tmp[b,s,m,:]=0
            # Merge By Con1D in backward direction => LSTM
            for i in range(self.batchsize):
                lstminput[i,:,:,:]=self.mergeConv1D(tmp[i,:,:,:])
        lstminput=lstminput.reshape(self.batchsize,self.seqlen,self.embedding_dim)
        o,state = self.RNN(lstminput,(h_state,c_state)) if (self.GRU==False) else self.RNN(lstminput,h_state)
        if(self.convpredict==True):
            o = o.permute(0,2,1) # N,H,L
            if(self.numOfConv1DBlock>0):
                rotate=self.kernelSize-1
                o = torch.cat([o,o[:,:,0:rotate]],dim=-1)
                o = self.conv1d(o)
            if(self.seqlen==self.outseqlen):
                rotate=self.predict_kernelSize-1
                o = torch.cat([o,o[:,:,0:rotate]],dim=-1)
        o = self.predict(o) # o.shape = N*L*1
        if(self.convpredict==False and self.softmax==True):
            o=o.permute(0,2,1)
        return o,state,mergeidx

    def init_state(self,double=False):
        if(self.GRU):
            batchsize=self.batchsize*2  if(double) else self.batchsize
            return (torch.zeros(self.bidirection*self.num_layers,batchsize, self.lstm_size))
        else:
            batchsize=self.batchsize*2  if(double) else self.batchsize
            return (torch.zeros(self.bidirection*self.num_layers,batchsize, self.lstm_size),
                    torch.zeros(self.bidirection*self.num_layers,batchsize, self.lstm_size))

class StateToImage(nn.Module):
    def __init__(self):
        super(StateToImage, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.decoder =  torch.nn.Sequential(
            torch.nn.ConvTranspose2d(16,512,3,stride=1),        #   (4-1)*1+(3-1)+1 = 3+2+1 = 6
            torch.nn.ReLU(), #(6,5),
            torch.nn.ConvTranspose2d(512,256,2,stride=2),        #  (6-1)*2+(2-1)+1 = 10+1+1 = 12
            torch.nn.ReLU(), #(6,5)
            torch.nn.ConvTranspose2d(256,64,2,stride=2),        #   (12-1)*2+(2-1)+1 = 22+1+1 = 24
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64,1,2,stride=2),        #  (24-1)*2+(2-1)+1 = 46+1+1 = 48
        ).to(self.device)

    def forward(self,state):
        o = self.decoder(state)
        return o
