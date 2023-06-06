import torch
from torch import nn
import numpy as np
from collections import OrderedDict
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.poisson import Poisson
#Reference
#https://www.kdnuggets.com/2020/07/pytorch-lstm-text-generation-tutorial.html

class PredictIntervention2Model(nn.Module):
    def __init__( self, wordsize,embeddingDim,seqlen,maxMerge,batchsize,num_layers,hidden_size,outdim=1,mergeRate=2):
        super(PredictIntervention2Model, self).__init__()
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
        self.model1Embedding={'embedding1':[],'embedding2':[]}
        self.model2Embedding={'embedding1':[],'embedding2':[]}
        self.embedding_numOfConvBlock=0
        self.model1_prediction_numOfConvBlock=0
        self.model2_numOfConvBlock=0
        self.mergeRate=mergeRate
        self.poisson = Poisson(torch.tensor([self.mergeRate],dtype=torch.float))
        assert self.maxMerge<self.seqlen , "max merge size can't > seqlen"
        ## calculate the conv block size of Merge Embedding
        nextlayer=self.maxMerge
        while nextlayer!=1:
            nextlayer= np.floor(((nextlayer-(self.kernelSize-1)-1)/1)+1)
            if(nextlayer>=1):
                self.embedding_numOfConvBlock+=1
            if(nextlayer==1):
                break
        print(f'Embedding Space 2 Conv1D block size:{self.embedding_numOfConvBlock}')
        ## calculate the conv block size of prediction
        nextlayer=self.num_layers
        while nextlayer!=self.outDim:
            nextlayer = np.floor(((nextlayer-(self.predict_kernelSize-1)-1)/1)+1)
            if(nextlayer>=self.outDim):
                self.model1_prediction_numOfConvBlock+=1
            if(nextlayer==self.outDim):
                break
        print(f'Model 1 Predict Conv1D Block size:{self.model1_prediction_numOfConvBlock}')
        ## calculate the conv block size of Model 2
        nextlayer=self.seqlen
        while nextlayer!=self.outDim:
            nextlayer = np.floor(((nextlayer-(self.predict_kernelSize-1)-1)/1)+1)
            if(nextlayer>=self.outDim):
                self.model2_numOfConvBlock+=1
            if(nextlayer==self.outDim):
                break
        print(f'Model 2 Conv1D Block size:{self.model2_numOfConvBlock}')
        # Preparing Embedding Space 1 and 2
        for i in range(self.batchsize):
            self.model1Embedding['embedding1'].append(
                nn.Embedding(
                    num_embeddings=self.n_vocab,
                    embedding_dim=self.embedding_dim,
                ).to(self.device)
            )
            self.model2Embedding['embedding1'].append(
                nn.Embedding(
                    num_embeddings=self.n_vocab,
                    embedding_dim=self.embedding_dim,
                ).to(self.device)
            )
            layer_list=list()
            for j in range(self.embedding_numOfConvBlock):
                layer_list.append((f'Conv1D_{j}_1',nn.Conv1d(self.embedding_dim,self.embedding_dim,kernel_size=self.kernelSize)))
                #layer_list.append((f'Conv1D_{i}_2',nn.Conv1d(self.embedding_dim,self.embedding_dim,kernel_size=self.kernelSize,padding=1,dilation=2)))
                layer_list.append((f'ReLU_{j}',nn.ReLU()))
            layerDict = OrderedDict(layer_list)
            self.model1Embedding['embedding2'].append(torch.nn.Sequential(layerDict).to(self.device))
            self.model2Embedding['embedding2'].append(torch.nn.Sequential(layerDict).to(self.device))

        # Create Conv1D prediction Block
        layer_list=list()
        #for i in range(self.model1_prediction_numOfConvBlock):
        #    layer_list.append((f'Predict_Conv1D_{i}_1',nn.Conv1d(self.lstm_size,
        #                                                self.lstm_size,
        #                                                kernel_size=self.predict_kernelSize)))
            #layer_list.append((f'Conv1D_{i}_2',nn.Conv1d(self.embedding_dim,self.embedding_dim,kernel_size=self.kernelSize,padding=1,dilation=2)))
        #    layer_list.append((f'Predict_ReLU_{i}',nn.ReLU()))
        layer_list.append(('Predict_flatten',nn.Flatten()))
        layer_list.append(('Predict_linear',nn.Linear(self.lstm_size*self.num_layers,1)))
        layerDict = OrderedDict(layer_list)
        self.model1_predictConv1D = torch.nn.Sequential(layerDict).to(self.device)
        self.model1_lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            batch_first=True,
        ).to(self.device)
        #self.predictAndCompetition=nn.Linear(self.batchsize,1,bias=False).to(self.device)

        layer_list=list()
        for i in range(self.model2_numOfConvBlock):
            layer_list.append((f'Model2_Conv1D_{i}_1',nn.Conv1d(self.embedding_dim,
                                                        self.embedding_dim,
                                                        kernel_size=self.predict_kernelSize)))
            #layer_list.append((f'Conv1D_{i}_2',nn.Conv1d(self.embedding_dim,self.embedding_dim,kernel_size=self.kernelSize,padding=1,dilation=2)))
            layer_list.append((f'Model2_ReLU_{i}',nn.ReLU()))
        layerDict = OrderedDict(layer_list)
        self.model2_Conv1D = torch.nn.Sequential(layerDict).to(self.device)
        self.model2_predict = nn.Conv1d(self.embedding_dim,self.outDim,kernel_size=self.predict_kernelSize).to(self.device)

    def forward(self, x, h_state,c_state,model2_previousOut,switch=0):
        x=torch.tensor(x).to(self.device)
        h_state=h_state.to(self.device)
        c_state=c_state.to(self.device)
        model2_previousOut=model2_previousOut.to(self.device)
        model1_lstminput=torch.zeros((self.batchsize,self.seqlen,self.embedding_dim,1),requires_grad=True).to(self.device)
        model2_conv1dinput=torch.zeros((self.batchsize,self.seqlen,self.embedding_dim,1),requires_grad=True).to(self.device)
        if(switch==0):
            model1_lstminput = self.model1Embedding['embedding1'][0](x).reshape(self.batchsize,self.seqlen,self.embedding_dim,1) # Embedding Layer
            model2_conv1dinput = self.model2Embedding['embedding1'][0](x).reshape(self.batchsize,self.seqlen,self.embedding_dim,1) # Embedding Layer
        else:
            # For each Token Strategy
            # Temp Tensor to hold input to Conv1D , Max Merge Token == 4 ,if no merge (ie. 1 token), the max Batch Size of Conv1D is == Seqlen
            for i in range(1,self.batchsize):
                model1_tmp=torch.zeros((self.seqlen,self.embedding_dim,self.maxMerge),dtype=torch.float,requires_grad=True).to(self.device)
                model2_tmp=torch.zeros((self.seqlen,self.embedding_dim,self.maxMerge),dtype=torch.float,requires_grad=True).to(self.device)
                token=self.poisson.sample([self.seqlen]).type(torch.LongTensor)+1# avoid zero
                token=torch.where(token>self.maxMerge,self.maxMerge,token)
                currentidx=self.seqlen
                j=self.seqlen-1
                # Merge By Con1D in backward direction => LSTM
                model1_embed = self.model1Embedding['embedding1'][i](x[i]) # Embedding Layer
                model2_embed = self.model2Embedding['embedding1'][i](x[i]) # Embedding Layer
                mode11_embed=model1_embed.permute(1,0) # (embeddingDim,seqlen)
                mode12_embed=model2_embed.permute(1,0) # (embeddingDim,seqlen)
                while(currentidx>0):
                    start=currentidx-token[j]
                    if(start<0):
                        start=0
                    model1_mergeToken = model1_embed[:, start:currentidx] # Get # embedded vectors accroding to Strategy to merge
                    model2_mergeToken = model2_embed[:, start:currentidx] # Get # embedded vectors accroding to Strategy to merge
                    if(currentidx-start<self.maxMerge): # Padding Zero if merge token < 4
                        remain=torch.zeros((self.embedding_dim,self.maxMerge-(currentidx-start))).to(self.device)
                        model1_mergeToken=torch.concat([model1_mergeToken,remain],dim=1)
                        model2_mergeToken=torch.concat([model2_mergeToken,remain],dim=1)
                    model1_tmp[j,:,:]=model1_mergeToken
                    model2_tmp[j,:,:]=model2_mergeToken
                    currentidx-=token[j]
                    j-=1
                model1_embed=self.model1Embedding['embedding2'][i](model1_tmp)
                model2_embed=self.model2Embedding['embedding2'][i](model2_tmp)
                model1_lstminput[i]=model1_embed
                model2_conv1dinput[i]=model2_embed

        model1_lstminput=model1_lstminput.reshape(self.batchsize,self.seqlen,self.embedding_dim)
        o, (h, c) = self.model1_lstm(model1_lstminput,(h_state,c_state))
        #out = torch.permute(c,(1,2,0)) # (batchsiz, num_layer, h_out) , layer as channel ?
        out = torch.permute(c,(1,2,0)) # (batchsiz, num_layer, h_out) , layer as channel ?
        out = self.model1_predictConv1D(out)

        model2_conv1dinput=model2_conv1dinput.reshape(self.batchsize,self.seqlen,self.embedding_dim)
        model2_conv1dinput=model2_conv1dinput.permute(0,2,1) # (embeddingDim,seqlen)
        model2out = self.model2_Conv1D(model2_conv1dinput)
        out2 = torch.concat([model2out,model2_previousOut],dim=-1)
        out2 = self.model2_predict(out2).reshape(self.outDim,1)
        model2_previousOut=model2out.reshape(1,self.embedding_dim,1)
        return out,(h,c) , out2 , model2_previousOut


    def init_state(self, batchSize):
        return (torch.zeros(self.num_layers, batchSize, self.lstm_size,requires_grad=True),
                torch.zeros(self.num_layers, batchSize, self.lstm_size,requires_grad=True),
                torch.zeros((1,self.embedding_dim,1),requires_grad=True))

