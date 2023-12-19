import pandas as pd
import numpy as np
import torch
import re
import os
import itertools
from datasets import load_dataset
import spacy
import argparse
import copy
from torch.utils.tensorboard import SummaryWriter
from Model.PredictLSTMIntervionP2 import PredictLSTMIntervionP
from torch.distributions.bernoulli import Bernoulli
from torchvision.utils import make_grid
import datetime
import pickle
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from sklearn.metrics import classification_report
import warnings
from collections import Counter
import time
warnings.filterwarnings('ignore')

#Reference https://muhark.github.io/python/ml/nlp/2021/10/21/word2vec-from-scratch.html

#nlp = spacy.load("en_core_web_sm")
nlp = spacy.load("en_core_web_md")

#https://www.kaggle.com/code/youssefamdouni/movie-review-classification-using-spacy
def clean_review(text):
    clean_text = re.sub('<.*?>', '', text)
    clean_text =re.sub(r"http\S+", "", clean_text)
    clean_text = re.sub(r"www\S+", "", clean_text)
    #clean_text = re.sub('[^a-zA-Z\']', ' ', clean_text)
    #clean_text = clean_text.lower()
    return clean_text.lower()

def rnnOutputSimility(negout,posout,minlen,flag=0):
    #shape : batch x seq x hiddenSize
    if(flag==0):
        negnorm = torch.norm(negout,dim=-1).detach().cpu().numpy()
        posnorm = torch.norm(posout,dim=-1).detach().cpu().numpy()
        negout = torch.nn.functional.normalize(negout,dim=-1)
        posout = torch.nn.functional.normalize(posout,dim=-1)
        #ns,ps,btws=np.zeros(seqlen),np.zeros(seqlen),np.zeros(seqlen)
        negoutT = torch.einsum('ijk->ikj',negout)
        posoutT = torch.einsum('ijk->ikj',posout)
        ns = torch.einsum('ijk,ikl->ijl',negout,negoutT).detach().cpu().numpy()
        ps = torch.einsum('ijk,ikl->ijl',posout,posoutT).detach().cpu().numpy()
        ridx,cidx=torch.triu_indices(ns.shape[1],ns.shape[2],offset=1)
        ns_flatten = ns[:,ridx,cidx]
        ridx,cidx=torch.triu_indices(ps.shape[1],ps.shape[2],offset=1)
        ps_flatten = ps[:,ridx,cidx]
        negout=negout[:minlen]
        posout=posout[:minlen]
        negout = torch.einsum('ijk->jik',negout) # sample*seqlen*dim -> seqlen*sample*dim
        posout = torch.einsum('ijk->jki',posout) # -> seqlen*dim*sample
        btwgroup = torch.einsum('ijk,ikl->ijl',negout,posout).detach().cpu().numpy() # seqlen*sample*sample
        ridx,cidx=torch.triu_indices(btwgroup.shape[1],btwgroup.shape[2],offset=1)
        btws_flatten = btwgroup[:,ridx,cidx]
        return ns_flatten,ps_flatten,btws_flatten,negnorm,posnorm,ns,ps
    else:
        negout = torch.nn.functional.normalize(negout,dim=-1)
        posout = torch.nn.functional.normalize(posout,dim=-1)
        posout = torch.einsum('ijk->ikj',posout)
        ns = torch.einsum('ijk,ikl->ijl',negout,posout)
        return torch.diagonal(ns,offset=0,dim1=-2,dim2=-1).detach().cpu().numpy(),ns.detach().cpu().numpy()


def CellStateSimility(nstate,pstate,minlen):
    #cstate=cellstate.view(hiddenLayer, -1, batchsize, hiddenSize*(2 if(bidirectional==True) else 1))
    #cstate=cellstate.view(hiddenLayer, 2 if(bidirectional==True) else 1, batchsize, hiddenSize)
    #negCellState = torch.nn.functional.normalize(cstate[-1,0,nidx,:].reshape(-1,hiddenSize))
    #posCellState = torch.nn.functional.normalize(cstate[-1,0,pidx,:].reshape(-1,hiddenSize))
    nstate = torch.nn.functional.normalize(nstate,dim=-1)
    pstate = torch.nn.functional.normalize(pstate,dim=-1)
    n=torch.triu(torch.mm(nstate,nstate.T),diagonal=1)
    p=torch.triu(torch.mm(pstate,nstate.T),diagonal=1)
    btwgroup = torch.triu(torch.mm(nstate[:minlen,:],pstate[:minlen,:].T),diagonal=1)
    nridx,ncidx=torch.triu_indices(n.shape[0],n.shape[1],offset=1)
    pridx,pcidx=torch.triu_indices(p.shape[0],p.shape[1],offset=1)
    btwgridx,btwgcidx=torch.triu_indices(btwgroup.shape[0],btwgroup.shape[1],offset=1)
    n = n[nridx,ncidx]
    p = p[pridx,pcidx]
    btwgroup=btwgroup[btwgridx,btwgcidx]
    negSimiality=torch.mean(n).detach().cpu().numpy()
    posSimiality=torch.mean(p).detach().cpu().numpy()
    btwGroupSimiality=torch.mean(btwgroup).detach().cpu().numpy()
    return negSimiality,posSimiality,btwGroupSimiality


def curl(input_,output):
    x=torch.autograd.grad(output[:,:,0],input_,torch.ones_like(output[:,:,0]),retain_graph=True)[0]
    P_y = x[:,:,1].detach().cpu().numpy()
    P_z = x[:,:,2].detach().cpu().numpy()
    P_x = x[:,:,0].detach().cpu().numpy()
    # Output dim 1 = Q , we need Q_x,Q_z
    x=torch.autograd.grad(output[:,:,1],input_,torch.ones_like(output[:,:,1]),retain_graph=True)[0]
    Q_x = x[:,:,0].detach().cpu().numpy()
    Q_y = x[:,:,1].detach().cpu().numpy()
    Q_z = x[:,:,2].detach().cpu().numpy()
    # Output dim 2 = R , we need R_x,R_y
    x=torch.autograd.grad(output[:,:,2],input_,torch.ones_like(output[:,:,2]),retain_graph=True)[0]
    R_x = x[:,:,0].detach().cpu().numpy()
    R_y = x[:,:,1].detach().cpu().numpy()
    R_z = x[:,:,2].detach().cpu().numpy()
    x=torch.autograd.grad(output[:,:,3],input_,torch.ones_like(output[:,:,3]),retain_graph=True)[0]
    BP_x = x[:,:,0].detach().cpu().numpy()
    BP_y = x[:,:,1].detach().cpu().numpy()
    BP_z = x[:,:,2].detach().cpu().numpy()
    x=torch.autograd.grad(output[:,:,4],input_,torch.ones_like(output[:,:,4]),retain_graph=True)[0]
    BR_x = x[:,:,0].detach().cpu().numpy()
    BR_y = x[:,:,1].detach().cpu().numpy()
    BR_z = x[:,:,2].detach().cpu().numpy()
    x=torch.autograd.grad(output[:,:,5],input_,torch.ones_like(output[:,:,5]),retain_graph=True)[0]
    BQ_x = x[:,:,0].detach().cpu().numpy()
    BQ_y = x[:,:,1].detach().cpu().numpy()
    BQ_z = x[:,:,2].detach().cpu().numpy()
    return R_y-Q_z,P_z-R_x,Q_x-P_y,P_x,Q_y,R_z,BR_y-BQ_z,BP_z-BR_x,BQ_x-BP_y,BP_x,BQ_y,BR_z

def curlPrediction(input_,output):
    x=torch.autograd.grad(output[:,:,0],input_,torch.ones_like(output[:,:,0]),retain_graph=True)[0]
    P_y = x[:,:,1].detach().cpu().numpy()
    P_z = x[:,:,2].detach().cpu().numpy()
    P_x = x[:,:,0].detach().cpu().numpy()
    BP_x = x[:,:,3].detach().cpu().numpy()
    BP_y = x[:,:,4].detach().cpu().numpy()
    BP_z = x[:,:,5].detach().cpu().numpy()
    # Output dim 1 = Q , we need Q_x,Q_z
    x=torch.autograd.grad(output[:,:,1],input_,torch.ones_like(output[:,:,1]),retain_graph=True)[0]
    Q_x = x[:,:,0].detach().cpu().numpy()
    Q_y = x[:,:,1].detach().cpu().numpy()
    Q_z = x[:,:,2].detach().cpu().numpy()
    BQ_x = x[:,:,3].detach().cpu().numpy()
    BQ_y = x[:,:,4].detach().cpu().numpy()
    BQ_z = x[:,:,5].detach().cpu().numpy()
    # Output dim 2 = R , we need R_x,R_y
    x=torch.autograd.grad(output[:,:,2],input_,torch.ones_like(output[:,:,2]),retain_graph=True)[0]
    R_x = x[:,:,0].detach().cpu().numpy()
    R_y = x[:,:,1].detach().cpu().numpy()
    R_z = x[:,:,2].detach().cpu().numpy()
    BR_x = x[:,:,3].detach().cpu().numpy()
    BR_y = x[:,:,4].detach().cpu().numpy()
    BR_z = x[:,:,5].detach().cpu().numpy()
    return R_y-Q_z,P_z-R_x,Q_x-P_y,P_x,Q_y,R_z,BR_y-BQ_z,BP_z-BR_x,BQ_x-BP_y,BP_x,BQ_y,BR_z

parser = argparse.ArgumentParser()
parser.add_argument('-epoch', type=int, default=20,
                    help='Epoch to run')

parser.add_argument('-batch', type=int, default=5,
                    help='BatchSize')

parser.add_argument('-maxlen', type=int, default=600,
                    help='Max text len')

parser.add_argument('-HPCTrain', type=int, default=0,
                    help='Use HPC Training Set')

parser.add_argument('-trainSize', type=int, default=15000,
                    help='Train Set : 15000 or 5000')

parser.add_argument('-optstep', type=int, default=1,
                    help='#Seq per Opt Step')

parser.add_argument('-dynamicOpt', type=int, default=1,
                    help='#Switching Opt Step')

parser.add_argument('-RMS', type=int, default=0,
                    help='RMSprop')

parser.add_argument('-codewordNorm', type=int, default=0,
                    help='RMSprop')

parser.add_argument('-preprocesstext', type=float, default=1,
                    help='Remove Stop word and Pun')

parser.add_argument('-embeddingDim', type=int, default=128,
                    help='Embedding Dim')

parser.add_argument('-seqlen', type=int, default=5,
                    help='Seq Len for reading token')

parser.add_argument('-slide', type=int, default=1,
                    help='Seqlen Sliding')

parser.add_argument('-GRU', type=int, default=0,
                    help='Use GRU')

parser.add_argument('-hiddenLayer', type=int, default=20,
                    help='Hidden Layer of LSTM')

parser.add_argument('-hiddenSize', type=int, default=128,
                    help='Hidden Size of LSTM')

parser.add_argument('-bidirection', type=int, default=0,
                    help='0 = false for bi-direction')

parser.add_argument('-withHiddenState', type=int, default=1,
                    help='Carry Forward the hidden and cell state')

parser.add_argument('-numconv1d', type=int, default=1,
                    help='Number of Conv1D')

parser.add_argument('-groupRelu', type=int, default=1,
                    help='1=ReLU, 2=Tanh')

parser.add_argument('-convpredict', type=int, default=0,
                    help='Use Conv1D predict')

parser.add_argument('-predictkernelsize', type=int, default=2,
                    help='Kernel Size for predict Conv1D')

parser.add_argument('-interventionP', type=float, default=0.5,
                    help='Probability for intervention')

parser.add_argument('-mergeRate', type=int, default=2,
                    help='Merge Rate')

parser.add_argument('-maxmerge', type=int, default=4,
                    help='max merge token size')

parser.add_argument('-minmerge', type=int, default=2,
                    help='min merge token size')

parser.add_argument('-remark', type=str, default='',
                    help='Remark on filename')

parser.add_argument('-inhibit', nargs="*", default=[],
                    help='list of pos for inhibit learning')

parser.add_argument('-inhibiteps', type=float, default=0.0,
                    help='variance for inhibit loss with mean=0.5')

parser.add_argument('-numClass', type=int, default=3,
                    help='num of Class 3 or 4')

parser.add_argument('-permuteidx', nargs="*", default=None,
                    help='Permute Token idx')

parser.add_argument('-onlyMerge', nargs="*", default=[],
                    help='Skip POS tag for Merge')

parser.add_argument('-skipPOS', nargs="*", default=[],
                    help='Skip POS')

parser.add_argument('-consecutive', type=int, default=0,
                    help='Consecutive merge')


args = parser.parse_args()
seqlen=args.seqlen
batchsize=args.batch
embeddingDim=args.embeddingDim
epochs = args.epoch
hiddenSize=args.hiddenSize
hiddenLayer=args.hiddenLayer
maxmerge=args.maxmerge
minmerge=args.minmerge
interventionP=args.interventionP
sliding=args.slide
bidirectional= True if (args.bidirection==1) else False
preprocess = args.preprocesstext
convpredict = True if (args.convpredict==1) else False
RMS = True if (args.RMS==1) else False
GRU = True if (args.GRU==1) else False
optstep = args.optstep
optstep = optstep*seqlen
dynamicOpt = True if (args.dynamicOpt==1) else False
retain=True if (optstep>1) else False
bias = True if (args.HPCTrain==1) else False
carryforward = True if (args.withHiddenState==1) else False
kernelsize= args.predictkernelsize
groupRelu = args.groupRelu
maxlen =args.maxlen
remark = args.remark
trainSize=args.trainSize
numOfConvBlock=args.numconv1d
inhibitlist=args.inhibit
neps = args.inhibiteps
numClass = args.numClass
permuteidx = list(range(seqlen)) if(args.permuteidx is None) else [ int(x) for x in args.permuteidx ]
mergeRate =args.mergeRate
onlyMerge=args.onlyMerge
skipPOS=args.skipPOS
consecutive = True if(args.consecutive==1) else False
mincodeword = True if(args.codewordNorm==1) else False

print(f'Run Para : {args}',flush=True)


#valbatchsize=2
#imdb_dataset = load_dataset('imdb', split=['train[10000:10010]', 'train[10000:10010]', 'test[:20]'])
#imdb_dataset = load_dataset('imdb')
with open('./testpd.plk','rb') as f:
    test_pd = pickle.load(f)
print(f'Test PD shape : {test_pd.shape}',flush=True)
print(test_pd["label"].describe().T)
vocab=None


with  open('./Master_vocab.pkl','rb') as f:
    vocab=pickle.load(f)


tok2id=vocab['tok2id']
id2tok=vocab['id2tok']



if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
startid=tok2id['s0s']
endid=tok2id['e0s']
n_vocab=len(vocab['wordlist'])
print(f'Vocab len: {n_vocab}',flush=True)
bernoulli= Bernoulli(torch.tensor([interventionP]))
weightPath=f'./LSTMWeight'
basecaseweightPath=f'./LSTMWeight/basecase_25_dim3_weight_bestVal.pt'
mergeweightPath=f'./LSTMWeight/RandMerge_weight_R100.pt'
model=torch.load(basecaseweightPath)
mergeModel=torch.load(mergeweightPath)
#model.mergeConv1D.load_state_dict(mergeModel.mergeConv1D.state_dict())
model.mergeConv1D=copy.deepcopy(mergeModel.mergeConv1D)
model.embeddingSpace=copy.deepcopy(mergeModel.embeddingSpace)
model.maxMerge=maxmerge
criterion = torch.nn.CrossEntropyLoss()
#val_criterion = torch.nn.CrossEntropyLoss()
#if(len(inhibitlist)>0):
#    criterion = torch.nn.CrossEntropyLoss(reduction='none')
test_text=test_pd['text']
test_label=test_pd['label']
vallosses=[]
valAvgAccy=[]
valAvgAccy2=[]
for d in range(0,len(test_text),batchsize):
    state=model.init_state()
    losses=[]
    switch=bernoulli.sample()
    sequences,targets,idxarray,tokenpos,negNorm,posNorm,blocknegNorm,blockposNorm=[],[],[],[],[],[],[],[]
    for i,x in enumerate(nlp.pipe(test_text[d:d+batchsize])):
        data=[]
        pos=[]
        for t in x:
            if(t.is_stop==False and t.is_punct==False):
                if(t.pos_ not in skipPOS):
                    data+= [tok2id[t.text]]
                    pos += [t.pos_ if(t.text!='s0s' and t.text!='e0e') else 's0e']
        tmp=np.zeros(maxlen)
        c=5 if(seqlen>=maxlen) else seqlen
        tmp[:c]=2
        tmp[c:]=test_label[d+i]
        idx=maxlen-1
        if(len(data)>maxlen):
            data=data[:maxlen-1]+[endid]
            pos=pos[:maxlen]
            tmp[-1]=2 if (numClass==3) else 3
        else:
            orglen=len(data)
            idx=orglen-1
            data=(data+([endid]*(maxlen-orglen)))
            pos=(pos+['e0s']*(maxlen-orglen))
            tmp[orglen:]=2 if (numClass==3) else 3
        idxarray.append(idx)
        targets.append(tmp)
        sequences.append(data)
        tokenpos.append(pos)
    idxarray=np.array(idxarray)
    targets=torch.tensor(np.array(targets),dtype=torch.long).to(device)
    sequences=np.array(sequences)
    tokenpos=np.array(tokenpos)
    predict_history=np.zeros((batchsize,maxlen,numClass))
    target=test_label[d:d+batchsize].to_numpy()
    nidx=np.where(target==0)[0]
    pidx=np.where(target==1)[0]
    #nmind=np.zeros((len(nidx),maxlen//seqlen))
    #pmind=np.zeros((len(pidx),maxlen//seqlen))
    c=1 if (seqlen>=maxlen) else (maxlen-seqlen)
    minlen =len(pidx)-1 if (len(nidx)>len(pidx)) else len(nidx)-1
    previousOutput=None
    for i in range(0,c,sliding):
        sequence=sequences[:,i:i+seqlen]
        t=targets[:,i:i+seqlen]
        poslist = tokenpos[:,i:i+seqlen]
        pred,output,input_,state,mergeidx =model(sequence,state,switch,permuteidx,onlyMerge,poslist,consecutive)
        loss=criterion(pred,t)
        losses.append(loss.item())
        pred=torch.nn.functional.softmax(pred,dim=1)
        pred=pred.permute(0,2,1)
        predict_history[:,i:i+seqlen,:]=pred.detach().cpu().numpy()
    avgloss=np.mean(losses)
    vallosses.append(avgloss)
    est_prediction=[]
    est_prediction2=[]
    est_magnitude=[]
    drawcount=0
    #[ predict_median.append(np.median(predict_history[i, :idxarray[i]-1])) for i in range(batchsize) ]
    #[ predict_mean.append(np.mean(predict_history[i, :idxarray[i]-1])) for i in range(batchsize) ]
    maxidx= [ np.argmax(predict_history[i, idxarray[i]-21:idxarray[i]-1, :],axis=-1) for i in range(batchsize)]

    for i,each in enumerate(maxidx):
        if(each.shape[0]==0):
            if(idxarray[i]<20):
                maxidx[i]=np.argmax(predict_history[i, idxarray[i]-6:idxarray[i]-1, :],axis=-1)
                drawcount+=1
    [ est_prediction.append(np.mean(maxidx[i],axis=-1)) for i in range(batchsize) ]
    [ est_magnitude.append(np.mean(np.diag(predict_history[i, idxarray[i]-maxidx[i].shape[0]:idxarray[i]-1,:].T[maxidx[i]]),axis=-1)) for i in range(batchsize) ]
    est_prediction = np.where(np.isnan(est_prediction)==True,2,est_prediction)
    if(d<batchsize):
        with open(f'{weightPath}/EstPred_validation.plk','wb') as f:
            pickle.dump(est_prediction,f)
    batchaccy=np.sum(np.abs(est_prediction-target)<0.05)/batchsize
    batchmagnitude=np.median(est_magnitude)
    cr=classification_report(target,np.round(est_prediction),output_dict=True)
    c=5 if(seqlen>=maxlen) else seqlen
    maxidx= [ np.argmax(predict_history[i, c:idxarray[i]-1, :],axis=-1) for i in range(batchsize)]
    [ est_prediction2.append(np.mean(maxidx[i],axis=-1)) for i in range(batchsize) ]
    batchaccy2=np.sum(np.abs(est_prediction2-target)<0.05)/batchsize
    key=list(cr.keys())
    del cr[key[0]]['support']
    del cr[key[1]]['support']
    del cr['macro avg']['support']
    valAvgAccy.append(batchaccy)
    valAvgAccy2.append(batchaccy2)
    print(f'Batch ValDoc#{(d+batchsize):5d}, Switch:{int(switch):1d}, AvgLoss:{avgloss:0.3f}, AvgAccy:{batchaccy:0.3f}, AvgAccy2:{batchaccy2:0.3f}',flush=True)
avgValloss=np.mean(vallosses)
valAvgAccy = np.mean(valAvgAccy)
valAvgAccy2 = np.mean(valAvgAccy2)
print(f'Validation Finished, Avg Loss:{avgValloss:0.6f}, AvgAccy:{valAvgAccy:0.3f}, AvgAccy2:{valAvgAccy2:0.3f}',flush=True)
#print(f'Epoch: {epoch:2d} Validation Finished, Merge: {valmergeStatistic.most_common(n=10)}',flush=True)
#with open(f'{tensorboardpath}/trainOverAllTop10_MergeStatistic_{epoch}.pkl','wb') as f:
#    pickle.dump(overallTop10Merge,f)
#with open(f'{tensorboardpath}/ValMergeStatistic_{epoch}.pkl','wb') as f:
#    pickle.dump(valmergeStatistic,f)
