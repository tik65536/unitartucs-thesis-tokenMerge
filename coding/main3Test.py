import pandas as pd
import numpy as np
import torch
import re
import os
import itertools
from datasets import load_dataset
import spacy
import argparse
from torch.utils.tensorboard import SummaryWriter
from Model.PredictLSTMIntervionP2 import PredictLSTMIntervionP
from torch.distributions.bernoulli import Bernoulli
from torchvision.utils import make_grid
import datetime
import pickle
from matplotlib import pyplot as plt
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
def CellStateSimility(cellstate,nidx,pidx,weight):
    minlen =len(pidx) if (len(nidx)>len(pidx)) else len(nidx)
    cstate=cellstate.view(hiddenLayer, 2 if(bidirectional==True) else 1, batchsize, hiddenSize)
    negCellState = torch.nn.functional.normalize(cstate[-1,0,nidx,:].reshape(-1,hiddenSize))
    posCellState = torch.nn.functional.normalize(cstate[-1,0,pidx,:].reshape(-1,hiddenSize))
    n=torch.triu(torch.mm(negCellState,negCellState.T),diagonal=1)
    p=torch.triu(torch.mm(posCellState,negCellState.T),diagonal=1)
    btwgroup = torch.triu(torch.mm(negCellState[:minlen,:],posCellState[:minlen,:].T),diagonal=1)
    nridx,ncidx=torch.triu_indices(n.shape[0],n.shape[1],offset=1)
    pridx,pcidx=torch.triu_indices(p.shape[0],p.shape[1],offset=1)
    btwgridx,btwgcidx=torch.triu_indices(btwgroup.shape[0],btwgroup.shape[1],offset=1)
    n = n[nridx,ncidx]
    p = p[pridx,pcidx]
    btwgroup=btwgroup[btwgridx,btwgcidx]
    negSimiality=torch.mean(n).detach().cpu().numpy()
    posSimiality=torch.mean(p).detach().cpu().numpy()
    btwGroupSimiality=torch.mean(btwgroup)
    n = torch.abs(torch.mm(negCellState,weight.T))
    p = torch.abs(torch.mm(posCellState,weight.T))
    negCellwordSimiality,negCellwordIdx = torch.min(n,dim=1)
    posCellwordSimiality,posCellwordIdx = torch.min(p,dim=1)
    negCellwordSimiality= negCellwordSimiality.detach().cpu().numpy()
    posCellwordSimiality= posCellwordSimiality.detach().cpu().numpy()
    negCellwordIdx= negCellwordIdx.detach().cpu().numpy()
    posCellwordIdx= posCellwordIdx.detach().cpu().numpy()
    return negSimiality,posSimiality,btwGroupSimiality,negCellwordIdx,posCellwordIdx,negCellwordSimiality,posCellwordSimiality



parser = argparse.ArgumentParser()
parser.add_argument('-epoch', type=int, default=20,
                    help='Epoch to run')

parser.add_argument('-batch', type=int, default=5,
                    help='BatchSize')

parser.add_argument('-maxlen', type=int, default=600,
                    help='Max text len')

parser.add_argument('-biasTrain', type=int, default=0,
                    help='Use Bias Training Set')

parser.add_argument('-trainSize', type=int, default=15000,
                    help='Train Set : 15000 or 5000')

parser.add_argument('-optstep', type=int, default=1,
                    help='#Seq per Opt Step')

parser.add_argument('-dynamicOpt', type=int, default=1,
                    help='#Switching Opt Step')

parser.add_argument('-RMS', type=int, default=0,
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
                    help='ReLU for Conv1d , 0,1,2')

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


parser.add_argument('-permuteidx', nargs="*", default=None,
                    help='Permute Token idx')

parser.add_argument('-onlyMerge', nargs="*", default=[],
                    help='Skip POS tag for Merge')

parser.add_argument('-consecutive', type=int, default=0,
                    help='Consecutive merge')

parser.add_argument('-weight', type=str, default=None,
                    help='Weight File Name')

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
bias = True if (args.biasTrain==1) else False
carryforward = True if (args.withHiddenState==1) else False
kernelsize= args.predictkernelsize
groupRelu = args.groupRelu
maxlen =args.maxlen
remark = args.remark
trainSize=args.trainSize
numOfConvBlock=args.numconv1d
permuteidx = list(range(seqlen)) if(args.permuteidx is None) else [ int(x) for x in args.permuteidx ]
mergeRate =args.mergeRate
onlyMerge=args.onlyMerge
consecutive = True if(args.consecutive==1) else False
weightfile= args.weight

assert weightfile is not None,"Weight File Name is None"
print(f'Run Para : {args}',flush=True)

weightPath=f'./LSTMWeight/{weightfile}'

#valbatchsize=2
#imdb_dataset = load_dataset('imdb', split=['train[10000:10010]', 'train[10000:10010]', 'test[:20]'])
#imdb_dataset = load_dataset('imdb')
if(bias):
    with open('./bias_pos_trainpd.plk','rb') as f:
        train_pd = pickle.load(f)
else:
    imdb_dataset = load_dataset('imdb', split=['train[5000:20000]']) if(trainSize==15000) else load_dataset('imdb',split=['train[10000:15000]'])
    train_pd=pd.DataFrame(columns=["text","label"])
    train_pd["text"]=imdb_dataset[0]['text']
    train_pd["label"]=imdb_dataset[0]['label']
    train_pd['text']=train_pd['text'].apply(lambda x: clean_review(x))
    start=''
    c = 5 if(seqlen>=maxlen) else seqlen
    for i in range(c):
        start+='s0s '
    train_pd['text']=start+train_pd['text']+' e0s'
with open('./testpd.plk','rb') as f:
    test_pd = pickle.load(f)
print(f'Traing PD shape : {train_pd.shape}',flush=True)
print(train_pd["label"].describe().T)
print(f'Test PD shape : {test_pd.shape}',flush=True)
print(test_pd["label"].describe().T)
vocab=None


with  open('./Master_vocab.pkl','rb') as f:
    vocab=pickle.load(f)

if(bias):
    with open('./bias_pos_traincounter.plk','rb') as f:
        traincounter=pickle.load(f)
    del traincounter['s0s']
    del traincounter['e0s']
else:
    with open('./traincounter.plk','rb') as f:
        traincounter=pickle.load(f)
reverse_traincounter=sorted(traincounter.items(), key=lambda pair: pair[1])


tok2id=vocab['tok2id']
id2tok=vocab['id2tok']



fig,axes=plt.subplots(4,1,figsize=(20,20), gridspec_kw={'height_ratios': [1,2,1,2]})
trainposlabel = np.array([ tok2id[pair[0]] for pair in traincounter.most_common(n=20) ])
trainneglabel = np.array([ tok2id[pair[0]] for pair in reverse_traincounter[:20] ])
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
startid=tok2id['s0s']
endid=tok2id['e0s']
n_vocab=len(vocab['wordlist'])
print(f'Vocab len: {n_vocab}',flush=True)
bernoulli= Bernoulli(torch.tensor([interventionP]))
preprocessP= Bernoulli(torch.tensor([preprocess]))
model=PredictLSTMIntervionP(n_vocab,embeddingDim,seqlen,seqlen,minmerge,maxmerge,batchsize,GRU,hiddenLayer,hiddenSize,bidirectional,True,numOfConvBlock,groupRelu,convpredict,kernelsize,mergeRate)
model.load_state_dict(torch.load(weightPath))
criterion = torch.nn.CrossEntropyLoss()
gate=['igone','forget','learn','output']
print(model.predict.weight.shape)
test_text=test_pd['text']
test_label=test_pd['label']
with torch.no_grad():
    vallosses=[]
    valAvgAccy=[]
    valAvgAccy2=[]
    valNegSimility=[]
    valPosSimility=[]
    valbtwGroupSimility=[]
    valNegSimilityMedian=[]
    valPosSimilityMedian=[]
    valbtwGroupSimilityMedian=[]
    switchcount=0
    tokendist=[]
    valavgdiffcount=[]
    valbatchcount=0
    for d in range(0,len(test_text),batchsize):
        diffcount=np.zeros((batchsize,))
        switch=bernoulli.sample()
        preprocessswitch=preprocessP.sample()
        state=model.init_state()
        losses=[]
        negCellStateSimility=[]
        posCellStateSimility=[]
        btwGroupSimility=[]
        sequences=[]
        targets=[]
        idxarray=[]
        tokenpos=[]
        for i,x in enumerate(nlp.pipe(test_text[d:d+batchsize])):
            data=[]
            pos=[]
            for t in x:
                if(preprocessswitch==0):
                    data+= [tok2id[t.text]]
                    pos+= [t.pos_]
                elif(t.is_stop==False and t.is_punct==False):
                    data+= [tok2id[t.text]]
                    pos+= [t.pos_]
            tmp=np.zeros(maxlen)
            c=5 if(seqlen>=maxlen) else seqlen
            tmp[:c]=2
            tmp[c:]=test_label[d+i]
            idx=maxlen-1
            if(len(data)>maxlen):
                data=data[:maxlen-1]+[endid]
                pos=pos[:maxlen]
                tmp[-1]=2
            else:
                orglen=len(data)
                idx=orglen-1
                data=(data+([endid]*(maxlen-orglen)))
                pos=(pos+['e0s']*(maxlen-orglen))
                tmp[orglen:]=2
            idxarray.append(idx)
            targets.append(tmp)
            sequences.append(data)
            tokenpos.append(pos)
        idxarray=np.array(idxarray)
        targets=torch.tensor(np.array(targets),dtype=torch.long).to(device)
        sequences=np.array(sequences)
        tokenpos=np.array(tokenpos)
        predict_history=np.zeros((batchsize,maxlen,3))
        target=test_label[d:d+batchsize].to_numpy()
        nidx=np.where(target==0)[0]
        pidx=np.where(target==1)[0]
        nmind=np.zeros((len(nidx),maxlen//seqlen))
        pmind=np.zeros((len(pidx),maxlen//seqlen))
        c=1 if (seqlen>=maxlen) else (maxlen-seqlen)
        for i in range(0,c,sliding):
            sequence=sequences[:,i:i+seqlen]
            t=targets[:,i:i+seqlen]
            poslist = tokenpos[:,i:i+seqlen]
            permuteposlist = poslist[:,permuteidx]
            diffcount+=np.sum(poslist!=permuteposlist,axis=-1)
            pred,state,mergeidx =model.testRotation(sequence,state,switch,permuteidx,onlyMerge,poslist,consecutive)
            g=time.time()
            loss=criterion(pred,t)
            losses.append(loss.item())
            pred=torch.nn.functional.softmax(pred,dim=1)
            pred=pred.permute(0,2,1).detach().cpu().numpy()
            predict_history[:,i:i+seqlen,:]=pred
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
        negsample=sequences[nidx]
        possample=sequences[pidx]
        negsampleidx = np.argmax(np.sum(np.isin(negsample, trainposlabel), axis=-1))
        possampleidx = np.argmax(np.sum(np.isin(possample, trainneglabel), axis=-1))
        c = 5 if(seqlen>=maxlen) else seqlen
        nprob=list(itertools.chain(*[  predict_history[n,c:idxarray[n],0] for n in nidx ]))
        pprob=list(itertools.chain(*[  predict_history[p,c:idxarray[p],1] for p in pidx ]))
        negSampleDiff=predict_history[nidx[negsampleidx],:,0]-predict_history[nidx[negsampleidx],:,1]
        posSampleDiff=predict_history[pidx[possampleidx],:,1]-predict_history[pidx[possampleidx],:,0]
        negsampletext = [ id2tok[tid].replace("$","\$") for tid in negsample[negsampleidx] ]
        possampletext = [ id2tok[tid].replace("$","\$") for tid in possample[possampleidx] ]
        negidx = [ negsampletext.index(id2tok[x]) for x in trainposlabel if(id2tok[x] in negsampletext)  ]
        negidx2 = [ negsampletext.index(id2tok[x]) for x in trainneglabel if(id2tok[x] in negsampletext)  ]
        posidx = [ possampletext.index(id2tok[x]) for x in trainneglabel if(id2tok[x] in possampletext) ]
        posidx2 = [ possampletext.index(id2tok[x]) for x in trainposlabel if(id2tok[x] in possampletext) ]
        negendidx=idxarray[nidx[negsampleidx]]
        posendidx=idxarray[pidx[possampleidx]]
        axes[1].set_title(f'GT:{target[nidx[negsampleidx]]} Pred:{est_prediction[nidx[negsampleidx]]},{est_prediction2[nidx[negsampleidx]]}')
        axes[1].plot(predict_history[nidx[negsampleidx],:negendidx,0].T,label='NegProb',marker='x')
        axes[1].plot(predict_history[nidx[negsampleidx],:negendidx,1].T,label='PosProb',marker='o')
        axes[1].set_xticks(list(range(negendidx)))
        axes[1].set_xticklabels(negsampletext[:negendidx], rotation=90, ha='right',fontdict={'fontsize':4})
        axes[3].set_title(f'GT:{target[pidx[possampleidx]]} Pred:{est_prediction[pidx[possampleidx]]},{est_prediction2[pidx[possampleidx]]}')
        axes[3].plot(predict_history[pidx[possampleidx],:posendidx,0].T,label='NegProb',marker='x')
        axes[3].plot(predict_history[pidx[possampleidx],:posendidx,1].T,label='PosProb',marker='o')
        axes[3].set_xticks(list(range(posendidx)))
        axes[3].set_xticklabels(possampletext[:posendidx], rotation=90, ha='right',fontdict={'fontsize':4})
        axes[2].plot(posSampleDiff[:posendidx].T,label='Diff',color='g',marker='o',markersize=0.7,alpha=0.5)
        axes[2].axhline(y=0,color='r')
        axes[0].plot(negSampleDiff[:negendidx].T,label='Diff',color='g',marker='o',markersize=0.7,alpha=0.5)
        axes[0].axhline(y=0,color='r')
        if(len(posidx)>0): [ axes[2].axvline(i,linewidth=2,c='#F39C12') for i in posidx ]
        if(len(posidx2)>0): [ axes[2].axvline(i,linewidth=2,c='#0000EE') for i in posidx2 ]
        if(len(negidx)>0): [ axes[0].axvline(i,linewidth=2,c='#F39C12') for i in negidx]
        if(len(negidx2)>0): [ axes[0].axvline(i,linewidth=2,c='#0000EE') for i in negidx2 ]
        axes[2].set_title(",".join([ id2tok[x] for x in trainneglabel if(id2tok[x] in possampletext)]))
        axes[0].set_title(",".join([ id2tok[x] for x in trainposlabel if(id2tok[x] in negsampletext)]))
        [ axes[i].legend() for i in range(4) ]
        fig.suptitle(f'Batch Accy : {batchaccy} {batchaccy2}')
        plt.tight_layout()
        fig.savefig(f'./Validation_{valbatchcount}_switch{int(switch)}_preprocess{int(preprocessswitch)}_{avgloss:0.6f}_plot.png',dpi=400)
        [ axes[i].clear() for i in range(4) ]
        plt.cla()#
        diffcount=np.mean(diffcount/idxarray)
        valavgdiffcount.append(diffcount)
        print(f'Batch ValDoc#{(d+batchsize):5d}, Switch:{int(switch):1d}, Preprocess:{int(preprocessswitch):1d}, AvgLoss:{avgloss:0.3f}, AvgAccy:{batchaccy:0.3f}, AvgAccy2:{batchaccy2:0.3f}, DiffPOS:{diffcount:2.3f}',flush=True)
        valbatchcount+=1
    avgValloss=np.mean(vallosses)
    valAvgAccy = np.mean(valAvgAccy)
    valAvgAccy2 = np.mean(valAvgAccy2)
    print(f'Validation Finished, Avg Loss:{avgValloss:0.6f}, AvgAccy:{valAvgAccy:0.3f}, AvgAccy2:{valAvgAccy2:0.3f}',flush=True)
print("")
