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

def rnnOutputSimility(negout,posout,minlen):
    #shape : batch x seq x hiddenSize
    negnorm = torch.norm(negout,dim=-1).detach().cpu().numpy()
    posnorm = torch.norm(posout,dim=-1).detach().cpu().numpy()
    negout = torch.nn.functional.normalize(negout,dim=-1)
    posout = torch.nn.functional.normalize(posout,dim=-1)
    #ns,ps,btws=np.zeros(seqlen),np.zeros(seqlen),np.zeros(seqlen)
    ns,ps,btws=[],[],[]
    for i in range(seqlen):
        n=torch.triu(torch.mm(negout[:,i,:],negout[:,i,:].T),diagonal=1)
        p=torch.triu(torch.mm(posout[:,i,:],posout[:,i,:].T),diagonal=1)
        btwgroup = torch.triu(torch.mm(negout[:minlen,i,:],posout[:minlen,i,:].T),diagonal=1)
        nridx,ncidx=torch.triu_indices(n.shape[0],n.shape[1],offset=1)
        pridx,pcidx=torch.triu_indices(p.shape[0],p.shape[1],offset=1)
        btwgridx,btwgcidx=torch.triu_indices(btwgroup.shape[0],btwgroup.shape[1],offset=1)
        n = n[nridx,ncidx]
        p = p[pridx,pcidx]
        btwgroup=btwgroup[btwgridx,btwgcidx]
        #ns[i]=torch.mean(n).detach().cpu().numpy()
        #ps[i]=torch.mean(p).detach().cpu().numpy()
        #btws[i]=torch.mean(btwgroup).detach().cpu().numpy()
        ns.append(n.detach().cpu().numpy())
        ps.append(p.detach().cpu().numpy())
        btws.append(btwgroup.detach().cpu().numpy())
    return ns,ps,btws,negnorm,posnorm

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
skipPOS=args.skipPOS
consecutive = True if(args.consecutive==1) else False
mincodeword = True if(args.codewordNorm==1) else False

print(f'Run Para : {args}',flush=True)

filename=f"{remark}seqlen{seqlen}_sldie{sliding}_batch{batchsize}_opt{optstep}_dynamicOpt{dynamicOpt}_train{trainSize}_ksize{kernelsize}_e{embeddingDim}_BiDirection{bidirectional}_HL{hiddenLayer}_HS{hiddenSize}_P{interventionP}_MaxMerge{maxmerge}_MinMerge{minmerge}_preprocess{preprocess}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
weightPath=f'./LSTMWeight/{filename}'
os.mkdir(weightPath)
tensorboardpath=f"./Tensorboard2/{filename}"

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
writer=SummaryWriter(log_dir=tensorboardpath)
bernoulli= Bernoulli(torch.tensor([interventionP]))
preprocessP= Bernoulli(torch.tensor([preprocess]))
model=PredictLSTMIntervionP(n_vocab,embeddingDim,seqlen,seqlen,minmerge,maxmerge,batchsize,GRU,hiddenLayer,hiddenSize,bidirectional,True,numOfConvBlock,groupRelu,convpredict,kernelsize,mergeRate)
weightHistory={}
gate=['igone','forget','learn','output']
print(model.predict.weight.shape)
for name,para in model.RNN.named_parameters():
    if("weight" in name):
        print(f'{name} {para.data.shape}')
        data=torch.abs(para.data.detach().cpu())
        weightHistory[name]={}
        weightHistory[name]['previous']=data
        for gidx in range(len(gate)):
            weightHistory[name][gate[gidx]]={}
            weightHistory[name][gate[gidx]]['birthrate']=[]
            weightHistory[name][gate[gidx]]['deathrate']=[]
            weightHistory[name][gate[gidx]]['previous_pop']=torch.count_nonzero(data[gidx*hiddenSize:(gidx*hiddenSize)+hiddenSize,:])
if(numOfConvBlock>0):
    for name,para in model.conv1d.named_parameters():
        if("weight" in name):
            data=torch.abs(para.data.detach().cpu())
            weightHistory[name]={}
            weightHistory[name]['previous']=data
            weightHistory[name]['birthrate']=[]
            weightHistory[name]['deathrate']=[]
            weightHistory[name]['previous_pop']=torch.count_nonzero(data)
for name,para in model.predict.named_parameters():
    if("weight" in name):
        data=torch.abs(para.data.detach().cpu())
        weightHistory[name]={}
        weightHistory[name]['previous']=data
        weightHistory[name]['birthrate']=[]
        weightHistory[name]['deathrate']=[]
        weightHistory[name]['previous_pop']=torch.count_nonzero(data)
criterion = torch.nn.CrossEntropyLoss()
criterion2 = torch.nn.MSELoss()
optimizer = torch.optim.RMSprop(model.parameters()) if (RMS==True) else torch.optim.Adam(model.parameters())
currentbestaccy=0
test_text=test_pd['text']
test_label=test_pd['label']
trainbatchcount=0
valbatchcount=0
fig,axes=plt.subplots(4,1,figsize=(20,20), gridspec_kw={'height_ratios': [1,2,1,2]})
optstep = np.full(epochs,optstep)
if(dynamicOpt):
    optstep[100:150]=optstep[0]//2
    optstep[150:200]=optstep[0]//4
    optstep[200:]=1
print(optstep)
overallTop10Merge=Counter()
valmergeStatistic=Counter()
for epoch in range(epochs):
    mergeStatistic=Counter()
    trainloss=[]
    trainaddloss=[]
    avgtrainaccy=[]
    avgtrainaccy2=[]
    train_pd=train_pd.sample(frac=1)
    train_pd=train_pd.reset_index(drop=True)
    train_text=train_pd['text']
    train_label=train_pd['label']
    #for d in range(0,len(train_text),batchsize):
    s=time.time()
    avgdiffcount=[]
    for d in range(0,batchsize,batchsize):
        weight = torch.nn.functional.normalize(model.embeddingSpace.weight,dim=-1)
        state=model.init_state()
        losses=[]
        switch=bernoulli.sample()
        preprocessswitch=preprocessP.sample()
        sequences=[]
        targets=[]
        idxarray=[]
        tokendist=[]
        inhibitlists=[]
        promotelists=[]
        tokenpos=[]
        diffcount=np.zeros((batchsize,))
        for i,x in enumerate(nlp.pipe(train_text[d:d+batchsize])):
            data=[]
            inhibit=[]
            promote=[]
            pos=[]
            for t in x:
                if(preprocessswitch==0):
                    data+= [tok2id[t.text]]
                    pos+= [t.pos_]
                elif(t.is_stop==False and t.is_punct==False):
                    if(t.pos_ not in skipPOS):
                        data+= [tok2id[t.text]]
                        pos+= [t.pos_]
            tmp=np.zeros(maxlen)
            c=5 if(seqlen>=maxlen) else seqlen
            tmp[:c]=2
            tmp[c:]=train_label[d+i]
            idx=maxlen-1
            if(len(data)>maxlen):
                data=data[:maxlen-1]+[endid]
                pos=pos[:maxlen]
                tmp[-1]=2
            else:
                orglen=len(data)
                idx=orglen-1
                data=(data+([endid]*(maxlen-orglen)))
                pos=(pos+(['e0s']*(maxlen-orglen)))
                tmp[orglen:]=2
            idxarray.append(idx)
            targets.append(tmp)
            sequences.append(data)
            tokenpos.append(pos)
        targets=torch.tensor(np.array(targets),dtype=torch.long).to(device)
        sequences=np.array(sequences)
        idxarray=np.array(idxarray)
        predict_history=np.zeros((batchsize,maxlen,3))
        tokenpos=np.array(tokenpos)
        c=1 if(seqlen>=maxlen) else (maxlen-seqlen)
        t=train_label[d:d+batchsize].to_numpy()
        nidx=np.where(t==0)[0]
        pidx=np.where(t==1)[0]
        s1=time.time()
        for i in range(0,c,sliding):
            sequence=sequences[:,i:i+seqlen]
            target=targets[:,i:i+seqlen]
            poslist = tokenpos[:,i:i+seqlen]
            permuteposlist = poslist[:,permuteidx]
            if(not carryforward):
                 state=model.init_state()
            pred,codeword, state ,mergeidx =  model(sequence, state, switch,permuteidx,onlyMerge,poslist,consecutive)
            diffcount+=np.sum(poslist!=permuteposlist,axis=-1)
            endidx = np.where(permuteposlist=='e0s')[0]
            endidx = np.in1d(range(permuteposlist.shape[0]),endidx)
            selectidx = np.arange(len(permuteposlist))[~endidx]
            if(len(mergeidx)==batchsize):
                totalpos=[]
                for idx in selectidx:
                    for midx in [ m for m in mergeidx[idx] if(len(m)>1) ]:
                        #midx.sort()
                        pos = '_'.join(permuteposlist[idx,midx])
                        totalpos.append(pos)
                mergeStatistic.update(totalpos)
            else:
                permuteposlist=permuteposlist[selectidx,:]
                for midx in [ m for m in mergeidx if(len(m)>1) ]:
                    pos = permuteposlist[:,midx]
                    pos=[ "_".join(pos[pidx]) for pidx in range(len(pos))  ]
                    mergeStatistic.update(pos)

            loss=criterion(pred,target)
            if(criterion2==True):
                codeword_norm=torch.mean(torch.mean(torch.norm(codeword,dim=-1),dim=1))
                loss+=criterion2(codeword_norm,torch.zeros_like(codeword_norm))
            losses.append(loss.item())
            trainloss.append(loss.item())
            pred=torch.nn.functional.softmax(pred,dim=1)
            pred=pred.permute(0,2,1).detach().cpu().numpy()
            predict_history[:,i:i+seqlen,:]=pred[:batchsize]
            if(GRU==False):
                h_state = state[0].detach()
                c_state = state[1].detach()
            else:
                state = state.detach()
            loss.backward()
            if(i%optstep[epoch]==0):
                optimizer.step()
                optimizer.zero_grad()
            if(GRU==False):
               state=(h_state,c_state)
        e1=time.time()
        overallTop10Merge.update(dict(mergeStatistic.most_common(n=10)))
        [writer.add_histogram(f'EmbeddingSpace_Weight_{name}',para.data,trainbatchcount) for name,para in model.embeddingSpace.named_parameters()]
        if(switch==1):
            [writer.add_histogram(f'Merge_Conv1D_Weight_{name}',para.data,trainbatchcount) for name,para in model.mergeConv1D.named_parameters()]
        for name,para in model.RNN.named_parameters():
            writer.add_histogram(f'LSTM_{name}',para.data,trainbatchcount)
            if("weight" in name):
                previousweight = weightHistory[name]['previous']
                currentweight = para.data.detach().cpu()
                weightHistory[name]['previous']=currentweight
                for gidx in range(len(gate)):
                    g = gate[gidx]
                    previousGateWeight = previousweight[gidx*hiddenSize:gidx*hiddenSize+hiddenSize,:]
                    currentGateWeight = currentweight[gidx*hiddenSize:gidx*hiddenSize+hiddenSize,:]
                    diff = torch.flatten(previousGateWeight-torch.abs(currentGateWeight))
                    br=len(torch.where(diff>0)[0])
                    dr=len(torch.where(diff<0)[0])
                    freeze=len(torch.where(diff==0)[0])
                    init=weightHistory[name][g]['previous_pop']
                    weightHistory[name][g]['previous_pop']=torch.count_nonzero(torch.abs(currentGateWeight))
                    weightHistory[name][g]['birthrate'].append(br/init)
                    weightHistory[name][g]['deathrate'].append(dr/init)
                    writer.add_scalars(f'{name}_{g}_population',{'birth':(br/init),'death':(dr/init),'freeze':(freeze/init)},trainbatchcount)
                    writer.add_scalar(f'{name}_{g}_PopNorm',torch.norm(currentGateWeight),trainbatchcount)
                    writer.add_histogram(f'{name}_{g}',currentGateWeight,trainbatchcount)
        if(numOfConvBlock>0):
            for name,para in model.conv1d.named_parameters():
                writer.add_histogram(f'Conv1D_{name}',para.data,trainbatchcount)
                if("weight" in name):
                    previousweight = weightHistory[name]['previous']
                    currentweight = torch.abs(para.data.detach().cpu())
                    diff = torch.flatten(previousweight-currentweight)
                    br=len(torch.where(diff>0)[0])
                    dr=len(torch.where(diff<0)[0])
                    init=weightHistory[name]['previous_pop']
                    weightHistory[name]['previous_pop']=torch.count_nonzero(currentweight)
                    weightHistory[name]['previous']=currentweight
                    weightHistory[name]['birthrate'].append(br/init)
                    weightHistory[name]['deathrate'].append(dr/init)
                    name=name.replace(".","_")
                    writer.add_scalars(f'{name}_population',{'birth':(br/init),'death':(dr/init)},trainbatchcount)
        for name,para in model.predict.named_parameters():
            writer.add_histogram(f'Predict_{name}',para.data,trainbatchcount)
            if("weight" in name):
                previousweight = weightHistory[name]['previous']
                currentweight = torch.abs(para.data.detach().cpu())
                diff = torch.flatten(previousweight-currentweight)
                br=len(torch.where(diff>0)[0])
                dr=len(torch.where(diff<0)[0])
                init=weightHistory[name]['previous_pop']
                weightHistory[name]['previous_pop']=torch.count_nonzero(currentweight)
                weightHistory[name]['previous']=currentweight
                weightHistory[name]['birthrate'].append(br/init)
                weightHistory[name]['deathrate'].append(dr/init)
                writer.add_scalars(f'PredictConv1D_{name}_population',{'birth':(br/init),'death':(dr/init)},trainbatchcount)
        avgloss=np.mean(losses)
        target=train_label[d:d+batchsize].to_numpy()
        nidx=np.where(target==0)[0]
        pidx=np.where(target==1)[0]
        est_prediction=[]
        est_prediction2=[]
        est_magnitude=[]
        drawcount=0
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
        cr=classification_report(target,np.round(est_prediction),output_dict=True)
        c=5 if(seqlen>=maxlen) else seqlen
        maxidx= [ np.argmax(predict_history[i, c:idxarray[i]-1, :],axis=-1) for i in range(batchsize)]
        [ est_prediction2.append(np.mean(maxidx[i],axis=-1)) for i in range(batchsize) ]
        batchaccy2=np.sum(np.abs(est_prediction2-target)<0.05)/batchsize
        key=list(cr.keys())
        del cr[key[0]]['support']
        del cr[key[1]]['support']
        del cr['macro avg']['support']
        batchmagnitude=np.median(est_magnitude)
        avgtrainaccy.append(batchaccy)
        avgtrainaccy2.append(batchaccy2)
        negsample=sequences[nidx]
        possample=sequences[pidx]
        negsampleidx = np.argmax(np.sum(np.isin(negsample, trainposlabel), axis=-1))
        possampleidx = np.argmax(np.sum(np.isin(possample, trainneglabel), axis=-1))
        c=5 if(seqlen>=maxlen) else seqlen
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
        writer.add_scalars(f'Batch NegProb',{'mean':np.mean(nprob),'q25':np.quantile(nprob,0.25),'median':np.median(nprob),'q75':np.quantile(nprob,0.75)},trainbatchcount)
        writer.add_scalars(f'Batch PosProb',{'mean':np.mean(pprob),'q25':np.quantile(pprob,0.25),'median':np.median(pprob),'q75':np.quantile(pprob,0.75)},trainbatchcount)
        writer.add_scalars('Training ClassificationReport Neg Class',cr[key[0]],trainbatchcount)
        writer.add_scalars('Training ClassificationReport Pos Class',cr[key[1]],trainbatchcount)
        writer.add_scalars('Training ClassificationReport MarcoAvg',cr['macro avg'],trainbatchcount)
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
        if(len(posidx2)>0): [ axes[2].axvline(i,linewidth=2,c='#0000EE') for i in posidx2]
        if(len(negidx)>0): [ axes[0].axvline(i,linewidth=2,c='#F39C12') for i in negidx]
        if(len(negidx2)>0): [ axes[0].axvline(i,linewidth=2,c='#0000EE') for i in negidx2]
        fig.suptitle(f'Batch Accy : {batchaccy} {batchaccy2}')
        axes[2].set_title(",".join([ id2tok[x] for x in trainneglabel if(id2tok[x] in possampletext)]))
        axes[0].set_title(",".join([ id2tok[x] for x in trainposlabel if(id2tok[x] in negsampletext)]))
        [ axes[i].legend() for i in range(4) ]
        plt.tight_layout()
        fig.savefig(f'{tensorboardpath}/Training_Epoch_{epoch}_batch_{trainbatchcount}_switch{int(switch)}_preprocess{int(preprocessswitch)}_{avgloss:0.6f}_plot.png',dpi=400)
        [ axes[i].clear() for i in range(4) ]
        plt.cla()
        diffcount=np.mean(diffcount/idxarray)
        avgdiffcount.append(diffcount)
        e=time.time()
        printout=f'Epoch: {epoch} Batch TrainDoc#{(d+batchsize):5d}, Switch:{int(switch):1d}, Preprocess:{int(preprocessswitch):1d}, AvgLoss:{avgloss:0.4f}, AvgAccy:{batchaccy:0.3f}, AvgAccy2:{batchaccy2:0.3f}, DiffPOS:{diffcount:2.3f}'
        #printout+=f'PStat:({predict_mean:0.3f},{quantile25:0.3f},{median:0.3f},{quantile75:0.3f}), AvgAccy:{batchaccy:0.3f} '
        print(printout,flush=True)
        trainbatchcount+=1
    avgtrainingloss=np.mean(trainloss)
    avgtrainingAccy=np.mean(avgtrainaccy)
    avgtrainingAccy2=np.mean(avgtrainaccy2)
    avgdiffcount=np.mean(avgdiffcount)
    diffcount=diffcount/(len(train_text)*((maxlen-seqlen)/sliding))
    writer.add_scalar(f'Epoch Training AvgLoss',avgtrainingloss,epoch)
    writer.add_scalar(f'Epoch Training AvgAccy(abs value)',avgtrainingAccy,epoch)
    writer.add_scalar(f'Epoch Training AvgAccy2(abs value)',avgtrainingAccy2,epoch)
    print(f'Epoch: {epoch:2d} Training Finished, AvgTrainingLoss:{avgtrainingloss:0.6f}, AvgAccy:{avgtrainingAccy:0.3f}, DiffPOS:{avgdiffcount:2.3f}, Time:{(e-s):2.4f}, Time2:{(e1-s1):2.4f}',flush=True)
    print(f'Epoch: {epoch:2d} Training Finished, Merge:{overallTop10Merge.most_common(n=10)}',flush=True)
    with torch.no_grad():
        vallosses=[]
        valAvgAccy=[]
        valAvgAccy2=[]
        valFNegSimility,valFPosSimility,valFbtwGroupSimility,valFNegSimilityMedian,valFPosSimilityMedian,valFbtwGroupSimilityMedian=[],[],[],[],[],[]
        valBNegSimility,valBPosSimility,valBbtwGroupSimility,valBNegSimilityMedian,valBPosSimilityMedian,valBbtwGroupSimilityMedian=[],[],[],[],[],[]
        valBlockNegSimility,valBlockPosSimility,valBlockbtwGroupSimility=[],[],[]
        valblocknegNorm,valblockposNorm=np.zeros(12),np.zeros(12)
        valnegNorm,valposNorm=[],[]
        valbtwGroupSimilityRaw,valNegSimilityRaw,valPosSimilityRaw=[],[],[]
        valbtwNegBlock,valbtwPosBlock=[],[]
        switchcount=0
        tokendist=[]
        valavgdiffcount=[]
        for d in range(0,len(test_text),batchsize):
            diffcount=np.zeros((batchsize,))
            switch=bernoulli.sample()
            preprocessswitch=preprocessP.sample()
            state=model.init_state()
            losses=[]
            negFCellStateSimility,posFCellStateSimility,negBCellStateSimility,posBCellStateSimility,FbtwGroupSimility,BbtwGroupSimility=[],[],[],[],[],[]
            negbtwBlock,posbtwBlock=[],[]
            negSimialityRaw,posSimialityRaw,btwSimilityRaw=[],[],[]
            img=torch.zeros((batchsize,1,48,48))
            sequences,targets,idxarray,tokenpos,negNorm,posNorm,blocknegNorm,blockposNorm=[],[],[],[],[],[],[],[]
            for i,x in enumerate(nlp.pipe(test_text[d:d+batchsize])):
                data=[]
                pos=[]
                for t in x:
                    if(preprocessswitch==0):
                        data+= [tok2id[t.text]]
                        pos+= [t.pos_]
                    elif(t.is_stop==False and t.is_punct==False):
                        if(t.pos_ not in skipPOS):
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
            #nmind=np.zeros((len(nidx),maxlen//seqlen))
            #pmind=np.zeros((len(pidx),maxlen//seqlen))
            c=1 if (seqlen>=maxlen) else (maxlen-seqlen)
            minlen =len(pidx)-1 if (len(nidx)>len(pidx)) else len(nidx)-1
            previousOutput=None
            for i in range(0,c,sliding):
                sequence=sequences[:,i:i+seqlen]
                t=targets[:,i:i+seqlen]
                poslist = tokenpos[:,i:i+seqlen]
                permuteposlist = poslist[:,permuteidx]
                diffcount+=np.sum(poslist!=permuteposlist,axis=-1)
                pred,output,state,mergeidx =model(sequence,state,switch,permuteidx,onlyMerge,poslist,consecutive)
                g=time.time()
                if(GRU==False):
                    ns,ps,btwgroups,nnorm,pnorm = rnnOutputSimility(output[nidx],output[pidx],minlen)
                    negFCellStateSimility.append(np.mean(ns))
                    posFCellStateSimility.append(np.mean(ps))
                    FbtwGroupSimility.append(np.mean(btwgroups))
                    negNorm.append(nnorm)
                    posNorm.append(pnorm)
                    blocknegNorm.append(np.mean(np.mean(nnorm,axis=0)))
                    blockposNorm.append(np.mean(np.mean(pnorm,axis=0)))
                    negSimialityRaw.append(np.array(ns))
                    posSimialityRaw.append(np.array(ps))
                    btwSimilityRaw.append(np.array(btwgroups))
                    if(i>0):
                        _,_,negbtw,_,_ = rnnOutputSimility(output[nidx],previousOutput[nidx],len(nidx))
                        _,_,posbtw,_,_ = rnnOutputSimility(output[pidx],previousOutput[pidx],len(pidx))
                        negbtwBlock.append(np.array(negbtw))
                        posbtwBlock.append(np.array(posbtw))
                previousOutput=output
                loss=criterion(pred,t)
                losses.append(loss.item())
                pred=torch.nn.functional.softmax(pred,dim=1)
                pred=pred.permute(0,2,1).detach().cpu().numpy()
                predict_history[:,i:i+seqlen,:]=pred
            if(GRU==False):
                valblocknegNorm+=np.array(blocknegNorm[1:13])
                valblockposNorm+=np.array(blockposNorm[1:13])
                negbtwBlock=np.array(negbtwBlock)
                posbtwBlock=np.array(posbtwBlock)
                negSimialityRaw=np.array(negSimialityRaw)
                posSimialityRaw=np.array(posSimialityRaw)
                valnegNorm.append(np.array(negNorm))
                valposNorm.append(np.array(posNorm))
                negFCellStateSimility=np.array(negFCellStateSimility).reshape(-1,)
                posFCellStateSimility=np.array(posFCellStateSimility).reshape(-1,)
                FbtwGroupSimility=np.array(FbtwGroupSimility).reshape(-1,)
                valFNegSimility.append(np.mean(negFCellStateSimility[25:325]))
                valFPosSimility.append(np.mean(posFCellStateSimility[25:325]))
                valNegSimilityRaw.append(negFCellStateSimility[25:325])
                valPosSimilityRaw.append(posFCellStateSimility[25:325])
                valFNegSimilityMedian.append(np.median(negFCellStateSimility[25:325]))
                valFPosSimilityMedian.append(np.median(posFCellStateSimility[25:325]))
                valFbtwGroupSimility.append(np.mean(FbtwGroupSimility[25:325]))
                valbtwGroupSimilityRaw.append(FbtwGroupSimility[25:325])
                valbtwNegBlock.append(negbtwBlock[:13,:])
                valbtwPosBlock.append(posbtwBlock[:13,:])
                valNegSimilityRaw.append(negSimialityRaw[:13,:])
                valPosSimilityRaw.append(posSimialityRaw[:13,:])
                valbtwGroupSimilityRaw.append(btwSimilityRaw[:13,:])
                valFbtwGroupSimilityMedian.append(np.median(FbtwGroupSimility[25:325]))
                valBlockNegSimility.append([ np.mean(negFCellStateSimility[i:i+seqlen+1]) for i in range(seqlen,325,seqlen)])
                valBlockPosSimility.append([ np.mean(posFCellStateSimility[i:i+seqlen+1]) for i in range(seqlen,325,seqlen)])
                valBlockbtwGroupSimility.append([ np.mean(FbtwGroupSimility[i:i+seqlen+1]) for i in range(seqlen,325,seqlen)])
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
            #nimg = make_grid(torch.logit(img[nidx[:10]]),nrow=2,padding=5,normalize=False,pad_value=0.8)
            #pimg = make_grid(torch.logit(img[pidx[:10]]),nrow=2,padding=5,normalize=False,pad_value=0.8)
            #nimg = nimg.unsqueeze(dim=1)
            #pimg = pimg.unsqueeze(dim=1)
            writer.add_scalars(f'Validation Batch NegProb',{'mean':np.mean(nprob),'q25':np.quantile(nprob,0.25),'median':np.median(nprob),'q75':np.quantile(nprob,0.75)},valbatchcount)
            writer.add_scalars(f'Validation Batch PosProb',{'mean':np.mean(pprob),'q25':np.quantile(pprob,0.25),'median':np.median(pprob),'q75':np.quantile(pprob,0.75)},valbatchcount)
            writer.add_scalars('Validation ClassificationReport Neg Class',cr[key[0]],valbatchcount)
            writer.add_scalars('Validation ClassificationReport Pos Class',cr[key[1]],valbatchcount)
            writer.add_scalars('Validation ClassificationReport MacroAvg',cr['macro avg'],valbatchcount)
            #writer.add_image(f'Validation NegImg', nimg, valbatchcount,dataformats='NCHW')
            #writer.add_image(f'Validation PosImg', pimg, valbatchcount,dataformats='NCHW')
            #if(GRU==False):
            #    writer.add_histogram('Neg Validation Mind',nmind,valbatchcount)
            #    writer.add_histogram('Pos Validation Mind',pmind,valbatchcount)
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
            fig.savefig(f'{tensorboardpath}/Validation_Epoch_{epoch}_batch_{valbatchcount}_switch{int(switch)}_preprocess{int(preprocessswitch)}_{avgloss:0.6f}_plot.png',dpi=400)
            [ axes[i].clear() for i in range(4) ]
            plt.cla()#
            diffcount=np.mean(diffcount/idxarray)
            valavgdiffcount.append(diffcount)
            print(f'Epoch: {epoch:2d} Batch ValDoc#{(d+batchsize):5d}, Switch:{int(switch):1d}, Preprocess:{int(preprocessswitch):1d}, AvgLoss:{avgloss:0.3f}, AvgAccy:{batchaccy:0.3f}, AvgAccy2:{batchaccy2:0.3f}, DiffPOS:{diffcount:2.3f}',flush=True)
            valbatchcount+=1
        avgValloss=np.mean(vallosses)
        valAvgAccy = np.mean(valAvgAccy)
        valAvgAccy2 = np.mean(valAvgAccy2)
        valavgdiffcount=np.mean(valavgdiffcount)
        valNS = np.mean(valFNegSimility)
        valPS = np.mean(valFPosSimility)
        valbtwS = np.mean(valFbtwGroupSimility)
        valNSM = np.mean(valFNegSimilityMedian)
        valPSM = np.mean(valFPosSimilityMedian)
        valbtwGM = np.mean(valFbtwGroupSimilityMedian)
        valBNS = np.mean(valBNegSimility)
        valBPS = np.mean(valBPosSimility)
        valBbtwS = np.mean(valBbtwGroupSimility)
        valBNSM = np.mean(valBNegSimilityMedian)
        valBPSM = np.mean(valBPosSimilityMedian)
        valBbtwGM = np.mean(valBbtwGroupSimilityMedian)
        valBlockNegSimility=np.mean(np.array(valBlockNegSimility),axis=0)
        valBlockPosSimility=np.mean(np.array(valBlockPosSimility),axis=0)
        valBlockbtwGroupSimility=np.mean(np.array(valBlockbtwGroupSimility),axis=0)
        if(valAvgAccy>currentbestaccy):
            currentbestaccy=valAvgAccy
            torch.save(model,f'{weightPath}/Val_Epoch_{epoch}_accy_{currentbestaccy}_loss_{avgValloss}.pt')
        writer.add_scalar(f'Validation AvgLoss',avgValloss,epoch)
        writer.add_scalar(f'Validation AvgAccy',np.mean(valAvgAccy),epoch)
        writer.add_scalar(f'Validation AvgAccy2',np.mean(valAvgAccy2),epoch)
        writer.add_scalars(f'Validation negNorm',[{f'negblockNorm_{i}':valblocknegNorm[i]/3 for i in range(12)}][0],epoch)
        writer.add_scalars(f'Validation posNorm',[{f'posblockNorm_{i}':valblockposNorm[i]/3 for i in range(12)}][0],epoch)
        writer.add_scalars(f'Validation negSimiality',[{f'block_{i}':valBlockNegSimility[i] for i in range(12)}][0],epoch)
        writer.add_scalars(f'Validation posSimiality',[{f'block_{i}':valBlockPosSimility[i] for i in range(12)}][0],epoch)
        writer.add_scalars(f'Validation BTWSimiality',[{f'block_{i}':valBlockbtwGroupSimility[i] for i in range(12)}][0],epoch)
        writer.add_scalars(f'Validation NegBlockSimiality',[{f'block_{i}':valbtwNegBlock[i] for i in range(13)}][0],epoch)
        writer.add_scalars(f'Validation POSBlockSimiality',[{f'block_{i}':valbtwPosBlock[i] for i in range(13)}][0],epoch)
        for block in range(1,13):
            ndata=np.vstack((valnegNorm[0][block,:,:],valnegNorm[1][block,:,:],valnegNorm[2][block,:,:]))
            pdata=np.vstack((valposNorm[0][block,:,:],valposNorm[1][block,:,:],valposNorm[2][block,:,:]))
            nsdata=np.vstack((valNegSimilityRaw[0][block,:],valNegSimilityRaw[1][block,:],valNegSimilityRaw[2][block,:]))
            psdata=np.vstack((valPosSimilityRaw[0][block,:],valPosSimilityRaw[1][block,:],valPosSimilityRaw[2][block,:]))
            btwdata=np.vstack(valbtwGroupSimilityRaw[0][block,:],valbtwGroupSimilityRaw[1][block,:],valbtwGroupSimilityRaw[2][block,:])
            btwNegBlockdata = np.vstack((valBlockNegSimility[0][block,:],valBlockNegSimility[1][block,:],valBlockNegSimility[2][block,:]))
            btwPosBlockdata = np.vstack((valBlockPosSimility[0][block,:],valBlockPosSimility[1][block,:],valBlockPosSimility[2][block,:]))
            writer.add_histogram(f'Validation neg norm dist {block}',ndata,epoch)
            writer.add_histogram(f'Validation pos norm dist {block}',pdata,epoch)
            writer.add_histogram(f'Validation neg sim Dist {block}',nsdata,epoch)
            writer.add_histogram(f'Validation pos sim Dist {block}',psdata,epoch)
            writer.add_histogram(f'Validation btw sim Dist {block}',btwdata,epoch)
            writer.add_histogram(f'Validation neg block sim Dist {block}',btwNegBlockdata,epoch)
            writer.add_histogram(f'Validation pos block sim Dist {block}',btwPosBlockdata,epoch)
    if(GRU==False):
        print(f'Epoch: {epoch:2d} Validation Finished, Avg Loss:{avgValloss:0.6f}, AvgAccy:{valAvgAccy:0.3f}, AvgAccy2:{valAvgAccy2:0.3f}, DiffPOS:{valavgdiffcount:2.3f}, Forward:{{({valNS:0.3f},{valNSM:0.3f}),({valPS:0.3f},{valPSM:0.3f}),({valbtwS:0.3f},{valbtwGM:0.3f})}}',flush=True)
        print(f'Epoch: {epoch:2d} Validation Finished, Forward:{{({valNS:0.3f},{valNSM:0.3f}),({valPS:0.3f},{valNS:0.3f}),({valbtwS:0.3f},{valbtwGM:0.3f})}} Backward:{{({valBNS:0.3f},{valBNSM:0.3f}),({valBPS:0.3f},{valBPSM:0.3f}),({valBbtwS:0.3f},{valBbtwGM:0.3f})}}',flush=True)
    else:
        print(f'Epoch: {epoch:2d} Validation Finished, Avg Loss:{avgValloss:0.6f}, AvgAccy:{valAvgAccy:0.3f}, AvgAccy2:{valAvgAccy2:0.3f}, DiffPOS:{valavgdiffcount:2.3f}',flush=True)
    #print(f'Epoch: {epoch:2d} Validation Finished, Merge: {valmergeStatistic.most_common(n=10)}',flush=True)
    with open(f'{tensorboardpath}/trainOverAllTop10_MergeStatistic_{epoch}.pkl','wb') as f:
        pickle.dump(overallTop10Merge,f)
    #with open(f'{tensorboardpath}/ValMergeStatistic_{epoch}.pkl','wb') as f:
    #    pickle.dump(valmergeStatistic,f)
    print("")
