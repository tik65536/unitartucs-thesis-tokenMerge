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
from Model.LSTMwithImg import LSTMwithImg
from torch.distributions.bernoulli import Bernoulli
from torchvision.utils import make_grid
from PIL import Image
import torchvision.transforms as transforms
import datetime
import pickle
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
import warnings
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

def loadImage(pos,neg):
    transform_list = [transforms.ToTensor()]
    transform = transforms.Compose(transform_list)
    poslist=np.array(os.listdir("./images/train/happy/"))
    neglist=np.array(os.listdir("./images/train/disgust/"))
    posidx=np.arange(len(poslist))
    negidx=np.arange(len(neglist))
    np.random.shuffle(posidx)
    np.random.shuffle(negidx)
    poslist=poslist[posidx[:neg]]
    neglist=neglist[negidx[:pos]]
    possample=torch.zeros((len(poslist),1,48,48))
    negsample=torch.zeros((len(neglist),1,48,48))
    for i,fname in enumerate(poslist):
        img = Image.open(f'./images/train/happy/{fname}')
        img = transform(img)
        possample[i]=img
    for i,fname in enumerate(neglist):
        img = Image.open(f'./images/train/disgust/{fname}')
        img = transform(img)
        negsample[i]=img
    return possample,negsample


parser = argparse.ArgumentParser()
parser.add_argument('-epoch', type=int, default=20,
                    help='Epoch to run')

parser.add_argument('-batch', type=int, default=5,
                    help='BatchSize')

parser.add_argument('-maxlen', type=int, default=600,
                    help='Max text len')

parser.add_argument('-preprocesstext', type=float, default=1,
                    help='Remove Stop word and Pun')

parser.add_argument('-embeddingDim', type=int, default=128,
                    help='Embedding Dim')

parser.add_argument('-seqlen', type=int, default=5,
                    help='Seq Len for reading token')

parser.add_argument('-slide', type=int, default=1,
                    help='Seqlen Sliding')

parser.add_argument('-hiddenLayer', type=int, default=20,
                    help='Hidden Layer of LSTM')

parser.add_argument('-hiddenSize', type=int, default=128,
                    help='Hidden Size of LSTM')

parser.add_argument('-bidirection', type=int, default=0,
                    help='0 = false for bi-direction')

parser.add_argument('-remark', type=str, default='',
                    help='Remark on filename')

parser.add_argument('-additionalloss', type=int, default=0,
                    help='0: No , 1:img , 2: similarity')

args = parser.parse_args()
seqlen=args.seqlen
batchsize=args.batch
embeddingDim=args.embeddingDim
epochs = args.epoch
hiddenSize=args.hiddenSize
hiddenLayer=args.hiddenLayer
sliding=args.slide
bidirectional= True if (args.bidirection==1) else False
preprocess = args.preprocesstext
maxlen =args.maxlen
remark = args.remark
additionalloss=args.additionalloss

print(f'Run Para : {args}',flush=True)

filename=f"{remark}seqlen{seqlen}_sldie{sliding}_batch{batchsize}_e{embeddingDim}_BiDirection{bidirectional}_HL{hiddenLayer}_HS{hiddenSize}_preprocess{preprocess}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
weightPath=f'./LSTMWeight/{filename}'
os.mkdir(weightPath)
tensorboardpath=f"./Tensorboard2/{filename}"

#valbatchsize=2
#imdb_dataset = load_dataset('imdb', split=['train[10000:10010]', 'train[10000:10010]', 'test[:20]'])
#imdb_dataset = load_dataset('imdb')
imdb_dataset = load_dataset('imdb', split=['train[5000:20000]'])
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
preprocessP= Bernoulli(torch.tensor([preprocess]))
model=LSTMwithImg(n_vocab,embeddingDim,seqlen,batchsize,hiddenLayer,hiddenSize,bidirectional,additionalloss)
weightHistory={}
for name,para in model.RNN.named_parameters():
    if("weight" in name):
        data=torch.abs(para.data[:seqlen,:].detach().cpu())
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
imgbce=torch.nn.BCEWithLogitsLoss()
mseloss=torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())
currentbestaccy=0
test_text=test_pd['text']
test_label=test_pd['label']
trainbatchcount=0
valbatchcount=0
fig,axes=plt.subplots(4,1,figsize=(20,20), gridspec_kw={'height_ratios': [1,2,1,2]})
for epoch in range(epochs):
    trainloss=[]
    trainaddloss=[]
    avgtrainaccy=[]
    avgtrainaccy2=[]
    train_pd=train_pd.sample(frac=1)
    train_pd=train_pd.reset_index(drop=True)
    train_text=train_pd['text']
    train_label=train_pd['label']
    #for d in range(0,len(train_text),batchsize):
    for d in range(0,batchsize,batchsize):
        weight = torch.nn.functional.normalize(model.embeddingSpace.weight,dim=-1)
        state=model.init_state()
        losses=[]
        addlosses=[]
        preprocessswitch=preprocessP.sample()
        sequences=[]
        targets=[]
        idxarray=[]
        tokendist=[]
        for i,x in enumerate(nlp.pipe(train_text[d:d+batchsize])):
            data=[]
            inhibit=[]
            prohibit=[]
            for t in x:
                if(preprocessswitch==0):
                    data+= [tok2id[t.text]]
                elif(t.is_stop==False and t.is_punct==False):
                    data+= [tok2id[t.text]]
            tmp=np.zeros(maxlen)
            c=5 if(seqlen>=maxlen) else seqlen
            tmp[:c]=2
            tmp[c:]=train_label[d+i]
            idx=maxlen-1
            if(len(data)>maxlen):
                data=data[:maxlen-1]+[endid]
                tmp[-1]=2
            else:
                orglen=len(data)
                idx=orglen-1
                data=(data+([endid]*(maxlen-orglen)))
                tmp[orglen:]=2
            idxarray.append(idx)
            targets.append(tmp)
            sequences.append(data)
        targets=torch.tensor(np.array(targets),dtype=torch.long).to(device)
        sequences=np.array(sequences)
        predict_history=np.zeros((batchsize,maxlen,3))
        c=1 if(seqlen>=maxlen) else (maxlen-seqlen)
        t=train_label[d:d+batchsize].to_numpy()
        nidx=np.where(t==0)[0]
        pidx=np.where(t==1)[0]
        for i in range(0,c,sliding):
            sequence=sequences[:,i:i+seqlen]
            target=targets[:,i:i+seqlen]
            pred,dream,similarity,state =  model(sequence, state,nidx,pidx)
            #pred = pred.detach()
            loss=criterion(pred,target)
            if(additionalloss==1):
                possample,negsample=loadImage(len(pidx),len(nidx))
                imgTarget=torch.zeros((batchsize,1,48,48))
                imgTarget[nidx]=possample
                imgTarget[pidx]=negsample
                imgTarget=imgTarget.to(device)
                addloss=imgbce(dream,imgTarget)
            if(additionalloss==2):
                starget=torch.full((1,1),-1)
                addloss=mseloss(similarity,starget)
            losses.append(loss.item())
            if(additionalloss>0):
                addlosses.append(addloss.item())
            trainloss.append(loss.item())
            pred=torch.nn.functional.softmax(pred,dim=1)
            pred=pred.permute(0,2,1).detach().cpu().numpy()
            predict_history[:,i:i+seqlen,:]=pred[:batchsize]
            loss = loss if(additionalloss==0) else loss+addloss
            hstate=state[0].detach()
            cstate=state[1].detach()
            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
            state=(hstate,cstate)

        [writer.add_histogram(f'EmbeddingSpace_Weight_{name}',para.data,trainbatchcount) for name,para in model.embeddingSpace.named_parameters()]
        for name,para in model.RNN.named_parameters():
            writer.add_histogram(f'LSTM_{name}',para.data,trainbatchcount)
            if("weight" in name):
                previousweight = weightHistory[name]['previous']
                currentweight = torch.abs(para.data[:seqlen,:].detach().cpu())
                diff = torch.flatten(previousweight-currentweight)
                br=len(torch.where(diff>0)[0])
                dr=len(torch.where(diff<0)[0])
                init=weightHistory[name]['previous_pop']
                weightHistory[name]['previous_pop']=torch.count_nonzero(currentweight)
                weightHistory[name]['previous']=currentweight
                weightHistory[name]['birthrate'].append(br/init)
                weightHistory[name]['deathrate'].append(dr/init)
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
        avgaddloss=np.mean(addlosses)
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
        fig.savefig(f'{tensorboardpath}/Training_Epoch_{epoch}_batch_{trainbatchcount}_preprocess{int(preprocessswitch)}_{avgloss:0.6f}_plot.png',dpi=400)
        [ axes[i].clear() for i in range(4) ]
        plt.cla()
        printout=f'Epoch: {epoch} Batch TrainDoc#{(d+batchsize):5d}, Preprocess:{int(preprocessswitch):1d}, AvgLoss:{avgloss:0.4f}, AvgAddLoss:{avgaddloss:0.4f} AvgAccy:{batchaccy:0.3f}, AvgAccy2:{batchaccy2:0.3f}'
        #printout+=f'PStat:({predict_mean:0.3f},{quantile25:0.3f},{median:0.3f},{quantile75:0.3f}), AvgAccy:{batchaccy:0.3f} '
        print(printout,flush=True)
        trainbatchcount+=1
    avgtrainingloss=np.mean(trainloss)
    avgtrainingAccy=np.mean(avgtrainaccy)
    avgtrainingAccy2=np.mean(avgtrainaccy2)
    writer.add_scalar(f'Epoch Training AvgLoss',avgtrainingloss,epoch)
    writer.add_scalar(f'Epoch Training AvgAccy(abs value)',avgtrainingAccy,epoch)
    writer.add_scalar(f'Epoch Training AvgAccy2(abs value)',avgtrainingAccy2,epoch)
    print(f'Epoch: {epoch:2d} Training Finished, AvgTrainingLoss:{avgtrainingloss:0.6f}, AvgAccy:{avgtrainingAccy:0.3f}',flush=True)
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
        weight = torch.nn.functional.normalize(model.embeddingSpace.weight,dim=-1)
        for d in range(0,len(test_text),batchsize):
            preprocessswitch=preprocessP.sample()
            state=model.init_state()
            losses=[]
            negCellStateSimility=[]
            posCellStateSimility=[]
            btwGroupSimility=[]
            img=torch.zeros((batchsize,1,48,48))
            sequences=[]
            targets=[]
            idxarray=[]
            for i,x in enumerate(nlp.pipe(test_text[d:d+batchsize])):
                data = [ tok2id[t.text] for t in x ] if(preprocessswitch==0) else [ tok2id[t.text] for t in x if(t.is_stop==False and t.is_punct==False) ]
                tmp=np.zeros(maxlen)
                c=5 if(seqlen>=maxlen) else seqlen
                tmp[:c]=2
                tmp[c:]=test_label[d+i]
                idx=maxlen-1
                if(len(data)>maxlen):
                    data=data[:maxlen-1]+[endid]
                    tmp[-1]=2
                else:
                    orglen=len(data)
                    idx=orglen-1
                    data=(data+([endid]*(maxlen-orglen)))
                    tmp[orglen:]=2
                idxarray.append(idx)
                targets.append(tmp)
                sequences.append(data)
            idxarray=np.array(idxarray)
            targets=torch.tensor(np.array(targets),dtype=torch.long).to(device)
            sequences=np.array(sequences)
            predict_history=np.zeros((batchsize,maxlen,3))
            c=1 if (seqlen>=maxlen) else (maxlen-seqlen)
            target=test_label[d:d+batchsize].to_numpy()
            nidx=np.where(target==0)[0]
            pidx=np.where(target==1)[0]
            nmind=np.zeros((len(nidx),maxlen//seqlen))
            pmind=np.zeros((len(pidx),maxlen//seqlen))
            for i in range(0,c,sliding):
                sequence=sequences[:,i:i+seqlen]
                t=targets[:,i:i+seqlen]
                pred,dream,similarity,state =model(sequence,state,nidx,pidx)
                ns,ps,btwgroups,nwidx,pwidx,_,_ = CellStateSimility(state[1],nidx,pidx,weight)
                negCellStateSimility.append(ns)
                posCellStateSimility.append(ps)
                btwGroupSimility.append(btwgroups.detach().cpu().numpy())
                nmind[:,i//seqlen]=nwidx
                pmind[:,i//seqlen]=pwidx
                #pred=pred.reshape(batchsize,seqlen) # N,L,1 -> L,1 for batchsize == 1
                loss=criterion(pred,t)
                losses.append(loss.item())
                pred=torch.nn.functional.softmax(pred,dim=1)
                pred=pred.permute(0,2,1).detach().cpu().numpy()
                predict_history[:,i:i+seqlen,:]=pred
                if((endid in sequence)):
                    ridx=np.where(sequence==endid)[0]
                    img[ridx]=dream[ridx].detach().cpu()
            avgloss=np.mean(losses)
            avgNegSimiality=np.mean(negCellStateSimility)
            avgPosSimiality=np.mean(posCellStateSimility)
            avgNegSimialityMedian=np.median(negCellStateSimility)
            avgPosSimialityMedian=np.median(posCellStateSimility)
            avgbtwGroupSimiality=np.mean(btwGroupSimility)
            avgbtwGroupSimialityMedian=np.median(btwGroupSimility)
            vallosses.append(avgloss)
            valNegSimility.append(avgNegSimiality)
            valPosSimility.append(avgPosSimiality)
            valbtwGroupSimility.append(avgbtwGroupSimiality)
            valNegSimilityMedian.append(avgNegSimialityMedian)
            valPosSimilityMedian.append(avgPosSimialityMedian)
            valbtwGroupSimilityMedian.append(avgbtwGroupSimialityMedian)
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
            nimg = make_grid(torch.logit(img[nidx[:10]]),nrow=2,padding=5,normalize=False,pad_value=0.8)
            pimg = make_grid(torch.logit(img[pidx[:10]]),nrow=2,padding=5,normalize=False,pad_value=0.8)
            nimg = nimg.unsqueeze(dim=1)
            pimg = pimg.unsqueeze(dim=1)
            writer.add_scalars(f'Validation Batch NegProb',{'mean':np.mean(nprob),'q25':np.quantile(nprob,0.25),'median':np.median(nprob),'q75':np.quantile(nprob,0.75)},valbatchcount)
            writer.add_scalars(f'Validation Batch PosProb',{'mean':np.mean(pprob),'q25':np.quantile(pprob,0.25),'median':np.median(pprob),'q75':np.quantile(pprob,0.75)},valbatchcount)
            writer.add_scalars('Validation ClassificationReport Neg Class',cr[key[0]],valbatchcount)
            writer.add_scalars('Validation ClassificationReport Pos Class',cr[key[1]],valbatchcount)
            writer.add_scalars('Validation ClassificationReport MacroAvg',cr['macro avg'],valbatchcount)
            writer.add_image(f'Validation NegImg', nimg, valbatchcount,dataformats='NCHW')
            writer.add_image(f'Validation PosImg', pimg, valbatchcount,dataformats='NCHW')
            writer.add_histogram('Neg Validation Mind',nmind,valbatchcount)
            writer.add_histogram('Pos Validation Mind',pmind,valbatchcount)
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
            fig.savefig(f'{tensorboardpath}/Validation_Epoch_{epoch}_batch_{valbatchcount}_preprocess{int(preprocessswitch)}_{avgloss:0.6f}_plot.png',dpi=400)
            [ axes[i].clear() for i in range(4) ]
            plt.cla()#
            print(f'Epoch: {epoch:2d} Batch ValDoc#{(d+batchsize):5d}, Preprocess:{int(preprocessswitch):1d}, AvgLoss:{avgloss:0.3f}, AvgAccy:{batchaccy:0.3f}, AvgAccy2:{batchaccy2:0.3f}, NS:({avgNegSimiality:0.3f},{avgNegSimialityMedian:0.3f}), PS:({avgPosSimiality:0.3f},{avgPosSimialityMedian:0.3f}), btwGroup:({avgbtwGroupSimiality:0.3f},{avgbtwGroupSimialityMedian:0.3f})',flush=True)
            valbatchcount+=1
        avgValloss=np.mean(vallosses)
        valAvgAccy = np.mean(valAvgAccy)
        valAvgAccy2 = np.mean(valAvgAccy2)
        valNS = np.mean(valNegSimility)
        valPS = np.mean(valPosSimility)
        valbtwS = np.mean(valbtwGroupSimility)
        valNSM = np.mean(valNegSimilityMedian)
        valPSM = np.mean(valPosSimilityMedian)
        valbtwGM = np.mean(valbtwGroupSimilityMedian)
        if(valAvgAccy>currentbestaccy):
            currentbestaccy=valAvgAccy
            torch.save(model,f'{weightPath}/Val_Epoch_{epoch}_accy_{currentbestaccy}_loss_{avgValloss}.pt')
        writer.add_scalar(f'Validation AvgLoss',avgValloss,epoch)
        writer.add_scalar(f'Validation AvgAccy',np.mean(valAvgAccy),epoch)
        writer.add_scalar(f'Validation AvgAccy2',np.mean(valAvgAccy2),epoch)
    print(f'Epoch: {epoch:2d} Validation Avg Loss:{avgValloss:0.6f}, AvgAccy:{valAvgAccy:0.3f}, AvgAccy2:{valAvgAccy2:0.3f}, AvgNS:({valNS:0.3f},{valNSM:0.3f}), AvgPS:({valPS:0.3f},{valNS:0.3f}), AvgBTG:({valbtwS:0.3f},{valbtwGM:0.3f})',flush=True)
    print("")
