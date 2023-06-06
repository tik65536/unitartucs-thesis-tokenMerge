import pandas as pd
import numpy as np
import torch
import re
import os
from datasets import load_dataset
import spacy
import argparse
from torch.utils.tensorboard import SummaryWriter
from Model.StrangeModel import StrangeModel
from torch.distributions.bernoulli import Bernoulli
import datetime
import pickle
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

parser = argparse.ArgumentParser()
parser.add_argument('-epoch', type=int, default=20,
                    help='Epoch to run')

parser.add_argument('-batch', type=int, default=5,
                    help='BatchSize')

parser.add_argument('-maxlen', type=int, default=600,
                    help='Max text len')

parser.add_argument('-futureBlock', type=int, default=2,
                    help='Max text len')

parser.add_argument('-preprocesstext', type=int, default=1,
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
futureBlock=args.futureBlock

print(f'Run Para : {args}',flush=True)

filename=f"Strange_seqlen{seqlen}_fblock{futureBlock}_sldie{sliding}_batch{batchsize}_e{embeddingDim}_BiDirection{bidirectional}_HL{hiddenLayer}_HS{hiddenSize}_preprocess{preprocess}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
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
model=StrangeModel(n_vocab,embeddingDim,seqlen,batchsize,hiddenLayer,hiddenSize,bidirectional)
weightHistory={}
criterion = torch.nn.CrossEntropyLoss()
criterion2 = torch.nn.CosineSimilarity()
optimizer = torch.optim.Adam(model.parameters())
currentbestaccy=0
test_text=test_pd['text']
test_label=test_pd['label']
trainbatchcount=0
valbatchcount=0
for epoch in range(epochs):
    trainloss=[]
    trainloss2=[]
    trainloss3=[]
    avgtrainaccy=[]
    avgtrainaccy2=[]
    train_pd=train_pd.sample(frac=1)
    train_pd=train_pd.reset_index(drop=True)
    train_text=train_pd['text']
    train_label=train_pd['label']
    #for d in range(0,len(train_text),batchsize):
    for d in range(0,batchsize,batchsize):
        (hstate,cstate)=model.init_state()
        losses=[]
        losses2=[]
        losses3=[]
        norm=[]
        sequences=[]
        targets=[]
        idxarray=[]
        tokendist=[]
        for i,x in enumerate(nlp.pipe(train_text[d:d+batchsize])):
            data = [ tok2id[t.text] for t in x ] if(preprocess==0) else [ tok2id[t.text] for t in x if(t.is_stop==False and t.is_punct==False) ]
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
        for i in range(0,c,sliding):
            sequence=sequences[:,i:i+seqlen]
            target=targets[:,i:i+seqlen]
            pred,currentState,(hstate,cstate)  =  model(sequence,(hstate,cstate))
            hstate = hstate.detach()
            cstate = cstate.detach()
            if((i+seqlen)+(seqlen*futureBlock)>maxlen):
                remain=maxlen-(i+seqlen)
                if(remain>0):
                    future=sequences[:,i+seqlen:remain]
                    tmp=np.full((batchsize,seqlen*futureBlock-remain),endid)
                    future = np.concatenate([future,tmp],axis=1)
                else:
                    future=np.full((batchsize,seqlen*futureBlock),endid)
            else:
                future = sequences[:,i+seqlen:(i+seqlen)+(seqlen*futureBlock)]
            currentState=currentState.detach()
            out2,approximate = model.approximate(currentState)
            output=model.forwardFuture(future,(hstate,cstate))
            #pred=pred.reshape(batchsize,seqlen,3) # N,L,1 -> L,1 for batchsize == 1
            loss= criterion(pred,target)
            loss2=0
            for r in range(0,len(output)):
                loss2 += torch.mean(torch.abs(criterion2(output[r], approximate)))
            loss3 = criterion(out2,target)
            losses.append(loss.item())
            losses2.append(loss2.item())
            losses3.append(loss3.item())
            trainloss.append(loss.item())
            trainloss2.append(loss2.item())
            trainloss3.append(loss3.item())
            sloss=loss2+loss3
            pred=torch.nn.functional.softmax(pred,dim=1)
            pred=pred.permute(0,2,1).detach().cpu().numpy()
            predict_history[:,i:i+seqlen,:]=pred[:batchsize]
            loss.backward()
            sloss.backward()
            optimizer.step()
            optimizer.zero_grad()
        avgloss=np.mean(losses)
        avgloss2=np.mean(losses2)
        avgloss3=np.mean(losses3)
        avgnorm = np.mean(norm)
        target=train_label[d:d+batchsize].to_numpy()
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
        writer.add_scalars('Training ClassificationReport Neg Class',cr[key[0]],trainbatchcount)
        writer.add_scalars('Training ClassificationReport Pos Class',cr[key[1]],trainbatchcount)
        writer.add_scalars('Training ClassificationReport MarcoAvg',cr['macro avg'],trainbatchcount)
        printout=f'Epoch: {epoch} Batch Training Doc#{(d+batchsize):5d}, Preprocess:{int(preprocess):1d}, AvgLoss:{avgloss:0.6f},AvgLoss2:{avgloss2:0.6f}, Avgloss3:{avgloss3:0.4f}, AvgAccy:{batchaccy:0.3f}, AvgAccy2:{batchaccy2:0.3f} '
        print(printout,flush=True)
        trainbatchcount+=1
    avgtrainingloss=np.mean(trainloss)
    avgtrainingloss2=np.mean(trainloss2)
    avgtrainingloss3=np.mean(trainloss3)
    avgtrainingAccy=np.mean(avgtrainaccy)
    avgtrainingAccy2=np.mean(avgtrainaccy2)
    writer.add_scalar(f'Epoch Training AvgLoss',avgtrainingloss,epoch)
    writer.add_scalar(f'Epoch Training AvgLoss2',avgtrainingloss2,epoch)
    writer.add_scalar(f'Epoch Training AvgAccy(abs value)',avgtrainingAccy,epoch)
    writer.add_scalar(f'Epoch Training AvgAccy2(abs value)',avgtrainingAccy2,epoch)
    print(f'Epoch: {epoch:2d} Training Finished, AvgTrainingLoss:{avgtrainingloss:0.6f}, AvgTrainingLoss2:{avgtrainingloss2:0.6f}, AvgTrainingLoss3:{avgtrainingloss3:0.3f} AvgAccy:{avgtrainingAccy:0.3f}',flush=True)
    with torch.no_grad():
        vallosses=[]
        valAvgAccy=[]
        valAvgAccy2=[]
        switchcount=0
        tokendist=[]
        for d in range(0,len(test_text),batchsize):
            (hstate,cstate)=model.init_state()
            losses=[]
            sequences=[]
            targets=[]
            predict_history=[]
            idxarray=[]
            for i,x in enumerate(nlp.pipe(test_text[d:d+batchsize])):
                data = [ tok2id[t.text] for t in x] if(preprocess==0) else [tok2id[t.text] for t in x if (t.is_stop==False and t.is_punct==False) ]
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
            for i in range(0,c,sliding):
                sequence=sequences[:,i:i+seqlen]
                target=targets[:,i:i+seqlen]
                pred,(hstate,cstate) =model.validation(sequence,(hstate,cstate))
                #pred=pred.reshape(batchsize,seqlen) # N,L,1 -> L,1 for batchsize == 1
                loss=criterion(pred,target)
                losses.append(loss.item())
                pred=torch.nn.functional.softmax(pred,dim=1)
                pred=pred.permute(0,2,1).detach().cpu().numpy()
                predict_history[:,i:i+seqlen,:]=pred
            avgloss=np.mean(losses)
            vallosses.append(avgloss)
            target=test_label[d:d+batchsize].to_numpy()
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
            writer.add_scalars('Validation ClassificationReport Neg Class',cr[key[0]],valbatchcount)
            writer.add_scalars('Validation ClassificationReport Pos Class',cr[key[1]],valbatchcount)
            writer.add_scalars('Validation ClassificationReport MacroAvg',cr['macro avg'],valbatchcount)
            print(f'Epoch: {epoch:2d} Batch Validation Doc#{(d+batchsize):5d}, Preprocess:{int(preprocess):1d}, AvgLoss:{avgloss:0.6f}, AvgAccy:{batchaccy:0.3f}, AvgAccy2:{batchaccy2:0.3f}',flush=True)
            valbatchcount+=1
        avgValloss=np.mean(vallosses)
        valAvgAccy = np.mean(valAvgAccy)
        valAvgAccy2 = np.mean(valAvgAccy2)
        if(valAvgAccy>currentbestaccy):
            currentbestaccy=valAvgAccy
            torch.save(model,f'{weightPath}/Val_Epoch_{epoch}_accy_{currentbestaccy}_loss_{avgValloss}.pt')
        writer.add_scalar(f'Validation AvgLoss',avgValloss,epoch)
        writer.add_scalar(f'Validation AvgAccy',np.mean(valAvgAccy),epoch)
        writer.add_scalar(f'Validation AvgAccy2',np.mean(valAvgAccy2),epoch)
    print(f'Epoch: {epoch:2d} Validation Avg Loss:{avgValloss:0.6f}, AvgAccy:{valAvgAccy:0.3f}, AvgAccy2:{valAvgAccy2:0.3f}',flush=True)
    print("")
