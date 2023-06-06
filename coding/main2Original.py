import pandas as pd
import numpy as np
import torch
import re
import os
from datasets import load_dataset
import spacy
import argparse
from torch.utils.tensorboard import SummaryWriter
from Model.PredictLSTMIntervionP import PredictLSTMIntervionP
from torch.distributions.bernoulli import Bernoulli
import datetime
import pickle
from matplotlib import pyplot as plt

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


parser.add_argument('-preprocesstext', type=int, default=0,
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

parser.add_argument('-interventionP', type=float, default=0.5,
                    help='Probability for intervention')

parser.add_argument('-maxmerge', type=int, default=4,
                    help='max merge token size')

parser.add_argument('-remark', type=str, default='',
                    help='Remark on filename')

args = parser.parse_args()
seqlen=args.seqlen
batchsize=args.batch
embeddingDim=args.embeddingDim
epochs = args.epoch
hiddenSize=args.hiddenSize
hiddenLayer=args.hiddenLayer
maxmerge=args.maxmerge
interventionP=args.interventionP
sliding=args.slide
bidirectional= True if (args.bidirection==1) else False
preprocess = True if (args.preprocesstext==1) else False
remark = args.remark

print(f'Run Para : {args}',flush=True)

filename=f"{remark}seqlen{seqlen}_sldie{sliding}_batch{batchsize}_e{embeddingDim}_BiDirection{bidirectional}_HL{hiddenLayer}_HS{hiddenSize}_P{interventionP}_MaxMerget{maxmerge}_preprocess{preprocess}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
weightPath=f'./LSTMWeight/{filename}'
os.mkdir(weightPath)
tensorboardpath=f"./Tensorboard/{filename}"

#valbatchsize=2
imdb_dataset = load_dataset('imdb', split=['train[5000:20000]', 'test[12000:13000]'])
#imdb_dataset = load_dataset('imdb', split=['train[10000:10010]', 'train[10000:10010]', 'test[:20]'])
#imdb_dataset = load_dataset('imdb')
train_pd=pd.DataFrame(columns=["text","label"])
test_pd=pd.DataFrame(columns=["text","label"])
train_pd["text"]=imdb_dataset[0]['text']
train_pd["label"]=imdb_dataset[0]['label']
test_pd["text"]=imdb_dataset[1]['text']
test_pd["label"]=imdb_dataset[1]['label']
train_pd['text']=train_pd['text'].apply(lambda x: clean_review(x))
test_pd['text']=test_pd['text'].apply(lambda x: clean_review(x))
start=''
for i in range(seqlen):
    start+='s0s '
train_pd['text']=start+train_pd['text']+' e0s'
test_pd['text']=start+test_pd['text']+' e0s'
print(f'Traing PD shape : {train_pd.shape}',flush=True)
print(train_pd["label"].describe().T)
print(f'Test PD shape : {test_pd.shape}',flush=True)
print(test_pd["label"].describe().T)
vocab=None
with  open('./Master_vocab.pkl','rb') as f:
    vocab=pickle.load(f)
test_pd=test_pd.sample(frac=1)
test_pd=test_pd.reset_index(drop=True)

tok2id=vocab['tok2id']
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
model=PredictLSTMIntervionP(n_vocab,embeddingDim,seqlen,maxmerge,batchsize,hiddenLayer,hiddenSize,bidirectional,False,1)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())
currentbestloss=np.inf
test_text=test_pd['text']

maxlen=300
trainbatchcount=0
valbatchcount=0
fig,axes=plt.subplots(1,2,figsize=(10,10))
for epoch in range(epochs):
    trainloss=[]
    avgtrainaccy=[]
    train_pd=train_pd.sample(frac=1)
    train_pd=train_pd.reset_index(drop=True)
    train_text=train_pd['text']
    for d in range(0,len(train_text),batchsize):
        (h_state,c_state)=model.init_state()
        losses=[]
        switch=bernoulli.sample()
        sequences=[]
        targets=[]
        idxarray=[]
        for i,x in enumerate(nlp.pipe(train_text[d:d+batchsize])):
            data = [ tok2id[t.text] for t in x ] if(preprocess==False) else [ tok2id[t.text] for t in x if(t.is_stop==False and t.is_punct==False) ]
            tmp=np.empty(maxlen)
            tmp.fill(train_pd['label'][d+i])
            tmp[:seqlen]=0.5
            idx=maxlen-1
            if(len(data)>maxlen):
                data=data[:maxlen-1]+[endid]
            else:
                orglen=len(data)
                idx=orglen-1
                data=(data+([endid]*(maxlen-orglen)))
                tmp[orglen:]=0.5
            idxarray.append(idx)
            targets.append(tmp)
            sequences.append(data)
        idxarray=np.array(idxarray)
        targets=torch.tensor(np.array(targets),dtype=torch.float).to(device)
        sequences=np.array(sequences)
        predict_history=np.zeros((batchsize,maxlen))
        for i in range(0,maxlen-seqlen,sliding):
            sequence=sequences[:,i:i+seqlen]
            target=targets[:,i:i+seqlen]
            pred,switch,(h_state,c_state)=model(sequence,h_state,c_state,switch)
            pred=pred.reshape(batchsize,seqlen) # N,L,1 -> L,1 for batchsize == 1
            loss=criterion(pred,target)
            losses.append(loss.item())
            trainloss.append(loss.item())
            pred=torch.sigmoid(pred).detach().cpu().numpy()
            predict_history[:,i:i+seqlen]=pred
            h_state=h_state.detach()
            c_state=c_state.detach()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if(switch==1):
            [writer.add_histogram(f'Merge_EmbeddingSpace_Weight_{name}',para.data,trainbatchcount) for name,para in model.embeddingSpace2.named_parameters()]
            [writer.add_histogram(f'Merge_Conv1D_Weight_{name}',para.data,trainbatchcount) for name,para in model.mergeConv1D.named_parameters()]
        else:
            [writer.add_histogram(f'Normal_EmbeddingSpace_Weight_{name}',para.data,trainbatchcount) for name,para in model.embeddingSpace.named_parameters()]
        [writer.add_histogram(f'LSTM_Weight_{name}',para.data,trainbatchcount) for name,para in model.lstm.named_parameters() ]
        [writer.add_histogram(f'Predict_Weight_{name}',para.data,trainbatchcount) for name,para in model.predict.named_parameters()]
        avgloss=np.mean(losses)
        target=train_pd.iloc[d:d+batchsize]['label'].values
        predict_mean=[]
        predict_median=[]
        est_prediction=[]
        [ predict_median.append(np.median(predict_history[i, :idxarray[i]-1])) for i in range(batchsize) ]
        [ predict_mean.append(np.mean(predict_history[i, :idxarray[i]-1])) for i in range(batchsize) ]
        [ est_prediction.append(np.mean(predict_history[i, idxarray[i]-10:idxarray[i]-1])) for i in range(batchsize) ]
        batchaccy=np.sum(np.abs(est_prediction-target)<0.05)/batchsize
        predict_mean=np.mean(predict_mean)
        avgtrainaccy.append(batchaccy)
        quantile75 =np.median(np.quantile(predict_history,0.75,axis=-1))
        quantile25 =np.median(np.quantile(predict_history,0.25,axis=-1))
        median=np.median(predict_median)
        writer.add_scalar(f'Batch Training_AvgLoss',avgloss,trainbatchcount)
        writer.add_scalar(f'Batch Training_AvgAccy',batchaccy,trainbatchcount)
        writer.add_scalars(f'Batch Training AvgPredictionByDoc',{'mean':predict_mean,'q25':quantile25,'q75':quantile75,'median':median},trainbatchcount)
        nidx=np.where(target==0)[0]
        pidx=np.where(target==1)[0]
        #xaxis=np.arange(maxlen)
        axes[0].plot(predict_history[nidx,:].T)
        axes[1].plot(predict_history[pidx,:].T)
        axes[0].set_title(np.mean(idxarray[nidx]))
        axes[1].set_title(np.mean(idxarray[pidx]))
        fig.savefig(f'{tensorboardpath}/Training_Epoch_{epoch}_batch_{trainbatchcount}_{avgloss:0.6f}_plot.png')
        axes[0].clear()
        axes[1].clear()
        plt.cla()
        printout=f'Epoch: {epoch} Batch Training Doc#{(d+batchsize):5d}, AvgLoss:{avgloss:0.6f},'
        printout+=f'PStat:({predict_mean:0.3f},{quantile25:0.3f},{median:0.3f},{quantile75:0.3f}), AvgAccy:{batchaccy:0.3f} '
        print(printout,flush=True)
        trainbatchcount+=1
    avgtrainingloss=np.mean(trainloss)
    avgtrainingAccy=np.mean(avgtrainaccy)
    writer.add_scalar(f'Epoch Training AvgLoss',avgtrainingloss,epoch)
    writer.add_scalar(f'Epoch Training AvgAccy',avgtrainingAccy,epoch)
    print(f'Epoch: {epoch:2d} Training Finished, AvgTrainingLoss:{avgtrainingloss:0.6f}, AvgAccy:{avgtrainingAccy:0.3f}',flush=True)
    with torch.no_grad():
        vallosses=[]
        valAvgAccy=[]
        switchcount=0
        for d in range(0,len(test_text),batchsize):
            switch=bernoulli.sample()
            (h_state,c_state)=model.init_state()
            losses=[]
            sequences=[]
            targets=[]
            predict_history=[]
            idxarray=[]
            for i,x in enumerate(nlp.pipe(test_text[d:d+batchsize])):
                data = [ tok2id[t.text] for t in x] if(preprocess==False) else [tok2id[t.text] for t in x if (t.is_stop==False and t.is_punct==False) ]
                tmp=np.empty(maxlen)
                tmp.fill(test_pd['label'][d+i])
                tmp[:seqlen]=0.5
                idx=maxlen-1
                if(len(data)>maxlen):
                    data=data[:maxlen-1]+[endid]
                else:
                    orglen=len(data)
                    idx=orglen-1
                    data=(data+([endid]*(maxlen-orglen)))
                    tmp[orglen:]=0.5
                idxarray.append(idx)
                targets.append(tmp)
                sequences.append(data)
            idxarray=np.array(idxarray)
            targets=torch.tensor(np.array(targets),dtype=torch.float).to(device)
            sequences=np.array(sequences)
            predict_history=np.zeros((batchsize,maxlen))
            for i in range(0,maxlen-seqlen,sliding):
                sequence=sequences[:,i:i+seqlen]
                target=targets[:,i:i+seqlen]
                pred,switch,(h_state,c_state) =model(sequence,h_state,c_state,switch)
                pred=pred.reshape(batchsize,seqlen) # N,L,1 -> L,1 for batchsize == 1
                loss=criterion(pred,target)
                losses.append(loss.item())
                pred=torch.sigmoid(pred).detach().cpu().numpy()
                predict_history[:,i:i+seqlen]=pred
            avgloss=np.mean(losses)
            vallosses.append(avgloss)
            target=test_pd['label'][d:d+batchsize]
            predict_median=[]
            predict_mean=[]
            est_prediction=[]
            [ predict_median.append(np.median(predict_history[i, :idxarray[i]-1])) for i in range(batchsize) ]
            [ predict_mean.append(np.mean(predict_history[i, :idxarray[i]-1])) for i in range(batchsize) ]
            [ est_prediction.append(np.mean(predict_history[i, idxarray[i]-10:idxarray[i]-1])) for i in range(batchsize) ]
            batchaccy=np.sum(np.abs(est_prediction-target)<0.05)/batchsize
            avgValPro=np.mean(predict_mean)
            valAvgAccy.append(batchaccy)
            quantile75 = np.median(np.quantile(predict_history,0.75,axis=-1))
            quantile25 = np.median(np.quantile(predict_history,0.25,axis=-1))
            median=np.median(predict_median)
            nidx=np.where(target==0)[0]
            pidx=np.where(target==1)[0]
            axes[0].plot(predict_history[nidx,:].T)
            axes[1].plot(predict_history[pidx,:].T)
            axes[0].set_title(np.mean(idxarray[nidx]))
            axes[1].set_title(np.mean(idxarray[pidx]))
            fig.savefig(f'{tensorboardpath}/Validation_Epoch_{epoch}_batch_{valbatchcount}_{avgloss:0.6f}_plot.png')
            axes[0].clear()
            axes[1].clear()
            plt.cla()
            writer.add_scalars(f'Batch Validation AvgPrediction',{'mean':avgValPro,'q25':quantile25,'q75':quantile75,'median':median},valbatchcount)
            print(f'Epoch: {epoch:2d} Batch Validation Doc#{(d+batchsize):5d}, AvgLoss:{avgloss:0.6f}, '+
                    f'AvgPredict:({avgValPro:0.3f},{quantile25:0.3f},{median:0.3f},{quantile75:0.3f}), AvgAccy:{batchaccy:0.3f}',flush=True)
            valbatchcount+=1
        avgValloss=np.mean(vallosses)
        if(avgValloss<currentbestloss):
            currentbestloss=avgValloss
            torch.save(model,f'{weightPath}/Val_Epoch_{epoch}_{currentbestloss}.pt')
        writer.add_scalar(f'Validation AvgLoss',avgValloss,epoch)
        writer.add_scalar(f'Validation AvgAccy',np.mean(valAvgAccy),epoch)
    print(f'Epoch: {epoch:2d} Validation Avg Loss:{avgValloss:0.6f}, AvgAccy:{np.mean(valAvgAccy):0.3f}',flush=True)
    print("")
