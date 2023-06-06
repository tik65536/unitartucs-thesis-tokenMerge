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
    clean_text = re.sub('<br\s?\/>|<br>', '', text)
    #clean_text = re.sub('[^a-zA-Z\']', ' ', clean_text)
    #clean_text = clean_text.lower()
    return clean_text

parser = argparse.ArgumentParser()
parser.add_argument('-embeddingDim', type=int, default=128,
                    help='Embedding Dim')

parser.add_argument('-maxmerge', type=int, default=4,
                    help='max merge token size')

parser.add_argument('-seqlen', type=int, default=5,
                    help='Seq Len for reading token')

parser.add_argument('-slide', type=int, default=1,
                    help='Seqlen Sliding')

parser.add_argument('-numberOfSample', type=int, default=100,
                    help='Number of Training Docs before Validation')

parser.add_argument('-hiddenLayer', type=int, default=20,
                    help='Hidden Layer of LSTM')

parser.add_argument('-hiddenSize', type=int, default=128,
                    help='Hidden Size of LSTM')

parser.add_argument('-batch', type=int, default=5,
                    help='BatchSize')

parser.add_argument('-interventionP', type=float, default=0.5,
                    help='Probability for intervention')

parser.add_argument('-epoch', type=int, default=20,
                    help='Epoch to run')


parser.add_argument('-desc', type=str, default=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                    help='desc')

args = parser.parse_args()
seqlen=args.seqlen
batchsize=args.batch
embeddingDim=args.embeddingDim
epochs = args.epoch
desc=args.desc
hiddenSize=args.hiddenSize
hiddenLayer=args.hiddenLayer
maxmerge=args.maxmerge
interventionP=args.interventionP
sliding=args.slide
valbatchsize=args.numberOfSample

print(f'Run Para : Seqlen:{seqlen}, batch:{batchsize}, embeddingDim:{embeddingDim}, hiddenSize:{hiddenSize}, hiddenLayer:{hiddenLayer}, maxmerge:{maxmerge}, InterventionP:{interventionP}, sliding:{sliding} ',flush=True)

weightPath=f"./LSTMWeight/{desc}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
os.mkdir(weightPath)
tensorboardpath=f"./Tensorboard/{desc}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

#valbatchsize=2
#imdb_dataset = load_dataset('imdb', split=['train[10000:15000]', 'train[10000:15000]', 'test[12250:12750]'])
imdb_dataset = load_dataset('imdb', split=['train[10000:10010]', 'train[10000:10010]', 'test[:20]'])
#imdb_dataset = load_dataset('imdb')
train_pd=pd.DataFrame(columns=["text","label"])
test_pd=pd.DataFrame(columns=["text","label"])
train_pd["text"]=imdb_dataset[0]['text']
train_pd["label"]=imdb_dataset[0]['label']
test_pd["text"]=imdb_dataset[2]['text']
test_pd["label"]=imdb_dataset[2]['label']
train_pd['text']=train_pd['text'].apply(lambda x: clean_review(x))
test_pd['text']=test_pd['text'].apply(lambda x: clean_review(x))
#start=''
#for i in range(seqlen):
#    start+='S0S '
#print(start)
train_pd['text']='S0S '+train_pd['text']+' E0S'
test_pd['text']='S0S '+test_pd['text']+' E0S'
print(f'Traing PD shape : {train_pd.shape}',flush=True)
print(train_pd["label"].describe().T)
print(f'Test PD shape : {test_pd.shape}',flush=True)
print(test_pd["label"].describe().T)
vocab=None
with  open('./Master_vocab.pkl','rb') as f:
    vocab=pickle.load(f)
test_pd=test_pd.sample(frac=1)
test_pd=test_pd.reset_index()
train_pd=train_pd.sample(frac=1)
train_pd=train_pd.reset_index()

docs=nlp.pipe(train_pd["text"])


tok2id=vocab['tok2id']
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
startid=tok2id['S0S']
endid=tok2id['E0S']
n_vocab=len(vocab['wordlist'])
writer=SummaryWriter(log_dir=tensorboardpath)
bernoulli= Bernoulli(torch.tensor([interventionP]))
samplerate = Bernoulli(torch.tensor([0.1]))
model=PredictLSTMIntervionP(n_vocab,embeddingDim,seqlen,maxmerge,batchsize,hiddenLayer,hiddenSize,1)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())
currentbestloss=np.inf
per100Dcount=0
fig,axes=plt.subplots(1,2,figsize=(10,10))

for epoch in range(epochs):
    docs=nlp.pipe(train_pd["text"])
    trainloss=[]
    trainprod=[]
    for d,doc in enumerate(docs,1):
        token_len=len(doc)
        (h_state,c_state)=model.init_state(batchsize)
        sequences=[]
        losses=[]
        predict_history=[]
        switch=bernoulli.sample()
        #targets=torch.full((batchsize,1),train_pd['label'][d],dtype=torch.float).to(device)
        targets=torch.full((seqlen,1),train_pd['label'][d-1],dtype=torch.float).to(device)
        for i in range(0,token_len-seqlen,sliding):
            sequence=[ tok2id[doc[i+x].text] for x in range(seqlen) ]
            sequences.append(sequence)
        if(len(sequences)>batchsize):
            prediction=torch.zeros((len(sequences)-batchsize,1),requires_grad=True)
            for i in range(0,len(sequences),batchsize):
                optimizer.zero_grad()
                sequence=np.array(sequences[i:i+batchsize])
                if(startid in sequence):
                    targets[0,0]=0.5
                if(endid in sequence):
                    endidx=sequence.index(endid)
                    targets[endidx,0]=0.5
                pred,switch,(h_state,c_state)=model(sequence,h_state,c_state,switch)
                pred=pred.squeeze(0) # N,L,1 -> L,1 for batchsize == 1
                #h_state=h_state.detach()
                #c_state=c_state.detach()
                loss=criterion(pred,targets)
                losses.append(loss.item())
                trainloss.append(loss.item())
                loss.backward(retain_graph=True)
                pred=torch.sigmoid(pred).detach().cpu().numpy()
                predict_history.append(pred.reshape(seqlen,))
            optimizer.step()
            if(switch==1 and interventionP!=1):
                if(samplerate.sample()==1):
                        torch.save(model,f'{weightPath}/Train_Epoch_{epoch}_Doc_{d}_switch_on_.pt')
            [writer.add_histogram(f'Epoch_{epoch}_LSTM_Weight_{name}',para.data,d) for name,para in model.lstm.named_parameters() ]
            if(batchsize>1):
                [writer.add_histogram(f'Epoch_{epoch}_EmbeddingSpace1_1_Weight_{name}',para.data,d) for name,para in model.embeddingSpace[1].named_parameters()]
                [writer.add_histogram(f'Epoch_{epoch}_EmbeddingSpace2_1_Weight_{name}',para.data,d) for name,para in model.embeddingSpace2[1].named_parameters()]
            [writer.add_histogram(f'Epoch_{epoch}_EmbeddingSpace1_0_Weight_{name}',para.data,d) for name,para in model.embeddingSpace[0].named_parameters()]
            [writer.add_histogram(f'Epoch_{epoch}_EmbeddingSpace2_0_Weight_{name}',para.data,d) for name,para in model.embeddingSpace2[0].named_parameters()]
            [writer.add_histogram(f'Epoch_{epoch}_Predict_Weight_{name}',para.data,d) for name,para in model.predict.named_parameters()]
            avgloss=np.mean(losses)
            predict_history=np.array(predict_history) # shape= (seqlen-batchsize),seqlen
            predict_mean=np.mean(np.mean(predict_history,axis=-1))
            quantile75 =np.mean(np.quantile(predict_history,0.75,axis=-1))
            quantile25 =np.mean(np.quantile(predict_history,0.25,axis=-1))
            median=np.mean(np.median(predict_history,axis=-1))
            trainprod.append(predict_mean)
            writer.add_scalar(f'Training_Epoch_{epoch}_AvgLossByDoc',avgloss,d)
            writer.add_scalars(f'Training_Epoch_{epoch}_AvgPredictionByDoc',{'mean':predict_mean,'q25':quantile25,'q75':quantile75,'median':median},d)
            if(train_pd['label'][d-1]==0):
                axes[0].plot(np.mean(predict_history,axis=-1))
            else:
                axes[1].plot(np.mean(predict_history,axis=-1))
            print(f'Doc#{d:5d}, AvgLoss:{avgloss:0.6f}, Target:{train_pd["label"][d-1]}, '+
                    f'PStat:({predict_mean:0.3f},{quantile25:0.3f},{median:0.3f},{quantile75:0.3f}), '+
                    f'1st: {np.mean(predict_history[0]):0.3f}, Last: {np.mean(predict_history[-1]):0.3f}',flush=True)
        else:
            print(f'Training Doc#{d:5d} len(sequences)<batchsize Skip')
        if(d%valbatchsize==0):
            fig.savefig(f'{tensorboardpath}/Training_per100Batch_{per100Dcount}_plot.png')
            axes[0].clear()
            axes[1].clear()
            per100Dcount+=1
            avgtrainingloss=np.mean(trainloss)
            avgtrainpro=np.mean(trainprod,axis=0)
            testdocs=nlp.pipe(test_pd["text"])
            writer.add_scalar(f'AvgTrainingLoss/100_Docs',avgtrainingloss,per100Dcount)
            print(f'Epoch:{epoch:2d} 100Docs Training Finished, AvgTrainingLoss:{avgtrainingloss:0.6f}',flush=True)
            trainloss=[]
            trainprod=[]
            fig,axes=plt.subplots(1,2,figsize=(10,10))
            with torch.no_grad():
                vallosses=[]
                valpredict=[]
                switchcount=0
                for td,testdoc in enumerate(testdocs):
                    token_len=len(testdoc)
                    switch=bernoulli.sample()
                    (h_state,c_state)=model.init_state(batchsize)
                    sequences=[]
                    losses=[]
                    predict_history=[]
                    targets=torch.full((seqlen,1),test_pd['label'][td],dtype=torch.float).to(device)
                    for i in range(0,token_len-seqlen,sliding):
                        sequence=[ tok2id[testdoc[i+x].text] for x in range(seqlen) ]
                        sequences.append(sequence)
                    previous=None
                    if(len(sequences)>batchsize):
                        for i in range(0,len(sequences),batchsize):
                            sequence=np.array(sequences[i:i+batchsize])
                            if(startid in sequence):
                                targets[0,0]=0.5
                            if(endid in sequence):
                                endidx=sequence.index(endid)
                                targets[endidx,0]=0.5
                            pred,switch,(h_state,c_state) =model(sequence,h_state,c_state,switch)
                            pred=pred.squeeze(0) # N,L,1 -> L,1 for batchsize == 1
                            loss=criterion(pred,targets)
                            losses.append(loss.item())
                            pred=torch.sigmoid(pred).detach().cpu().numpy()
                            predict_history.append(pred.reshape(seqlen,))
                        predict_history=np.array(predict_history)
                        avgloss=np.mean(losses)
                        vallosses.append(avgloss)
                        avgValPro=np.mean(np.mean(predict_history,axis=-1))
                        quantile75 = np.mean(np.quantile(predict_history,0.75,axis=-1))
                        quantile25 = np.mean(np.quantile(predict_history,0.25,axis=-1))
                        median=np.mean(np.median(predict_history,axis=-1))
                        if(test_pd['label'][td]==0):
                            axes[0].plot(np.mean(predict_history,axis=-1))
                        else:
                            axes[1].plot(np.mean(predict_history,axis=-1))
                        writer.add_scalars(f'Val_Epoch_{epoch}_AvgPrediction/{valbatchsize}',{'mean':avgValPro,'q25':quantile25,'q75':quantile75,'median':median},td)
                        print(f'Val Doc#{td:5d}, AvgLoss:{avgloss:0.6f},  Target:{test_pd["label"][td]}, '+
                                f'AvgPredict:({avgValPro:0.3f},{quantile25:0.3f},{median:0.3f},{quantile75:0.3f}),'+
                                f'1st: {np.mean(predict_history[0]):0.3f}, Last: {np.mean(predict_history[-1]):0.3f}',flush=True)
                avgValloss=np.mean(vallosses)
                if(avgValloss<currentbestloss):
                    currentbestloss=avgValloss
                    torch.save(model,f'{weightPath}/Val_Epoch_{epoch}_100DCount_{per100Dcount}_{currentbestloss}_.pt')
                writer.add_scalar(f'Val_Epoch_{epoch}_AvgLoss/{valbatchsize} Docs',avgValloss,per100Dcount)
                fig.savefig(f'{tensorboardpath}/Validation_per100Batch_{per100Dcount}_plot.png')
            print(f'Epoch:{epoch:2d} 100Docs Avg Val Loss:{avgValloss:0.6f}',flush=True)
            print("")
