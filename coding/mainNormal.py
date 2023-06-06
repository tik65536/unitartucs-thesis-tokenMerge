import pandas as pd
import numpy as np
import torch
import re
import os
from datasets import load_dataset
import spacy
import argparse
from torch.utils.tensorboard import SummaryWriter
from Model.PredictLSTMNormal import PredictLSTMNormal
from torch.distributions.bernoulli import Bernoulli
import datetime
import pickle

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


print(f'Run Para : {args}',flush=True)

filename=f"Normal_seqlen{seqlen}_sldie{sliding}_batch{batchsize}_e{embeddingDim}_BiDirection{bidirectional}_HL{hiddenLayer}_HS{hiddenSize}_P{interventionP}_MaxMerget{maxmerge}_preprocess{preprocess}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
weightPath=f'./LSTMWeight/{filename}'
os.mkdir(weightPath)
tensorboardpath=f"./Tensorboard/{filename}"

imdb_dataset = load_dataset('imdb', split=['train[5000:20000]', 'test[12000:13000]'])
#imdb_dataset = load_dataset('imdb', split=['train[:25000]','test[12000:13000]'])
train_pd=pd.DataFrame(columns=["text","label"])
test_pd=pd.DataFrame(columns=["text","label"])
train_pd["text"]=imdb_dataset[0]['text']
train_pd["label"]=imdb_dataset[0]['label']
test_pd["text"]=imdb_dataset[1]['text']
test_pd["label"]=imdb_dataset[1]['label']
train_pd['text']=train_pd['text'].apply(lambda x: clean_review(x))
test_pd['text']=test_pd['text'].apply(lambda x: clean_review(x))
test_pd=test_pd.sample(frac=1)
test_pd.reset_index(drop=True)
#for i in range(seqlen):
#    start+='S0S '
#print(start)
print(f'Training Size : {train_pd.shape}',flush=True)
print(f'Train lable distributions: {np.mean(train_pd["label"])}, {np.quantile(train_pd["label"],0.25)}, {np.median(train_pd["label"])}, {np.quantile(train_pd["label"],0.75)}')
print(f'Test Size : {test_pd.shape}',flush=True)
print(f'Test lable distributions: {np.mean(test_pd["label"])}, {np.quantile(test_pd["label"],0.25)}, {np.median(test_pd["label"])}, {np.quantile(test_pd["label"],0.75)}')
vocab=None
with  open('./Master_vocab.pkl','rb') as f:
    vocab=pickle.load(f)
tok2id=vocab['tok2id']
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
n_vocab=len(vocab['wordlist'])
print(f'Vocab Len: {n_vocab}',flush=True)
writer=SummaryWriter(log_dir=tensorboardpath)
bernoulli= Bernoulli(torch.tensor([interventionP]))
samplerate = Bernoulli(torch.tensor([0.1]))
model=PredictLSTMNormal(n_vocab,embeddingDim,seqlen,maxmerge,batchsize,hiddenLayer,hiddenSize,bidirectional)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())
currentbestloss=np.inf

for epoch in range(epochs):
    losses=[]
    predict_history=[]
    batchcount=0
    train_pd=train_pd.sample(frac=1)
    train_pd.reset_index(drop=True)
    for d in range(0,train_pd.shape[0],batchsize):
        (h_state,c_state)=model.init_state()
        sequences=[]
        for x in nlp.pipe(train_pd.iloc[d:d+batchsize]['text']):
            sequence = [ tok2id[t.text] for t in x ] if(preprocess==False) else [ tok2id[t.text] for t in x if(t.is_stop==False and t.is_punct==False) ]
            if(len(sequence)>seqlen):
                sequences.append(sequence[:seqlen])
            else:
                sequences.append(([0]*(seqlen-len(sequence)) + sequence))
        sequences=np.array(sequences)
        switch=bernoulli.sample()
        targets=torch.tensor(train_pd.iloc[d:d+batchsize]['label'].values,dtype=torch.float).reshape(batchsize,1).to(device)
        pred=model(sequences,h_state,c_state,switch)
        loss=criterion(pred,targets)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        pred=torch.sigmoid(pred).detach().cpu().numpy()
        predict_history.append(pred.reshape(batchsize))
        batchaccy = np.sum(pred==targets.detach().cpu().numpy())/batchsize
        print(f'Training Doc#{(d+batchsize):4d}, Batch:{batchcount:3d} Loss:{loss.item():0.6f}, Batch Acct:{batchaccy}',flush=True)
        if(switch==1 and interventionP!=1):
            if(samplerate.sample()==1):
                torch.save(model,f'{weightPath}/Train_Epoch_{epoch}_Doc_{d}_switch_on_.pt')
        [writer.add_histogram(f'Epoch_{epoch}_LSTM_Weight_{name}',para.data,batchcount) for name,para in model.lstm.named_parameters() ]
        [writer.add_histogram(f'Epoch_{epoch}_EmbeddingSpace1_0_Weight_{name}',para.data,batchcount) for name,para in model.embeddingSpace[0].named_parameters()]
        #[writer.add_histogram(f'Epoch_{epoch}_EmbeddingSpace2_0_Weight_{name}',para.data,d) for name,para in model.embeddingSpace2[0].named_parameters()]
        [writer.add_histogram(f'Epoch_{epoch}_Predict_Weight_{name}',para.data,batchcount) for name,para in model.predict.named_parameters()]
        batchcount+=1
    predict_history=np.array(predict_history).reshape(-1)
    trainaccy = np.sum(predict_history==train_pd['label'])/train_pd.shape[0]
    writer.add_scalar(f'Train_AvgLoss',np.mean(losses),epoch)
    writer.add_scalar(f'Train_AvgAccy',trainaccy,epoch)
    print(f'Epoch:{epoch:2d}, Training Avgloss:{np.mean(losses):0.4f}, AvgAccy:{(trainaccy):0.4f}',flush=True)
    with torch.no_grad():
        losses=[]
        predict_history=[]
        switchcount=0
        for d in range(0,test_pd.shape[0],batchsize):
            (h_state,c_state)=model.init_state()
            sequences=[]
            for x in nlp.pipe(test_pd.iloc[d:d+batchsize]['text']):
                sequence = [ tok2id[t.text] for t in x] if(preprocess==False) else [ tok2id[t.text] for t in x if(t.is_stop==False and t.is_punct==False) ]
                if(len(sequence)>seqlen):
                    sequences.append(sequence[:seqlen])
                else:
                    sequences.append(([0]*(seqlen-len(sequence)) + sequence))
            switch=bernoulli.sample()
            targets=torch.tensor(test_pd.iloc[d:d+batchsize]['label'].values,dtype=torch.float).reshape(batchsize,1).to(device)
            pred =model(sequences,h_state,c_state,switch)
            loss=criterion(pred,targets)
            losses.append(loss.item())
            pred=torch.sigmoid(pred).detach().cpu().numpy()
            predict_history.append(pred.reshape(batchsize))
        predict_history=np.array(predict_history).reshape(-1)
        avgValloss=np.mean(losses)
        valaccy = np.sum(predict_history==test_pd['label'])/test_pd.shape[0]
        print(f'Epoch:{epoch:2d}, Validation Avgloss:{avgValloss:0.6f}, avgAccy:{(valaccy):0.3f}',flush=True)
        if(avgValloss<currentbestloss):
            currentbestloss=avgValloss
            torch.save(model,f'{weightPath}/Val_Epoch_{epoch}_{currentbestloss}_.pt')
        writer.add_scalar(f'Val_AvgLoss',avgValloss,epoch)
        writer.add_scalar(f'Val_AvgAccy',valaccy,epoch)
