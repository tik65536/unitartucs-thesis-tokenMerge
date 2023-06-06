import pandas as pd
import numpy as np
import torch
import re
import os
from datasets import load_dataset
import spacy
import argparse
from torch.utils.tensorboard import SummaryWriter
from Model.PredictIntervention2Model import PredictIntervention2Model
from torch.distributions.bernoulli import Bernoulli
import datetime
import pickle

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
print(f'2Model Run Para : Seqlen:{seqlen}, batch:{batchsize}, embeddingDim:{embeddingDim}, hiddenSize:{hiddenSize}, hiddenLayer:{hiddenLayer}, maxmerge:{maxmerge}, InterventionP:{interventionP}, sliding:{sliding} ',flush=True)
weightPath=f"./LSTMWeight/{desc}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
os.mkdir(weightPath)
tensorboardpath=f"./Tensorboard/{desc}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

valbatchsize=500
#valbatchsize=2
imdb_dataset = load_dataset('imdb', split=['train[10000:15000]', 'train[10000:15000]', 'test[12000:13000]'])
#imdb_dataset = load_dataset('imdb', split=['train[10000:10010]', 'train[10000:10010]', 'test[:20]'])
#imdb_dataset = load_dataset('imdb')
train_pd=pd.DataFrame(columns=["text","label"])
test_pd=pd.DataFrame(columns=["text","label"])
train_pd["text"]=imdb_dataset[0]['text']
train_pd["label"]=imdb_dataset[0]['label']
test_pd["text"]=imdb_dataset[2]['text']
test_pd["label"]=imdb_dataset[2]['label']
train_pd['text']=train_pd['text'].apply(lambda x: clean_review(x))
test_pd['text']=test_pd['text'].apply(lambda x: clean_review(x))
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

n_vocab=len(vocab['wordlist'])
writer=SummaryWriter(log_dir=tensorboardpath)
bernoulli= Bernoulli(torch.tensor([interventionP]))
samplerate = Bernoulli(torch.tensor([0.05]))
model=PredictIntervention2Model(n_vocab,embeddingDim,seqlen,maxmerge,batchsize,hiddenLayer,hiddenSize,1)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())
m1currentbestloss=np.inf
m2currentbestloss=np.inf
per100Dcount=0
for epoch in range(epochs):
    docs=nlp.pipe(train_pd["text"])
    model1_trainloss=[]
    model1_trainprod=[]
    model2_trainloss=[]
    model2_trainprod=[]
    for d,doc in enumerate(docs,1):
        token_len=len(doc)
        h_state,c_state, model2_previousOut=model.init_state(batchsize)
        sequences=[]
        model1_losses=[]
        model2_losses=[]
        model1_predict_history=[]
        model2_predict_history=[]
        switch=bernoulli.sample()
        #targets=torch.full((batchsize,1),train_pd['label'][d],dtype=torch.float).to(device)
        targets=torch.full((batchsize,1),train_pd['label'][d-1],dtype=torch.float).to(device)
        for i in range(0,token_len-seqlen,sliding):
            sequence=[ tok2id[doc[i+x].text] for x in range(seqlen) ]
            sequences.append(sequence)
        if(len(sequences)>batchsize):
            for i in range(0,len(sequences)-batchsize,batchsize):
                optimizer.zero_grad()
                sequence=np.array(sequences[i:i+batchsize])
                model1_pred,(h_state,c_state), model2_pred ,model2_previousOut =model(sequence,h_state,c_state,model2_previousOut,switch)
                h_state=h_state.detach()
                c_state=c_state.detach()
                model2_previousOut=model2_previousOut.detach()
                model1loss=criterion(model1_pred,targets)
                model2loss=criterion(model2_pred,targets)
                model1_losses.append(model1loss.item())
                model2_losses.append(model2loss.item())
                model1_trainloss.append(model1loss.item())
                model2_trainloss.append(model2loss.item())
                model1loss.backward()
                model2loss.backward()
                #loss.backward(retain_graph=True)
                optimizer.step()
                model1_pred=torch.sigmoid(model1_pred).detach().cpu().numpy()
                model2_pred=torch.sigmoid(model2_pred).detach().cpu().numpy()
                model1_predict_history.append(model1_pred.reshape(batchsize))
                model2_predict_history.append(model2_pred.reshape(batchsize))
                #selection.append(model.competition.detach().cpu().numpy())
            if(switch==1 and interventionP!=1):
                if(samplerate.sample()==1):
                        torch.save(model,f'{weightPath}/Train_Epoch_{epoch}_Doc_{d}_switch_on_.pt')
            [writer.add_histogram(f'Epoch_{epoch}_LSTM_Weight_{name}',para.data,d) for name,para in model.model1_lstm.named_parameters() ]
            #if(batchsize>1):
            #    [writer.add_histogram(f'Epoch_{epoch}_M1_EmbeddingSpace1_1_Weight_{name}',para.data,d) for name,para in model.model1Embedding['embedding1'][1].named_parameters()]
            #    [writer.add_histogram(f'Epoch_{epoch}_M1_EmbeddingSpace2_1_Weight_{name}',para.data,d) for name,para in model.model1Embedding['embedding2'][1].named_parameters()]
            #    [writer.add_histogram(f'Epoch_{epoch}_M2_EmbeddingSpace1_1_Weight_{name}',para.data,d) for name,para in model.model2Embedding['embedding1'][1].named_parameters()]
            #    [writer.add_histogram(f'Epoch_{epoch}_M2_EmbeddingSpace2_1_Weight_{name}',para.data,d) for name,para in model.model2Embedding['embedding2'][1].named_parameters()]
            [writer.add_histogram(f'Epoch_{epoch}_Model1_EmbeddingSpace1_0_Weight_{name}',para.data,d) for name,para in model.model1Embedding['embedding1'][0].named_parameters()]
            [writer.add_histogram(f'Epoch_{epoch}_Model1_EmbeddingSpace2_0_Weight_{name}',para.data,d) for name,para in model.model1Embedding['embedding2'][0].named_parameters()]
            [writer.add_histogram(f'Epoch_{epoch}_Model1_PredictConv1D_Weight_{name}',para.data,d) for name,para in model.model1_predictConv1D.named_parameters()]
            [writer.add_histogram(f'Epoch_{epoch}_Model2_EmbeddingSpace1_0_Weight_{name}',para.data,d) for name,para in model.model2Embedding['embedding1'][0].named_parameters()]
            [writer.add_histogram(f'Epoch_{epoch}_Model2_EmbeddingSpace2_0_Weight_{name}',para.data,d) for name,para in model.model2Embedding['embedding2'][0].named_parameters()]
            [writer.add_histogram(f'Epoch_{epoch}_Model2_Conv_Weight_{name}',para.data,d) for name,para in model.model2_Conv1D.named_parameters()]
            [writer.add_histogram(f'Epoch_{epoch}_Model2_Predict_Weight_{name}',para.data,d) for name,para in model.model2_predict.named_parameters()]
            m1avgloss=np.mean(model1_losses)
            m2avgloss=np.mean(model2_losses)
            model1_predict_history=np.array(model1_predict_history)
            model2_predict_history=np.array(model2_predict_history)
            m1predict_mean=np.mean(np.mean(model1_predict_history,axis=0))
            m1quantile75 =np.mean(np.quantile(model1_predict_history,0.75,axis=0))
            m1quantile25 =np.mean(np.quantile(model1_predict_history,0.25,axis=0))
            m1median=np.mean(np.median(model1_predict_history,axis=0))
            m2predict_mean=np.mean(np.mean(model2_predict_history,axis=0))
            m2quantile75 =np.mean(np.quantile(model2_predict_history,0.75,axis=0))
            m2quantile25 =np.mean(np.quantile(model2_predict_history,0.25,axis=0))
            m2median=np.mean(np.median(model2_predict_history,axis=0))
            model1_trainprod.append(m1predict_mean)
            model2_trainprod.append(m2predict_mean)
            writer.add_scalar(f'Training_Epoch_{epoch}_M1_AvgLossByDoc',m1avgloss,d)
            writer.add_scalar(f'Training_Epoch_{epoch}_M2_AvgLossByDoc',m2avgloss,d)
            writer.add_scalars(f'Training_Epoch_{epoch}_M1_AvgPredictionByDoc',{'mean':m1predict_mean,'q25':m1quantile25,'q75':m1quantile75,'median':m1median},d)
            writer.add_scalars(f'Training_Epoch_{epoch}_M2_AvgPredictionByDoc',{'mean':m2predict_mean,'q25':m2quantile25,'q75':m2quantile75,'median':m2median},d)
            print(f'M1 Doc#{d:5d}, AvgLoss:{m1avgloss:0.6f},  Target:{train_pd["label"][d-1]}, '+
                    f'AvgPredict:({m1predict_mean:0.3f},{m1quantile25:0.3f},{m1median:0.3f},{m1quantile75:0.3f}), Last Prediction:({np.mean(model1_predict_history[-1]):0.3f},{np.quantile((model1_predict_history[-1]),0.25):0.3f},{np.median(model1_predict_history[-1]):0.3f},{np.quantile(model1_predict_history[-1],0.75):0.3f})',flush=True)
            print(f'M2 Doc#{d:5d}, AvgLoss:{m2avgloss:0.6f},  Target:{train_pd["label"][d-1]}, '+
                    f'AvgPredict:({m2predict_mean:0.3f},{m2quantile25:0.3f},{m2median:0.3f},{m2quantile75:0.3f}), Last Prediction:({np.mean(model2_predict_history[-1]):0.3f},{np.quantile((model2_predict_history[-1]),0.25):0.3f},{np.median(model2_predict_history[-1]):0.3f},{np.quantile(model2_predict_history[-1],0.75):0.3f})',flush=True)
        else:
            print(f'Training Doc#{d:5d} len(sequences)<batchsize Skip')
        if(d%valbatchsize==0):
            per100Dcount+=1
            m1avgtrainingloss=np.mean(model1_trainloss)
            m1avgtrainpro=np.mean(model1_trainprod,axis=0)
            m2avgtrainingloss=np.mean(model1_trainloss)
            m2avgtrainpro=np.mean(model1_trainprod,axis=0)
            testdocs=nlp.pipe(test_pd["text"])
            writer.add_scalar(f'M1 AvgTrainingLoss/100_Docs',m1avgtrainingloss,per100Dcount)
            writer.add_scalar(f'M2 AvgTrainingLoss/100_Docs',m2avgtrainingloss,per100Dcount)
            print(f'Epoch:{epoch:2d} 100Docs Training Finished, M1 AvgTrainingLoss:{m1avgtrainingloss:0.6f},M2 AvgTrainingLoss:{m2avgtrainingloss:0.6f}',flush=True)
            model1_trainloss,model1_trainprod=[],[]
            model2_trainloss,model2_trainprod=[],[]
            with torch.no_grad():
                m1vallosses,m2vallosses=[],[]
                m1predict_history,m2predict_history=[],[]
                switchcount=0
                for td,doc in enumerate(testdocs):
                    token_len=len(doc)
                    switch=bernoulli.sample()
                    h_state,c_state,model2_previousOut=model.init_state(batchsize)
                    sequences=[]
                    m1predict_history,m2predict_history=[],[]
                    targets=torch.full((batchsize,1),test_pd['label'][td],dtype=torch.float).to(device)
                    for i in range(0,token_len-seqlen,sliding):
                        sequence=[ tok2id[doc[i+x].text] for x in range(seqlen) ]
                        sequences.append(sequence)
                    if(len(sequences)>batchsize):
                        for i in range(0,len(sequences)-batchsize,batchsize):
                            sequence=np.array(sequences[i:i+batchsize])
                            m1pred,(h_state,c_state),m2pred,model2_previousOut=model(sequence,h_state,c_state,model2_previousOut,switch)
                            m1loss=criterion(m1pred,targets)
                            m2loss=criterion(m2pred,targets)
                            m1vallosses.append(m1loss.item())
                            m2vallosses.append(m2loss.item())
                            m1pred=torch.sigmoid(m1pred).detach().cpu().numpy()
                            m2pred=torch.sigmoid(m2pred).detach().cpu().numpy()
                            m1predict_history.append(m1pred)
                            m2predict_history.append(m2pred)
                m1avgValloss=np.mean(m1vallosses)
                m2avgValloss=np.mean(m2vallosses)
                m1avgValPro=np.mean(np.mean(m1predict_history,axis=0))
                m2avgValPro=np.mean(np.mean(m2predict_history,axis=0))
                m1quantile75 = np.mean(np.quantile(m1predict_history,0.75,axis=0))
                m2quantile75 = np.mean(np.quantile(m2predict_history,0.75,axis=0))
                m1quantile25 = np.mean(np.quantile(m1predict_history,0.25,axis=0))
                m2quantile25 = np.mean(np.quantile(m2predict_history,0.25,axis=0))
                m1median=np.median(m1predict_history)
                m2median=np.median(m2predict_history)
                if(m1avgValloss<m1currentbestloss):
                    m1currentbestloss=m1avgValloss
                    torch.save(model,f'{weightPath}/Val_Epoch_{epoch}_M1_100DCount_{per100Dcount}_{m1currentbestloss}_.pt')
                if(m2avgValloss<m2currentbestloss):
                    m2currentbestloss=m2avgValloss
                    torch.save(model,f'{weightPath}/Val_Epoch_{epoch}_M1_100DCount_{per100Dcount}_{m2currentbestloss}_.pt')
                writer.add_scalar(f'M1 AvgValLoss/100_Docs',m1avgValloss,per100Dcount)
                writer.add_scalar(f'M2 AvgValLoss/100_Docs',m2avgValloss,per100Dcount)
                [writer.add_histogram(f'Val_Epoch_{epoch}_LSTM_Weight_{name}',para.data,per100Dcount) for name,para in model.model1_lstm.named_parameters() ]
                #if(batchsize>1):
                #    [writer.add_histogram(f'Val_Epoch_{epoch}_EmbeddingSpace1_1_Weight_{name}',para.data,per100Dcount) for name,para in model.embeddingSpace[1].named_parameters()]
                #    [writer.add_histogram(f'Val_Epoch_{epoch}_EmbeddingSpace2_1_Weight_{name}',para.data,per100Dcount) for name,para in model.embeddingSpace2[1].named_parameters()]
                #[writer.add_histogram(f'Val_Epoch_{epoch}_EmbeddingSpace1_0_Weight_{name}',para.data,per100Dcount) for name,para in model.embeddingSpace[0].named_parameters()]
                #[writer.add_histogram(f'Val_Epoch_{epoch}_EmbeddingSpace2_0_Weight_{name}',para.data,per100Dcount) for name,para in model.embeddingSpace2[0].named_parameters()]
                #[writer.add_histogram(f'Val_Epoch_{epoch}_PredictConv1D_Weight_{name}',para.data,per100Dcount) for name,para in model.predictConv1D.named_parameters()]
                [writer.add_histogram(f'Val_Epoch_{epoch}_Model1_EmbeddingSpace1_0_Weight_{name}',para.data,d) for name,para in model.model1Embedding['embedding1'][0].named_parameters()]
                [writer.add_histogram(f'Val_Epoch_{epoch}_Model1_EmbeddingSpace2_0_Weight_{name}',para.data,d) for name,para in model.model1Embedding['embedding2'][0].named_parameters()]
                [writer.add_histogram(f'Val_Epoch_{epoch}_Model1_PredictConv1D_Weight_{name}',para.data,d) for name,para in model.model1_predictConv1D.named_parameters()]
                [writer.add_histogram(f'Val_Epoch_{epoch}_Model2_EmbeddingSpace1_0_Weight_{name}',para.data,d) for name,para in model.model2Embedding['embedding1'][0].named_parameters()]
                [writer.add_histogram(f'Val_Epoch_{epoch}_Model2_EmbeddingSpace2_0_Weight_{name}',para.data,d) for name,para in model.model2Embedding['embedding2'][0].named_parameters()]
                [writer.add_histogram(f'Val_Epoch_{epoch}_Model2_Conv_Weight_{name}',para.data,d) for name,para in model.model2_Conv1D.named_parameters()]
                [writer.add_histogram(f'Val_Epoch_{epoch}_Model2_Predict_Weight_{name}',para.data,d) for name,para in model.model2_predict.named_parameters()]
                print(f'Epoch:{epoch:2d} M1 100Docs Avg Val Loss:{m1avgValloss:0.6f} , Avg Val Predict:({m1avgValPro:0.3f}, '+
                        f'{m1quantile25:0.3f},{m1median:0.3f},{m1quantile75:0.3f}) ',flush=True)
                print(f'Epoch:{epoch:2d} M2 100Docs Avg Val Loss:{m2avgValloss:0.6f} , Avg Val Predict:({m2avgValPro:0.3f}, '+
                        f'{m2quantile25:0.3f},{m2median:0.3f},{m2quantile75:0.3f}) ',flush=True)
                print("")
