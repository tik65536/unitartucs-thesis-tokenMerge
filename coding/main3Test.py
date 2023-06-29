import numpy as np
import torch
import re
import itertools
import spacy
import argparse
from torch.distributions.bernoulli import Bernoulli
import pickle
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
import warnings
import time
import datetime
import os
import n_sphere
import math
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

parser.add_argument('-batch', type=int, default=5,
                    help='BatchSize')

parser.add_argument('-maxlen', type=int, default=600,
                    help='Max text len')


parser.add_argument('-preprocesstext', type=float, default=1,
                    help='Remove Stop word and Pun')


parser.add_argument('-seqlen', type=int, default=5,
                    help='Seq Len for reading token')

parser.add_argument('-slide', type=int, default=1,
                    help='Seqlen Sliding')


parser.add_argument('-interventionP', type=float, default=0.5,
                    help='Probability for intervention')

parser.add_argument('-permuteidx', nargs="*", default=None,
                    help='Permute Token idx')

parser.add_argument('-onlyMerge', nargs="*", default=[],
                    help='Skip POS tag for Merge')

parser.add_argument('-consecutive', type=int, default=0,
                    help='Consecutive merge')

parser.add_argument('-weight', type=str, default=None,
                    help='Weight File Name')

parser.add_argument('-rotate', type=int, default=0,
                    help='Rotate or not')

parser.add_argument('-degree', type=float, default=0,
                    help='Degree to rotate')
args = parser.parse_args()
seqlen=args.seqlen
batchsize=args.batch
interventionP=args.interventionP
sliding=args.slide
preprocess = args.preprocesstext
maxlen =args.maxlen
permuteidx = list(range(seqlen)) if(args.permuteidx is None) else [ int(x) for x in args.permuteidx ]
onlyMerge=args.onlyMerge
consecutive = True if(args.consecutive==1) else False
weightfile= args.weight
rotate = True if (args.rotate==1) else False
degree = args.degree

assert weightfile is not None,"Weight File Name is None"
print(f'Run Para : {args}',flush=True)

weightPath=f'./LSTMWeight/{weightfile}'

validationPlot=f"./validationPlot_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_Degree_{degree}/"
os.mkdir(validationPlot)

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
#model=PredictLSTMIntervionP(n_vocab,embeddingDim,seqlen,seqlen,minmerge,maxmerge,batchsize,GRU,hiddenLayer,hiddenSize,bidirectional,True,numOfConvBlock,groupRelu,convpredict,kernelsize,mergeRate)
#model.load_state_dict(torch.load(weightPath))
model=torch.load(weightPath)
criterion = torch.nn.CrossEntropyLoss()
gate=['igone','forget','learn','output']
test_text=test_pd['text']
test_label=test_pd['label']
print(f'Embedding Shape: {model.embeddingSpace.weight.shape}')
if(rotate):
    w=model.embeddingSpace.weight.detach().cpu().numpy()
    for widx in range(len(w)):
        try:
            t=n_sphere.convert_spherical(w[widx])
            t[1]+=(degree*math.pi)
            t[2]+=(degree*math.pi)
            t=n_sphere.convert_rectangular(t)
            w[widx]=t.astype(float)
        except ValueError:
            pass
    model.embeddingSpace.weight=torch.nn.Parameter(torch.tensor(w,dtype=float).to(device))

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
        fig.savefig(f'./{validationPlot}Validation_{valbatchcount}_switch{int(switch)}_preprocess{int(preprocessswitch)}_{avgloss:0.6f}_plot.png',dpi=400)
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
