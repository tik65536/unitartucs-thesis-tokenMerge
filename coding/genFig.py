#!/usr/bin/python3
import pickle
import os
import glob
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import spacy
import pandas as pd
import pymp
# For better quality video
nlp=spacy.load("en_core_web_md")

with open('./first1000_testset.plk','rb') as f:
    test1=pickle.load(f)
with open('./test_totalCorpora.plk','rb') as f:
    totalCount=pickle.load(f)
with open('./test_posCorpora.plk','rb') as f:
    posCount=pickle.load(f)
with open('./test_negCorpora.plk','rb') as f:
    negCount=pickle.load(f)

accy=pd.read_csv('./ResultLog/Curl_BaseCase_val.csv')

for k in totalCount.keys():
    total=totalCount[k]
    neg=negCount[k]
    pos=negCount[k]
    negCount[k]=neg/total
    posCount[k]=pos/total
posidx=np.array(test1[test1['label']==1].index)
negidx=np.array(test1[test1['label']==0].index)
samplesize=200
nsmpidx=np.random.choice(negidx,samplesize,replace=False)
psmpidx=np.random.choice(posidx,samplesize,replace=False)
with open('./Master_vocab.pkl','rb') as f:
    vocab=pickle.load(f)
tok2id=vocab['tok2id']
id2tok=vocab['id2tok']
endid=tok2id['e0s']
docs=nlp.pipe(test1['text'])
negtextFreq = [ [ negCount[token.text] for token in doc if(token.is_stop==False and token.is_punct==False)] for doc in docs ]
for i,r in enumerate(negtextFreq):
    if len(r)<175:
        negtextFreq[i]=(r+([negCount['e0s']]*(175-len(r))))
    else:
        negtextFreq[i]=negtextFreq[i][:175]
docs=nlp.pipe(test1['text'])
postextFreq = [ [ posCount[token.text] for token in doc if(token.is_stop==False and token.is_punct==False)] for doc in docs ]
for i,r in enumerate(postextFreq):
    if len(r)<175:
        postextFreq[i]=(r+([posCount['e0s']]*(175-len(r))))
    else:
        postextFreq[i]=postextFreq[i][:175]

negtextFreq=np.array(negtextFreq)
postextFreq=np.array(postextFreq)
path='/home/dick/RunpodData/Curl_BaseCase_R1_seqlen25_sldie25_batch1000_opt25_dynamicOptFalse_train15000_ksize2_e3_BiDirectionTrue_HL3_HS3_P0.0_MaxMerge2_MinMerge2_preprocess1.0_20230724-111115'
curdataList = glob.glob(f'{path}/val_curldata_*.plk')
divdataList = glob.glob(f'{path}/val_divdata_*.plk')
curldata={}
divdata={}
for file in curdataList:
    with open(f'{file}','rb') as f:
        try:
            tmp=pickle.load(f)
            i=list(tmp.keys())[0]
            curldata[i]=tmp[i]
        except EOFError:
            pass
for file in divdataList:
    with open(f'{file}','rb') as f:
        tmp=pickle.load(f)
        i=list(tmp.keys())[0]
        divdata[i]=tmp[i]


#negFreq=negtextFreq[:]
#posFreq=postextFreq[:]
plt.rcParams['grid.color'] = "black"
negcolormap = cm.bwr
norm = Normalize()
colors=np.zeros((1000,175))
colors[posidx,:]=1
norm.autoscale(colors)
nco=negcolormap(norm(colors))
#norm.autoscale(negFreq)
#nco=negcolormap(norm(negFreq))
keys=list(curldata[0]['forward'].keys())
with pymp.Parallel(5) as p:
    for epoch in p.range(10,432):
        direction='forward'
        epoch_curl=curldata[epoch][direction][0][:,25:,:]
        epoch_div=divdata[epoch][direction][0][:,25:,:]
        for k in range(1,len(keys)):
            epoch_curl = np.hstack((epoch_curl,curldata[epoch][direction][keys[k]][:,25:,:]))
            epoch_div = np.hstack((epoch_div,divdata[epoch][direction][keys[k]][:,25:,:]))
        l=np.linalg.norm(epoch_curl,axis=-1)
        epoch_curl=epoch_curl/l.reshape(1000,175,1)
        sample_curl=epoch_curl[:]
        start,stop=0,175
        plt.close('all')
        divfig=plt.figure(figsize=(10,10))
        curlax=divfig.add_subplot(projection='3d')
        for i in range(start,stop):
            #n=epoch_in[:,i:i+1,:]
            #n=np.linalg.norm(n,axis=-1).reshape(-1,)
            #curlax.quiver(epoch_in[:,i:i+1,0], epoch_in[:,i:i+1,1], epoch_in[:,i:i+1,2],
            #              epoch_curl[:,i:i+1,0] , epoch_curl[:,i:i+1,1], epoch_curl[:,i:i+1,2] ,
            #              length=0.2,colors=nco[:,i:i+1],normalize=True,linewidths=n)#
            curlax.scatter3D(sample_curl[:,i:i+1,0], sample_curl[:,i:i+1,1], sample_curl[:,i:i+1,2],c=nco[:,i:i+1],s=1)
        curlax.set_xlabel('LSTM output dim 1 (Curl)')
        curlax.set_ylabel('LSTM output dim 2 (Curl)')
        curlax.set_zlabel('LSTM output dim 3 (Curl)')
        curlax.set_facecolor('gray')
        curlax.view_init(elev=20, azim=-80)
        sm = plt.cm.ScalarMappable(cmap=cm.bwr, norm=norm)
        cbar=divfig.colorbar(sm,ax=curlax,shrink=0.5,pad=0.2)
        cbar.set_label('Positive=1, Negative=0')
        curlax.set_title(f"MergePROPN Curl of output dimension, Epoch {epoch} Accy:{accy.loc[epoch]['Accy']} \n ValidationSet (1000 Samples) ColorMap=Positive/Negative Sentiment")
        plt.tight_layout()
        plt.savefig(f'./GenFig/epoch_{epoch}')
