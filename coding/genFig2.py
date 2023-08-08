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
with open('./test_negCorpora.plk','rb') as f:
    negCount=pickle.load(f)

accy=pd.read_csv('./ResultLog/Curl_BaseCase_val.csv')

for k in totalCount.keys():
    total=totalCount[k]
    neg=negCount[k]
    negCount[k]=neg/total

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
negtextFreq = []
poslist=[]
for doc in docs:
    countlist=[]
    plist=[]
    for token in doc:
        if(token.is_stop==False and token.is_punct==False):
            countlist.append(negCount[token.text])
            if(token.text!='s0s' and token.text!='e0s'):
                plist.append(token.pos)
            elif(token.text=='s0s'):
                plist.append(70)
            else:
                plist.append(78)
    negtextFreq.append(countlist)
    poslist.append(plist)

for i,r in enumerate(negtextFreq):
    if len(r)<175:
        negtextFreq[i]=(r+([negCount['e0s']]*(175-len(r))))
    else:
        negtextFreq[i]=negtextFreq[i][:175]
for i,r in enumerate(poslist):
    if len(r)<175:
        poslist[i]=(r+([78]*(175-len(r))))
    else:
        poslist[i]=poslist[i][:175]

negtextFreq=np.array(negtextFreq)
poslist=np.array(poslist)
#postextFreq=np.array(postextFreq)
path='/home/dick/RunpodData/Curl_BaseCase_R1_seqlen25_sldie25_batch1000_opt25_dynamicOptFalse_train15000_ksize2_e3_BiDirectionTrue_HL3_HS3_P0.0_MaxMerge2_MinMerge2_preprocess1.0_20230724-111115'
curdataList = glob.glob(f'{path}/val_curldata_*.plk')
curldata={}
for file in curdataList:
    with open(f'{file}','rb') as f:
        try:
            tmp=pickle.load(f)
            i=list(tmp.keys())[0]
            curldata[i]=tmp[i]
        except EOFError:
            pass
lastepoch=sorted(list(curldata.keys()))[-1]
print(f'Last Epoch: {lastepoch}')
negFreq=negtextFreq[:]
#posFreq=postextFreq[:]
negcolormap = cm.bwr
norm = Normalize()
norm.autoscale(negFreq)
nco=negcolormap(norm(negFreq))
keys=list(curldata[0]['forward'].keys())
plt.rcParams['grid.color'] = "black"
with pymp.Parallel(6) as p:
    for epoch in p.range(10,lastepoch+1):
        direction='forward'
        epoch_in=curldata[epoch][direction][0][:,:25,:]
        epoch_curl=curldata[epoch][direction][0][:,25:,:]

        for k in range(1,len(keys)):
            epoch_in = np.hstack((epoch_in,curldata[epoch][direction][keys[k]][:,:25,:]))
            epoch_curl = np.hstack((epoch_curl,curldata[epoch][direction][keys[k]][:,25:,:]))
        in_l=np.linalg.norm(epoch_in,axis=-1)
        epoch_in_unit = epoch_in/ in_l.reshape(1000,175,1)
        curl_l=np.linalg.norm(epoch_curl,axis=-1)
        epoch_curl_unit = epoch_curl/ curl_l.reshape(1000,175,1)

        sample=epoch_in_unit[:]
        sample_curl=epoch_curl_unit[:]
        start,stop=0,175
        plt.close('all')
        divfig=plt.figure(figsize=(10,10))
        curlax=divfig.add_subplot(projection='3d')
        for i in range(start,stop):
            #n=epoch_in[:,i:i+1,:]
            #n=np.linalg.norm(n,axis=-1).reshape(-1,)
            keep=np.ceil((i/test1['length'])*100)
            keep=np.where(keep>=100,-1,keep)
            p = [ int(k) for k in keep if(k!=-1)]
            keep=np.where(keep!=-1)[0]
            curlax.scatter3D(sample_curl[keep,i:i+1,0], sample_curl[keep,i:i+1,1], 
                             sample_curl[keep,i:i+1,2],c=nco[keep,i:i+1],s=0.05)
            #curlax.quiver(sample[keep,i:i+1,0], sample[keep,i:i+1,1], sample[keep,i:i+1,2],
            #              sample_curl[keep,i:i+1,0] , sample_curl[keep,i:i+1,1], sample_curl[keep,i:i+1,2] ,
            #              length=0.2,colors=nco[keep,i:i+1],normalize=True)#
        curlax.set_xlabel('LSTM output dim 1')
        curlax.set_ylabel('LSTM output dim 2')
        curlax.set_zlabel('LSTM output dim 3')
        #curlax.view_init(elev=60, azim=-90)
        curlax.set_facecolor('gray')
        sm = plt.cm.ScalarMappable(cmap=cm.bwr, norm=norm)
        cbar=divfig.colorbar(sm,ax=curlax,shrink=0.5,pad=0.2)
        cbar.set_label('Negative Tokens Relative Frequence')
        curlax.set_title(f"BaseCase LSTM output unit norm Epoch {epoch} Accy:{accy.loc[epoch]['Accy']} \n ValidationSet (1000 Samples) ColorMap=Negative Sentiment Token Frequence")
        plt.tight_layout()
        plt.savefig(f'./GenFig/epoch_{epoch}')
