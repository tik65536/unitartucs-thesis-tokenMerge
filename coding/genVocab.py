import pandas as pd
import numpy as np
import re
from datasets import load_dataset
import spacy
from collections import Counter
import pickle
import pymp

#Reference https://muhark.github.io/python/ml/nlp/2021/10/21/word2vec-from-scratch.html

#nlp = spacy.load("en_core_web_sm")
nlp = spacy.load("en_core_web_md")

#https://www.kaggle.com/code/youssefamdouni/movie-review-classification-using-spacy
def clean_review(text):
    clean_text = re.sub('<.*?>', '', text)
    clean_text =re.sub("https?\S+", "", clean_text)
    clean_text = re.sub("www\S+", "", clean_text)
    #clean_text = re.sub('[^a-zA-Z\']', ' ', clean_text)
    #clean_text = clean_text.lower()
    return clean_text.lower()



imdb_dataset = load_dataset('imdb', split=['train[5000:20000]', 'test[11000:14000]'])
#imdb_dataset = load_dataset('imdb')
train_pd=pd.DataFrame(columns=["text","label"])
test_pd=pd.DataFrame(columns=["text","label"])
train_pd["text"]=imdb_dataset[0]['text']
train_pd["label"]=imdb_dataset[0]['label']
test_pd["text"]=imdb_dataset[1]['text']
test_pd["label"]=imdb_dataset[1]['label']
train_pd['text']='S0S '+train_pd['text']+' E0S'
test_pd['text']='S0S '+test_pd['text']+' E0S'
train_pd['text']=train_pd['text'].apply(lambda x: clean_review(x))
test_pd['text']=test_pd['text'].apply(lambda x: clean_review(x))
#master_pd=pd.concat([train_pd,test_pd])
train_pos_pd=train_pd[train_pd['label']==1]
train_neg_pd=train_pd[train_pd['label']==0]
test_pos_pd=test_pd[test_pd['label']==1]
test_neg_pd=test_pd[test_pd['label']==0]
counter=[ None,None ]
data=[{'pos':nlp.pipe(train_pos_pd['text']),'neg':nlp.pipe(train_neg_pd['text'])}, {'pos':nlp.pipe(test_pos_pd['text']),'neg':nlp.pipe(test_neg_pd['text'])}]
for i in range(2):
    docs=data[i]['pos']
    counter[i] = Counter([token.text for doc in docs  for token in doc if(token.is_stop==False and token.is_punct==False)])
    docs=data[i]['neg']
    for doc in docs:
        for token in doc:
            if(token.is_stop==False and token.is_punct==False):
                counter[i][token.text]=-1 if(counter[i][token.text]==0) else counter[i][token.text]-1
with open('./traincounter.plk','wb') as f:
    pickle.dump(counter[0],f)
with open('./textcounter.plk','wb') as f:
    pickle.dump(counter[1],f)
print(counter[0].most_common(10))
print(counter[1].most_common(10))
