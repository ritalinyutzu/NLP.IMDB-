#!/usr/bin/env python
# coding: utf-8


#以IMDB影評數據做文字處理
#電影評論數據集下載處: http://ai.stanford.edu/~amaas/data/sentiment/   (大概84.1MB)
#pip install tarfile

import tarfile
import os
os.chdir("C:\\Users\\ritayutzu.lin\\Downloads")
os.getcwd()



#懶得下載7-zip解壓縮,可以在python中直接使用Gzip壓縮包,但要花點時間
with tarfile.open('aclImdb_v1.tar.gz','r:gz') as tar:
    def is_within_directory(directory, target):
        
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
    
        prefix = os.path.commonprefix([abs_directory, abs_target])
        
        return prefix == abs_directory
    
    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
    
        for member in tar.getmembers():
            member_path = os.path.join(path, member.name)
            if not is_within_directory(path, member_path):
                raise Exception("Attempted Path Traversal in Tar File")
    
        tar.extractall(path, members, numeric_owner=numeric_owner) 
        
    
    safe_extract(tar)


import pyprind
import pandas as pd

basepath = 'aclImdb'

labels = {'pos':1,'neg':0}
pbar = pyprind.ProgBar(50000)
df = pd.DataFrame()

for s in ('test','train'):
    for l in ('pos','neg'):
        path = os.path.join(basepath,s,l)
        for file in os.listdir(path):
            with open(os.path.join(path,file),
                      'r',encoding='utf-8') as infile:
                txt = infile.read()
            df = df.append([[txt,labels[l]]],
                           ignore_index=True)
            pbar.update()
            
#初始化一個進度條物件pbar,含五萬次的迭代
#使用巢狀for loop重複拜訪aclImdb主資料夾,train子目錄和test子目錄,讀取pos子目錄與neg子目錄其中的個別文本檔案
#將這些檔案加到df的dataFrame後面,加一個表示極性的整數"類別標籤" (1是正面評價,0是負面評價)

df.columns = ['review','sentiment']

import numpy as np

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('movie_data.csv',index=False,encoding='utf-8')

#快速確認是否讀取成功
df = pd.read_csv('./movie_data.csv')
df.head(5)

#詞袋模型(bag-of-words)->將字串轉成數字特徵向量(sentiment)
#(1)對於每個唯一字符(token)可以建立一個詞彙(vocabulary) 例如完整文件集中的字詞(word)就可以當作詞彙(vocabulary)
#(2)對於每份文件,建立一個特徵向量,在這個向量裡,紀錄每個在這份特定文件出現的"字詞"和"出現的次數"
#(3)文件中,word只佔token中的一小部分,因此sentiment中絕大部分是0,稱其為"稀疏"(sparse)

#將word轉換成sentiment->利用scikit-learn中的CountVectorizer類別,依照每份文件word出現的次數,建立詞袋模型
#CountVectorizer類別輸入一個array的文本數據,可以是許多文件,或許多句子,來建立詞袋模型

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer()
docs = np.array([
    'The sun is shining',
    'The weather is sweet',
    'The sun is shining, and the weather is sweet, and one and one is two'
])
bag = count.fit_transform(docs)

#呼叫CountVectorizer類別裡的fit_transform方法,產生詞袋模型的詞彙(word),將剛剛三個句子轉換成稀疏特徵向量
print(count.vocabulary_)
#result: word存在python的dictionary中

#print特徵向量
print(bag.toarray())
#result: 又稱為原始詞頻(raw term)
#函式:tf(t,d)->在一份文件d中,字詞term(t)出現次數

#函示:idf(t,d)= log(nd/(1+df(d,t)))->反向文件頻率
#字詞(word)出現在許多個文件之中,這些頻繁出現的字詞,沒有幫助判讀上下語意資訊的意義,簡稱廢詞
#idf(term frequency-inverse document frequency,tf-idf)
#nd = 文件總數, df(d,t) 字詞t的文件檔d的數量
#log確保低頻文件不會被賦予太大的加權

#sk-learn某轉換器,TfidfTransformer類別,從CountVectorizer類別輸入"原始詞頻"並轉換成tf-idfs
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer(use_idf=True,
                        norm='l2',  #L2正規化
                        smooth_idf=True)
np.set_printoptions(precision=2)
print(tfidf.fit_transform(count.fit_transform(docs))
     .toarray())

#先顯示電影評論集最後50個字元
df.loc[0,'review'][-50:]
#跑出一串垃圾,含有html標記,標點符號,其他非字母字元

#使用Python正規表示式(Regular expression)函式庫,re函式庫
import re
# 第一個正規表示式長這樣 <[^>]*> 用來刪除電影評論數據集影評中所有的html標記,清除html標記後用其他正規表示式來找"表情符號"
#(雖然有些工程式不建議用正規表示式來解析html)
#正規表示式 [\w]+刪除本字中的所有非字元符號,並且轉換所有字元為小寫
#暫時將儲存的表情符號(emoticons)加到處裡完畢的文件字串之後,刪除表情符號中的鼻子(-)
#(我也想知道為什麼要刪掉鼻子)

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                         text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text

#確認預處裡器有正常運轉,清理過後的文件與字串最後附加表情符號
preprocessor(df.loc[0, 'review'][-50:])

#進行測試,裡面混表情符號
preprocessor("</a>This :) is :( :( a test :-)!")

#對DataFrame中所有的電影評論,套用Preprocessor函數
df['review'] = df['review'].apply(preprocessor)

#成功準備好電影評論數據集後,需要考慮如何將文本語料庫(text corpora)分割成個別字符(token)元素 -> 字符化
#字符化:依照空白字元(whitespace characters)分割文件成為單獨的字詞(token)
def tokenizer(text):
    return text.split()
tokenizer('runners like running and thus they run')
#result>>>: ['runner', 'like', 'run', 'and', 'thu', 'they', 'run']

#token情境下,另一技術"詞幹"(word stemming),將字詞轉變成字根形式的過程,對應相關的字詞到詞幹->"波特詞幹還原演算法"(Porter stemmer algorithm)
#NLTK是波特詞幹還原演算法的實作 -> visit website (http://www.nltk.org/book/)
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]
tokenizer_porter('runner like running and thus they run')
#result>>>: ['runner', 'like', 'run', 'and', 'thu', 'they', 'run']
#跟字符化可以獲得一樣的成果

#波特詞幹還原演算法是最古老的詞幹還原演算法,另一種較新且受歡迎的還有"雪球詞幹還原器"(Snowball stemmer, Porter2) & Lancaster stemmer

from nltk.corpus import stopwords
stop = stopwords.words('english')
[w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:] if w not in stop]
#result>>>:['runner', 'like', 'run', 'run', 'lot'] 沒有出現重複字詞

