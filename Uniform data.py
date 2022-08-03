# In[ ]:


import tensorflow as tf
tf.compat.v1.disable_eager_execution()
#import shap
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from keras.utils import np_utils
from tensorflow import keras
import math

from sklearn.metrics import r2_score 


# In[28]:


#load given train sequences (shuffled)
data = np.loadtxt("train_shuff.txt", dtype=str)


# In[55]:


#separate data into list of sequences and list of expressions
seq=[]
score=[]
for i in range(len(data)):
    seq.append(data[i][0])
    score.append(float(data[i][1]))
    


# In[ ]:


#distribution of the scores
plt.hist(score,bins=10)
plt.title('Distribution of expression levels')
plt.xlabel('Bin number')
plt.ylabel('Number of sequences')


# In[ ]:


#round expressions to the closest whole number (bin number)
scores_rounded=[]
for i in score: 
    i=round(i)
    scores_rounded.append(i)
print(scores_rounded[:100])    


# In[ ]:


#histogram of rounded scores
plt.hist(scores_rounded)


# In[61]:


#count number of sequences for each bin
from collections import Counter
a=Counter(scores_rounded)


# In[ ]:


#select a certain number of sequences for each bin
hits=np.zeros((18,))
uniform_scores=[]
uniform_seq=[]
for i in tqdm.tqdm(range(len(seq))): 
    if hits[scores_rounded[i]]<=30000:
        hits[scores_rounded[i]]+=1
        uniform_scores.append(score[i])
        uniform_seq.append(seq[i])
print(uniform_scores[:5])
print(uniform_seq[:5])


# In[ ]:


#distribution of uniformed scores
plt.hist(uniform_scores,bins=15)
plt.title('Uniformed distribution of expression levels')
plt.xlabel('Bin number')
plt.ylabel('Number of sequences')


# In[49]:


#create a file with sequences and corresponding uniformed scores
with open("uniformed data.txt", "w") as f:
    for i,seq in enumerate(uniform_seq):
        print(seq,uniform_scores[i], file=f)

