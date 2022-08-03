#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind
from scipy.stats import pearsonr
import itertools


# In[ ]:


#load 10 000 train seqs
data = np.loadtxt("train_seq.text", dtype=str,max_rows=10000)


# In[6]:


#separate data into sequences and corresponding scores
seq=[]
score=[]
for i in range(len(data)):
    seq.append(data[i][0])
    score.append(float(data[i][1]))
    
    
#print(seq)
#print(score)


# In[ ]:


#find max length and min length of traon sequences
max_seq=[]
min_seq=[]
for i in seq: 
    if len(i)>len(max_seq):
        max_seq=i
print(len(max_seq))


# In[ ]:


#count occurences of each base across all sequences and store in lists
count_A_list=[]
count_T_list=[]
count_G_list=[]
count_C_list=[]
count_A=0
count_T=0
count_G=0
count_C=0
for i in seq:
    for j in i:
        if j=='A' :
            count_A+=1
        if j=='T' :
            count_T+=1
        if j=='G':
            count_G+=1
        if j=='C':
            count_C+=1
    count_A_list.append(count_A)
    count_T_list.append(count_T)
    count_G_list.append(count_G)
    count_C_list.append(count_C)

print(count_A_list)
print(count_T_list)
print(count_G_list)
print(count_C_list)
print('A=',count_A)
print('T=',count_T)
print('G=',count_G)
print('C=',count_C)


# In[14]:


#create list of fractions of base A in each sequence
counterA=[]
for i in seq:
    counterA.append(i.count('A')/len(i))
    
print(counterA[:1])


# In[15]:


#create list of fractions of base T in each sequence
counterT=[]
for i in seq:
    counterT.append(i.count('T')/len(i))
    


# In[16]:


#create list of fractions of base G in each sequence
counterG=[]
for i in seq:
    counterG.append(i.count('G')/len(i))


# In[17]:


#create list of fractions of base C in each sequence
counterC=[]
for i in seq:
    counterC.append(i.count('C')/len(i))


# In[18]:


#plot and calculate correlation between content of base A and expression
from scipy.stats import pearsonr
corr, _ = pearsonr(counterA, score)
plt.scatter(counterA,score)
plt.show()
print('R=',corr)


# In[19]:


#plot and calculate correlation between content of base T and expression
corr, _ = pearsonr(counterT, score)
plt.scatter(counterT,score)
plt.show()
print('R=',corr)


# In[ ]:


#plot and calculate correlation between content of base G and expression
corr, _ = pearsonr(counterG, score)
scipy.stats.chisquare(counterG, score)
plt.scatter(counterG,score)
plt.show()
print('R=',corr)


# In[ ]:


#plot and calculate correlation between content of base C and expression
corr, _ = pearsonr(counterC, score)
plt.scatter(counterC,score)
plt.show()
print('R=',corr)


# In[28]:


#Reb1 motif finder
#key='CCGGGTA'
key='CCGGGTAA'
reb1_motif=[]
no_reb1_motif=[]
reb1_mot_scores=[]
no_reb1_mot_scores=[]
pos=0
for i in seq:
    if key in i:
        reb1_motif.append(i)
        reb1_mot_scores.append(score[pos])
        pos+=1
    else:
        no_reb1_motif.append(i)
        no_reb1_mot_scores.append(score[pos])
        pos+=1
        
#print(no_reb1_motif)
print(reb1_motif)
print(reb1_mot_scores)


# In[29]:


#Rap1 motif finder
key2='TGTACGGGTG'
rap1_motif=[]
no_rap1_motif=[]
rap1_mot_scores=[]
no_rap1_mot_scores=[]
pos=0
for i in seq:
    if key2 in i:
        rap1_motif.append(i)
        rap1_mot_scores.append(score[pos])
        pos+=1
    else:
        no_rap1_motif.append(i)
        no_rap1_mot_scores.append(score[pos])
        pos+=1
        
#print(no_reb1_motif)
print(rap1_motif)
print(rap1_mot_scores)


# In[39]:


#plot expressions for sequences with and without Reb1 motif , calculate p value
box_plot_data=[reb1_mot_scores,no_reb1_mot_scores]
plt.boxplot(box_plot_data,patch_artist=True,labels=['Reb1 Motif','No Reb1 Motif'])
plt.title('Comparison of expression levels between sequences with Reb1 motif and without')
plt.ylabel('Expression level')
plt.show()
res = ttest_ind(reb1_mot_scores,no_reb1_mot_scores).pvalue

print('P-value=' ,res)


# In[38]:


#plot expressions for sequences with and without Rap1 motif , calculate p value
box_plot_data2=[rap1_mot_scores,no_rap1_mot_scores]
plt.boxplot(box_plot_data2,patch_artist=True,labels=['Rap1 Motif','No Rap1 Motif'])
plt.title('Comparison of expression levels between sequences with Rap1 motif and without')
plt.ylabel('Expression level')
plt.show()
res = ttest_ind(rap1_mot_scores,no_rap1_mot_scores).pvalue

print('P-value=' ,res)


# In[23]:


#count AT content
counterA=np.array(counterA)
counterT=np.array(counterT)
ATcontent=counterA+counterT
print(ATcontent)


# In[40]:


#correlation between AT content and expressions
corr, _ = pearsonr(ATcontent, score)
plt.scatter(ATcontent,score)
plt.title('AT base content vs. Expression level')
plt.xlabel('AT content as a fraction of total content in a sequence')
plt.ylabel('Expression level')
plt.show()
print('R=',corr)


# In[41]:


#count GC content
counterG=np.array(counterG)
counterC=np.array(counterC)
GCcontent=counterG+counterC
print(GCcontent)


# In[43]:


#correlation between GC content and expressions
corr, _ = pearsonr(GCcontent, score)
plt.scatter(GCcontent,score)
plt.title('GC base content vs. Expression level')
plt.xlabel('GC content as a fraction of total content in a sequence')
plt.ylabel('Expression level')
plt.show()
print('R=',corr)


# In[25]:


#make a list of sequence lengthes
seq_len=[]
for i in seq:
    seq_len.append(len(i))  


# In[44]:


#plot correlation between lengths and expression
corr, _ = pearsonr(seq_len, score)
plt.scatter(seq_len,score)
plt.title('Sequence length vs. Expression level')
plt.xlabel('Sequence length')
plt.ylabel('Expression level')
plt.show()
print('R=',corr)

