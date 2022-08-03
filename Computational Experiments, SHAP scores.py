from tensorflow import keras
import re
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow import keras
import shap
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from keras.utils import np_utils
from tensorflow import keras
import math
import re
from sklearn.metrics import r2_score 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.ion()
import logomaker as lm

#load the best model
model = keras.models.load_model('Trial_5.model')

#load uniformed shuffled data
data_uni= np.loadtxt("uniformed_shuffled.txt", dtype=str)

#load test sequences
data_test= np.loadtxt("test_sequences.txt", dtype=str)

#separate data into sequences and scores
seq_uni=[]
score_uni=[]
for i in range(len(data_uni)):
    seq_uni.append(data_uni[i][0])
    score_uni.append(float(data_uni[i][1]) / 18)



#function that onehotencodes sequences
def one_hot_encode(sequence):
    table=np.empty((len(sequence),4),dtype='int8')
    ordseq=np.fromstring(sequence,np.int8)
    table[:,0]=(ordseq==ord('A'))
    table[:,1]=(ordseq==ord('C'))
    table[:,2]=(ordseq==ord('G'))
    table[:,3]=(ordseq==ord('T'))
    return table


#function that makes sequences of equal length 135bp and onehotencodes them
def Equal_Len(seqs,seq_resize_encoded)   :  
    for i,seq in enumerate(tqdm.tqdm(seqs)):
            if len(seq) <=135:
                diff=135-len(seq)
                if diff==1:
                    seq_resize=seq+right_DNA_str[0]
                    seq_encoded=one_hot_encode(seq_resize)
                    seq_resize_encoded.append(seq_encoded)
                elif diff ==0:
                    seq_encoded=one_hot_encode(seq)
                    seq_resize_encoded.append(seq_encoded)
                elif (diff % 2) == 0:
                    seq_resize=left_DNA_str[int(-(diff/2)):]+seq+right_DNA_str[:int(diff/2)]
                    seq_encoded=one_hot_encode(seq_resize)
                    seq_resize_encoded.append(seq_encoded)
                elif (diff % 2) != 0:
                    seq_resize=left_DNA_str[int(-(diff//2)):]+seq+right_DNA_str[:int(((diff//2)+1))]
                    seq_encoded=one_hot_encode(seq_resize)
                    seq_resize_encoded.append(seq_encoded)
    return seq_resize_encoded

#vector parts that were used to elongate short sequences
right_DNA_str='TCTTAATTAAAAAAAGATAGAAAACATTAGGAGT'

left_DNA_str='CTAGCAGGAATGATGCAAAAGGTTCCCGATTCGAAC'


#function that only elongates sequences to 135bp
def Resized(seqs,seq_resized)   :  
    for i,seq in enumerate(tqdm.tqdm(seqs)):
            if len(seq) <=135:
                diff=135-len(seq)
                if diff==1:
                    seq_resize=seq+right_DNA_str[0]
                    seq_resized.append(seq_resize)
                elif diff ==0:
                    seq_resized.append(seq_resize)
                elif (diff % 2) == 0:
                    seq_resize=left_DNA_str[int(-(diff/2)):]+seq+right_DNA_str[:int(diff/2)]
                    seq_resized.append(seq_resize)
                elif (diff % 2) != 0:
                    seq_resize=left_DNA_str[int(-(diff//2)):]+seq+right_DNA_str[:int(((diff//2)+1))]
                    seq_resized.append(seq_resize)
    return seq_resized


#function that elongtes sequences to 80 bp
def Resized_80(seqs,seq_resized)   :  
    for i,seq in enumerate(tqdm.tqdm(seqs)):
            if len(seq) >80:
                diff=len(seq)-80
                if diff==1:
                    seq_resize=seq[:-1]
                    seq_resized.append(seq_resize)
                elif (diff % 2) == 0:
                    seq_resize=seq[int(diff/2):-int(diff/2)]
                    seq_resized.append(seq_resize)
                elif (diff % 2) != 0:
                    seq_resize=seq[int(diff/2):-(int(diff/2)+1)]
                    seq_resized.append(seq_resize)
    return seq_resized

#resize original sequences to 80bp
seq_resized_80=[]
Resized_80(seq_uni,seq_resized_80)
pass


#resize original sequences to 135bp
seq_resized=[]
Resized(seq_uni,seq_resized)
pass


#resize original sequences and encode them
seq_resize_encoded=[]
Equal_Len(seq_uni,seq_resize_encoded)
pass


#select only sequences of length 110 and corresponding scores 
seq_uni_110=[]
score_uni_110=[]
for i in range(len(seq_uni)):
    if len(seq_uni[i])==110:
        seq_uni_110.append(seq_uni[i])
        score_uni_110.append(score_uni[i])


#create reference sequences for shap out of firts 1000 seqs
reference_seq=[]
Equal_Len(seq_uni[:1000],reference_seq)
pass


#shap initiation
def standard_combine_mult_and_diffref(mult,orig_inp,bg_data):
    to_return=[(mult[l]*(orig_inp[l]-bg_data[l])).mean(0)
                for l in range(len(orig_inp))]
    return to_return


#create reference sequences
black_reference_images=np.array(reference_seq)


#calculate shap scores for 1000 sequences
scores=[]
x=model.output
outputtarget=tf.reduce_sum(x,axis=0,keepdims=True)
black_reference_explainer=shap.TFDeepExplainer((model.input,outputtarget),black_reference_images,combine_mult_and_diffref=standard_combine_mult_and_diffref)
blackShapScores=black_reference_explainer.shap_values(np.array(seq_resize_encoded[1000:2000]),progress_message=100)
scores.append(blackShapScores)


#select shap importances for base A in every seqeunce and every position, 
#average the importances for each position in the sequence across all sequences,
#result is stored avergaed across 1000 sequences shap score for A being at position 1, 2, ... and so on along the sequence
a_scores_means_positions=[]
for i in tqdm.tqdm(range(135)):
    a_scores=[]
    for idx,seq in tqdm.tqdm(enumerate(seq_resized[1000:2000])):
        if seq[i]=='A':
            a_scores.append(scores[0][0][idx][i][0])
    if len(a_scores)==0:
        a_scores_mean=0
        a_scores_means_positions.append(a_scores_mean)
    else:
        a_scores_mean=sum(a_scores)/len(a_scores)
        a_scores_means_positions.append(a_scores_mean)


#select shap importances for base C in every seqeunce and every position, 
#average the importances for each position in the sequence across all sequences,
#result is stored avergaed across 1000 sequences shap score for C being at position 1, 2, ... and so on along the sequence
c_scores_means_positions=[]
for i in tqdm.tqdm(range(135)):
    c_scores=[]
    for idx,seq in tqdm.tqdm(enumerate(seq_resized[1000:2000])):
        if seq[i]=='C':
            c_scores.append(scores[0][0][idx][i][0])
    if len(c_scores)==0:
        c_scores_mean=0
        c_scores_means_positions.append(c_scores_mean)
    else:
        c_scores_mean=sum(c_scores)/len(c_scores)
        c_scores_means_positions.append(c_scores_mean)


#select shap importances for base G in every seqeunce and every position, 
#average the importances for each position in the sequence across all sequences,
#result is stored avergaed across 1000 sequences shap score for G being at position 1, 2, ... and so on along the sequence
g_scores_means_positions=[]
for i in tqdm.tqdm(range(135)):
    g_scores=[]
    for idx,seq in tqdm.tqdm(enumerate(seq_resized[1000:2000])):
        if seq[i]=='G':
            g_scores.append(scores[0][0][idx][i][0])
    if len(g_scores)==0:
        g_scores_mean=0
        g_scores_means_positions.append(g_scores_mean)
    else:
        g_scores_mean=sum(g_scores)/len(g_scores)
        g_scores_means_positions.append(g_scores_mean)


#select shap importances for base T in every seqeunce and every position, 
#average the importances for each position in the sequence across all sequences,
#result is stored avergaed across 1000 sequences shap score for T being at position 1, 2, ... and so on along the sequence
t_scores_means_positions=[]
for i in tqdm.tqdm(range(135)):
    t_scores=[]
    for idx,seq in tqdm.tqdm(enumerate(seq_resized[1000:2000])):
        if seq[i]=='T':
            t_scores.append(scores[0][0][idx][i][0])
    if len(t_scores)==0:
        t_scores_mean=0
        t_scores_means_positions.append(t_scores_mean)
    else:
        t_scores_mean=sum(t_scores)/len(t_scores)
        t_scores_means_positions.append(t_scores_mean)



#make an array of lists that store importances for each base
array_scores = np.array( [ a_scores_means_positions ,c_scores_means_positions , g_scores_means_positions,t_scores_means_positions])


#plot the array into a matrix 
label=['A','C','G','T']
plt.rcParams['figure.figsize']=[15,5]
fig, ax = plt.subplots()
im = ax.imshow(array_scores,
               interpolation='nearest',
               cmap='bwr',
               vmin=-0.001,
               vmax=0.001)

fig.colorbar(im, ax=ax)
ax.set_yticks(np.arange(len(label)), labels=label)


fig.tight_layout()



#code to generate random sequences of certain lengths
seq_length = 80
r=[]
for i in range (1000):
    r.append(''.join(np.random.choice(('C','G','T','A'), seq_length )))


# EXPERIMENT EXPRESSION VS REB1 POSITION


#insert reb1 on every position in a 80 bp sequence, make model to predict corresponding expressions, 
#and average the predictions across 1000 random seuencessequenc
means_exp=[]
for i in range(80):
    ran_sequences_reb1=[]
    for seq in ran_sequences:
        if i==0:
            final_string='CGGGTAA' + seq
            ran_sequences_reb1.append(final_string)
        elif i==80:
            final_string=seq+'CGGGTAA'
            ran_sequences_reb1.append(final_string)
        else:
            final_string=seq[:i]+'CGGGTAA' + seq[i:]
            ran_sequences_reb1.append(final_string)

            
    test_batches=DNAseqload(ran_sequences_reb1)
    means_exp.append(np.mean(model.predict(test_batches,verbose=2)))


#select random sequences where reb1 is at position 33 
i=33
ran_sequences_reb1_pos33=[]
for seq in ran_sequences:
        final_string=seq[:i]+'CGGGTAA' + seq[i:]
        ran_sequences_reb1_pos33.append(final_string)



#select random sequences where reb1 is at position 0 
i=0
ran_sequences_reb1_pos0=[]
for seq in ran_sequences:
        final_string='CGGGTAA' + seq
        ran_sequences_reb1_pos0.append(final_string)


#prepare sequences for shap where reb1 is at position 33
list_of_shap_seq_reb1_pos33=[]
Equal_Len(ran_sequences_reb1_pos33,list_of_shap_seq_reb1_pos33)
pass


#prepare sequences for shap where reb1 is at position 0
list_of_shap_seq_reb1_pos0=[]
Equal_Len(ran_sequences_reb1_pos0,list_of_shap_seq_reb1_pos0)
pass


#plot average expression levels as a function of reb1 pisition
plt.plot(means_exp)
plt.ylabel('Expression level',fontsize=14)
plt.xlabel('Position of Reb1',fontsize=14)
plt.title('Expression levels as a function of Reb1 position in RANDOM sequences',fontsize=14)
plt.show()


#find shap scores for random sequences where reb1 is in position 33
imp_scores_reb1_pos33=[]
x=model.output
outputtarget=tf.reduce_sum(x,axis=0,keepdims=True)
black_reference_explainer=shap.TFDeepExplainer((model.input,outputtarget),black_reference_images,combine_mult_and_diffref=standard_combine_mult_and_diffref)
blackShapScores=black_reference_explainer.shap_values(np.array(list_of_shap_seq_reb1_pos33),progress_message=100)
imp_scores_reb1_pos33.append(blackShapScores)


#draw the corresponding logo
logo=lm.Logo(av_im_scores_random(imp_scores_reb1_pos33,list_of_shap_seq_reb1_pos33))
logo.highlight_position_range(57,63,alpha=0.3,color='lightblue',edgecolor='black')
logo.ax.set_title('Average Logo of Reb1 in 1000 random sequences at position 33',fontsize=14)


for i in range(5):
    df = pd.DataFrame(imp_scores_reb1_pos33[0][0][i], columns=['A',  'C','G','T'])
    df_sum=df.sum(axis=1)
    a=pd.DataFrame(list_of_shap_seq_reb1_pos33[i],columns=['A',  'C','G','T'])
    df_summed_encoded= a.apply(lambda x: np.asarray(x) * np.asarray(df_sum))
    logo=lm.Logo(df_summed_encoded)
    logo.highlight_position_range(57,63,alpha=0.3,color='lightblue',edgecolor='black')
    logo.ax.set_title('Logo of Reb1 in random sequences at position 33',fontsize=14)


#importances for random sequences where reb1 at position 0
imp_scores_reb1_pos0=[]
x=model.output
outputtarget=tf.reduce_sum(x,axis=0,keepdims=True)
black_reference_explainer=shap.TFDeepExplainer((model.input,outputtarget),black_reference_images,combine_mult_and_diffref=standard_combine_mult_and_diffref)
blackShapScores=black_reference_explainer.shap_values(np.array(list_of_shap_seq_reb1_pos0),progress_message=100)
imp_scores_reb1_pos0.append(blackShapScores)


#function to generate a matrix for average logo (averages importances across all sequences)
def av_im_scores_random(importances,sequences):
    av_df=0
    for i in tqdm.tqdm(range(len(importances[0][0]))):
        df = pd.DataFrame(importances[0][0][i], columns=['A',  'C','G','T'])
        df_sum=df.sum(axis=1)
        a=pd.DataFrame(sequences[i],columns=['A',  'C','G','T'])
        df_summed_encoded= a.apply(lambda x: np.asarray(x) * np.asarray(df_sum))
        av_df=av_df+df_summed_encoded

    av_df=av_df/len(importances[0][0])
    return av_df


av_im_scores_random(imp_scores_reb1_pos0,list_of_shap_seq_reb1_pos0)
logo=lm.Logo(av_df)
logo.highlight_position_range(24,30,alpha=0.3,color='lightblue',edgecolor='black')
logo.ax.set_title('Average Logo of Reb1 in 1000 random sequences at position 0',fontsize=14)



#cuts the sequences to 80 bp
seq_uni_cut=[]
score_uni_cut=[]
for i in range(len(data_uni)):
    if len(seq_uni[i][17:-13])==80:
        seq_uni_cut.append(seq_uni[i][17:-13])
        score_uni_cut.append(score_uni[i])




#select seqs withg reb1 out of the cut sequences
seq_uni_cut_reb1=[]
scores_uni_cut_reb1=[]
for idx,seq in enumerate(seq_uni_cut):
    if 'CGGGTAA' in seq:
        seq_uni_cut_reb1.append(seq)
        scores_uni_cut_reb1.append(score_uni_cut[idx])
        



#the same experiment of reb1 insertion but with real sequences
mean_scores=[]
seq_count=[]
for i in range(80):
    scores=[]
    for idx,seq in tqdm.tqdm(enumerate(seq_uni_cut_reb1)):
        if seq[i:i+7]=='CGGGTAA':
            scores.append(scores_uni_cut_reb1[idx])
    mean_scores.append(np.mean(scores))
    seq_count.append(len(scores))
        




plt.plot(mean_scores)
plt.ylabel('Expression level')
plt.xlabel('Position of Reb1')
plt.title('Expression levels as a function of Reb1 position in REAL sequences')
plt.show()





#find real seqs were reb1 is at position 0
real_sequences_reb1_pos0=[]
for seq in seq_uni_cut_reb1:
        if seq[:7]=='CGGGTAA':
            real_sequences_reb1_pos0.append(seq)



#prepare the seqs for shap
list_of_shap_realseq_reb1_pos0=[]
Equal_Len(real_sequences_reb1_pos0,list_of_shap_realseq_reb1_pos0)
pass



#find scores
imp_scores_realseq_reb1_pos0=[]
x=model.output
outputtarget=tf.reduce_sum(x,axis=0,keepdims=True)
black_reference_explainer=shap.TFDeepExplainer((model.input,outputtarget),black_reference_images,combine_mult_and_diffref=standard_combine_mult_and_diffref)
blackShapScores=black_reference_explainer.shap_values(np.array(list_of_shap_realseq_reb1_pos0),progress_message=100)
imp_scores_realseq_reb1_pos0.append(blackShapScores)



#plot the average logo 
logo=lm.Logo(av_im_scores_random(imp_scores_realseq_reb1_pos0,list_of_shap_realseq_reb1_pos0))
logo.highlight_position_range(27,33,alpha=0.3,color='lightblue',edgecolor='black')
logo.ax.set_title('Average Logo of Reb1 in 95 REAL sequences at position 0',fontsize=14)



for i in range(5):
    df = pd.DataFrame(imp_scores_realseq_reb1_pos0[0][0][i], columns=['A',  'C','G','T'])
    df_sum=df.sum(axis=1)
    a=pd.DataFrame(list_of_shap_realseq_reb1_pos0[i],columns=['A',  'C','G','T'])
    df_summed_encoded= a.apply(lambda x: np.asarray(x) * np.asarray(df_sum))
    logo=lm.Logo(df_summed_encoded)
    logo.highlight_position_range(27,33,alpha=0.3,color='lightblue',edgecolor='black')
    logo.ax.set_title('Logo of Reb1 in REAL sequences at position 0',fontsize=14)


# The same process as above but for sequences were reb1 is at position 1

real_sequences_reb1_pos1=[]
for seq in seq_uni_cut_reb1:
        if seq[1:8]=='CGGGTAA':
            real_sequences_reb1_pos1.append(seq)

list_of_shap_realseq_reb1_pos1=[]
Equal_Len(real_sequences_reb1_pos1,list_of_shap_realseq_reb1_pos1)
pass


imp_scores_realseq_reb1_pos1=[]
x=model.output
outputtarget=tf.reduce_sum(x,axis=0,keepdims=True)
black_reference_explainer=shap.TFDeepExplainer((model.input,outputtarget),black_reference_images,combine_mult_and_diffref=standard_combine_mult_and_diffref)
blackShapScores=black_reference_explainer.shap_values(np.array(list_of_shap_realseq_reb1_pos1),progress_message=100)
imp_scores_realseq_reb1_pos1.append(blackShapScores)


logo=lm.Logo(av_im_scores_random(imp_scores_realseq_reb1_pos1,list_of_shap_realseq_reb1_pos1))
logo.highlight_position_range(28,34,alpha=0.3,color='lightblue',edgecolor='black')
logo.ax.set_title('Average Logo of Reb1 in 75 REAL sequences at position 1',fontsize=14)

for i in range(5):
    df = pd.DataFrame(imp_scores_realseq_reb1_pos1[0][0][i], columns=['A',  'C','G','T'])
    df_sum=df.sum(axis=1)
    a=pd.DataFrame(list_of_shap_realseq_reb1_pos1[i],columns=['A',  'C','G','T'])
    df_summed_encoded= a.apply(lambda x: np.asarray(x) * np.asarray(df_sum))
    logo=lm.Logo(df_summed_encoded)
    logo.highlight_position_range(28,34,alpha=0.3,color='lightblue',edgecolor='black')
    logo.ax.set_title('Logo of Reb1 in REAL sequences at position 1',fontsize=14)


# EXPERIMENT RAP1 POSITION VS EXPRESSION

# The same experiment, method and functions but for rap1 motif

means_exp_rap1=[]
for i in range(80):
    ran_sequences_rap1=[]
    for seq in ran_sequences:
        if i==0:
            final_string='TGTATGGGTG' + seq
            ran_sequences_rap1.append(final_string)
        elif i==80:
            final_string=seq+'TGTATGGGTG'
            ran_sequences_rap1.append(final_string)
        else:
            final_string=seq[:i]+'TGTATGGGTG' + seq[i:]
            ran_sequences_rap1.append(final_string)

            
    test_batches=DNAseqload(ran_sequences_rap1)
    means_exp_rap1.append(np.mean(model.predict(test_batches,verbose=2)))


plt.plot(means_exp_rap1)
plt.ylabel('Expression level',fontsize=14)
plt.xlabel('Position of Rap1',fontsize=14)
plt.title('Expression levels as a function of Rap1 position in 1000 RANDOM sequences',fontsize=14)
plt.show()


i=33
ran_sequences_reb1_pos33=[]
for seq in ran_sequences:
        final_string=seq[:i]+'CGGGTAA' + seq[i:]
        ran_sequences_reb1_pos33.append(final_string)

ran_sequences_rap1_pos0=[]
for seq in ran_sequences:
        final_string='TGTATGGGTG' + seq
        ran_sequences_rap1_pos0.append(final_string)


list_of_shap_seq_rap1_pos0=[]
Equal_Len(ran_sequences_rap1_pos0,list_of_shap_seq_rap1_pos0)
pass



imp_scores_rap1_pos0=[]
x=model.output
outputtarget=tf.reduce_sum(x,axis=0,keepdims=True)
black_reference_explainer=shap.TFDeepExplainer((model.input,outputtarget),black_reference_images,combine_mult_and_diffref=standard_combine_mult_and_diffref)
blackShapScores=black_reference_explainer.shap_values(np.array(list_of_shap_seq_rap1_pos0),progress_message=100)
imp_scores_rap1_pos0.append(blackShapScores)



logo=lm.Logo(av_im_scores_random(imp_scores_rap1_pos0,list_of_shap_seq_rap1_pos0))
logo.highlight_position_range(22,31,alpha=0.3,color='lightblue',edgecolor='black')
logo.ax.set_title('Average Logo of Rap1 in 1000 random sequences at position 0',fontsize=14)


for i in range(5):
    df = pd.DataFrame(imp_scores_rap1_pos0[0][0][i], columns=['A',  'C','G','T'])
    df_sum=df.sum(axis=1)
    a=pd.DataFrame(list_of_shap_seq_rap1_pos0[i],columns=['A',  'C','G','T'])
    df_summed_encoded= a.apply(lambda x: np.asarray(x) * np.asarray(df_sum))
    logo=lm.Logo(df_summed_encoded)
    logo.highlight_position_range(22,31,alpha=0.3,color='lightblue',edgecolor='black')
    logo.ax.set_title('Logo of Rap1 in random sequences at position 0',fontsize=14)

ran_sequences_rap1_pos75=[]
for seq in ran_sequences:
        final_string=seq[:75]+'TGTATGGGTG' + seq[75:]
        ran_sequences_rap1_pos75.append(final_string)

list_of_shap_seq_rap1_pos75=[]
Equal_Len(ran_sequences_rap1_pos75,list_of_shap_seq_rap1_pos75)
pass


imp_scores_rap1_pos75=[]
x=model.output
outputtarget=tf.reduce_sum(x,axis=0,keepdims=True)
black_reference_explainer=shap.TFDeepExplainer((model.input,outputtarget),black_reference_images,combine_mult_and_diffref=standard_combine_mult_and_diffref)
blackShapScores=black_reference_explainer.shap_values(np.array(list_of_shap_seq_rap1_pos75),progress_message=100)
imp_scores_rap1_pos75.append(blackShapScores)


logo=lm.Logo(av_im_scores_random(imp_scores_rap1_pos75,list_of_shap_seq_rap1_pos75))
logo.highlight_position_range(97,106,alpha=0.3,color='lightblue',edgecolor='black')
logo.ax.set_title('Average Logo of Rap1 in 1000 random sequences at position 75',fontsize=14)



for i in range(5):
    df = pd.DataFrame(imp_scores_rap1_pos75[0][0][i], columns=['A',  'C','G','T'])
    df_sum=df.sum(axis=1)
    a=pd.DataFrame(list_of_shap_seq_rap1_pos75[i],columns=['A',  'C','G','T'])
    df_summed_encoded= a.apply(lambda x: np.asarray(x) * np.asarray(df_sum))
    logo=lm.Logo(df_summed_encoded)
    logo.highlight_position_range(97,106,alpha=0.3,color='lightblue',edgecolor='black')
    logo.ax.set_title('Logo of Rap1 in random sequences at position 75',fontsize=14)




seq_uni_cut_rap1=[]
scores_uni_cut_rap1=[]
for idx,seq in enumerate(seq_uni_cut):
    if 'TGTATGGGTG' in seq:
        seq_uni_cut_rap1.append(seq)
        scores_uni_cut_rap1.append(score_uni_cut[idx])


# In[ ]:


mean_scores_rap1=[]
seq_count_rap1=[]
for i in range(80):
    scores_rap1=[]
    for idx,seq in tqdm.tqdm(enumerate(seq_uni_cut_rap1)):
        if seq[i:i+10]=='TGTATGGGTG':
            scores_rap1.append(scores_uni_cut_rap1[idx])
    mean_scores_rap1.append(np.mean(scores_rap1))
    seq_count_rap1.append(len(scores_rap1))


# In[287]:


plt.plot(mean_scores_rap1)
plt.ylabel('Expression level')
plt.xlabel('Position of Rap1')
plt.title('Expression levels as a function of Rap1 position in REAL sequences')
plt.show()


# In[387]:


real_sequences_rap1_pos0=[]
for seq in seq_uni_cut_rap1:
        if seq[:10]=='TGTATGGGTG':
            real_sequences_rap1_pos0.append(seq)


# In[ ]:


list_of_shap_realseq_rap1_pos0=[]
Equal_Len(real_sequences_rap1_pos0,list_of_shap_realseq_rap1_pos0)
pass


# In[571]:


imp_scores_realseq_rap1_pos0=[]
x=model.output
outputtarget=tf.reduce_sum(x,axis=0,keepdims=True)
black_reference_explainer=shap.TFDeepExplainer((model.input,outputtarget),black_reference_images,combine_mult_and_diffref=standard_combine_mult_and_diffref)
blackShapScores=black_reference_explainer.shap_values(np.array(list_of_shap_realseq_rap1_pos0),progress_message=100)
imp_scores_realseq_rap1_pos0.append(blackShapScores)


# In[643]:


logo=lm.Logo(av_im_scores_random(imp_scores_realseq_rap1_pos0,list_of_shap_realseq_rap1_pos0))
logo.highlight_position_range(27,36,alpha=0.3,color='lightblue',edgecolor='black')
logo.ax.set_title('Average Logo of Rap1 in 3 Real sequences at position 0',fontsize=14)


# In[392]:


df = pd.DataFrame(imp_scores_realseq_rap1_pos0[0][0][i], columns=['A',  'C','G','T'])
df_sum=df.sum(axis=1)
a=pd.DataFrame(list_of_shap_realseq_rap1_pos0[i],columns=['A',  'C','G','T'])
df_summed_encoded= a.apply(lambda x: np.asarray(x) * np.asarray(df_sum))
logo=lm.Logo(df_summed_encoded)
logo.highlight_position_range(27,36,alpha=0.3,color='lightblue',edgecolor='black')
logo.ax.set_title('Logo of Rap1 in REAL sequences at position 0',fontsize=14)


# In[393]:


real_sequences_rap1_pos1=[]
for seq in seq_uni_cut_rap1:
        if seq[1:11]=='TGTATGGGTG':
            real_sequences_rap1_pos1.append(seq)


# In[394]:


list_of_shap_realseq_rap1_pos1=[]
Equal_Len(real_sequences_rap1_pos1,list_of_shap_realseq_rap1_pos1)
pass


# In[395]:


imp_scores_realseq_rap1_pos1=[]
x=model.output
outputtarget=tf.reduce_sum(x,axis=0,keepdims=True)
black_reference_explainer=shap.TFDeepExplainer((model.input,outputtarget),black_reference_images,combine_mult_and_diffref=standard_combine_mult_and_diffref)
blackShapScores=black_reference_explainer.shap_values(np.array(list_of_shap_realseq_rap1_pos1),progress_message=100)
imp_scores_realseq_rap1_pos1.append(blackShapScores)


# In[646]:


for i in range(1):
    df = pd.DataFrame(imp_scores_realseq_rap1_pos1[0][0][i], columns=['A',  'C','G','T'])
    df_sum=df.sum(axis=1)
    a=pd.DataFrame(list_of_shap_realseq_rap1_pos1[i],columns=['A',  'C','G','T'])
    df_summed_encoded= a.apply(lambda x: np.asarray(x) * np.asarray(df_sum))
    logo=lm.Logo(df_summed_encoded)
    logo.highlight_position_range(28,37,alpha=0.3,color='lightblue',edgecolor='black')
    logo.ax.set_title('Logo of Rap1 in 1 REAL sequence at position 1',fontsize=14)


# EXPERIMENT ABF1 POSITION VS EXPRESSION

# the same code as above, but for abf1

# In[ ]:


means_exp_abf1=[]
for i in range(74):
    ran_sequences_abf1=[]
    for seq in ran_sequences:
        if i==0:
            final_string='ATCAC' + seq[:5]+'ACG'+seq[5:]
            ran_sequences_abf1.append(final_string)
        elif i==75:
            final_string=seq[:i]+'ATCAC'+seq[i:]+'ACG'
            ran_sequences_abf1.append(final_string)
        else:
            final_string=seq[:i]+'ATCAC' + seq[i:i+5]+'ACG'+seq[i+5:]
            ran_sequences_abf1.append(final_string)

            
    test_batches=DNAseqload(ran_sequences_abf1)
    means_exp_abf1.append(np.mean(model.predict(test_batches,verbose=2)))


# In[661]:


plt.plot(means_exp_abf1)
plt.ylabel('Expression level')
plt.xlabel('Position of ABF1')
plt.title('Expression levels as a function of ABF1 position in 1000 RANDOM sequences')
plt.show()


# In[427]:


ran_sequences_abf1_pos0=[]
for seq in ran_sequences:
        final_string='ATCAC' + seq[:5]+'ACG'+seq[5:]
        ran_sequences_abf1_pos0.append(final_string)


# In[ ]:


ran_sequences_abf1_pos80=[]
for seq in ran_sequences:
        final_string=seq[:75]+'ATCAC'+seq[75:]+'ACG'
        ran_sequences_abf1_pos80.append(final_string)


# In[ ]:


list_of_shap_seq_abf1_pos0=[]
Equal_Len(ran_sequences_abf1_pos0,list_of_shap_seq_abf1_pos0)
pass


# In[649]:


list_of_shap_seq_abf1_pos80=[]
Equal_Len(ran_sequences_abf1_pos80,list_of_shap_seq_abf1_pos80)
pass


# In[437]:


imp_scores_abf1_pos0=[]
x=model.output
outputtarget=tf.reduce_sum(x,axis=0,keepdims=True)
black_reference_explainer=shap.TFDeepExplainer((model.input,outputtarget),black_reference_images,combine_mult_and_diffref=standard_combine_mult_and_diffref)
blackShapScores=black_reference_explainer.shap_values(np.array(list_of_shap_seq_abf1_pos0),progress_message=100)
imp_scores_abf1_pos0.append(blackShapScores)


# In[650]:


imp_scores_abf1_pos80=[]
x=model.output
outputtarget=tf.reduce_sum(x,axis=0,keepdims=True)
black_reference_explainer=shap.TFDeepExplainer((model.input,outputtarget),black_reference_images,combine_mult_and_diffref=standard_combine_mult_and_diffref)
blackShapScores=black_reference_explainer.shap_values(np.array(list_of_shap_seq_abf1_pos80),progress_message=100)
imp_scores_abf1_pos80.append(blackShapScores)


# In[648]:


logo=lm.Logo(av_im_scores_random(imp_scores_abf1_pos0,list_of_shap_seq_abf1_pos0))
logo.highlight_position_range(23,35,alpha=0.3,color='lightblue',edgecolor='black')
logo.ax.set_title('Average Logo of Abf1 in 1000 Random sequences at position 0',fontsize=14)


# In[ ]:





# In[663]:


logo=lm.Logo(av_im_scores_random(imp_scores_abf1_pos80,list_of_shap_seq_abf1_pos80))
logo.highlight_position_range(98,110,alpha=0.3,color='lightblue',edgecolor='black')
logo.ax.set_title('Average Logo of Abf1 in 1000 Random sequences at position 80',fontsize=14)


# In[439]:


for i in range(5):
    df = pd.DataFrame(imp_scores_abf1_pos0[0][0][i], columns=['A',  'C','G','T'])
    df_sum=df.sum(axis=1)
    a=pd.DataFrame(list_of_shap_seq_abf1_pos0[i],columns=['A',  'C','G','T'])
    df_summed_encoded= a.apply(lambda x: np.asarray(x) * np.asarray(df_sum))
    logo=lm.Logo(df_summed_encoded)
    logo.highlight_position_range(23,35,alpha=0.3,color='lightblue',edgecolor='black')
    logo.ax.set_title('Logo of ABF1 in random sequences at position 0',fontsize=14)


# In[447]:


seq_uni_cut_abf1=[]
scores_uni_cut_abf1=[]
for idx,seq in enumerate(seq_uni_cut):
    if re.search('[AGC]TC[AG][CT].....ACG',seq)!=None:
        seq_uni_cut_abf1.append(seq)
        scores_uni_cut_abf1.append(score_uni_cut[idx])


# In[ ]:


mean_scores_abf1=[]
seq_count_abf1=[]
for i in range(67):
    scores_abf1=[]
    for idx,seq in tqdm.tqdm(enumerate(seq_uni_cut_abf1)):
        if seq[i:i+5]=='ATCAC' and seq[i+10:i+13]=='ACG':
            scores_abf1.append(scores_uni_cut_abf1[idx])
    mean_scores_abf1.append(np.mean(scores_abf1))
    seq_count_abf1.append(len(scores_abf1))


# In[452]:


plt.plot(mean_scores_abf1)
plt.ylabel('Expression level')
plt.xlabel('Position of ABF1')
plt.title('Expression levels as a function of ABF1 position in REAL sequences')
plt.show()


# In[453]:


real_sequences_abf1_pos0=[]
for seq in seq_uni_cut_abf1:
        if seq[:5]=='ATCAC' and seq[10:13]=='ACG':
            real_sequences_abf1_pos0.append(seq)


# In[454]:


list_of_shap_realseq_abf1_pos0=[]
Equal_Len(real_sequences_abf1_pos0,list_of_shap_realseq_abf1_pos0)
pass


# In[455]:


imp_scores_realseq_abf1_pos0=[]
x=model.output
outputtarget=tf.reduce_sum(x,axis=0,keepdims=True)
black_reference_explainer=shap.TFDeepExplainer((model.input,outputtarget),black_reference_images,combine_mult_and_diffref=standard_combine_mult_and_diffref)
blackShapScores=black_reference_explainer.shap_values(np.array(list_of_shap_realseq_abf1_pos0),progress_message=100)
imp_scores_realseq_abf1_pos0.append(blackShapScores)


# In[665]:


logo=lm.Logo(av_im_scores_random(imp_scores_realseq_abf1_pos0,list_of_shap_realseq_abf1_pos0))
logo.highlight_position_range(27,39,alpha=0.3,color='lightblue',edgecolor='black')
logo.ax.set_title('Average Logo of Abf1 in 9 Real sequences at position 0',fontsize=14)


# In[457]:


for i in range(5):
    df = pd.DataFrame(imp_scores_realseq_abf1_pos0[0][0][i], columns=['A',  'C','G','T'])
    df_sum=df.sum(axis=1)
    a=pd.DataFrame(list_of_shap_realseq_abf1_pos0[i],columns=['A',  'C','G','T'])
    df_summed_encoded= a.apply(lambda x: np.asarray(x) * np.asarray(df_sum))
    logo=lm.Logo(df_summed_encoded)
    logo.highlight_position_range(27,39,alpha=0.3,color='lightblue',edgecolor='black')
    logo.ax.set_title('Logo of ABF1 in REAL sequences at position 0',fontsize=14)


# In[405]:


ran_sequences_rap1_pos0_reb1_pos25=[]
for seq in ran_sequences:
        final_string='TGTATGGGTG' + seq[:25]+'CGGGTAA'+seq[25:]
        ran_sequences_rap1_pos0_reb1_pos25.append(final_string)


# In[407]:


list_of_shap_seq_rap1_pos0_reb1_pos25=[]
Equal_Len(ran_sequences_rap1_pos0_reb1_pos25,list_of_shap_seq_rap1_pos0_reb1_pos25)
pass


# In[408]:


imp_scores_rap1_pos0_reb1_pos25=[]
x=model.output
outputtarget=tf.reduce_sum(x,axis=0,keepdims=True)
black_reference_explainer=shap.TFDeepExplainer((model.input,outputtarget),black_reference_images,combine_mult_and_diffref=standard_combine_mult_and_diffref)
blackShapScores=black_reference_explainer.shap_values(np.array(list_of_shap_seq_rap1_pos0_reb1_pos25),progress_message=100)
imp_scores_rap1_pos0_reb1_pos25.append(blackShapScores)


# In[410]:


for i in range(5):
    df = pd.DataFrame(imp_scores_rap1_pos0_reb1_pos25[0][0][i], columns=['A',  'C','G','T'])
    df_sum=df.sum(axis=1)
    a=pd.DataFrame(list_of_shap_seq_rap1_pos0_reb1_pos25[i],columns=['A',  'C','G','T'])
    df_summed_encoded= a.apply(lambda x: np.asarray(x) * np.asarray(df_sum))
    logo=lm.Logo(df_summed_encoded)
    logo.highlight_position_range(19,28,alpha=0.3,color='lightblue',edgecolor='black')
    logo.highlight_position_range(54,60,alpha=0.3,color='lightblue',edgecolor='black')
    logo.ax.set_title('Logo of Rap1-Reb1 pair in random sequences at position 0 and 25 respectively',fontsize=14)


# A couple of logos when there are 2 motifs in a sequence (sequences created below)

# In[412]:


ran_sequences_reb1_pos0_rap1_pos0=[]
for seq in ran_sequences:
        final_string='CGGGTAA'+'TGTATGGGTG'+seq
        ran_sequences_reb1_pos0_rap1_pos0.append(final_string)


# In[413]:


list_of_shap_seq_reb1_pos0_rap1_pos0=[]
Equal_Len(ran_sequences_reb1_pos0_rap1_pos0,list_of_shap_seq_reb1_pos0_rap1_pos0)
pass


# In[414]:


imp_scores_reb1_pos0_rap1_pos0=[]
x=model.output
outputtarget=tf.reduce_sum(x,axis=0,keepdims=True)
black_reference_explainer=shap.TFDeepExplainer((model.input,outputtarget),black_reference_images,combine_mult_and_diffref=standard_combine_mult_and_diffref)
blackShapScores=black_reference_explainer.shap_values(np.array(list_of_shap_seq_reb1_pos0_rap1_pos0),progress_message=100)
imp_scores_reb1_pos0_rap1_pos0.append(blackShapScores)


# In[458]:


for i in range(5):
    df = pd.DataFrame(imp_scores_reb1_pos0_rap1_pos0[0][0][i], columns=['A',  'C','G','T'])
    df_sum=df.sum(axis=1)
    a=pd.DataFrame(list_of_shap_seq_reb1_pos0_rap1_pos0[i],columns=['A',  'C','G','T'])
    df_summed_encoded= a.apply(lambda x: np.asarray(x) * np.asarray(df_sum))
    logo=lm.Logo(df_summed_encoded)
    logo.highlight_position_range(19,28,alpha=0.3,color='lightblue',edgecolor='black')
    logo.highlight_position_range(28,34,alpha=0.3,color='lightblue',edgecolor='black')
    logo.ax.set_title('Logo of Reb1-Rap1 pair in random sequences at position 0 and 0 respectively',fontsize=14)


# EXPERIMENT (keep reb1 and rap1, reb1 solo, rap1 solo)

# In[ ]:


#position 2 motifs and individual motifs in the same sequences and find the average predicted expression for each case
expr_2motifs=[]
expr_reb1motif=[]
expr_rap1motif=[]
for i in range(20,45):
    seq_2motifs=[]
    seq_reb1_motif=[]
    seq_rap1_motif=[]
    for seq in ran_sequences:
        seq_2motifs.append(seq[:i]+'CGGGTAA'+seq[i:i+15]+'TGTATGGGTG'+seq[i+15:])
        seq_reb1_motif.append(seq[:i]+'CGGGTAA'+seq[i:])
        seq_rap1_motif.append(seq[:i+15]+'TGTATGGGTG'+seq[i+15:])
    
    motifs2_batches=DNAseqload(seq_2motifs)
    expr_2motifs.append(np.mean(model.predict(motifs2_batches,verbose=2)))

    motif_reb1_batches=DNAseqload(seq_reb1_motif)
    expr_reb1motif.append(np.mean(model.predict(motif_reb1_batches,verbose=2)))

    motif_rap1_batches=DNAseqload(seq_rap1_motif)
    expr_rap1motif.append(np.mean(model.predict(motif_rap1_batches,verbose=2)))

expr_2motifs_av=np.mean(expr_2motifs)
expr_reb1motif_av=np.mean(expr_reb1motif)
expr_rap1motif_av=np.mean(expr_rap1motif)


test_batches=DNAseqload(ran_sequences)
seq_no_motif=np.mean(model.predict(test_batches,verbose=2))



# EXPERIMENT POSITION OF MOTIF PAIR UP TO 20 BASES APART ACROSS SEQUENCES (15 TO 65)

# similar code to the above, just place motifs in the positions we want at a given spacing, generate model predictions, and average them, then plot

# In[ ]:


means_exp_reb1_20_rap1=[]
for i in range(15,45):
    ran_sequences_reb1_20_rap1=[]
    for seq in ran_sequences:
            final_string=seq[:i]+'TGTATGGGTG' + seq[i:i+20]+'CGGGTAA'+seq[i+20:]
            ran_sequences_reb1_20_rap1.append(final_string)

            
    test_batches=DNAseqload(ran_sequences_reb1_20_rap1)
    means_exp_reb1_20_rap1.append(np.mean(model.predict(test_batches,verbose=2)))


# In[ ]:


means_exp_reb1_20_rap1=[]
means_exp_rap1_=[]
means_exp_reb1_=[]
for i in range(15,45):
    ran_sequences_reb1_20_rap1=[]
    ran_sequences_rap1_=[]
    ran_sequences_reb1_=[]
    for seq in ran_sequences:
            final_string=seq[:i]+'TGTATGGGTG' + seq[i:i+20]+'CGGGTAA'+seq[i+20:]
            ran_sequences_reb1_20_rap1.append(final_string)
            
            final_stringrap1=seq[:i]+'TGTATGGGTG' + seq[i:]
            ran_sequences_rap1_.append(final_stringrap1)
            
            final_stringreb1=seq[:i+20]+'CGGGTAA'+seq[i+20:]
            ran_sequences_reb1_.append(final_stringreb1)
            
            
    test_batches=DNAseqload(ran_sequences_reb1_20_rap1)
    means_exp_reb1_20_rap1.append(np.mean(model.predict(test_batches,verbose=2)))
    
    test_batchesrap1=DNAseqload(ran_sequences_rap1_)
    means_exp_rap1_.append(np.mean(model.predict(test_batchesrap1,verbose=2)))
    
    test_batchesreb1=DNAseqload(ran_sequences_reb1_)
    means_exp_reb1_.append(np.mean(model.predict(test_batchesreb1,verbose=2)))


# In[746]:


plt.plot(l,means_exp_reb1_20_rap1)
plt.plot(l,means_exp_rap1_)
plt.plot(l,means_exp_reb1_)
plt.legend(['Both', 'Rap1','Reb1'], loc='upper right')
plt.ylabel('Expression level',fontsize=14)
plt.xlabel('Position of Rap1-Reb1 pair',fontsize=14)
plt.title('Expression levels as a function of Rap1-Reb1 pair (20b apart) and individual motifs position in RANDOM sequences',fontsize=14)
plt.show()


# In[367]:


plt.plot(means_exp_reb1_20_rap1)
plt.ylabel('Expression level',fontsize=14)
plt.xlabel('Position of Rap1-Reb1 pair',fontsize=14)
plt.title('Expression levels as a function of Rap1-Reb1 pair (20b apart) position in RANDOM sequences',fontsize=14)
plt.show()


# In[ ]:


means_exp_reb1_10_rap1=[]
for i in range(15,55):
    ran_sequences_reb1_10_rap1=[]
    for seq in ran_sequences:
            final_string=seq[:i]+'TGTATGGGTG' + seq[i:i+10]+'CGGGTAA'+seq[i+10:]
            ran_sequences_reb1_10_rap1.append(final_string)

            
    test_batches=DNAseqload(ran_sequences_reb1_10_rap1)
    means_exp_reb1_10_rap1.append(np.mean(model.predict(test_batches,verbose=2)))


# In[ ]:


means_exp_reb1_10_rap1=[]
means_exp_rap1_10=[]
means_exp_reb1_10=[]
for i in range(15,55):
    ran_sequences_reb1_10_rap1=[]
    ran_sequences_rap1_=[]
    ran_sequences_reb1_=[]
    for seq in ran_sequences:
            final_string=seq[:i]+'TGTATGGGTG' + seq[i:i+10]+'CGGGTAA'+seq[i+10:]
            ran_sequences_reb1_10_rap1.append(final_string)
            
            final_stringrap1=seq[:i]+'TGTATGGGTG' + seq[i:]
            ran_sequences_rap1_.append(final_stringrap1)
            
            final_stringreb1=seq[:i+10]+'CGGGTAA'+seq[i+10:]
            ran_sequences_reb1_.append(final_stringreb1)
            
            
    test_batches=DNAseqload(ran_sequences_reb1_10_rap1)
    means_exp_reb1_10_rap1.append(np.mean(model.predict(test_batches,verbose=2)))
    
    test_batchesrap1=DNAseqload(ran_sequences_rap1_)
    means_exp_rap1_10.append(np.mean(model.predict(test_batchesrap1,verbose=2)))
    
    test_batchesreb1=DNAseqload(ran_sequences_reb1_)
    means_exp_reb1_10.append(np.mean(model.predict(test_batchesreb1,verbose=2)))


# In[752]:


plt.plot(l,means_exp_reb1_10_rap1)
plt.plot(l,means_exp_rap1_10)
plt.plot(l,means_exp_reb1_10)
plt.legend(['Both', 'Rap1','Reb1'], loc='upper right')
plt.ylabel('Expression level',fontsize=14)
plt.xlabel('Position of Rap1-Reb1 pair',fontsize=14)
plt.title('Expression levels as a function of Rap1-Reb1 pair (10b apart) and individual motifs position in RANDOM sequences',fontsize=14)
plt.show()


# In[687]:


plt.plot(l,means_exp_reb1_10_rap1)
plt.ylabel('Expression level')
plt.xlabel('Position of Rap1-Reb1 pair')
plt.title('Expression levels as a function of Rap1-Reb1 (10b apart) pair position in RANDOM sequences')
plt.show()


# In[751]:


#a list with correct x labels to put on graphs
l=[]
for i in range(15,55):
    l.append(i)


# SPACING EXPERIMENT

# In[ ]:


#code to place 2 motifs in all possible positions relative to each other and the sequence, find predictions, avergae them , plot into the matrix
means_spacing_nested=[]
means_idx_nested=[]
reb1='CGGGTAA'
rap1='TGTATGGGTG'
for i in tqdm.tqdm(range(81)):
    means_spacing=[]
    means_idx=[]
    for k in tqdm.tqdm(range(81)):
        reb1_rap1_seqs=[]
        for seq in ran_sequences:
            #reb1 on the left
            if k<i:  
                if k==0:
                    final_string=reb1+seq[:i]+rap1+seq[i:]
                    reb1_rap1_seqs.append(final_string)
                else:
                    final_string=seq[:k]+reb1+seq[k:i]+rap1+seq[i:]
                    reb1_rap1_seqs.append(final_string)
                    
            #rap1 on the left
            if k>i:
                if i==0:
                    final_string=rap1+seq[:k]+reb1+seq[k:]
                    reb1_rap1_seqs.append(final_string)
                else:
                    final_string=seq[:i]+rap1+seq[i:k]+reb1+seq[k:]
                    reb1_rap1_seqs.append(final_string)
                    
            if k==i:
                #reb1 on the left
                final_string=reb1+rap1+seq
                reb1_rap1_seqs.append(final_string)
                

                

        means_spacing.append(reb1_rap1_seqs)
        means_idx.append(k)
        
    means_spacing_nested.append(means_spacing)
    means_idx_nested.append(means_idx)


# In[ ]:


#predictions part
means_nested=[]
for j in range(len(means_spacing_nested)):
    means=[]
    for i in range(len(means_spacing_nested[j])):
        test_batches=DNAseqload(means_spacing_nested[j][i])
        means.append(np.mean(model.predict(test_batches,verbose=0)))
    means_nested.append(means)


# In[774]:


#df_motif_spacing= pd.DataFrame(means_nested)
plt.figure(figsize = (15,15))
plt.ylabel('Rap1')
plt.xlabel('Reb1')
sn.set()
s=sn.heatmap(df_motif_spacing)
s.set_xlabel('Reb1 motif position in 1000 random sequences', fontsize = 17)
s.set_ylabel('Rap1 motif position in 1000 random sequences', fontsize = 17)
s.set_title('Average association between Reb1 and Rap1 spacing and expression level across 1000 random sequences', fontsize = 17)



# The same spacing experiment, but for the different motifs. Also, spacing method is different. Here, I take the middle of the motif as index and substitute it into the sequences, keeping the lenght constant, instead of inserting motifd in the sequence and elongating them like in the experiment above

# In[865]:


Ace2='CCAGC'
Fkh1='GTAAACA'


# In[ ]:


means_spacing_nested2=[]
Ace2='CCAGC'
Fkh1='GTAAACA'
#fkh1
for i in tqdm.tqdm(range(4,77)):
    means_spacing2=[]

    #ace2
    for k in tqdm.tqdm(range(3,78)):
        ace2_fkh1_seqs=[]
        for seq in ran_sequences:
            #ace2 on the left
            if k<i and (i-k)>=6:  
                    final_string=seq[:k-3]+Ace2+seq[k+2:i-4]+Fkh1+seq[i+3:]
                    ace2_fkh1_seqs.append(final_string)
                    
            #fkh1 on the left
            if k>i and (k-i)>=6:
                    final_string=seq[:i-4]+Fkh1+seq[i+3:k-3]+Ace2+seq[k+2:]
                    ace2_fkh1_seqs.append(final_string)

         

        means_spacing2.append(ace2_fkh1_seqs)
 
        
    means_spacing_nested2.append(means_spacing2)



# In[993]:


#ace2 labels for the datafram
ace2_labels=[]
for k in range(3,78):
    ace2_labels.append(k)


# In[994]:


#fkh1 labels for the dataframe
fkh1_labels=[]
for k in range(4,77):
    fkh1_labels.append(k)


# In[ ]:


#predictions part of the experiment (just like in the first spacing experiment this part is separated friom the sequence generating part above because together computer crushes)
means_nested2=[]
for j in range(len(means_spacing_nested2)):
    means=[]
    for i in range(len(means_spacing_nested2[j])):
        if len(means_spacing_nested2[j][i])==0:
            means.append(1)
        else:
            test_batches=DNAseqload(means_spacing_nested2[j][i])
            means.append(np.mean(model.predict(test_batches,verbose=0)))
    means_nested2.append(means)


# In[1013]:


df = pd.DataFrame(means_nested2)


# In[1018]:


df.columns =ace2_labels
  
df.index =fkh1_labels


df=df.replace({1: np.nan})



s=sn.heatmap(df)
s.set_xlabel('Ace2 motif position in 1000 random sequences', fontsize = 17)
s.set_ylabel('Fkh1 motif position in 1000 random sequences', fontsize = 17)




#generate sequences and find predictions  for individual motifs and plot its positioning against expression
means_exp_ace2=[]
for i in range(3,78):
    ran_sequences_ace2=[]
    for seq in ran_sequences:
            final_string=seq[:i-3]+Ace2+seq[i+2:]
            ran_sequences_ace2.append(final_string)

            
    test_batches=DNAseqload(ran_sequences_ace2)
    means_exp_ace2.append(np.mean(model.predict(test_batches,verbose=2)))



# In[1027]:


plt.plot(ace2_labels,means_exp_ace2)
plt.ylabel('Expression level')
plt.xlabel('Position of ACE2')
plt.title('Expression levels as a function of ACE2 position in 1000 RANDOM sequences')
plt.show()


# In[ ]:


#the same thing for this motif
means_exp_fkh1=[]
for i in range(4,77):
    ran_sequences_fkh1=[]
    for seq in ran_sequences:
            final_string=seq[:i-4]+Fkh1+seq[i+3:]
            ran_sequences_fkh1.append(final_string)

            
    test_batches=DNAseqload(ran_sequences_fkh1)
    means_exp_fkh1.append(np.mean(model.predict(test_batches,verbose=2)))


# In[1029]:


plt.plot(fkh1_labels,means_exp_fkh1)
plt.ylabel('Expression level')
plt.xlabel('Position of FKH1')
plt.title('Expression levels as a function of FKH1 position in 1000 RANDOM sequences')
plt.show()


# In[556]:


#fill diagonal with NaN, if needed
np.fill_diagonal(df_motif_spacing.values, None)



#find sequences and corresponding scores (this time, predicted by the model) out of original train sequences with rap1 motif and without 
key2='TGTACGGGTG'
rap1_motif=[]
no_rap1_motif=[]
rap1_mot_scores=[]
no_rap1_mot_scores=[]
pos=0

for i in seq_uni:
    if key2 in i:
        rap1_motif.append(i)
        rap1_mot_scores.append(pred[pos])
        pos+=1
    else:
        no_rap1_motif.append(i)
        no_rap1_mot_scores.append(pred[pos])
        pos+=1


# In[736]:


#the same for reb1 motif
reb1_motif=[]
no_reb1_motif=[]
reb1_mot_scores=[]
no_reb1_mot_scores=[]
pos=0
for i in seq_uni:
    if reb1 in i:
        reb1_motif.append(i)
        reb1_mot_scores.append(pred[pos])
        pos+=1
    else:
        no_reb1_motif.append(i)
        no_reb1_mot_scores.append(pred[pos])
        pos+=1


# In[709]:


#predict expression for train sequences 
batches=DNAseqload(seq_uni)
pred=model.predict(batches)


# In[ ]:


pred.tolist()


# In[741]:


#plot the predicted expression for reb1 and no reb1 sequences, to compare with the same boxplot we did before but with real expressions to see if the model learned correct
from scipy.stats import ttest_ind
box_plot_data=[rap1_mot_scores_list,no_rap1_mot_scores_list]
plt.boxplot(box_plot_data,patch_artist=True,labels=['Rap1 Motif','No Rap1 Motif'])
plt.title('Comparison of predicted expression levels between sequences with Rap1 motif and without')
plt.ylabel('Expression level')
plt.show()
res = ttest_ind(rap1_mot_scores_list,no_rap1_mot_scores_list).pvalue

print('P-value=' ,res)


# In[742]:


#same as above but different motif
box_plot_data=[reb1_mot_scores_list,no_reb1_mot_scores_list]
plt.boxplot(box_plot_data,patch_artist=True,labels=['Reb1 Motif','No Reb1 Motif'])
plt.title('Comparison of predicted expression levels between sequences with Reb1 motif and without')
plt.ylabel('Expression level')
plt.show()
res = ttest_ind(reb1_mot_scores_list,no_reb1_mot_scores_list).pvalue

print('P-value=' ,res)


# In[728]:


#bring the lists to the right format 
rap1_mot_scores_list=[]
for i in range(len(rap1_mot_scores)):
     rap1_mot_scores_list.append(rap1_mot_scores[i][0])


# In[730]:


no_rap1_mot_scores_list=[]
for i in range(len(no_rap1_mot_scores)):
     no_rap1_mot_scores_list.append(no_rap1_mot_scores[i][0])


# In[737]:


no_reb1_mot_scores_list=[]
for i in range(len(no_reb1_mot_scores)):
     no_reb1_mot_scores_list.append(no_reb1_mot_scores[i][0])


# In[739]:


reb1_mot_scores_list=[]
for i in range(len(reb1_mot_scores)):
     reb1_mot_scores_list.append(reb1_mot_scores[i][0])

