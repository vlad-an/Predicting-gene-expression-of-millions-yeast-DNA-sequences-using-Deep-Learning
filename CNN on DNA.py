#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import necessary libraries
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import shap
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from keras.utils import np_utils
from tensorflow import keras
import math

from sklearn.metrics import r2_score 

 


# In[224]:


#load uniformed and shuffled data (uniformed by selecting 30000 seqs for each bin)
data_uni= np.loadtxt("uniformed_shuffled.txt", dtype=str)


# In[225]:


#separate data into lists of sequences and corresponding scores
#scores are scaled
seq_uni=[]
score_uni=[]
for i in range(len(data_uni)):
    seq_uni.append(data_uni[i][0])
    score_uni.append(float(data_uni[i][1]) / 18)


# In[ ]:


#distribution of data
plt.hist(score_uni,bins=20)


# In[ ]:


#loading of test data
data_test= np.loadtxt("test_sequences.txt", dtype=str)


# In[ ]:


#creating a list of test sequences (unnecessary i think)
seq_test=[]
for i in range(len(data_test)):
    seq_test.append(data_test[i][0])


# In[138]:


#left and right vector sequences
right_DNA_str='TCTTAATTAAAAAAAGATAGAAAACATTAGGAGT'

left_DNA_str='CTAGCAGGAATGATGCAAAAGGTTCCCGATTCGAAC'


# In[275]:


#data generator class to load the data for the model, also bring it to the same 135 bp length and one hot encode
class DNAseqload(keras.utils.Sequence):
    def __init__(self, seqs,score, batch_size=100):
        self.seq_arr=np.zeros((len(seqs),135,4))
        for i,seq in enumerate(tqdm.tqdm(seqs)):
            if len(seq) <=135:
                diff=135-len(seq)
                if diff==1:
                    seq_resize=seq+right_DNA_str[0]
                    seq_encoded=one_hot_encode(seq_resize)
                    try:
                        self.seq_arr[i,:,:]=seq_encoded
                    except ValueError:
                        print(diff)
                        print(len(seq))
                        print(i)
                        raise
                elif diff ==0:
                    seq_encoded=one_hot_encode(seq)
                    self.seq_arr[i,:,:]=seq_encoded 
                elif (diff % 2) == 0:
                    seq_resize=left_DNA_str[int(-(diff/2)):]+seq+right_DNA_str[:int(diff/2)]
                    seq_encoded=one_hot_encode(seq_resize)
                    try:
                        self.seq_arr[i,:,:]=seq_encoded
                    except ValueError:
                        print(len(seq))
                        print(len(seq_resize))
                        print(i)
                        raise
                elif (diff % 2) != 0:
                    seq_resize=left_DNA_str[int(-(diff//2)):]+seq+right_DNA_str[:int(((diff//2)+1))]
                    seq_encoded=one_hot_encode(seq_resize)
                    self.seq_arr[i,:,:]=seq_encoded  
                    
        self.y =  np.array(score)
        self.batch_size = batch_size
 

    def __len__(self):
        return math.ceil(len(self.seq_arr) / self.batch_size)

    def __getitem__(self, idx):
        sequences = self.seq_arr[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        expression_levels = self.y[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        return sequences, expression_levels
    
    
def one_hot_encode(sequence):
    table=np.empty((len(sequence),4),dtype='int8')
    ordseq=np.fromstring(sequence,np.int8)
    table[:,0]=(ordseq==ord('A'))
    table[:,1]=(ordseq==ord('C'))
    table[:,2]=(ordseq==ord('G'))
    table[:,3]=(ordseq==ord('T'))
    return table
    

    


# In[ ]:


#sequences for model training
train_batches=DNAseqload(seq_uni[:420000],score_uni[:420000])


# In[ ]:


#sequences for model validation
val_batches=DNAseqload(seq_uni[420000:430000],score_uni[420000:430000])


# In[ ]:


#sequences for model testing (need to change the class function inputs a little bit since we don't have any scores to pass to it)
test_batches=DNAseqload(seq_test)

#decode just to check that encoding works
decoded_seq=[]
for i in data_batches[1][0]:
        for k in range(135):
            if np.array_equal(i[k],[1,0,0,0])==True:
                decoded_seq.append('A')
            if np.array_equal(i[k],[0,1,0,0])==True:
                decoded_seq.append('C')
            if np.array_equal(i[k],[0,0,1,0])==True:
                decoded_seq.append('G')
            if np.array_equal(i[k],[0,0,0,1])==True:
                decoded_seq.append('T')

print(decoded_seq)
# In[299]:


#create model architecture
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv1D(22, kernel_size=7, padding='valid',activation='relu',input_shape=(135,4)),
  tf.keras.layers.Conv1D(22, kernel_size=7, padding='valid', activation='relu',dilation_rate=8),
  tf.keras.layers.Conv1D(22, kernel_size=7, padding='valid',activation='relu',dilation_rate=10),
  tf.keras.layers.Conv1D(22, kernel_size=7,padding='valid',activation='relu'),
  tf.keras.layers.Conv1D(22, kernel_size=7,padding='valid',activation='relu'),
  tf.keras.layers.Conv1D(1, kernel_size=9,padding='valid',activation='relu'),
  tf.keras.layers.Flatten()
    
])


# In[3]:


#just a brief journal to document and test some parameters while keeping others constant
filters_list=[5,10,15 ,20,21,22,23,24,25,30]

#8 layers
#val_loss_list=[16.860803022384644  ,17.02685516357422, 17.498201208114622,17.570484657287597 ]
#train_loss_list=[11.766885192871094 , 12.791529314041139, 7.904084845066071, 7.730988000392914]


#6 layers 
#val_loss_list=[16.58041666030884  ,17.73933310508728, 19.452689352035524, 20.511082887649536]
#train_loss_list=[12.669113101005554, 8.679510335445404, 5.8911036875247955,4.505417094707489,]


#4 layers
#val_loss_list=[17.760606517791746,18.670783710479736,19.124719619750977]
#train_loss_list=[13.544923670291901,10.734375729560853,10.166399903297425]




#filters_list=[3,4,6,7]
#6 layers - 617 params
#val_loss_list=[16.538017654418944,16.317375087738036,17.174409294128417,16.66775999069214]
#train_loss_list=[14.440428113937378, 13.927452249526977,12.567588663101196, 12.966246843338013]


#kernel_size=[5,9]
#5 layers - size=9 - ~617 params
#8 layers - size=5 - ~617 params
#val_loss_list=[17.042402839660646,16.488963317871093]
#train_loss_list=[14.377278566360474,13.642703819274903]



val_loss_list=[10.195351729393005,9.939123039245606,9.855611300468444,9.803793950080872,9.794187469482422 ,
               9.616924996376037,9.869128727912903,9.779841914176941 ,9.893263940811158,10.074967136383057]

train_loss_list=[10.48224326634407,9.882239612817765,9.873206666469574,9.236203590154648, 9.47342668890953,
                 9.349031442403794 ,9.537240002393723,9.151386786937714 ,9.287998747587205, 9.147918011903762]

#plot the dependance on a certain parameter while others constant 
plt.plot(filters_list,train_loss_list)
plt.plot(filters_list,val_loss_list)
plt.ylabel('Loss')
plt.xlabel('Filters')
plt.title('Model loss as a function of the number of fi
plt.legend(['train', 'test'], loc='upper right')


# In[300]:


#summary of the model
a=model.summary()
a


# In[301]:


#compile the model, define optimizers and learning rates
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=['mse'])


# In[302]:


#set a callback to automatically adjust learning rate
rlronp=tf.keras.callbacks.ReduceLROnPlateau( monitor="val_loss", factor=0.5,
           patience=3, verbose=1)


# In[303]:


#set a callback to stop training if the model is not improving for a long time
early_stop=tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=6,
    verbose=1,
    mode="auto",
)


# In[ ]:


#train the model
history=model.fit(train_batches, epochs=40,validation_data=val_batches,callbacks=[rlronp,early_stop])


# In[ ]:


#save model
model.save('Trial_5.model')


# In[408]:


#plot the training and validation losses as the model trains to see when it plateous 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model 3 (6 layers, 22 filters, size=7, dil=8,10) loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
#plt.ylim(14,32)
plt.legend(['Train', 'Validation'], loc='upper right')
#import os
#i = 0
#while os.path.exists('{}{:d}.png'.format("#24f Model 3(more data).png", i)):
 #   i += 1
#plt.savefig('{}{:d}.png'.format("#24f Model 3(more data).png", i))



# In[ ]:


#generate model predictions and plot them versus actual values to see correlation
pred = model.predict(val_batches,verbose=2)
actual = np.array(score_uni[420000:430000])
plt.plot(pred, actual, ',')
pear_corr = pearsonr(re, actual)
#sp_corr=spearmanr(pred, actual)
#R_square = r2_score(pred, actual)
print('R=',pear_corr)
#print('Spearman=',sp_corr)
#print('R^2 =',R_square)


# In[ ]:


import seaborn as sns


# In[ ]:


#generate nice correlation plot, like the code above but with distributions of predictions on top of the plot
g = sns.jointplot(x=re, y=actual, kind="reg", scatter=False, color='k')
sns.scatterplot(x=re, y=actual,
               palette='husl',
               sizes=(10, 200), legend='full',
                ax=g.ax_joint)
g.ax_joint.set_xlabel('Predicted expression')
g.ax_joint.set_ylabel('Actual expression')
#g.fig.suptitle("Model predictions vs. Actual data")
#plt.suptitle("Model predictions vs. Actual data")
pear_corr = pearsonr(re, actual)
print('R =',pear_corr[0])
sp_corr=spearmanr(pred, actual)
R_square = pear_corr[0]**2
print('Spearman R =',sp_corr[0])
print('R^2 =',R_square)


# In[ ]:


#needed to keep the dimensions consistent for the plot above
re=np.resize(pred, (10000,))


# In[124]:


#write model summary and statistics to the notebook document
with open("notebook.txt", "a") as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))
    print('Loss =',history.history['loss'][-1],file=f)
    print('Val_loss =',history.history['val_loss'][-1], file=f)
    print('PearsonR =',pear_corr[0],file=f)
    print('SpearmanR =',sp_corr[0], file=f)

