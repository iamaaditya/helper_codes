
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns; sns.set()
import torch


# In[3]:

get_ipython().magic(u'cd coupled_ensembles/')


# In[255]:

ba = torch.load('./SmallNet_E2_cifar10/branch_activations_20.pth')


# In[261]:

[j.size() for j in i for i in ba]


# In[263]:

[j.size() for j in ba[0]]


# In[13]:

[j.size() for j in i for i in ba]


# In[15]:

num_layers = len(ba) 
batch_size = len(ba[0][0])


# In[26]:

ba0 = [i.sum(0) for i in ba[0]]
ba1 = [i.sum(0) for i in ba[1]]


# In[29]:

[i.size() for i in ba0]


# In[41]:

k = torch.rand(16,32,32,32)
m = torch.rand(16,32,32,32)


# In[43]:

d = []
for a in [k,m]:
    a = a.numpy()
    a_s = a.shape
    d.append(a.reshape(a_s[0], -1))
    

z = np.corrcoef(*d)
z.shape


# In[56]:

for filters in zip(*ba):
    d = []
    for f in filters:
        f = f.data.cpu().numpy()
        f_s = f.shape
        d.append(f.reshape(f_s[0], -1))
    
    z = np.corrcoef(*d)
    sns.heatmap(z,cbar=False)
    sns.plt.show()


# In[105]:

def plot_corr(corr, file):
    
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        ax = sns.heatmap(corr, mask=mask, vmax=.3, square=True, cbar=False, yticklabels=False, xticklabels=False)
        ax.hlines([corr.shape[0]/2], *ax.get_xlim())
        ax.vlines([corr.shape[0]/2], *ax.get_xlim())
        #ax.set_ylabel('')    
        #ax.set_xlabel('')
        
        sns.plt.show()
        sns.plt.savefig(file+'.png')


# In[98]:

ba = torch.load('./SmallNet_E2_cifar10/branch_activations_0.pth')
for filters in zip(*ba):
    d = []
    for f in filters:
        if f.dim() <=2: continue
        
        f = f.data.cpu().numpy()
        f = f.sum(0)
        
        f_s = f.shape
        d.append(f.reshape(f_s[0], -1))
        corr = np.corrcoef(*d)
        
        plot_corr(corr)
        break
        
        
        
    


# In[167]:

def plot_ba(file, number):
    ba = torch.load(file)
    fig, axs = plt.subplots(nrows=3)
    fig.set_size_inches(10, 20)
    index=-1
    for filters in zip(*ba):
        d = []
        index+=1
        for f in filters:
            if f.dim() <=2: continue
            
            f = f.data.cpu().numpy()
            f = f.sum(0)
            f_s = f.shape
            d.append(f.reshape(f_s[0], -1))
        
        
        if not len(d): break # skip the fully connected layers
        corr = np.corrcoef(*d)
            
        # plotting code
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True
        with sns.axes_style("white"):
            ax = sns.heatmap(corr, ax=axs[index], mask=mask, vmax=.3, square=True, cbar=False, yticklabels=False, xticklabels=False)
            ax.hlines([corr.shape[0]/2], *ax.get_xlim())
            ax.vlines([corr.shape[0]/2], *ax.get_xlim())
            #ax.set_ylabel('')    
            #ax.set_xlabel('')
            ax.set_title('Layer: ' + str(index) + '  Epoch: ' + str(number))
        
            
            
        #axs[0].title('test')
        #sns.plt.show()
        fig.savefig(file+'.png')
        
    


# In[168]:

file = './SmallNet_E2_cifar10/branch_activations_40.pth'
plot_ba(file, 40)


# In[254]:

for i in range(0,100,10):
    file = './SmallNet_E2_cifar10/branch_activations_'+str(i)+'.pth'
    plot_ba(file, i)


# In[193]:

#def plot_ba(file, number):
#if True:
for i in range(0,100,10):
    #i = 20
    file='./SmallNet_E2_cifar10/branch_activations_'+str(i)+'.pth'
    ba = torch.load(file)
    #fig, axs = plt.subplots(nrows=3)
    #fig.set_size_inches(10, 20)
    index=-1
    sum_corr=[i]
    for filters in zip(*ba):
        d = []
        index+=1
        for f in filters:
            if f.dim() <=2: continue
            
            f = f.data.cpu().numpy()
            f = f.sum(0)
            f_s = f.shape
            d.append(f.reshape(f_s[0], -1))
        
        
        if not len(d): break # skip the fully connected layers
        corr = np.corrcoef(*d)
        z = len(corr)//2
        zz = z*z
        #print(z)
        sum_corr.extend([corr[:z, :z].sum()/zz, corr[z:, :z].sum()/zz, corr[z:, z:].sum()/zz])
        #print(corr.shape)
    print(sum_corr)
        
        


# In[189]:

from pprint import pprint


# In[ ]:

#def plot_ba(file, number):
#if True:
for i in range(0,100,10):
    #i = 20
    file='./SmallNet_E2_cifar10/branch_weights_'+str(i)+'.pth'
    ba = torch.load(file)
    #fig, axs = plt.subplots(nrows=3)
    #fig.set_size_inches(10, 20)
    index=-1
    sum_corr=[i]
    for filters in zip(*ba):
        d = []
        index+=1
        for f in filters:
            if f.dim() <=2: continue
            
            f = f.data.cpu().numpy()
            f = f.sum(0)
            f_s = f.shape
            d.append(f.reshape(f_s[0], -1))
        
        
        if not len(d): break # skip the fully connected layers
        corr = np.corrcoef(*d)
        z = len(corr)//2
        zz = z*z
        #print(z)
        sum_corr.extend([corr[:z, :z].sum()/zz, corr[z:, :z].sum()/zz, corr[z:, z:].sum()/zz])
        #print(corr.shape)
    print(sum_corr)


# In[194]:

bw = torch.load('./SmallNet_E2_cifar10/branch_weights_10.pth')


# In[195]:

type(bw)


# In[196]:

bw.keys()


# In[236]:

nets = []
for i in xrange(2):
    nets.append([bw[j] for j in bw if str(i)+'.conv' in j and 'weight' in j])

for i in nets:
    for j in i:
        print(j.size())


# In[216]:

[len(i) for i in nets]


# In[203]:

[j for j in bw if '0' in j and 'conv' in j and 'weight' in j]


# In[207]:

z = bw['nets.0.conv1.weight']


# In[217]:

nets


# In[240]:

#def plot_ba(file, number):
#if True:
for index_file in range(0,100,10):
    #i = 20
    file='./SmallNet_E2_cifar10/branch_weights_'+str(index_file)+'.pth'
    bw = torch.load(file)
    nets = []
    for i in xrange(2):
        nets.append([bw[j] for j in bw if str(i)+'.conv' in j and 'weight' in j])
    #fig, axs = plt.subplots(nrows=3)
    #fig.set_size_inches(10, 20)
    index=-1
    sum_corr=[index_file]
    for filters in zip(*nets):
        d = []
        index+=1
        for f in filters:
            if f.dim() <=2: continue
            
            f = f.cpu().numpy()
            f = f.sum(1)
            f_s = f.shape
            d.append(f.reshape(f_s[0], -1))
        
        
        if not len(d): break # skip the fully connected layers
        corr = np.corrcoef(*d)
        z = len(corr)//2
        zz = z*z
        #print(z)
        sum_corr.extend([corr[:z, :z].sum()/zz, corr[z:, :z].sum()/zz, corr[z:, z:].sum()/zz])
        #print(corr.shape)
    print(sum_corr)


# In[251]:

def plot_bw(file, number):
    ba = torch.load(file)
    fig, axs = plt.subplots(nrows=3)
    fig.set_size_inches(10, 20)
    nets = []
    for i in xrange(2):
        nets.append([ba[j] for j in ba if str(i)+'.conv' in j and 'weight' in j])
    #fig, axs = plt.subplots(nrows=3)
    #fig.set_size_inches(10, 20)
    index=-1
    
    for filters in zip(*nets):
        d = []
        index+=1
        for f in filters:
            if f.dim() <=2: continue
            
            f = f.cpu().numpy()
            f = f.sum(1)
            f_s = f.shape
            d.append(f.reshape(f_s[0], -1))
        
        
        if not len(d): break # skip the fully connected layers
        corr = np.corrcoef(*d)
            
        # plotting code
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True
        with sns.axes_style("white"):
            ax = sns.heatmap(corr, ax=axs[index], mask=mask, vmax=.3, square=True, cbar=False, yticklabels=False, xticklabels=False)
            ax.hlines([corr.shape[0]/2], *ax.get_xlim())
            ax.vlines([corr.shape[0]/2], *ax.get_xlim())
            #ax.set_ylabel('')    
            #ax.set_xlabel('')
            ax.set_title('Layer: ' + str(index) + '  Epoch: ' + str(number))
        
            
            
        #axs[0].title('test')
        #sns.plt.show()
        fig.savefig(file+'.png')
        
    


# In[252]:

for i in range(0,100,10):
    file = './SmallNet_E2_cifar10/branch_weights_'+str(i)+'.pth'
    plot_bw(file,i)


# In[ ]:



