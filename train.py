#!/usr/bin/env python
# coding: utf-8

# In[1]:


import envmodules

envmodules.set_auto_fix_sys_path(1)
envmodules.load('cuda')
envmodules.load('cudnn')
envmodules.load('pytorch')


# In[2]:


import sys
import os

sys.path.append('/home/jmunshi/scratch/wildfires')


# In[3]:


import numpy as np
import time
import os
import torch
import torch.nn as nn
from utils.tools import *
from model.Networks import unet
from dataset.Sen2Fire_Dataset import Sen2FireDataSet
from torch.optim import Adam # type: ignore
from torch.utils import data
import torch.backends.cudnn as cudnn
import random
from tqdm import tqdm


# In[4]:


# cuda stuff
use_cuda = torch.cuda.is_available()
if not use_cuda:
    print('Not using GPU')
device = torch.device('cuda' if use_cuda else 'cpu')
host = os.environ['SLURMD_NODENAME']
print(f'Using device {"gpu" if use_cuda else "cpu"} on {host}')


# In[5]:


# define the prediction classes
name_classes = np.array(['non-fire', 'fire'], dtype=str)

# very small nonzero number, to avoid division by zero errors
epsilon = 1e-14


# In[6]:


# dataset arguments
data_dir = '../Sen2Fire/'
train_list = './dataset/train.txt'
val_list = './dataset/val.txt'
test_list = './dataset/test.txt'
num_classes = 2
mode = 5 # defines the input type (0-all_bands, 1-all_bands_aerosol,...)


# In[7]:


# network arguments
train_kwargs = {
    'batch_size': 16
}
test_kwargs = {
    'batch_size': 50
}
val_kwargs = {
    'batch_size': 1
}
if use_cuda:
    n_workers = int(os.environ['SLURM_CPUS_PER_TASK'])
    cuda_kwargs = {
        'num_workers': n_workers,
        'pin_memory': True,
        'shuffle': True
    }
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)
    val_kwargs.update(cuda_kwargs)
else:
    n_workers = int(os.environ['SLURM_CPUS_PER_TASK']) - 1
    if n_workers <= 0:
        n_workers = 1
    cpu_kwargs = {
        'shuffle': True,
        'batch_size': n_workers,
        'num_workers': n_workers
    }
    train_kwargs.update(cpu_kwargs)
    test_kwargs.update(cpu_kwargs)
    val_kwargs.update(cpu_kwargs)
    
print(f'num_workers: {n_workers}')
    
epochs = 5
learning_rate = 1e-4
weight_decay = 5e-4 # regularization parameter for L2 loss
weight = 10 # ce weight


# In[8]:


modename = ['all_bands',                        #0
            'all_bands_aerosol',                #1
            'rgb',                              #2
            'rgb_aerosol',                      #3
            'swir',                             #4
            'swir_aerosol',                     #5
            'nbr',                              #6
            'nbr_aerosol',                      #7   
            'ndvi',                             #8
            'ndvi_aerosol',                     #9 
            'rgb_swir_nbr_ndvi',                #10
            'rgb_swir_nbr_ndvi_aerosol',]       #11


# In[9]:


snapshot_dir = './Exp/quantum_' + modename[mode] + '/weight_' + str(weight) + '_time' + time.strftime('%m%d_%H%M', time.localtime(time.time())) + '/'
    
if os.path.exists(snapshot_dir) == False:
    os.makedirs(snapshot_dir)
f = open(snapshot_dir + 'Training_log.txt', 'w')


# In[10]:


input_size_train = (512, 512)
torch.manual_seed(1234)

# create network
if mode == 0:
    model = unet(n_classes=num_classes, n_channels=12)
elif mode == 1:
    model = unet(n_classes=num_classes, n_channels=13)
elif mode == 2 or mode == 4 or mode == 6 or mode == 8:
    model = unet(n_classes=num_classes, n_channels=3)
elif mode == 3 or mode == 5 or mode == 7 or mode == 9:       
    model = unet(n_classes=num_classes, n_channels=4)
elif mode == 10:       
    model = unet(n_classes=num_classes, n_channels=6)
elif mode == 11:       
    model = unet(n_classes=num_classes, n_channels=7)
    
# put model in train mode
model = model.to(device)
model.train()


# In[11]:


# define data loaders
train_dataset = Sen2FireDataSet(data_dir, train_list, mode=mode)
test_dataset = Sen2FireDataSet(data_dir, test_list, mode=mode)
val_dataset = Sen2FireDataSet(data_dir, val_list, mode=mode)

print(f'len(train_dataset): {len(train_dataset)}')
print(f'len(test_dataset): {len(test_dataset)}')
print(f'len(val_dataset): {len(val_dataset)}')


train_loader = data.DataLoader(
    Sen2FireDataSet(data_dir, train_list, mode=mode), 
    **train_kwargs)

test_loader = data.DataLoader(
    Sen2FireDataSet(data_dir, test_list, mode=mode),
    **test_kwargs)

val_loader = data.DataLoader(
    Sen2FireDataSet(data_dir, val_list, mode=mode),
    **val_kwargs)

print(f'Number of batches in train_loader: {len(train_loader)} | batch size: {train_kwargs["batch_size"]}')
print(f'Number of batches in test_loader: {len(test_loader)} | batch size: {test_kwargs["batch_size"]}')
print(f'Number of batches in val_loader: {len(val_loader)} | batch size: {val_kwargs["batch_size"]}')


# In[12]:


# define optimizer
optimizer = Adam(model.parameters(), lr=learning_rate,
                weight_decay=weight_decay)

# interpolation for the probability maps and labels
interp = nn.Upsample(size=(input_size_train[1], input_size_train[0]),
                    mode='bilinear')

# define loss function
class_weights = [1, weight]
L_seg = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights).to(device))


# In[13]:


# training loop

model_name = 'best_model.pth'
hist = []
F1_best = 0.

model.train()
for epoch in range(1, epochs + 1):
    print(f'Starting epoch {epoch} on device {device}')
    
    for batch_idx, (patches, labels, _, _) in enumerate(tqdm(train_loader, desc=f'Training epoch {epoch}')):
        torch.cuda.empty_cache()
        
        start_time = time.time()
        
        patches, labels = patches.to(device), labels.to(device).long()
        optimizer.zero_grad()

        preds = interp(model(patches))

        loss = L_seg(preds, labels)

        # calculating metrics
        _, pred_labels = torch.max(preds, 1)
        lbl_pred = pred_labels.detach().cpu().numpy()
        lbl_true = labels.detach().cpu().numpy()
        metrics_batch = []
        for lt, lp in zip(lbl_true, lbl_pred):
            _, _, mean_iu, _ = label_accuracy_score(lt, lp, n_class=num_classes)
            metrics_batch.append(mean_iu)
            
        batch_miou = np.nanmean(metrics_batch, axis=0)
        batch_oa = np.sum(lbl_pred==lbl_true)*1./len(lbl_true.reshape(-1))
        hist.append([
            loss.item(),
            batch_oa,
            batch_miou,
            time.time() - start_time
        ])

        # stepping optimizer
        loss.backward()
        optimizer.step()


        if (batch_idx+1) % 10 == 0:
            print(f"Iter {batch_idx+1}/{len(train_loader)} | Seg Loss: {hist[-1][0]} | OA: {hist[-1][1]} | mIOU: {hist[-1][2]} | Time: {hist[-1][3]}")
            f.write(f"Iter {batch_idx+1}/{len(train_loader)} | Seg Loss: {hist[-1][0]} | OA: {hist[-1][1]} | mIOU: {hist[-1][2]} | Time: {hist[-1][3]}")
            f.flush


    # evaluation after each epoch       
    print('Validating..........')  
    f.write('Validating..........\n')  
    model.eval()
    TP_all = np.zeros((num_classes, 1))
    FP_all = np.zeros((num_classes, 1))
    TN_all = np.zeros((num_classes, 1))
    FN_all = np.zeros((num_classes, 1))
    n_valid_sample_all = 0
    F1 = np.zeros((num_classes, 1))
    IoU = np.zeros((num_classes, 1))
    tbar = tqdm(val_loader, desc="Validating")
    for _, batch in enumerate(tbar):  
        image, label,_,_ = batch
        label = label.squeeze().numpy()
        image = image.float().to(device)
        with torch.no_grad():
            pred = model(image)
        _,pred = torch.max(interp(nn.functional.softmax(pred,dim=1)).detach(), 1)
        pred = pred.squeeze().data.cpu().numpy()                       
        TP,FP,TN,FN,n_valid_sample = eval_image(pred.reshape(-1),label.reshape(-1),num_classes)
        TP_all += TP
        FP_all += FP
        TN_all += TN
        FN_all += FN
        n_valid_sample_all += n_valid_sample
    OA = np.sum(TP_all)*1.0 / n_valid_sample_all
    for i in range(num_classes):
        P = TP_all[i]*1.0 / (TP_all[i] + FP_all[i] + epsilon)
        R = TP_all[i]*1.0 / (TP_all[i] + FN_all[i] + epsilon)
        F1[i] = 2.0*P*R / (P + R + epsilon)
        IoU[i] = TP_all[i]*1.0 / (TP_all[i] + FP_all[i] + FN_all[i] + epsilon)
        if i==1:
            print('===>' + name_classes[i] + ' Precision: %.2f'%(P.item() * 100))
            print('===>' + name_classes[i] + ' Recall: %.2f'%(R.item() * 100))            
            print('===>' + name_classes[i] + ' IoU: %.2f'%(IoU[i].item() * 100))              
            print('===>' + name_classes[i] + ' F1: %.2f'%(F1[i].item() * 100))   
            f.write('===>' + name_classes[i] + ' Precision: %.2f\n'%(P.item() * 100))
            f.write('===>' + name_classes[i] + ' Recall: %.2f\n'%(R.item() * 100))            
            f.write('===>' + name_classes[i] + ' IoU: %.2f\n'%(IoU[i].item() * 100))              
            f.write('===>' + name_classes[i] + ' F1: %.2f\n'%(F1[i].item() * 100))   
    mF1 = np.mean(F1)   
    mIoU = np.mean(F1)           
    print('===> mIoU: %.2f mean F1: %.2f OA: %.2f'%(mIoU*100,mF1*100,OA*100))
    f.write('===> mIoU: %.2f mean F1: %.2f OA: %.2f\n'%(mIoU*100,mF1*100,OA*100))
    if F1[1]>F1_best:
        F1_best = F1[1]
        print('Save Model')   
        f.write('Save Model\n')   
        torch.save(model.state_dict(), os.path.join(snapshot_dir, model_name))

saved_state_dict = torch.load(os.path.join(snapshot_dir, model_name))  
model.load_state_dict(saved_state_dict)

print('Testing..........')  
f.write('Testing..........\n')  

model.eval()
TP_all = np.zeros((num_classes, 1))
FP_all = np.zeros((num_classes, 1))
TN_all = np.zeros((num_classes, 1))
FN_all = np.zeros((num_classes, 1))
n_valid_sample_all = 0
F1 = np.zeros((num_classes, 1))
IoU = np.zeros((num_classes, 1))

print("Starting test loop...")
tbar = tqdm(test_loader, desc="Testing")
for _, batch in enumerate(tbar):  
    image, label,_,_ = batch
    label = label.squeeze().numpy()
    image = image.float().to(device)
    with torch.no_grad():
        pred = model(image)
    _,pred = torch.max(interp(nn.functional.softmax(pred,dim=1)).detach(), 1)
    pred = pred.squeeze().data.cpu().numpy()                       
    TP,FP,TN,FN,n_valid_sample = eval_image(pred.reshape(-1),label.reshape(-1),num_classes)
    TP_all += TP
    FP_all += FP
    TN_all += TN
    FN_all += FN
    n_valid_sample_all += n_valid_sample

OA = np.sum(TP_all)*1.0 / n_valid_sample_all
for i in range(num_classes):
    P = TP_all[i]*1.0 / (TP_all[i] + FP_all[i] + epsilon)
    R = TP_all[i]*1.0 / (TP_all[i] + FN_all[i] + epsilon)
    F1[i] = 2.0*P*R / (P + R + epsilon)
    IoU[i] = TP_all[i]*1.0 / (TP_all[i] + FP_all[i] + FN_all[i] + epsilon)

    if i==1:
        print('===>' + name_classes[i] + ' Precision: %.2f'%(P.item() * 100))
        print('===>' + name_classes[i] + ' Recall: %.2f'%(R.item() * 100))            
        print('===>' + name_classes[i] + ' IoU: %.2f'%(IoU[i].item() * 100))              
        print('===>' + name_classes[i] + ' F1: %.2f'%(F1[i].item() * 100))   
        f.write('===>' + name_classes[i] + ' Precision: %.2f\n'%(P.item() * 100))
        f.write('===>' + name_classes[i] + ' Recall: %.2f\n'%(R.item() * 100))            
        f.write('===>' + name_classes[i] + ' IoU: %.2f\n'%(IoU[i].item() * 100))              
        f.write('===>' + name_classes[i] + ' F1: %.2f\n'%(F1[i].item() * 100))   

mF1 = np.mean(F1)   
mIoU = np.mean(F1)           
print('===> mIoU: %.2f mean F1: %.2f OA: %.2f'%(mIoU*100,mF1*100,OA*100))
f.write('===> mIoU: %.2f mean F1: %.2f OA: %.2f\n'%(mIoU*100,mF1*100,OA*100))        
f.close()
saved_state_dict = torch.load(os.path.join(snapshot_dir, model_name))  
np.savez(snapshot_dir+'Precision_'+str(int(P * 10000))+'Recall_'+str(int(R * 10000))+'F1_'+str(int(F1[1] * 10000))+'_hist.npz',hist=hist) 


# In[ ]:


import matplotlib.pyplot as plt

# Unpack history
seg_losses, oas, mious, times = zip(*hist)

# Plot Loss
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(seg_losses, label='Segmentation Loss')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.title('Segmentation Loss')
plt.legend()

# Plot OA
plt.subplot(2, 2, 2)
plt.plot(oas, label='Overall Accuracy')
plt.xlabel('Batch')
plt.ylabel('OA')
plt.title('Overall Accuracy')
plt.legend()

# Plot mIOU
plt.subplot(2, 2, 3)
plt.plot(mious, label='Mean IOU')
plt.xlabel('Batch')
plt.ylabel('mIOU')
plt.title('Mean IOU')
plt.legend()

# Plot Time
plt.subplot(2, 2, 4)
plt.plot(times, label='Time per Batch')
plt.xlabel('Batch')
plt.ylabel('Time (s)')
plt.title('Time per Batch')
plt.legend()

plt.tight_layout()
plt.show()
plt.savefig(os.path.join(snapshot_dir, 'training_plot.png'))


# # 
