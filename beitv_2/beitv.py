import random
import pandas as pd
import numpy as np
import os
from PIL import Image
# from autoaugment import ImageNetPolicy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
#import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.transforms import ToPILImage
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from tqdm.auto import tqdm

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

import warnings
warnings.filterwarnings(action='ignore')  

from urllib.request import urlopen
import timm
from utils import *


import GPUtil
GPUtil.showUtilization()




device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("device: ",device)



CFG = {
    'IMG_SIZE':188,
    'EPOCHS':50,
    'LEARNING_RATE':9e-5, #best : 1e-4
    'BATCH_SIZE':256,
    'SEED':41,
    "base_model":"beivt2",
    "saved_model": None,
    "best_mAP": 0.96,
}

if CFG["saved_model"]!=None: 
    model.load_state_dict(torch.load(f'/home1/shk00315/beitv2{CFG["saved_model"]}.pt'))
    
seed_everything(CFG['SEED'])

df = pd.read_csv('./train.csv')
df = df.sample(frac=1)
train_len = int(len(df) * 0.8)
train_df = df[:train_len]
val_df = df[train_len:]

train_labels = get_labels(train_df)
val_labels = get_labels(val_df)

   
# class CustomModel(nn.Module):
#     def __init__(self):
#         super(CustomModel,self).__init__()
#         self.backbone = timm.create_model('tresnet_v2_l.miil_in21k', pretrained=True, num_classes=60).to(device)
#     def forward(self, x):
#         x = self.backbone(x)
#         x = F.sigmoid(x)
#         return x
    
    
train_transform = transforms.Compose([
    transforms.AutoAugment(),
    transforms.ToTensor(),
    # transforms.RandomResizedCrop(CFG['IMG_SIZE'],scale = (0.5,1.0)),
    transforms.RandomInvert(),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = CustomDataset(train_df['img_path'].values, train_labels, train_transform)
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=2)

val_dataset = CustomDataset(val_df['img_path'].values, val_labels, test_transform)
val_loader = DataLoader(val_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=2)


def train(model, optimizer, train_loader, val_loader, device):
    # for n,p in model.named_parameters():
    #     if n in ['fc.weight','fc.bias']:
    #         p.requires_grad=True
    #     else: 
    #         p.requires_grad=False
    
    model.to(device)        
    model.train()
    criterion = AsymmetricLoss().to(device)
    # best_val_loss = float('inf') 
    best_model = None
    
    for epoch in range(1, CFG['EPOCHS']+1):
        train_loss = []
        print("LR: ",optimizer.param_groups[0]['lr'])
        for imgs, labels in tqdm(iter(train_loader),ncols=50):
            imgs = imgs.float().to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # out = model.forward(imgs)
            out = model(imgs)
            loss = criterion(out.float(), labels.float()) # out : list/ labels : Tensor(64,60)
            # loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            del imgs, labels,out
            torch.cuda.empty_cache()
        _train_loss = np.mean(train_loss)
        scheduler.step() 
        _val_loss, mAP = validation(model, criterion, val_loader, device)
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] mAP : [{mAP:.5f}]')

            
        if CFG["best_mAP"] < mAP:
            CFG["best_mAP"] = mAP
            best_model = model
            torch.save(best_model.state_dict(), '/home1/shk00315/beitv_2'+str(epoch)+str(mAP)+'.pt')
    
    return best_model



def validation(model, criterion, val_loader, device):
    model.to(device)
    model.eval()
    val_loss = []
    total_mAP = []
    cnt = 1
    with torch.no_grad():
        for imgs, labels in tqdm(iter(val_loader),ncols=50): 
            imgs = imgs.float().to(device)
            labels = labels.to(device) 
            # print(len(labels[1]))

            probs = model(imgs)
            
            loss = criterion(probs, labels)

            val_loss.append(loss.item())
            
            batch_mAP = cal_mAP(probs,labels)
            # print(f'배치mAP 사이즈 : {batch_mAP.shape}')
            total_mAP.append(batch_mAP)
            # print(f'total mAP 길이 : {len(total_mAP)}')
            del imgs, labels, batch_mAP, probs,loss
            torch.cuda.empty_cache()
        _val_loss = np.mean(val_loss)
        total_mAP = np.mean(total_mAP)
        
    return _val_loss,total_mAP

def inference(model, test_loader, device):
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for imgs in tqdm(iter(test_loader),ncols=100):
            imgs = imgs.float().to(device)
            
            probs = model(imgs)

            probs  = probs.cpu().detach().numpy()
            predictions += probs.tolist()
    return predictions


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
model = beitv2_base()
model = nn.DataParallel(model, output_device= 0)
model.to(device)

optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
scheduler = CosineAnnealingLR(optimizer, T_max=CFG['EPOCHS'])

infer_model = train(model, optimizer, train_loader, val_loader, device)

test = pd.read_csv('./test.csv')
test_dataset = CustomDataset(test['img_path'].values, None, test_transform)
test_loader = DataLoader(test_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=2)

preds = inference(infer_model, test_loader, device)

submit = pd.read_csv('./sample_submission.csv')
submit.iloc[:,1:] = preds
submit.to_csv('./beitv_submit.csv', index=False)