import pandas as pd
import numpy as np



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
import torchvision.models as models
from torchvision.transforms import ToPILImage

from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

from tqdm.auto import tqdm

import warnings
warnings.filterwarnings(action='ignore') 

from urllib.request import urlopen
import timm

from transformers import AutoFeatureExtractor,CvtForImageClassification


from utils import *

import GPUtil
GPUtil.showUtilization()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("device: ",device)

CFG = {
    'IMG_SIZE':224,
    'EPOCHS':200,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':256,
    'SEED':41,
    "base_model":"CvT",
    "saved_model": None,
    "best_mAP": 0.94,
}

seed_everything(CFG['SEED']) # Seed 고정


df = pd.read_csv('./train.csv')

df = df.sample(frac=1)
train_len = int(len(df) * 0.8)
train_df = df[:train_len]
val_df = df[train_len:]

train_labels = get_labels(train_df)
val_labels = get_labels(val_df)

train_transform = transforms.Compose([
    # transforms.AutoAugment(),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    # transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.5),
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



    
class TResnet_Model(nn.Module):
    def __init__(self):
        super(TResnet_Model,self).__init__()
        self.model = timm.create_model('tresnet_v2_l.miil_in21k_ft_in1k', pretrained=True).to(device)
        self.model.head.fc =  nn.Linear(2048, 60,bias=True) 
        self.data_config = timm.data.resolve_model_data_config(self.model)
        self.topilimage = ToPILImage()
        self.transforms = timm.data.create_transform(**self.data_config, is_training=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x_list = torch.split(x,1,dim=0)
        output= []
        for x in x_list:
            x = self.topilimage(x.view(-1,CFG['IMG_SIZE'],CFG['IMG_SIZE']))
            x = self.transforms(x).unsqueeze(0).to(device)
            x = self.model(x)
            x = F.sigmoid(x)
            output.append(x)
        x=torch.cat(output,dim=0)
        return x

def rand_bbox(size,lam) : # size : [B,C,W,H]
    W = size[2] # 이미지 width
    H = size[3] # 이미지 Height
    cut_rat = np.sqrt(1. - lam) # 패치 크기의 비율 정하기 
    cut_w = np.int64(W*cut_rat) # 패치의 너비 
    cut_h = np.int64(H*cut_rat) # 패치의 높이 
    
    # 기존 이미지의 크기에서 랜덤하게 값을 가져옴 ( 중간 좌표 추출 )
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx-cut_w//2,0,W)
    bby1 = np.clip(cy-cut_h//2,0,H)
    bbx2 = np.clip(cx+cut_w//2,0,W)
    bby2 = np.clip(cy+cut_h//2,0,H)
    
    return bbx1,bby1,bbx2,bby2

def train(model, optimizer, train_loader, val_loader, device):
    model.to(device)
    criterion = AsymmetricLoss().to(device) 
    #criterion = nn.BCELoss().to(device)
    
    best_model = None
    
    for epoch in range(1, CFG['EPOCHS']+1):
        model.train()
        train_loss = []
        for imgs, labels in tqdm(iter(train_loader), ncols=50):
            imgs = imgs.float().to(device)
            labels = labels.to(device)
            
            if np.random.random()>0.5: # cutmix 작동될 확률      
                lam = np.random.beta(1.0, 1.0)
                rand_index = torch.randperm(imgs.size()[0]).to(device)
                target_a = labels
                target_b = labels[rand_index]            
                bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
                imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[rand_index, :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size()[-1] * imgs.size()[-2]))
                outputs = model(imgs)
                loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
            
            else :     
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            
            del imgs, labels,outputs
            torch.cuda.empty_cache()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())

        # for imgs, labels in tqdm(iter(train_loader),ncols=50):
        #     imgs = imgs.float().to(device)
        #     labels = labels.to(device)
            
        #     optimizer.zero_grad()
            
        #     output = model(imgs)
        #     loss = criterion(output, labels)
            
        #     loss.backward()
        #     optimizer.step()
        #     train_loss.append(loss.item())
        #     del imgs, labels,output
        #     torch.cuda.empty_cache()
            
        _val_loss, mAP = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        print(
            f'Epoch [{epoch}] LR : [{(optimizer.param_groups[0]["lr"])}] Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] mAP : [{mAP:.5f}]',"\n")
        scheduler.step()
        if CFG["best_mAP"] < mAP:
            CFG["best_mAP"] = mAP
            best_model = model
            torch.save(best_model.state_dict(), f'/home1/shk00315/CvT/{CFG["base_model"]}_model/{CFG["base_model"]}_'+str(epoch)+'.pt')
    
    return best_model

def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    total_mAP = []
    with torch.no_grad():
        for imgs, labels in tqdm(iter(val_loader),ncols=50):
            imgs = imgs.float().to(device)
            labels = labels.to(device)
            
            probs = model(imgs)
            loss = criterion(probs, labels)
            val_loss.append(loss.item())
            batch_mAP = cal_mAP(probs, labels)
            total_mAP.append(batch_mAP)
            
            del imgs, labels, batch_mAP, probs,loss
            torch.cuda.empty_cache()

        _val_loss = np.mean(val_loss)
        total_mAP = np.mean(total_mAP)
    return _val_loss, total_mAP

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
model = Cvt_Model()
model = nn.DataParallel(model, output_device= 0)

if CFG["saved_model"]!=None: 
    model.load_state_dict(torch.load(f'/home1/shk00315/CvT/{CFG["base_model"]}_model/{CFG["saved_model"]}.pt'))
    
optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
scheduler = CosineAnnealingLR(optimizer, T_max=CFG['EPOCHS']) 
#scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=20, T_mult=1, eta_max=0.001,  T_up=2, gamma=0.5)
infer_model = train(model, optimizer, train_loader, val_loader, device)


test = pd.read_csv('./test.csv')
test_dataset = CustomDataset(test['img_path'].values, None, test_transform)
test_loader = DataLoader(test_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=2)

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
preds = inference(infer_model, test_loader, device)


submit = pd.read_csv('./sample_submission.csv')
submit.iloc[:,1:] = preds
submit.to_csv('./cvt_submit.csv', index=False)

