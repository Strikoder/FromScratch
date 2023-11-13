import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from  model import UNET
# from utils import(
#     load_checkpoint,
#     save_checkpoint,
#     get_loaders,
#     check_accuracy,
#     save_predicionts_as_imgs
# )

#Hyper para
LEARNING_RATE=1e-4
DEVICE='cuda' if torch.cuda.is_available() else"cpu"
BATCH_SIZE=16 #32
NUM_EPOCHS=3 #100
NUM_WORKERS=2
IMG_HEIGHT=160 #1280
IMG_WIDTH=240 #1918
PIN_MEMORY=True
LOAD_MODEL=True
TRAIN_IMG_DIR="data/train_images/"
TRAIN_MASK_DIR='data/train_masks/'
VAL_IMG_DIR="data/val_images/"
VAL_MASK_DIR="data/val_masks/"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop=tqdm(loader)

    for batch_idx, (data,targets) in enumerate(loop):
        data=data.to(device=DEVICE)
        targets=targets.float().unsqueeze(1).to(device=DEVICE) #Here to float since its a binary prediction

        #forward
        with torch.cuda.amp.autocast():
            predicitons=model(data)
            loss=loss_fn(predicitons,targets)
        
        #backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

def main():
    # Transformations for the training and validation sets
    train_transform = A.Compose(
        [
            A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    val_transforms = A.Compose(
        [
            A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model=UNET(in_channels=3,out_channels=1).to(DEVICE)
    loss_fn=nn.BCEWithLogitsLoss()#binary cross entropy (I cant ommit this if used sigmoid in model.py)
    optimzier=optim.adam(model.parameters(),lr=LEARNING_RATE)

    train_loader,val_loader=get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    scaler=torch.cuda.amp.grad_scaler()
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader,model,optimizer,loss_fn,scaler)
if __name__ =='__main__':
    main()