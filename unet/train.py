import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from  model import UNET
#from model_UNETR2D_with_backbone import UNETR2D
#from model_UNTER_with_Backbone import UNETR
from model_UNTER_with_Backbone import UNETWithBackbone

# When testing change LOAD_MODEL to True and un comment the row after the if statement

# Run this if u had a problem with env
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


from utils import(
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs
)

#Hyper param


LEARNING_RATE=1e-4
DEVICE='cuda' if torch.cuda.is_available() else"cpu"
BATCH_SIZE=16 #32
NUM_EPOCHS=3 #100
NUM_WORKERS=2
IMG_HEIGHT=160 #1280
IMG_WIDTH=240 #1918
PIN_MEMORY=True
LOAD_MODEL=False #For testing
TRAIN_IMG_DIR="data/train_images"
TRAIN_MASK_DIR='data/train_masks'
VAL_IMG_DIR="data/val_images"
VAL_MASK_DIR="data/val_masks"


# config = {
#         "image_size": (1918, 1280),
#         "num_layers": 12,
#         "hidden_dim": 768,
#         "mlp_dim": 3072,
#         "num_heads": 12,
#         "dropout_rate": 0.1,
#         "patch_size": 20,
#         "num_channels": 3,
#         "num_patches": (1920 // 20) * (1280 // 20)

# } 


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
            A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH), # In new pytorch versions this might be needed to be entered as a param antialias=True
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
    
    
    #model=UNET(in_channels=3,out_channels=1).to(DEVICE) #UNET
    # model = UNETR2D(config).to(DEVICE) UNTR
    # model = UNETR(in_channels=3, out_channels=1).to(DEVICE) # UNTR with backbone VGG
    model = UNETWithBackbone(in_channels=3, out_channels=1).to(DEVICE) 
    loss_fn=nn.BCEWithLogitsLoss()#binary cross entropy (I cant ommit this if used sigmoid in model.py)
    optimzier=optim.Adam(model.parameters(),lr=LEARNING_RATE)

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
    if LOAD_MODEL:
        load_checkpoint(torch.load('my_checkpoint.pth.tar'),model)
        check_accuracy(val_loader,model,device=DEVICE)
    
    scaler=torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader,model,optimzier,loss_fn,scaler)

        #Save model
        checkpoint={
            "state_dict":model.state_dict(),
            "optimizer":optimzier.state_dict()
        }
        save_checkpoint(checkpoint)
        #Check acc
        check_accuracy(val_loader,model,device=DEVICE)

        #Print some example in a folder
        save_predictions_as_imgs(
            val_loader,model,folder='saved_images/', device=DEVICE
        )
if __name__ =='__main__':
    main()