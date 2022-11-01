import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse

import os
from tqdm import tqdm
import time

from dataloader import MSVDDataset
from S2VT import S2VT_Baseline
from utils import Masking_Loss, EarlyStopping

# Create the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--captions_dir', type=str, default="D:\S2VT Ravialdy\S2VT-video-caption(own-version)\\generated_used_captions.json")
parser.add_argument('--features_dir', type=str, default="D:\S2VT Ravialdy\S2VT-video-caption(own-version)\extracted_features\\model-vgg16")
parser.add_argument('--mode', type=str, default="train")
parser.add_argument('--early_patiance', type=int, help="Early stopping patiance", default=400)
parser.add_argument('--lr_patiance', type=int, help="Learning rate patiance", default=200)
parser.add_argument('--epochs', type=int, help="Total epochs used for training", default=6000)
parser.add_argument('--save_per', type=int, help="Save the model for every n epoch", default=250)
parser.add_argument('--save_path', type=str, help="Directory of saved pytorch model", default="./checkpoint")

args = vars(parser.parse_args())

# prepare data
train_data = MSVDDataset(args['captions_dir'], args['features_dir'])
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
valid_data = MSVDDataset(args['captions_dir'], args['features_dir'], mode='valid')
valid_loader = DataLoader(valid_data, batch_size=16, shuffle=False)
word2idx = train_data.word2idx
idx2word = train_data.idx2word
vocab_size = len(word2idx)

model_s2vt = S2VT_Baseline(vocab_size=vocab_size, feature_dim=4096, length=80, hidden_dim=500, embedding_dim=500,
                        dropout_lstm=0.3, p_dropout=0.3).cuda()

optimizer = optim.Adam(model_s2vt.parameters(), lr=0.0001, betas=(0.9, 0.999))

# dynamic learning rate
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=args['lr_patiance'])
ckpt_time = time.strftime('%y_%m_%d_%H_%M_%S-', time.localtime())
early_stopping = EarlyStopping(patience=args['early_patiance'], verbose=True,
                                path=os.path.join(args['save_path'], ckpt_time + 'stop.pth'))
mask_criterion = Masking_Loss()

# Training Loop

for epoch in range(args['epochs']):

    # Training phase

    train_running_loss = 0.0
    loss_count = 0
    for index, (feats, mask_labels, name_videos, tensor_masks) in enumerate(
            tqdm(train_loader, desc="epoch:{}".format(epoch))):
  
        optimizer.zero_grad()
        model_s2vt.train()

        probs = model_s2vt(feats, mask_labels=mask_labels[:, :-1], mode='train')

        loss = mask_criterion(probs, mask_labels, tensor_masks)

        loss.backward()
        optimizer.step()

        train_running_loss += loss.item()
        loss_count += 1

    train_running_loss /= loss_count

    # Validation phase
    
    valid_running_loss = 0.0
    loss_count = 0
    for index, (feats, mask_labels, name_videos, tensor_masks) in enumerate(valid_loader):
        model_s2vt.eval()

        with torch.no_grad():
            probs = model_s2vt(feats, mask_labels=mask_labels[:, :-1], mode='train')
            loss = mask_criterion(probs, mask_labels, tensor_masks)

        valid_running_loss += loss.item()
        loss_count += 1

    valid_running_loss /= loss_count

    print("train loss:{} valid loss: {}".format(train_running_loss, valid_running_loss))
    lr_scheduler.step(valid_running_loss)

    # early stop
    early_stopping(valid_running_loss, model_s2vt)
    if early_stopping.early_stop:
        print("Early stopping")
        break

    # save checkpoint
    if epoch % args['save_per'] == 0:
        print('epoch:{}, saving checkpoint'.format(epoch))
        torch.save(model_s2vt, os.path.join(args['save_path'],
                                        ckpt_time + str(epoch) + '.pth'))

# save model
torch.save(model_s2vt, os.path.join(args['save_path'], ckpt_time + 'final.pth'))