import torch
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import json
import re
import argparse
from tqdm import tqdm

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge

from dataloader import MSVDDataset

import json

# Create the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--annot_txt', type=str, default="D:\S2VT Ravialdy\S2VT-video-caption(own-version)\\AllVideoDescriptions.txt")
parser.add_argument('--dict_path', type=str, default="D:\S2VT Ravialdy\S2VT-video-caption(own-version)\\gt_dict.json")
parser.add_argument('--caption_dir', type=str, default="D:\S2VT Ravialdy\S2VT-video-caption(own-version)\\generated_used_captions.json")
parser.add_argument('--features_path', type=str, default="D:\S2VT Ravialdy\S2VT-video-caption(own-version)\extracted_features\\model-vgg16")
parser.add_argument('--batch_size', type=int, help="Batch size used for inference", default=10)

args = vars(parser.parse_args())

# prepare data
test_set = MSVDDataset(args['caption_dir'], args['features_path'], mode='test')
test_loader = DataLoader(test_set, batch_size=args['batch_size'], shuffle=False)
word2idx = test_set.word2idx
idx2word = test_set.idx2word
vocab_size = len(word2idx)

# load model
model_s2vt = torch.load(args['model_path']).cuda()

###
### start test
###

pred_dict = {}

for index, (feats, labels, name_video, tensor_masks) in enumerate(tqdm(test_loader, desc="test")):
    # get prediction and cal loss
    model_s2vt.eval()
    with torch.no_grad():
        preds = model_s2vt(feats, mode='test')  # preds [B, L]
    # save result
    for video_id, pred in zip(name_video, preds):
        word_preds = [idx2word[str(i.item())] for i in pred]
        if '<eos>' in word_preds:
            word_preds = word_preds[:word_preds.index('<eos>')]
        pred_dict[video_id] = ' '.join(word_preds)

list_videos = sorted(list(pred_dict.keys()))

with open(args['dict_path'], encoding='utf-8') as f:
    gts = json.load(f)['gts']

gts_dict_sorted = {}
pred_dict_sorted = {}

sorted_pred_videos = sorted(list(gts.keys()))

for video_name in sorted_pred_videos:
    if video_name in list(pred_dict.keys()):
        gts_dict_sorted[video_name] = gts[video_name]
        val_pred_video = pred_dict[video_name]
        if type(val_pred_video) is not list: 
            pred_dict_sorted[video_name] = [val_pred_video]

def bleu():
    scorer = Bleu(n=4)
    # scorer += (hypo[0], ref1)   # hypo[0] = 'word1 word2 word3 ...'
    #                                 # ref = ['word1 word2 word3 ...', 'word1 word2 word3 ...']
    score, scores = scorer.compute_score(gts_dict_sorted, pred_dict_sorted)

    print('belu = %s' % score)

def cider():
    scorer = Cider()
    # scorer += (hypo[0], ref1)
    (score, scores) = scorer.compute_score(gts_dict_sorted, pred_dict_sorted)
    print('cider = %s' % score)

def meteor():
    scorer = Meteor()
    score, scores = scorer.compute_score(gts_dict_sorted, pred_dict_sorted)
    print('meteor = %s' % score)

def rouge():
    scorer = Rouge()
    score, scores = scorer.compute_score(gts_dict_sorted, pred_dict_sorted)
    print('rouge = %s' % score)


def main():
    bleu()
    cider()
    meteor()
    rouge()
    
main()