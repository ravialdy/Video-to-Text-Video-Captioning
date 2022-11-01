import torch
import glob
import re
from collections import Counter
import pandas as pd
import tqdm
import numpy as np
import json

list_video_id = []
list_gt_description = []
list_used_description = []
counter = Counter()
gt_dict = {}
max_cap_idx = {}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("AllVideoDescriptions.txt", "r") as file:
    for line in file:
        if line[0] != "#" and len(line) > 5:
            video_id, description = line.split(" ")[0], ' '.join(line.split(" ")[1:])
            used_description = description[:-1]
            video_caption = used_description.lower()
            video_caption = re.sub(r'[~\\/().!,;?:]', ' ', video_caption)
            ground_truth_caption = video_caption
            caption_used = video_caption.split()
            caption_used = ['<bos>'] + caption_used + ['<eos>']
            counter.update(caption_used)

            if video_id in gt_dict:
                max_cap_idx[video_id] += 1
                gt_dict[video_id].append({
                    u'video_id': video_id,
                    u'caption_id': max_cap_idx[video_id],
                    u'caption': used_description,
                    u'tokenized': ground_truth_caption
                })

            else:
                max_cap_idx[video_id] = 0
                gt_dict[video_id] = [{
                    u'video_id': video_id,
                    u'caption_id': 0,
                    u'caption': used_description,
                    u'tokenized': ground_truth_caption
                }]

            list_video_id.append(video_id)
            list_gt_description.append(ground_truth_caption[:-1])
            list_used_description.append(caption_used)

# dictionary of lists 
dict = {'video_id': list_video_id, 'ground truth descriptions': list_gt_description, 'used description' : list_used_description} 
    
df = pd.DataFrame(dict)

# use collections.Counter() to build vocab
all_words = counter.most_common()

word2idx = {'<pad>': 0, '<unk>': 1}
for idx, (word, freq) in enumerate(all_words, start=2):
    if freq <= 0:
        continue
    word2idx[word] = idx
idx2word = {v: k for k, v in word2idx.items()}

# turn words into index 
captions_idx = [[word2idx.get(w, word2idx['<unk>']) for w in caption] for caption in list_used_description]

caption_dict = {}
for name, caption in zip(list_video_id, captions_idx):
    if name not in caption_dict.keys():
        caption_dict[name] = []
    caption_dict[name].append(caption)

# split dataset
data_split = [1200, 100]  # train valid test
video_names = list(caption_dict.keys())
train_videos = video_names[:data_split[0]]
valid_videos = video_names[data_split[0]:data_split[0] + data_split[1]]
test_videos = video_names[data_split[0] + data_split[1]:]

print('number of words in vocab: {}'.format(len(word2idx)))
print("train:{} valid:{} test:{}".format(len(train_videos), len(valid_videos), len(test_videos)))

captions_file = "D:\S2VT Ravialdy\S2VT-video-caption(own-version)\\generated_used_captions.json"
gt_dict_file = "D:\S2VT Ravialdy\S2VT-video-caption(own-version)\\gt_dict1.json"

# save files
with open(captions_file, 'w+', encoding='utf-8') as f:
    json.dump(
        {'word2idx': word2idx,
            'idx2word': idx2word,
            'captions': caption_dict,
            'splits': {'train': train_videos, 'valid': valid_videos, 'test': test_videos}}, f
    )

with open(gt_dict_file, 'w+', encoding='utf-8') as f:
    json.dump({'gts': gt_dict}, f)