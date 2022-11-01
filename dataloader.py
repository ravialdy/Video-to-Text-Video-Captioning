from torch.utils.data import Dataset
import torch
import numpy as np
import json
import pathlib as plb
import glob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MSVDDataset(Dataset):

    """MSVD dataset for DataLoader.

    Parameters
    ----------
    captions_dir : str
        Directory of generated captions file.
    features_dir : str
        Directory of extracted features.
    max_length : int
        Fixed length of the time steps.
    mode : str
        Mode when processing the data. It can be in training phase (mode = 'train') or validation (mode = 'valid')

    Attributes
    ----------
    word2idx : 
        Convert selected word to corresponding index.
    idx2word : 
        Convert selected index to corresponding word.
    captions : 
        List of captions for selected video.
    splits : 
        Split the entire data to get the selected one (Training, Validation, or Testing data)
    """

    def __init__(self, captions_dir, features_dir, max_length=80, mode='train'):
        with open(captions_dir, encoding='utf-8') as f:
            data = json.load(f)
            self.word2idx = data['word2idx']
            self.idx2word = data['idx2word']
            self.captions = data['captions'] 
            self.splits = data['splits']
            self.base_path = features_dir

        all_feat_paths = [i for i in glob.glob(features_dir+'/*.npy')]
        self.videos = self.splits[mode]
        self.max_length = max_length

        self.feat_paths = [path for path in all_feat_paths if path.split("\\")[-1][:-4] in self.videos]

    def __getitem__(self, index):

        name_video = self.feat_paths[index].split("\\")[-1][:-4]

        feat = np.load(str(self.feat_paths[index]))
        feat = torch.tensor(feat, dtype=torch.float, device=device, requires_grad=True)

        labels = self.captions[name_video]
        label = np.random.choice(labels, 1)[0]
        mask_label = torch.zeros([self.max_length], dtype=torch.long, device=device) 
        if len(label) > self.max_length:
            label = label[:self.max_length]
        mask_label[:len(label)] = torch.tensor(label, dtype=torch.long, device=device)
        tensor_mask = torch.zeros([self.max_length], dtype=torch.float, device=device)
        tensor_mask[:len(label)] = 1

        return feat, mask_label, name_video, tensor_mask

    def __len__(self):
        return len(self.feat_paths)

# # Debungging
# if __name__ == '__main__':   
#     trainset = MSVDDataset('D:\S2VT Ravialdy\S2VT-video-caption\generated_used_captions.json', 'D:\S2VT Ravialdy\S2VT-video-caption\extracted_features\model-vgg16')
#     train_loader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True)
#     a = next(iter(train_loader))
