import pretrainedmodels
from pretrainedmodels import utils
import torch
import cv2

import os
import subprocess
import shutil
import numpy as np
from tqdm import tqdm
import glob
import argparse

import torch
import torchvision.transforms as transforms
import torchvision.models as models

# Create the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--base_path', type=str, default="D:\S2VT Ravialdy\S2VT-video-caption(own-version)\YouTubeClips")
parser.add_argument('--target_path', type=str, default="D:\S2VT Ravialdy\S2VT-video-caption(own-version)\extracted_features\\target_frames")
parser.add_argument('--features_path', type=str, default="D:\S2VT Ravialdy\S2VT-video-caption(own-version)\extracted_features\model-vgg16(own)")

args = vars(parser.parse_args())

base_path = args['base_path']
target_path = args['target_path']
features_path = args['features_path']

def vid2frames(video_file, target_path):

    """Convert videos to frames.

    Parameters
    ----------
    base_path : str
        The directory for all MSVD videos.
    target_path : str
        The directory for converted MSVD frames.

    """

    vs = cv2.VideoCapture(base_path + "/" + video_file)
    count = 0
    while True:
        (grabbed, frame) = vs.read()

        if not grabbed:
            break
    
        count += 1
        file_path = target_path + "/" + video_file[:-4]+" frame_{}.jpg".format(count)
        cv2.imwrite(file_path, frame) 


# def extract_frames(video, dst):
#     """
#     extract all frames of a video to dst
#     :param video: path of video
#     :param dst: img output file folder
#     :return:
#     """
#     with open(os.devnull, "w") as ffmpeg_log:
#         if os.path.exists(dst):
#             print(" cleanup: " + dst + "/")
#             shutil.rmtree(dst)
#         os.makedirs(dst)
#         video_to_frames_command = ["ffmpeg",
#                                    # (optional) overwrite output file if it exists
#                                    '-y',
#                                    '-i', video,  # input file
#                                    '-vf', "scale=400:300",  # input file
#                                    '-qscale:v', "2",  # quality for JPEG
#                                    '{0}/%06d.jpg'.format(dst)]  # %06d 6位数字
#         subprocess.call(video_to_frames_command,
#                         stdout=ffmpeg_log, stderr=ffmpeg_log, shell=True)


def extract_frames(video_name, frames_path, features_path, frame_num=80):

    """Get extracted features from frames for every video.

    Parameters
    ----------
    video_name : str
        Name of the video.
    frames_path : str
        The directory for all converted frames.
    features_path : str
        The directory for the result of extracted features.
    frames_path : int
        The frame number used for extracting features.

    """

    # preprocess the images
    transformations = transforms.Compose([
        transforms.Resize(256), #Resize the images where the shorter side is 256 pixels, keeping the aspect ratio
        transforms.CenterCrop(227), #Crop out the center 227x227 portion of the image
        transforms.RandomHorizontalFlip(), #Flip the given image horizontally with probability p =0.5
        transforms.RandomRotation(50), #Rotate images by range of 50 degrees
        transforms.ToTensor(), #Convert a PIL Image or numpy.ndarray to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #Normalize a tensor image with mean and standard deviation for rgb channel
    ])

    # load model
    C, H, W = 3, 224, 224
    #Use pretrained VGG16 model to extract features of rgb images
    model_vgg = models.vgg16(pretrained=True)
    model_vgg.classifier = torch.nn.Sequential(*list(model_vgg.classifier.children())[:-1]) #eliminate the last layer which is to predict 1000 classes on ImageNet dataset

    # load data
    img_list = sorted(glob.glob(frames_path+'/*.jpg'))
    # get index
    selected_imgs = np.linspace(0, len(img_list)-1, frame_num).astype(int)
    list_selected_img = [img_list[i] for i in selected_imgs]
    # build tensor
    imgs = torch.zeros([len(list_selected_img), C, H, W])
    for i in range(len(list_selected_img)):
        model_vgg = model_vgg.cuda()
        model_vgg.eval()
        img = transformations(list_selected_img[i])
        # load_image_fn = utils.LoadTransformImage(model_vgg)
        # img = load_image_fn(list_selected_img[i])
        imgs[i] = img
    with torch.no_grad():
        features = model_vgg(imgs.cuda())
    features = features.cpu().numpy()
    # save
    np.save(os.path.join(features_path, video_name + ".npy"), features)


def get_features(video_path, features_path):

    """Extracting features process all videos.

    Parameters
    ----------
    video_path : str
        The directory for the video.
    features_path : str
        The directory of the result of extracted features.

    """
    # check paths
    list_videos = glob.glob(video_path+"/*.avi")
    video_name = video.split("\\")[-1][:-4]
    # list_npys = os.listdir("D:\S2VT Ravialdy\S2VT-video-caption\extracted_features\model-vgg16")
    # prev_progress_idx = list_videos.index("D:\S2VT Ravialdy\S2VT-video-caption\YouTubeClips\\X_NLV2KCnIE_60_70.avi")
    # list_videos_baru = list_videos[prev_progress_idx:]

    for video in tqdm(list_videos, desc='Extracting~'):
        # get frames to a temp dir
        vid2frames(video_file=video, target_path=target_path)
        # get features
        extract_frames(video_name=video_name, feats_path=target_path, features_path=features_path)