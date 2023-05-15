#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
# adapted from https://github.com/dharwath

import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel as DDP
import torchaudio
from dataloaders import *
from models.setup import *
from models.util import *
from models.GeneralModels import *
from models.multimodalModels import *
from training.util import *
from evaluation.calculations import *
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from training import validate
import time
from tqdm import tqdm

import numpy as trainable_parameters
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import scipy
import scipy.signal
from scipy.spatial import distance
import librosa
import matplotlib.lines as lines

import itertools
import seaborn as sns
from torchvision.io import read_image
from torchvision.models import *
import shutil

BACKEND = "nccl"
INIT_METHOD = "tcp://localhost:54321"

aud_dir = Path('../../Datasets/spokencoco/SpokenCOCO')

config_library = {
    "multilingual": "English_Hindi_DAVEnet_config.json",
    "multilingual+matchmap": "English_Hindi_matchmap_DAVEnet_config.json",
    "english": "English_DAVEnet_config.json",
    "english+matchmap": "English_matchmap_DAVEnet_config.json",
    "hindi": "Hindi_DAVEnet_config.json",
    "hindi+matchmap": "Hindi_matchmap_DAVEnet_config.json",
}
def LoadAudio(path, alignment, audio_conf):
    if alignment is None:
        aud, sr = torchaudio.load(path)
    else:
        offset = int(alignment[0] * 16000)
        frames = int(alignment[1] * 16000)
        aud, sr = torchaudio.load(path, frame_offset=offset, num_frames=frames)

    return aud

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--resume", action="store_true", dest="resume",
        help="load from exp_dir if True")
parser.add_argument("--config-file", type=str, default='matchmap', choices=['matchmap'], help="Model config file.")
parser.add_argument("--restore-epoch", type=int, default=-1, help="Epoch to generate accuracies for.")
parser.add_argument("--image-base", default="../..", help="Model config file.")
command_line_args = parser.parse_args()
restore_epoch = command_line_args.restore_epoch

# Setting up model specifics
heading(f'\nSetting up model files ')
args, image_base = modelSetup(command_line_args, True)
rank = 'cuda'
 
concepts = []
with open('./data/test_keywords.txt', 'r') as f:
    for keyword in f:
        concepts.append(keyword.strip())

alignments = {}
prev = ''
prev_wav = ''
prev_start = 0
with open(Path('../../Datasets/spokencoco/SpokenCOCO/words.txt'), 'r') as f:
    for line in f:
        wav, start, stop, label = line.strip().split()
        if label in concepts or (label == 'hydrant' and prev == 'fire' and wav == prev_wav):
            if wav not in alignments: alignments[wav] = {}
            if label == 'hydrant' and prev == 'fire': 
                label = prev + " " + label
                start = prev_start
            if label not in alignments[wav]: alignments[wav][label] = (float(start), float(stop))
        prev = label
        prev_wav = wav
        prev_start = start

transcriptions = {}
prev = ''
prev_wav = ''
prev_start = 0

with open(Path('../../Datasets/spokencoco/SpokenCOCO/words.txt'), 'r') as f:
    for line in f:
        wav, start, stop, label = line.strip().split()
        
        if wav not in transcriptions: transcriptions[wav] = []
        transcriptions[wav].append((label, float(start), float(stop)))
        # prev = label
        # prev_wav = wav
        # prev_start = start

audio_conf = args["audio_config"]
target_length = audio_conf.get('target_length', 1024)
padval = audio_conf.get('padval', 0)
image_conf = args["image_config"]
crop_size = image_conf.get('crop_size')
center_crop = image_conf.get('center_crop')
RGB_mean = image_conf.get('RGB_mean')
RGB_std = image_conf.get('RGB_std')

resize = transforms.Resize((256, 256))
to_tensor = transforms.ToTensor()
image_normalize = transforms.Normalize(mean=RGB_mean, std=RGB_std)

image_resize = transforms.transforms.Resize((256, 256))
trans = transforms.ToPILImage()

image_base = Path('../../Datasets/spokencoco/')
episodes = np.load(args["episodes_test"], allow_pickle=True)['episodes'].item()

save_dir = Path('./examples')

with torch.no_grad():

    # for i, im in enumerate(episodes['matching_set']):

    #     imgpath = image_base / im
    #     name = Path(im).stem
    #     save_fn = save_dir / Path('few-shot_retrieval_images') / Path(f'{i}_{name}.jpg')
    #     save_fn.parent.mkdir(parents=True, exist_ok=True)
    #     shutil.copyfile(imgpath, save_fn)


    # episode_names = list(episodes.keys())
    # episode_names.remove('matching_set')

    # episode_num = 1
    # episode = episodes[episode_num]
    # for w in episode['matching_set']:
    #     imgpath = image_base / episode['matching_set'][w]
    #     save_fn = save_dir / Path('few-shot_classification_images') / Path(f'{episode_num}_{w}.jpg')
    #     save_fn.parent.mkdir(parents=True, exist_ok=True)
    #     shutil.copyfile(imgpath, save_fn)


    # for w in episode['queries']:
    #     wav, spkr = episode['queries'][w]
    #     lookup = str(Path(wav).stem)
    #     if lookup in alignments:
    #         if w in alignments[lookup]:

    #             aud = LoadAudio(image_base / 'SpokenCOCO' / wav, alignments[lookup][w], audio_conf).squeeze().numpy()
    #             plt.figure(figsize=(15, 5))
    #             plt.plot(aud, c='darkgreen')
    #             plt.axis('off')
    #             save_fn = save_dir / Path('few-shot_classification_audio_queries') / Path(f'{episode_num}_{w}.jpg')
    #             save_fn.parent.mkdir(parents=True, exist_ok=True)
    #             plt.savefig(save_fn, bbox_inches='tight',pad_inches = 0)

    # ss_save_fn = 'support_set/support_set_5.npz'
    # support_set = np.load(ss_save_fn, allow_pickle=True)['support_set'].item()
    # counts = {}
    # for name in support_set:
    #     wav, img, spkr, start, end, word = support_set[name]
    #     if word not in counts: counts[word] = 1
    #     imgpath = image_base / img
    #     save_fn = save_dir / Path('few-shot_support_set') / Path(f'{word}_{str(counts[word])}_image.jpg')
    #     save_fn.parent.mkdir(parents=True, exist_ok=True)
    #     shutil.copyfile(imgpath, save_fn)

    #     aud = LoadAudio(Path('support_set') / Path(wav).parent / Path(f'{name}_{word}.wav'), None, audio_conf).squeeze().numpy()
    #     plt.figure(figsize=(15, 5))
    #     plt.plot(aud, c='darkgreen')
    #     plt.axis('off')
    #     save_fn = save_dir / Path('few-shot_support_set') / Path(f'{word}_{str(counts[word])}_audio.jpg')
    #     save_fn.parent.mkdir(parents=True, exist_ok=True)
    #     plt.savefig(save_fn, bbox_inches='tight',pad_inches = 0)

    #     counts[word] += 1

    key = np.load(Path('data/label_key.npz'), allow_pickle=True)['id_to_word_key'].item()
    train_id_lookup = np.load(Path("data/train_lookup.npz"), allow_pickle=True)['lookup'].item()
    neg_images = np.load(Path("data/train_lookup.npz"), allow_pickle=True)['base_negs']
    c_aud = 1
    c_im = 1
    c = 1
    for id in train_id_lookup:

        pred_word = key[id]

        for name in list(train_id_lookup[id]['audio'].keys())[0:100]: 
            if name not in transcriptions: continue
            wav = train_id_lookup[id]['audio'][name][0]
            aud = LoadAudio(aud_dir / wav, None, audio_conf).squeeze().numpy()
            plt.figure(figsize=(15, 5))
            plt.plot(aud, c='darkgreen')
            plt.axis('off')
            flag = False
            for label, start, stop in transcriptions[name]:
                if label.lower() in concepts: flag = True
                s = int(float(start)*16000)
                maxx = np.max(aud)
                minn = np.min(aud)
                plt.plot([s, s], [minn, maxx], color = 'black')
                plt.text(int(16000*(start)), minn, label, horizontalalignment='left', verticalalignment='center')

            if flag:
                save_fn = save_dir / Path('few-shot_unlabelled_audio') / Path(f'{c_aud}_audio.jpg')
                save_fn.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_fn, bbox_inches='tight',pad_inches = 0)
                c_aud += 1
            else:
                save_fn = save_dir / Path('background') / Path(f'{c_aud}_audio.jpg')
                save_fn.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_fn, bbox_inches='tight',pad_inches = 0)
                c_aud += 1
        #     break

        # for name in list(train_id_lookup[id]['images'].keys())[0:50]: 
        #     img = train_id_lookup[id]['images'][name][0]
        #     imgpath = image_base / img
        #     save_fn = save_dir / Path('few-shot_unlabelled_images') / Path(f'{c}_image.jpg')
        #     save_fn.parent.mkdir(parents=True, exist_ok=True)
        #     shutil.copyfile(imgpath, save_fn)
        #     c += 1

    # for i, img in enumerate(neg_images): 
    #     if i == 500: continue
    #     imgpath = image_base / img
    #     save_fn = save_dir / Path('background_images') / Path(f'{c_im}_image.jpg')
    #     save_fn.parent.mkdir(parents=True, exist_ok=True)
    #     shutil.copyfile(imgpath, save_fn)
    #     c_im += 1
        #     break
        # break