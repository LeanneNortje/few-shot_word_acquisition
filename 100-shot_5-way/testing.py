#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
# adapted from https://github.com/dharwath

import os
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from losses import compute_matchmap_similarity_matrix_loss
from dataloaders import *
from models.setup import *
from models.util import *
from models.multimodalModels import *
from models.GeneralModels import *
from evaluation.calculations import *
from training.util import *
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import time
from torchvision.io import read_image
from torchvision.models import *

BACKEND = "nccl"
INIT_METHOD = "tcp://localhost:54321"

def spawn_training(rank, world_size, image_base, args):

    # # Create dataloaders
    
    torch.manual_seed(42)

    if rank == 0: writer = SummaryWriter(args["exp_dir"] / "tensorboard")
    best_epoch, best_acc = 0, 0
    global_step, start_epoch = 0, 0
    info = {}
    loss_tracker = valueTracking()

    if rank == 0: heading(f'\nSetting up Audio model ')
    audio_model = mutlimodal(args).to(rank)

    if rank == 0: heading(f'\nSetting up image model ')
    seed_model = alexnet(pretrained=False)
    image_model = nn.Sequential(*list(seed_model.features.children()))

    last_layer_index = len(list(image_model.children()))
    image_model.add_module(str(last_layer_index),
        nn.Conv2d(256, args["audio_model"]["embedding_dim"], kernel_size=(3,3), stride=(1,1), padding=(1,1)))
    image_model = image_model.to(rank)

    if rank == 0: heading(f'\nSetting up attention model ')
    attention = ScoringAttentionModule(args).to(rank)

    if rank == 0: heading(f'\nSetting up contrastive loss ')
    contrastive_loss = ContrastiveLoss(args).to(rank)

    
    model_with_params_to_update = {
        "audio_model": audio_model,
        "attention": attention,
        "contrastive_loss": contrastive_loss,
        "image_model": image_model
        }
    model_to_freeze = {
        }
    trainable_parameters = getParameters(model_with_params_to_update, model_to_freeze, args)

    if args["optimizer"] == 'sgd':
        optimizer = torch.optim.SGD(
            trainable_parameters, args["learning_rate_scheduler"]["initial_learning_rate"],
            momentum=args["momentum"], weight_decay=args["weight_decay"]
            )
    elif args["optimizer"] == 'adam':
        optimizer = torch.optim.Adam(
            trainable_parameters, args["learning_rate_scheduler"]["initial_learning_rate"],
            weight_decay=args["weight_decay"]
            )
    else:
        raise ValueError('Optimizer %s is not supported' % args["optimizer"])

    # audio_model = DDP(audio_model, device_ids=[rank])
    # image_model = DDP(image_model, device_ids=[rank]) 
    # attention = DDP(attention, device_ids=[rank])

    checkpoint_fn = Path("pretrained") / Path("best_ckpt.pt")
    checkpoint = torch.load(checkpoint_fn, map_location={'cuda:%d' % 0: 'cuda:%d' % rank})
    print(checkpoint['image_model'].keys())

    print(image_model.state_dict().keys())
    




    # # if args["resume"] is False and args['cpc']['warm_start']: 
    # #     if rank == 0: print("Loading pretrained acoustic weights")
    # #     audio_model = loadPretrainedWeights(audio_model, args, rank)
        
    # if args["resume"]:

    #     if "restore_epoch" in args:
    #         info, start_epoch, global_step, best_epoch, best_acc = loadModelAttriburesAndTrainingAtEpochAMP(
    #             args["exp_dir"], audio_model, image_model, attention, contrastive_loss, optimizer, rank, args["restore_epoch"]
    #             )
    #         if rank == 0: print(f'\nEpoch particulars:\n\t\tepoch = {start_epoch}\n\t\tglobal_step = {global_step}\n\t\tbest_epoch = {best_epoch}\n\t\tbest_acc = {best_acc}\n')
    #     else:
    #         info, start_epoch, global_step, best_epoch, best_acc = loadModelAttriburesAndTrainingAMP(
    #             args["exp_dir"], audio_model, image_model, attention, contrastive_loss, optimizer, rank
    #             )
    #         if rank == 0: print(f'\nEpoch particulars:\n\t\tepoch = {start_epoch}\n\t\tglobal_step = {global_step}\n\t\tbest_epoch = {best_epoch}\n\t\tbest_acc = {best_acc}\n')
    # else:

        # print(f'Loading pretrained model...')
        # checkpoint_fn = Path("pretrained") / Path("best_ckpt.pt")
        # checkpoint = torch.load(checkpoint_fn, map_location={'cuda:%d' % 0: 'cuda:%d' % rank})
    
        # audio_model.load_state_dict(checkpoint["audio_model"])
        # image_model.load_state_dict(checkpoint["image_model"])
        # # attention.load_state_dict(checkpoint["attention"])
        # # contrastive_loss.load_state_dict(checkpoint["contrastive_loss"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--resume", action="store_true", dest="resume",
            help="load from exp_dir if True")
    parser.add_argument("--config-file", type=str, default='matchmap', choices=['matchmap'],
            help="Model config file.")
    parser.add_argument("--restore-epoch", type=int, default=-1, help="Epoch to resore training from.")
    parser.add_argument("--image-base", default="../..", help="Path to images.")
    command_line_args = parser.parse_args()

    # Setting up model specifics
    heading(f'\nSetting up model files ')
    args, image_base = modelSetup(command_line_args)

    world_size = torch.cuda.device_count()
    spawn_training(
        0, world_size, image_base, args
    )