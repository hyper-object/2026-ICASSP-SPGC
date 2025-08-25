import os 
import pickle
import argparse
import numpy as np 
import matplotlib.pyplot as plt 

from config import track2_cfg as config
from utils.track2 import utils, torchmodel, metrics

from datasets.hyper_object import HyperObjectDataset
from datasets.base import JointTransform
from datasets.transform import random_flip

from baselines import mstpp_up

import torch
import torchinfo
from torchvision import transforms
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# python track2_train.py --data_dir /mnt/data/2026-Hyper-Object-Data --model_name MST_Plus_Plus_Up

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type = str, default = 'E:/hyper-skin-data/Hyper-Skin(MSI, NIR)', required=True, help = 'data directory')
parser.add_argument('--camera_type', type = str, default = 'CIE', required=False, help = 'camera response function used to generate the RGB')
parser.add_argument('--model_name', type = str, default = 'MST_Plus_Plus', required=True, help = 'the model')
parser.add_argument('--saved_dir', type = str, default = 'runs/track2/saved-models', required=False, help = 'directory to save the trained model')
parser.add_argument('--logged_dir', type = str, default = 'runs/track2/log', required=False, help = 'directory to save the results')
parser.add_argument('--reconstructed_dir', type = str, default = 'runs/track2/reconstructed-hsi', required=False, help = 'directory to save the reconstructed HSI')
parser.add_argument('--saved_predicted', type = bool, default = True, required=False, help = 'whether to save the predicted HSI')
parser.add_argument('--external_dir', type = str, default = None, required=False, help = 'create and save the log, reconstruction and trained model to external directory')


if __name__ == '__main__':
    args = parser.parse_args()


    ds_train = HyperObjectDataset(
        data_root=f'{args.data_dir}/track2',
        track=2,
        train=True,
        transforms=JointTransform(random_flip),
    )

    ds_val = HyperObjectDataset(
        data_root=f'{args.data_dir}/track2',
        track=2,
        train=False,
    )

    
    # create the saved  and log folders if they not exist
    exp_logged_dir, exp_saved_dir, exp_reconstructed_dir = utils.create_folders_for(
                                                            saved_dir = args.saved_dir, 
                                                            logged_dir = args.logged_dir, 
                                                            reconstructed_dir = args.reconstructed_dir,
                                                            model_name = args.model_name,
                                                            camera_type = args.camera_type,
                                                            external_dir = args.external_dir)
    logger = utils.initiate_logger(f'{exp_logged_dir}/log', 'train')
    #################################################################


    # define the dataloaders
    train_loader = torch.utils.data.DataLoader(
                                    dataset = ds_train, 
                                    batch_size = config.batch_size, 
                                    shuffle = True, 
                                    pin_memory = True, 
                                    drop_last = True,
                                    num_workers = 0)

    valid_loader = torch.utils.data.DataLoader(
                                    dataset = ds_val, 
                                    batch_size = 1, 
                                    shuffle = False, 
                                    num_workers = 2,
                                    pin_memory = False)
    

    #################################################################
    # define the model
    if args.model_name == 'MST_Plus_Plus_Up':
        print("Late Upsampling Strategy")
        model_architecture = mstpp_up.MST_Plus_Plus_LateUpsample(
            in_channels=3, out_channels=61, n_feat=61, stage=3, upscale_factor=2
        )

            
    total_model_parameters = sum(p.numel() for p in model_architecture.parameters())

    msg = f"[Experiment Metadata]:\n" +\
            f"Model: {args.model_name}\n" +\
            f"Total Model Parameters: {total_model_parameters}\n" +\
            f"Trained Model will be saved at: {exp_saved_dir}\n" +\
            f"Log file available at: {exp_logged_dir}\n"+\
            "==================================================================================================================="
    logger.info(msg)
    print(msg)

    optimizer = torch.optim.Adam(model_architecture.parameters(), lr=config.init_lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader), eta_min=1e-6)
    loss_fn = torch.nn.L1Loss()

    model = torchmodel.create(
            model_architecture = model_architecture,
            device=device,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs = config.epochs,
            logger=logger,
            model_saved_path=f'{exp_saved_dir}/model.pt'
    )

    # training
    best_valid_loss, history = model.train(
        train_loader = train_loader,
        valid_loader = valid_loader,
        best_valid_loss = 100,
    )
