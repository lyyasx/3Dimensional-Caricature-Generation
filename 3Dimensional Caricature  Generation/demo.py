import os

import configargparse
import torch
from torch.utils.data import DataLoader

import utils
from dataset_caricshop3d import CaricShop3D, CaricShop3DTrain
from surface_net import SurfaceDeformationField
from training_loop_surface import train

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

p = configargparse.ArgumentParser()
p.add_argument('--config', required=True, is_config_file=True, help='Evaluation configuration')
p.add_argument('--dir_caricshop', type=str,default='./3dcaricshop', help='3DCaricShop dataset root')
p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--summary_root', type=str, default='./summaries', help='root for summary')
p.add_argument('--checkpoint_path', type=str, default='', help='checkpoint to use for eval')
p.add_argument('--experiment_name', type=str, default='default',
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=256, help='training batch size.')
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-4')
p.add_argument('--epochs', type=int, default=8000, help='Number of epochs to train for.')

p.add_argument('--epochs_til_checkpoint', type=int, default=10,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=100,
               help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument('--model_type', type=str, default='sine',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')

p.add_argument('--latent_dim', type=int,default=128, help='latent code dimension.')
p.add_argument('--hidden_num', type=int,default=128, help='hidden layer dimension of deform-net.')
p.add_argument('--num_hidden_layers', type=int,default=3, help='number of hidden layers of deform-net.')
p.add_argument('--hyper_hidden_layers', type=int,default=1, help='number of hidden layers hyper-net.')
p.add_argument('--start_distance', type=float, default=-0.01, help='Start point for manipulation in latent space. (default: -3.0)')
p.add_argument('--end_distance', type=float, default=0.01, help='End point for manipulation in latent space. (default: 3.0)')
p.add_argument('--steps', type=int, default=11, help='Number of steps for image editing. (default: 10)')
p.add_argument('--model_dir', type=str, default='', help='model_dir')
p.add_argument('--summary_dir', type=str, default='', help='summary_dir')

# load configs
opt = p.parse_args()
meta_params = vars(opt)

# define DIF-Net
model = SurfaceDeformationField(531, **meta_params)
if meta_params['checkpoint_path']:
    print(f'load checkpoint_path: {meta_params["checkpoint_path"]}...')
    model.load_state_dict(torch.load(meta_params['checkpoint_path']))

def compute_model_para(mdl):
    import numpy as np
    total_params = 0
    trainable_params = 0
    not_trainable_params = 0
    for param in mdl.parameters():
        mul_value = np.prod(param.size())
        total_params += mul_value
        if param.requires_grad:
            trainable_params += mul_value
        else:
            not_trainable_params += mul_value
    print(f'total_params: {total_params}\n'
          f'trainable_params: {trainable_params}\n'
          f'not_trainable_params: {not_trainable_params}')
    return total_params, trainable_params, not_trainable_params

model.cuda()
model.train()

# create save path
root_path = os.path.join(meta_params['logging_root'], meta_params['experiment_name'])
utils.cond_mkdir(root_path)

latent_codes = []
attr_scores = []

trainset_length = 531
num_samples = 17600

# db_path = './sort_info.txt'
# map = _build_map(db_path)

caric = CaricShop3DTrain(meta_params['dir_caricshop'], trainset_length, num_samples)

data_loader = DataLoader(caric, meta_params['batch_size'], True)

train(model, data_loader, **meta_params)
