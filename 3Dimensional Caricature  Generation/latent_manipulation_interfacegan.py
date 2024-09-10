import sys
import os
import traceback

sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import yaml
import tqdm
import io
import numpy as np
import dataset_caricshop3d as dataset
import training_loop_surface as training_loop
import utils, loss, modules, meta_modules

import torch
from torch.utils.data import DataLoader
import configargparse
from torch import nn
from surface_net import SurfaceDeformationField
from surface_deformation import create_mesh_single

from attr.process_attr import load_dict
from attr.train_boundary import _train_boundary
from attr.helper.manipulator import linear_interpolate

#----------------------------------------------------------------------------

def _build_map(PATH):
    """
    Build map: index --> path in the dataset
    """
    result = {}
    with open(PATH, 'r') as fin:
        lines = fin.readlines()
    for line in lines:
        tokens = line.split()
        value = ' '.join(tokens[:-2])
        key = int(tokens[-1]) - 1

        result[key] = value 

    return result
    
#----------------------------------------------------------------------------

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
p.add_argument('--attr_index', type=int, default=1, help='index of target attribute')
p.add_argument('--start_distance', type=float, default=-0.01, help='Start point for manipulation in latent space. (default: -3.0)')
p.add_argument('--end_distance', type=float, default=0.01, help='End point for manipulation in latent space. (default: 3.0)')
p.add_argument('--steps', type=int, default=11, help='Number of steps for image editing. (default: 10)')

# load configs
opt = p.parse_args()
meta_params = vars(opt)

# define DIF-Net
model = SurfaceDeformationField(531, **meta_params)
model.load_state_dict(torch.load(meta_params['checkpoint_path']))
# The network should be fixed for evaluation.

if hasattr(model, 'hyper_net'):
    for param in model.hyper_net.parameters():
        param.requires_grad = False
model.cuda()

# create save path
root_path = os.path.join(meta_params['logging_root'], meta_params['experiment_name'])
utils.cond_mkdir(root_path)

attr_dict = load_dict("./attr/attr_data/attr_dict.pkl")
attr_list = load_dict("./attr/attr_data/attr_list.pkl")

latent_codes = []
attr_scores = []

attr_index = opt.attr_index
print(f"A target attribute is {attr_list[attr_index]}.")

trainset_length = 531

db_path = './sort_info_new.txt'
map = _build_map(db_path)

for i in range(trainset_length):
    path_name = map[i]
    id_name = os.path.dirname(path_name)
    f_name = os.path.splitext(os.path.basename(path_name))[0]
    latent = model.latent_codes(torch.Tensor([i]).long().cuda())

    latent_codes.append(latent.detach().cpu().squeeze().numpy())
    attr_scores.append(attr_dict[id_name][f_name][attr_index])

latent_codes = np.array(latent_codes)
attr_scores = np.expand_dims(np.array(attr_scores), axis=1)

try:
    boundary = _train_boundary(f"./attr/attr_data/{attr_list[attr_index]}_boundary", latent_codes, attr_scores, split_ratio=0.8)
except Exception as e:
    print(traceback.format_exc())
    boundary = np.load(f'./attr/attr_data/{attr_list[attr_index]}_boundary/boundary.npy')

print(np.linalg.norm(boundary, ord=2))
caric = dataset.CaricShop3D(meta_params['dir_caricshop'], skip=True)
train_dataset = dataset.CaricShop3DTrain(meta_params['dir_caricshop'], 531, 0)

for i in [360]:
    dir_path = os.path.join(root_path, str(i))
    ckpt_path = os.path.join(dir_path, "checkpoints")
    utils.cond_mkdir(ckpt_path)
    latent = model.latent_codes(torch.Tensor([i]).long().cuda())
    latent = latent.detach().cpu().squeeze().unsqueeze(0).numpy()
    rgb_latent = model.rgb_latent_codes(torch.Tensor([i]).long().cuda())
    rgb_latent = rgb_latent.detach().cpu().squeeze().unsqueeze(0).numpy()
    interpolations = linear_interpolate(latent,
                                        boundary,
                                        start_distance=opt.start_distance,
                                        end_distance=opt.end_distance,
                                        steps=opt.steps)
    for j, intp in enumerate(interpolations) :
        distance = opt.start_distance + (opt.end_distance - opt.start_distance) * j / (opt.steps - 1)
        # print(f'train_dataset.ds.all_instances: {len(train_dataset.ds.all_instances)}\n'
        #       f'i: {i}\n')
        create_mesh_single(
            model,
            os.path.join(ckpt_path, f'{(i):04d}_{attr_list[attr_index]}_{distance:.4f}.obj'),
            torch.Tensor(caric.V_ref),
            caric.F,
            torch.Tensor(train_dataset.ds.all_instances[i].img).float().cuda().reshape(1, *train_dataset.ds.all_instances[i].img.shape),
            embedding=torch.Tensor(intp).cuda().reshape((1, 256)),
            rgb_embedding=torch.Tensor(rgb_latent).cuda().reshape((1, 256))
        )
