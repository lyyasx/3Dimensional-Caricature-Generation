import os

import configargparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from dataset_caricshop3d import CaricShop3D, CaricShop3DTrain
from surface_deformation import save_obj
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


class Evaluater:
    def __init__(self, trainer, data):
        self.trainer = trainer

        self.latent_dim = trainer.latent_dim
        self.latent_codes = trainer.latent_codes
        self.rgb_latent_codes = trainer.rgb_latent_codes
        self.latent_codes.requires_grad = False
        self.rgb_latent_codes.requires_grad = True

        self.deform_net = trainer.deform_net
        self.rgb_deform_net = trainer.rgb_deform_net
        self.hyper_net = trainer.hyper_net
        self.rgb_hyper_net = trainer.rgb_hyper_net
        self.deform_net.requires_grad = False
        self.rgb_deform_net.requires_grad = False
        self.hyper_net.requires_grad = False
        self.rgb_hyper_net.requires_grad = False

        self.data = data

        # for param in self.rgb_hyper_net.parameters():
        #     param.requires_grad = False

    def fitting_2d(self, img_idx, v_idx):
        torch.cuda.empty_cache()

        img_input, img_gt = self.data[img_idx]
        img_input, img_gt = img_input[0], img_gt[0]
        v_input, v_gt = self.data[v_idx]
        v_input, v_gt = v_input[0], v_gt[0]
        img_input['img'] = img_input['img'].reshape(1, *img_input['img'].shape)
        for k, v in img_input.items():
            img_input[k] = v.cuda()
        for k, v in img_gt.items():
            img_gt[k] = v.cuda()
        rgb_code = self.rgb_latent_codes(torch.tensor(v_idx).cuda())
        rgb_code = rgb_code.reshape(1, *rgb_code.shape)
        rgb_code = torch.nn.Parameter(rgb_code.cuda().requires_grad_())
        params = []
        params.append({'params': rgb_code, 'lr': 0.005})
        optimizer = torch.optim.Adam(params, betas=(0.9, 0.999))
        steps = 100
        loop = tqdm(range(steps))
        dir_path = f'{edit_rgb_dir}/{img_idx, v_idx}'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        for i in loop:

            rgb_loss = torch.tensor(0.0).cuda()
            optimizer.zero_grad()
            pre_v_rgb = self.trainer.forward_for_edit_rgb(img_input, rgb_code)[:img_gt['v_rgb'].shape[0]]

            rgb_loss += torch.abs(pre_v_rgb - img_gt['v_rgb']).mean()

            rgb_loss.backward()
            optimizer.step()

            if i != 0 and i % 10 == 0:
                v = np.concatenate([v_gt['positions'][:35200].detach().cpu().numpy(), pre_v_rgb.detach().cpu().numpy()], axis=1)
                save_obj(f'{dir_path}/{i}.obj', v, self.data.caricshop.F)

            loop.set_description('Step [{}/{}] Total Loss: {:.4f}'.format(i, steps, rgb_loss.item()))




if __name__ == '__main__':
    # load configs
    opt = p.parse_args()
    meta_params = vars(opt)

    model = SurfaceDeformationField(531, **meta_params)

    model.load_state_dict(torch.load(meta_params['checkpoint_path']))
    model.cuda()
    model.eval()

    trainset_length = 531
    num_samples = 17600
    edit_rgb_dir = 'edit_rgb_dir/1119_1'
    if not os.path.exists(edit_rgb_dir):
        os.makedirs(edit_rgb_dir)

    caric = CaricShop3DTrain(meta_params['dir_caricshop'], trainset_length, num_samples)

    # data_loader = DataLoader(caric, meta_params['batch_size'], True)

    evaluater = Evaluater(model, caric)

    idx_list = [94, 399]

    for i in idx_list:
        for j in idx_list:
            if i == j:
                continue
            evaluater.fitting_2d(j, i)
