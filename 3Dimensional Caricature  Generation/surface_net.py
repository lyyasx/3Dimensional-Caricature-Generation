import torch
from torch import nn
import modules
from meta_modules import HyperNetwork
from loss import *
from new_siren.model import NeRFNetwork
from new_siren.multi_head_mapping import MultiHeadMappingNetwork
import staticdata

class SurfaceDeformationField(nn.Module):
    def __init__(self, num_instances, latent_dim=128, model_type='sine', hyper_hidden_layers=1,
                 num_hidden_layers=3, hyper_hidden_features=256,hidden_num=128, **kwargs):
        super().__init__()

        # latent code embedding for training subjects
        self.latent_dim = latent_dim
        self.latent_codes = nn.Embedding(num_instances, self.latent_dim)
        self.rgb_latent_codes = nn.Embedding(num_instances, self.latent_dim)
        nn.init.normal_(self.latent_codes.weight, mean=0, std=0.01)
        nn.init.normal_(self.rgb_latent_codes.weight, mean=0, std=0.01)

        # Deform-Net
        # self.deform_net=modules.SingleBVPNet(type=model_type,mode='mlp', hidden_features=hidden_num, num_hidden_layers=num_hidden_layers, in_features=3,out_features=3)
        # self.deform_net = NeRFNetwork()  # todo 新网络
        self.deform_net = NeRFNetwork(hidden_dim=hidden_num, style_dim=latent_dim)  # todo 新网络
        self.rgb_deform_net = NeRFNetwork(hidden_dim=hidden_num, style_dim=latent_dim, is_rgb=True)  # todo 新网络
        # Hyper-Net
        # self.hyper_net = HyperNetwork(hyper_in_features=self.latent_dim, hyper_hidden_layers=hyper_hidden_layers, hyper_hidden_features=hyper_hidden_features,
        #                               hypo_module=self.deform_net)
        self.hyper_net = MultiHeadMappingNetwork(latent_dim, hidden_num)  # todo 新网络
        self.rgb_hyper_net = MultiHeadMappingNetwork(latent_dim, hidden_num)  # todo 新网络
        print(self)

    def get_hypo_net_weights(self, model_input):
        instance_idx = model_input['instance_idx']
        embedding = self.latent_codes(instance_idx)
        hypo_params = self.hyper_net(embedding)
        return hypo_params, embedding

    def get_latent_code(self,instance_idx):
        embedding = self.latent_codes(instance_idx)
        return embedding

    # for generation
    def inference(self, coords, embedding, img, rgb_embedding=None):
        with torch.no_grad():
            model_in = {'coords': coords, 'img': img}
            hypo_params = self.hyper_net(embedding)  # todo
            # hypo_params_v = self.hyper_net(embedding[:, :128])
            # hypo_params_rgb = self.hyper_net(embedding[:, 128:])
            # hypo_params_v = self.hyper_net(embedding)
            # hypo_params_rgb = self.hyper_net(rgb_embedding)
            model_output = self.deform_net(model_in, params=hypo_params)
            # model_output = self.deform_net(model_in, params_v=hypo_params_v, params_rgb=hypo_params_rgb)
            rgb_hypo_params = self.rgb_hyper_net(rgb_embedding)
            rgb_model_output = self.rgb_deform_net(model_in, params=rgb_hypo_params)
            # todo 新网络
            if len(model_output['model_out'].shape) == 2:
                model_output['model_out'] = model_output['model_out'].reshape((1, *model_output['model_out'].shape))
            if len(rgb_model_output['rgb_model_out'].shape) == 2:
                rgb_model_output['rgb_model_out'] = rgb_model_output['rgb_model_out'].reshape((1, *rgb_model_output['rgb_model_out'].shape))
            deformation = model_output['model_out']
            v_rgb = rgb_model_output['rgb_model_out']
            new_coords = coords + deformation[:, :, :3]  # todo rgb 融合
            return new_coords, v_rgb  # todo rgb 融合
            # return new_coords, rgb_model_output['rgb_model_out']

    def inference_back(self, coords, embedding, img, rgb_embedding=None):
        model_in = {'coords': coords, 'img': img}
        # hypo_params = self.hyper_net(embedding)
        hypo_params_v = self.hyper_net(embedding)
        hypo_params_rgb = self.rgb_hyper_net(rgb_embedding)
        # model_output = self.deform_net(model_in, params=hypo_params)
        # model_output = self.deform_net(model_in, params_v=hypo_params_v, params_rgb=hypo_params_rgb)
        model_output = self.deform_net(model_in, params=hypo_params_v)
        rgb_model_output = self.rgb_deform_net(model_in, params=hypo_params_rgb)

        deformation = model_output['model_out']
        new_coords = coords + deformation[:, :, :3]
        v_rgb = rgb_model_output['rgb_model_out']
        return new_coords, v_rgb

    # for training
    def forward(self, model_input, gt):
        instance_idx = model_input['instance_idx']
        coords = model_input['coords'] # 3 dimensional input coordinates

        # get network weights for Deform-net using Hyper-net 
        embedding = self.latent_codes(instance_idx)
        rgb_embedding = self.rgb_latent_codes(instance_idx)
        hypo_params = self.hyper_net(embedding)
        rgb_hypo_params = self.rgb_hyper_net(rgb_embedding)

        model_output = self.deform_net(model_input, params=hypo_params)
        rgb_model_output = self.rgb_deform_net(model_input, params=rgb_hypo_params)
        # model_output = self.deform_net(model_input, params_v=hypo_params, params_rgb=rgb_hypo_params)
        displacement = model_output['model_out'].squeeze()
        displacement[:, :, :3] = coords + displacement[:, :, :3]  # deform into template space
        # displacement[:, :, :3] = coords + (displacement[:, :, :3] / 10)  # todo 变形缩小1000倍
        v_rgb = rgb_model_output['rgb_model_out'].squeeze()
        displacement = torch.cat([displacement, v_rgb], dim=2)

        # todo 使用相机参数增加映射
        front_v = displacement[:, staticdata.rawlandmarks.CaricShop3D.MAP_ORIG_TO_SLICED, :3]
        front_wv = torch.cat([front_v, torch.ones([*front_v.shape[:2], 1]).cuda()], 2).transpose(1, 2)
        v_xy = model_input['cam'] @ front_wv
        v_uv = torch.round(v_xy[:, :2] / v_xy[:, 2:]).long().transpose(1, 2)
        v_ind = (v_uv[..., 0] * 512 + v_uv[..., 1]).unsqueeze(-1).expand(-1, -1, 3)
        v_ind = torch.clamp(v_ind, 0, 512*512-1)
        ori_img = model_input['img']
        img_uv_rgb = torch.gather(ori_img.view(ori_img.shape[0], -1, ori_img.shape[-1]), 1, v_ind)

        model_out = {
            'model_in':model_output['model_in'],
            'model_out':displacement,
            'latent_vec':embedding,
            'img_uv_rgb': img_uv_rgb,  # todo as map img from generated 3d model
            # 'rgb_model_out': rgb_model_output['rgb_model_out'].squeeze(),
            'rgb_latent_vec': rgb_embedding,
            # 'hypo_params': hypo_params,
            # 'rbg_hypo_params': rgb_hypo_params,
        }

        losses = surface_deformation_pos_loss(model_out, gt)
        return losses

    # for evaluation
    def embedding(self, embed, model_input, gt, landmarks=None, dims_lmk=3):
        coords = model_input['coords'] # 3 dimensional input coordinates
        embedding = embed
        hypo_params = self.hyper_net(embedding)

        model_output = self.deform_net(model_input, params=hypo_params)
        displacement = model_output['model_out'].squeeze()
        V_new = coords + displacement # deform into template space

        model_out = {
            'model_in':model_output['model_in'],
            'model_out':V_new[:,:dims_lmk],
            'latent_vec':embedding, 
            'hypo_params':hypo_params}
        gt['positions'] = gt['positions'][:,:dims_lmk]

        losses = surface_deformation_pos_loss(model_out, gt)
        return losses

    # 编辑纹理使用
    def forward_for_edit_rgb(self, model_input, rgb_embedding):

        rgb_hypo_params = self.rgb_hyper_net(rgb_embedding)
        # model_input['img'] = model_input['img'].reshape(1, *model_input['img'].shape)
        rgb_model_output = self.rgb_deform_net(model_input, params=rgb_hypo_params)
        v_rgb = rgb_model_output['rgb_model_out'].squeeze()
        return v_rgb
