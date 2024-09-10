import torch
import torch.nn.functional as F

import staticdata.rawlandmarks


def surface_deformation_pos_loss(model_output, gt):
    gt_pos = gt['positions']
    gt_rgb = gt['v_rgb']
    uv_rgb = model_output['img_uv_rgb']

    # V_new = model_output['model_out']
    # # rgb_new = model_output['rgb_model_out']
    V_new = model_output['model_out'][:, :, :3]
    rgb_new = model_output['model_out'][:, :, 3:]
    rgb_front = rgb_new[:, staticdata.rawlandmarks.CaricShop3D.MAP_ORIG_TO_SLICED]

    embeddings = model_output['latent_vec']
    rgb_embeddings = model_output['rgb_latent_vec']

    data_constraint = torch.nn.functional.mse_loss(V_new, gt_pos)
    embeddings_constraint = torch.mean(embeddings ** 2)
    rgb_data_constraint = torch.nn.functional.mse_loss(rgb_new[:, :gt_rgb.shape[1], :], gt_rgb)
    rgb_embeddings_constraint = torch.mean(rgb_embeddings ** 2)
    map_img_constraint = torch.nn.functional.mse_loss(rgb_front, uv_rgb)  # 台湾文章新增loss

    # -----------------
    return {'data_constraint': data_constraint * 3e3, 
            'embeddings_constraint': embeddings_constraint.mean() * 1e6,
            'rgb_data_constraint': rgb_data_constraint * 3e3,
            'rgb_embeddings_constraint': rgb_embeddings_constraint.mean() * 1e6,
            'map_img_constraint': map_img_constraint,
            }
