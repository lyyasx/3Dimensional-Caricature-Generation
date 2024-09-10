import cyobj.io
import igl
import numpy as np
import time
import torch

def create_mesh(model, filename, V_ref, F, embedding=None, img=None, rgb_embedding=None):
    """
    From trained model and embeddings, create meshes
    """
    device = embedding.device
    coords = V_ref.to(device)
    img = img.reshape([1, *img.shape]).to(device)
    V, v_rgb = model.inference(coords, embedding, img, rgb_embedding)
    V = V.cpu().numpy()[0]
    v_rgb = v_rgb.cpu().numpy()[0][:V.shape[0], :]
    # igl.write_obj(filename, V, F)
    save_obj(filename, np.concatenate([V, v_rgb], axis=1), F)
    # cyobj.io.write_obj(filename, V.astype(np.double), F, VT.astype(np.double))


def save_obj(path, v, f):
    with open(path, 'w') as file:
        for i in range(len(v)):
            if len(v[i]) == 6:
                file.write('v %f %f %f %f %f %f\n' % (v[i][0], v[i][1], v[i][2], v[i][3], v[i][4], v[i][5]))
            elif len(v[i]) == 3:
                file.write('v %f %f %f\n' % (v[i][0], v[i][1], v[i][2]))
        file.write('\n')
        for i in range(len(f)):
            file.write('f %d %d %d\n' % (f[i, 0]+1, f[i, 1]+1, f[i, 2]+1))
    file.close()


def create_mesh_single(model, filename, V_ref, F, img, embedding=None, rgb_embedding=None):
    """
    From trained model and embeddings, create meshes
    """
    device = embedding.device
    coords = V_ref.to(device)
    V, v_rgb = model.inference(coords, embedding, img, rgb_embedding)
    V = V.cpu().numpy()[0]
    v_rgb = v_rgb.cpu().numpy()[0][:V.shape[0]]
    # igl.write_obj(filename, V, F)
    save_obj(filename, np.concatenate([V, v_rgb], axis=1), F)

