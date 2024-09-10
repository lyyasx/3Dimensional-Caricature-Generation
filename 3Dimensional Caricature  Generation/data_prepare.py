import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import igl
import meshplot as mp
from PIL import Image
from pylab import imshow, plot


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


def get_color_mesh(name, no, mesh_dir, tgt_dir):
    try:
        img = Image.open(f'dataset/RGB3DCaricShop_raw_t_mesh_plus/processedData/img/{name}/{no}.jpg')
    except:
        img = Image.open(f'dataset/RGB3DCaricShop_raw_t_mesh_plus/processedData/img/{name}/{no.lower()}.jpg')
    # v, f = igl.read_triangle_mesh(f'dataset/3DCaricShop/processedData/rawMesh/{name_no}.obj')
    try:
        # v, f = igl.read_triangle_mesh(f'dataset/3DCaricShop/processedData/{mesh_type}Mesh/{name}/{no}.obj')
        v, f = igl.read_triangle_mesh(f'{mesh_dir}/{name}/{no}.obj')
    except:
        # v, f = igl.read_triangle_mesh(f'dataset/3DCaricShop/processedData/{mesh_type}Mesh/{name}/{no.lower()}.obj')
        v, f = igl.read_triangle_mesh(f'{mesh_dir}/{name}/{no.lower()}.obj')
    wv = np.row_stack([v.T, np.ones([len(v)])])
    try:
        cam = sio.loadmat(f'dataset/RGB3DCaricShop_raw_t_mesh_plus/processedData/calib/{name}/{no}.mat')
    except:
        cam = sio.loadmat(f'dataset/RGB3DCaricShop_raw_t_mesh_plus/processedData/calib/{name}/{no.lower()}.mat')

    ext = cam['Ext']
    int_p = cam['Int']
    inp = np.column_stack([np.row_stack([int_p, np.array([0., 0., 1.])]), np.zeros([3])])
    zc_uv = inp @ ext @ wv
    uv = zc_uv / zc_uv[2]
    # imshow(img)
    # plot(uv[0], uv[1], 'r.')
    v_list = []
    for i in range(len(v)):
        try:
            vc = np.clip(img.getpixel((int(uv[0, i]), int(uv[1, i]))), 0, 255)
            vct = list(vc/255)
            v_list.append([*list(v[i]), *vct])
        except:
            v_list.append([*list(v[i]), 0., 0., 0.])
    save_obj(f'{tgt_dir}/{name}/{no}.obj', np.array(v_list), f)


if __name__ == '__main__':
    # mesh_type = 't'
    data_base_dir = './dataset/RGB3DCaricShop_raw_t_mesh_plus/processedData'
    mesh_dir = './raw_t_mesh_plus'
    tgt_dir = './raw_t_mesh_plus_rgb'
    # name_list = os.listdir(f'{data_base_dir}/img')
    name_list = os.listdir(mesh_dir)
    for name in name_list:
        print(f'INFO: dealing {name}...')
        name_dir = f'{tgt_dir}/{name}'
        if not os.path.exists(name_dir):
            os.makedirs(name_dir)
        # no_list = [item[-12:-4] for item in os.listdir(f'{data_base_dir}/img/{name}')]
        no_list = [item[-12:-4] for item in os.listdir(f'{mesh_dir}/{name}')]
        for no in no_list:
            try:
                get_color_mesh(name, no.upper(), mesh_dir, tgt_dir)
            except Exception as e:
                print(f'WARN: generate {name}/{no}.obj failed, detail: {e}')
