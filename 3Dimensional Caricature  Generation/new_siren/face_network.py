# import tensorflow as tf
import torch
import numpy as np
import platform
from psbody.mesh import Mesh
from renderer import mesh_renderer
import math as m
is_windows = platform.system() == "Windows"


class carimodel():
	def __init__(self):
		model_path = './weights/cari_model/mean_face.obj'
		lmk_path = './weights/cari_model/best_68.txt'
		template = Mesh(filename=model_path)
		faces_by_vertex = [[] for i in range(len(template.v))]
		for i, face in enumerate(template.f):
			faces_by_vertex[face[0]].append(i)
			faces_by_vertex[face[1]].append(i)
			faces_by_vertex[face[2]].append(i)

		for i in range(len(faces_by_vertex)):
			if len(faces_by_vertex[i]) > 8:  # 有30个的数量为10
				faces_by_vertex[i].pop(-1)
				faces_by_vertex[i].pop(-1)
			while len(faces_by_vertex[i]) < 8:
				faces_by_vertex[i].append(faces_by_vertex[i][-1])
		self.point_buf = tf.constant(np.array(faces_by_vertex))  # (6144, 8)


		self.face_buf = tf.constant(template.f)
		self.front_mask_render= tf.squeeze(tf.constant(np.loadtxt('./weights/cari_model/small_index.txt')))
		self.mask_face_buf = tf.constant(Mesh(filename='./weights/cari_model/949_small.obj').f)
		self.keypoints = tf.squeeze(tf.constant(np.loadtxt(lmk_path)))


class Face3D():
	def __init__(self):
		self.facemodel = carimodel()

	# analytic 3D face reconstructions with coefficients from R-Net
	def Reconstruction_Block(self,coeff, opt, mean, std):
		self.mean = mean
		self.std = std
		shape_tex_coeff, angles,translation,camera_scale,f_scale = self.Split_coeff(coeff)
		vertice_pred, color_pred, face_shape, face_texture, mesh_color, mesh_color_r = self.decode(shape_tex_coeff, opt)


		# [batchsize,3,3] rotation matrix for face shape
		rotation = self.Compute_rotation_matrix(angles)
		face_shape_t = self.Rigid_transform_block(face_shape,rotation,translation)
		face_landmark_t = self.Compute_landmark(face_shape_t,self.facemodel)  # 68 lmk 3d
		landmark_p = self.Projection_block(face_landmark_t,camera_scale,f_scale) # 68 lmk 2d

		face_norm = self.Compute_norm(face_shape, self.facemodel)
		norm_r = tf.matmul(face_norm, rotation)

		#face_color = self.Illumination_block(face_texture, norm_r, gamma)

		render_imgs0, img_mask0, img_mask_crop0 = self.Render_block(face_shape_t, norm_r, face_texture[0], camera_scale,f_scale, self.facemodel, opt.batch_size, opt.is_train)
		render_imgs1, img_mask1, img_mask_crop1 = self.Render_block(face_shape_t, norm_r, face_texture[1], camera_scale,f_scale, self.facemodel, opt.batch_size, opt.is_train)
		render_imgs2, img_mask2, img_mask_crop2 = self.Render_block(face_shape_t, norm_r, face_texture[2], camera_scale, f_scale, self.facemodel, opt.batch_size, opt.is_train)
		render_imgs3, img_mask3, img_mask_crop3 = self.Render_block(face_shape_t, norm_r, face_texture[3], camera_scale,f_scale, self.facemodel, opt.batch_size, opt.is_train)

		render_imgs = tf.concat([render_imgs0, render_imgs1, render_imgs2, render_imgs3], axis=0)
		img_mask = tf.concat([img_mask0, img_mask1, img_mask2, img_mask3], axis=0)
		img_mask_crop = tf.concat([img_mask_crop0, img_mask_crop1, img_mask_crop2, img_mask_crop3], axis=0)

		self.shape_tex_coeff = shape_tex_coeff
		self.landmark_p = landmark_p
		self.vertice_pred = vertice_pred
		self.color_pred = color_pred
		self.mesh_color = mesh_color   #  四种固定风格  # [4, 16, 6144, 6]
		self.mesh_color_r = mesh_color_r  # 一种随机风格

		self.render_imgs = render_imgs
		self.img_mask = img_mask
		self.img_mask_crop = img_mask_crop
		#self.gamma = gamma

	#---------------------------------------------------------------------------------------------
	def style(self, onehot):
		fc1 = tf.layers.dense(onehot, 64, activation=tf.nn.relu, name='f_c-style1',reuse=tf.AUTO_REUSE)
		fc2 = tf.layers.dense(fc1, 128, activation=tf.nn.relu, name='f_c-style2',reuse=tf.AUTO_REUSE)
		fc2 = tf.cast(fc2, tf.float32)
		return fc2

	def decode(self, shape_tex_coeff, opt):
		coma = networks.coma(opt)
		z = shape_tex_coeff
		z_geo = z[:,:z.shape[1] // 2]
		z_tex = z[:, z.shape[1] // 2:]  # (16, 128)
		#print('self.std: ',self.std.shape)   # self.std:  (5, 6144, 3)
		vertice_pred = coma.decode(z_geo, name='decoder_geo')
		face_shape = vertice_pred*self.std[4]+self.mean[4]

		style_0 = tf.constant([[1.0,0.,0.,0.]])
		style_0 = tf.tile(input=style_0, multiples=[opt.batch_size,1])
		style_0 = self.style(style_0)
		z_tex_0 = z_tex + style_0
		tex_0 = coma.decode(z_tex_0, name='decoder_tex')
		face_texture0 = (tex_0*self.std[0]+self.mean[0])
		#print('sss', face_shape.shape,face_texture0.shape )


		style_1 = tf.constant([[0., 1., 0., 0.]])
		style_1 = tf.tile(input=style_1, multiples=[opt.batch_size, 1])
		style_1 = self.style(style_1)
		z_tex_1 = z_tex + style_1
		tex_1 = coma.decode(z_tex_1, name='decoder_tex')
		face_texture1 = (tex_1 * self.std[1] + self.mean[1])

		style_2 = tf.constant([[0., 0., 1., 0.]])
		style_2 = tf.tile(input=style_2, multiples=[opt.batch_size, 1])
		style_2 = self.style(style_2)
		z_tex_2 = z_tex + style_2
		tex_2 = coma.decode(z_tex_2, name='decoder_tex')
		face_texture2 = (tex_2 * self.std[2] + self.mean[2])

		style_3 = tf.constant([[0., 0., 0., 1.]])
		style_3 = tf.tile(input=style_3, multiples=[opt.batch_size, 1])
		style_3 = self.style(style_3)
		z_tex_3 = z_tex + style_3
		tex_3 = coma.decode(z_tex_3, name='decoder_tex')
		face_texture3 = (tex_3 * self.std[3] + self.mean[3])

		color_pred = tf.concat([tex_0, tex_1, tex_2, tex_3], axis= 0)
		face_texture_ = tf.stack([face_texture0, face_texture1, face_texture2, face_texture3], axis=0)
		#face_texture_ = tf.clip_by_value(face_texture_, 0, 1)
		face_texture = face_texture_*255



		#print('face_shape:', face_shape.shape) #  (16, 6144, 3)
		#print('face_texture3',face_texture3.shape)
		#print('face_texture_[0]',face_texture_[0].shape) #  (16, 6144, 3)

		mesh_color_0 = tf.concat([face_shape,face_texture_[0]], axis=2)
		mesh_color_1 = tf.concat([face_shape, face_texture_[1]], axis=2)
		mesh_color_2 = tf.concat([face_shape, face_texture_[2]], axis=2)
		mesh_color_3 = tf.concat([face_shape, face_texture_[3]], axis=2)
		mesh_color = tf.stack([mesh_color_0,mesh_color_1, mesh_color_2, mesh_color_3], axis=0)  # [4, 16, 6144, 6]

		self.regularization = coma.regularizers

		style_r = tf.random_normal([1,4], name='style_of_texture')
		#style_r = tf.constant([[0., 1., 0.,0.]])  #############################################################################################
		#with tf.Session() as sess:
		#	d = sess.run(style_r)
		#	print(d)


		style_r = tf.tile(input=style_r, multiples=[opt.batch_size, 1])
		#print('style_r', style_r.shape)   # (16, 4)
		style_r = self.style(style_r)
		z_tex_r = z_tex + style_r
		tex_r = coma.decode(z_tex_r, name='decoder_tex')
		face_texture_r = (tex_r * self.std[0] + self.mean[0])
		#face_texture_r = tf.clip_by_value(face_texture_r, 0, 1)
		mesh_color_r = tf.concat([face_shape, face_texture_r], axis=2)

		return vertice_pred, color_pred, face_shape, face_texture, mesh_color, mesh_color_r


	def Compute_landmark(self,face_shape,facemodel):
		# compute 3D landmark postitions with pre-computed 3D face shape
		keypoints_idx = facemodel.keypoints
		#keypoints_idx = tf.cast(keypoints_idx - 1,tf.int32)    #*********************************
		keypoints_idx = tf.cast(keypoints_idx-1, tf.int32)
		face_landmark = tf.gather(face_shape,keypoints_idx,axis = 1)

		return face_landmark
	def Projection_block(self,face_shape,camera_scale,f_scale):
		# pre-defined camera focal for pespective projection
		focal = tf.constant(1015.0)
		focal = focal*f_scale
		focal = tf.reshape(focal,[-1,1])
		batchsize = tf.shape(focal)[0]

		# define camera position
		camera_pos = tf.reshape(tf.constant([0.0,0.0,10.0]),[1,1,3])*tf.reshape(camera_scale,[-1,1,1])
		reverse_z = tf.tile(tf.reshape(tf.constant([1.0,0,0,0,1,0,0,0,-1.0]),[1,3,3]),[tf.shape(face_shape)[0],1,1])

		# compute projection matrix
		p_matrix = tf.concat([focal,tf.zeros([batchsize,1]),112.*tf.ones([batchsize,1]),tf.zeros([batchsize,1]),focal,112.*tf.ones([batchsize,1]),tf.zeros([batchsize,2]),tf.ones([batchsize,1])],axis = 1)
		p_matrix = tf.reshape(p_matrix,[-1,3,3])

		# convert z in world space to the distance to camera
		face_shape = tf.matmul(face_shape,reverse_z) + camera_pos
		aug_projection = tf.matmul(face_shape,tf.transpose(p_matrix,[0,2,1]))

		# [batchsize, N,2] 2d face projection
		face_projection = aug_projection[:,:,0:2]/tf.reshape(aug_projection[:,:,2],[tf.shape(face_shape)[0],tf.shape(aug_projection)[1],1])
		return face_projection

	def Compute_rotation_matrix(self,angles):
		n_data = tf.shape(angles)[0]
		# compute rotation matrix for X-axis, Y-axis, Z-axis respectively
		rotation_X = tf.concat([tf.ones([n_data,1]),
			tf.zeros([n_data,3]),
			tf.reshape(tf.cos(angles[:,0]),[n_data,1]),
			-tf.reshape(tf.sin(angles[:,0]),[n_data,1]),
			tf.zeros([n_data,1]),
			tf.reshape(tf.sin(angles[:,0]),[n_data,1]),
			tf.reshape(tf.cos(angles[:,0]),[n_data,1])],
			axis = 1
			)
		rotation_Y = tf.concat([tf.reshape(tf.cos(angles[:,1]),[n_data,1]),
			tf.zeros([n_data,1]),
			tf.reshape(tf.sin(angles[:,1]),[n_data,1]),
			tf.zeros([n_data,1]),
			tf.ones([n_data,1]),
			tf.zeros([n_data,1]),
			-tf.reshape(tf.sin(angles[:,1]),[n_data,1]),
			tf.zeros([n_data,1]),
			tf.reshape(tf.cos(angles[:,1]),[n_data,1])],
			axis = 1
			)
		rotation_Z = tf.concat([tf.reshape(tf.cos(angles[:,2]),[n_data,1]),
			-tf.reshape(tf.sin(angles[:,2]),[n_data,1]),
			tf.zeros([n_data,1]),
			tf.reshape(tf.sin(angles[:,2]),[n_data,1]),
			tf.reshape(tf.cos(angles[:,2]),[n_data,1]),
			tf.zeros([n_data,3]),
			tf.ones([n_data,1])],
			axis = 1
			)
		rotation_X = tf.reshape(rotation_X,[n_data,3,3])
		rotation_Y = tf.reshape(rotation_Y,[n_data,3,3])
		rotation_Z = tf.reshape(rotation_Z,[n_data,3,3])

		# R = RzRyRx
		rotation = tf.matmul(tf.matmul(rotation_Z,rotation_Y),rotation_X)
		rotation = tf.transpose(rotation, perm = [0,2,1])
		return rotation

	def Rigid_transform_block(self, face_shape, rotation, translation):
		# do rigid transformation for 3D face shape
		face_shape_r = tf.matmul(face_shape, rotation)
		face_shape_t = face_shape_r + tf.reshape(translation, [tf.shape(face_shape)[0], 1, 3])
		return face_shape_t

	def Split_coeff(self,coeff):
		shape_tex_coeff = coeff[:,:256]
		angles = coeff[:,256:259]  #3
		#gamma = coeff[:,259:286]  #27
		translation = coeff[:,259:262]
		#gamma = coeff[:,262:289]
		camera_scale = tf.ones([tf.shape(coeff)[0],1])
		f_scale = tf.ones([tf.shape(coeff)[0],1])
		return shape_tex_coeff,angles,translation,camera_scale,f_scale#,gamma

	def Compute_norm(self, face_shape, facemodel):
		shape = face_shape
		face_id = facemodel.face_buf
		point_id = facemodel.point_buf

		# face_id and point_id index starts from 1
		face_id = tf.cast(face_id, tf.int32)  # tf.cast 数据类型转换
		point_id = tf.cast(point_id, tf.int32)

		# compute normal for each face     # tf.gather 从params的axis维根据indices的参数值获取切片
		v1 = tf.gather(shape, face_id[:, 0], axis=1)  # 一个face的三个顶点, F个face的第一个顶点
		v2 = tf.gather(shape, face_id[:, 1], axis=1)  # F个face的第二个顶点
		v3 = tf.gather(shape, face_id[:, 2], axis=1)
		e1 = v1 - v2
		e2 = v2 - v3
		face_norm = tf.cross(e1, e2)

		face_norm = tf.nn.l2_normalize(face_norm, dim=2)  # normalized face_norm first
		face_norm = tf.concat([face_norm, tf.zeros([tf.shape(face_shape)[0], 1, 3])], axis=1)  # 在第一维 拼接张量

		# compute normal for each vertex using one-ring neighborhood
		v_norm = tf.reduce_sum(tf.gather(face_norm, point_id, axis=1), axis=2)
		v_norm = tf.nn.l2_normalize(v_norm, dim=2)

		return v_norm

	def Render_block(self,face_shape,face_norm,face_color,camera_scale,f_scale,facemodel,batchsize,is_train=True):
		if is_train and is_windows:
			raise ValueError('Not support training with Windows environment.')

		if is_windows:
			return [],[],[]

		# render reconstruction images
		# n_vex = int(facemodel.idBase.shape[0].value/3)
		n_vex = 6144                                           #*********************************
		fov_y = 2*tf.atan(112./(1015.*f_scale))*180./m.pi
		fov_y = tf.reshape(fov_y,[batchsize])
		# full face region
		face_shape = tf.reshape(face_shape,[batchsize,n_vex,3])
		face_norm = tf.reshape(face_norm,[batchsize,n_vex,3])
		face_color = tf.reshape(face_color,[batchsize,n_vex,3])

		# pre-defined cropped face region   # 是否需要减去1 呢
		mask_face_shape = tf.gather(face_shape, tf.cast(facemodel.front_mask_render-1 , tf.int32), axis=1)
		mask_face_norm = tf.gather(face_norm, tf.cast(facemodel.front_mask_render-1, tf.int32), axis=1)
		mask_face_color = tf.gather(face_color, tf.cast(facemodel.front_mask_render-1, tf.int32), axis=1)

		# setting cammera settings
		camera_position = tf.constant([[0,0,10.0]])*tf.reshape(camera_scale,[-1,1])
		camera_lookat = tf.constant([0,0,0.0])
		camera_up = tf.constant([0,1.0,0])

		# setting light source position(intensities are set to 0 because we have computed the vertex color)
		light_positions = tf.tile(tf.reshape(tf.constant([0,0,1e5]),[1,1,3]),[batchsize,1,1])
		light_intensities = tf.tile(tf.reshape(tf.constant([0.0,0.0,0.0]),[1,1,3]),[batchsize,1,1])
		ambient_color = tf.tile(tf.reshape(tf.constant([1.0,1,1]),[1,3]),[batchsize,1])

		#using tf_mesh_renderer for rasterization (https://github.com/google/tf_mesh_renderer)
		# img: [batchsize,224,224,3] images in RGB order (0-255)
		# mask:[batchsize,224,224,1] transparency for img ({0,1} value)
		img_rgba = mesh_renderer.mesh_renderer(face_shape,  # vertices,
			tf.cast(facemodel.face_buf,tf.int32),  # triangles               #*********************************
			face_norm,  # normals
			face_color,  # diffuse_colors: 3D float32 tensor with shape[b,v,3].The RGB diffuse reflection in the range [0,1] for each vertex
			camera_position = camera_position,
			camera_lookat = camera_lookat,
			camera_up = camera_up,
			light_positions = light_positions,
			light_intensities = light_intensities,
			image_width = 224,
			image_height = 224,
			fov_y = fov_y,
			near_clip = 0.01,
			far_clip = 50.0,
			ambient_color = ambient_color)

		img = img_rgba[:,:,:,:3]
		mask = img_rgba[:,:,:,3:]

		#img = tf.cast(img[:,:,:,::-1],tf.float32) #transfer RGB to BGR, tf.cast:类型转换
		mask = tf.cast(mask,tf.float32) # full face region

		if is_train:
			# compute mask for small face region
			img_crop_rgba = mesh_renderer.mesh_renderer(mask_face_shape,
														tf.cast(facemodel.mask_face_buf, tf.int32),
														mask_face_norm,
														mask_face_color,
														camera_position=camera_position,
														camera_lookat=camera_lookat,
														camera_up=camera_up,
														light_positions=light_positions,
														light_intensities=light_intensities,
														image_width=224,
														image_height=224,
														fov_y=fov_y,
														near_clip=0.01,
														far_clip=50.0,
														ambient_color=ambient_color)
			mask_f = mask                              #*********************************
			mask_f = img_crop_rgba[:, :, :, 3:]
			mask_f = tf.cast(mask_f, tf.float32)  # small face region
			return img,mask,mask_f

		img_rgba = tf.cast(tf.clip_by_value(img_rgba,0,255),tf.float32)
		#tf.clip_by_value ()函数可以将一个张量中的数值限制在一个范围之内

		return img_rgba,mask,mask


	def Illumination_block(self,face_texture,norm_r,gamma):
		n_data = tf.shape(gamma)[0]
		n_point = tf.shape(norm_r)[1]
		gamma = tf.reshape(gamma,[n_data,3,9])
		# set initial lighting with an ambient lighting
		init_lit = tf.constant([0.8,0,0,0,0,0,0,0,0])
		gamma = gamma + tf.reshape(init_lit,[1,1,9])

		# compute vertex color using SH function approximation
		a0 = m.pi
		a1 = 2*m.pi/tf.sqrt(3.0)
		a2 = 2*m.pi/tf.sqrt(8.0)
		c0 = 1/tf.sqrt(4*m.pi)
		c1 = tf.sqrt(3.0)/tf.sqrt(4*m.pi)
		c2 = 3*tf.sqrt(5.0)/tf.sqrt(12*m.pi)

		Y = tf.concat([tf.tile(tf.reshape(a0*c0,[1,1,1]),[n_data,n_point,1]),
			tf.expand_dims(-a1*c1*norm_r[:,:,1],2),
			tf.expand_dims(a1*c1*norm_r[:,:,2],2),
			tf.expand_dims(-a1*c1*norm_r[:,:,0],2),
			tf.expand_dims(a2*c2*norm_r[:,:,0]*norm_r[:,:,1],2),
			tf.expand_dims(-a2*c2*norm_r[:,:,1]*norm_r[:,:,2],2),
			tf.expand_dims(a2*c2*0.5/tf.sqrt(3.0)*(3*tf.square(norm_r[:,:,2])-1),2),
			tf.expand_dims(-a2*c2*norm_r[:,:,0]*norm_r[:,:,2],2),
			tf.expand_dims(a2*c2*0.5*(tf.square(norm_r[:,:,0])-tf.square(norm_r[:,:,1])),2)],axis = 2)

		color_r = tf.squeeze(tf.matmul(Y,tf.expand_dims(gamma[:,0,:],2)),axis = 2)  # tf.squeeze 移除大小为1的维度
		color_g = tf.squeeze(tf.matmul(Y,tf.expand_dims(gamma[:,1,:],2)),axis = 2) # expand_dims 插入尺寸索引处的1维axis的input的形状
		color_b = tf.squeeze(tf.matmul(Y,tf.expand_dims(gamma[:,2,:],2)),axis = 2)

		#[batchsize,N,3] vertex color in RGB order
		face_color = tf.stack([color_r*face_texture[:,:,0],color_g*face_texture[:,:,1],color_b*face_texture[:,:,2]],axis = 2)

		return face_color