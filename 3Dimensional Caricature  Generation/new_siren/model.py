import logging

import torch
from einops import rearrange
from tl2 import tl2_utils
from tl2.proj.pytorch import init_func, torch_utils
from torch import nn

from new_siren import film_layer, nerf_network


class NeRFNetwork(nn.Module):
  """Same architecture as TALLSIREN but adds a UniformBoxWarp to map input points to -1, 1"""

  def __repr__(self):
    return tl2_utils.get_class_repr(self)

  def __init__(self,
               in_dim=3,
               hidden_dim=128,  # todo rgb 原128，增加rgb潜码空间后 256
               hidden_layers=2,
               style_dim=128,  # todo 原维度 128，增加rgb潜码空间后 256
               rgb_dim=3,
               device=None,
               name_prefix='nerf',
               is_rgb=False,  # todo 是否针对rgb训练
               **kwargs):
    """

    :param z_dim:
    :param hidden_dim:
    :param rgb_dim:
    :param device:
    :param kwargs:
    """
    super().__init__()

    self.repr_str = tl2_utils.dict2string(dict_obj={
      'in_dim': in_dim,
      'hidden_dim': hidden_dim,
      'hidden_layers': hidden_layers,
      'style_dim': style_dim,
      'rgb_dim': rgb_dim,
    })

    self.device = device
    self.in_dim = in_dim
    self.hidden_dim = hidden_dim
    self.rgb_dim = rgb_dim
    self.style_dim = style_dim
    self.hidden_layers = hidden_layers
    self.name_prefix = name_prefix
    self.is_rgb = is_rgb
    if self.is_rgb:  # todo rgb
      self.rgb_conv = torch.nn.Conv2d(3, 3, 3, 1, 1)
      self.rgb_pool = torch.nn.MaxPool2d(4, 4)
      # self.rgb_linear = torch.nn.Linear(128*128, 11551 + 11581)  # todo 增加num_samples
      self.rgb_linear = torch.nn.Linear(128*128, 35200 + 17600)  # plus todo 增加num_samples eye_data
      # self.rgb_linear = torch.nn.Linear(128*128, 34649)  # todo plus_rgb

    # self.xyz_emb = pigan_utils.PosEmbedding(max_logscale=9, N_freqs=10)
    # dim_xyz_emb = self.xyz_emb.get_out_dim()
    # self.dir_emb = pigan_utils.PosEmbedding(max_logscale=3, N_freqs=4)
    # dim_dir_emb = self.dir_emb.get_out_dim()

    self.module_name_list = []

    self.style_dim_dict = {}

    # self.network = nn.ModuleList([
    #   FiLMLayer(3, hidden_dim),
    #   # FiLMLayer(dim_xyz_emb, hidden_dim),
    #   FiLMLayer(hidden_dim, hidden_dim),
    #   # FiLMLayer(hidden_dim, hidden_dim),
    #   # FiLMLayer(hidden_dim, hidden_dim),
    #   # FiLMLayer(hidden_dim, hidden_dim),
    #   # FiLMLayer(hidden_dim, hidden_dim),
    #   # FiLMLayer(hidden_dim, hidden_dim),
    #   # FiLMLayer(hidden_dim, hidden_dim),
    # ])

    self.network = nn.ModuleList()
    self.module_name_list.append('network')
    _out_dim = in_dim
    for idx in range(hidden_layers):
      _in_dim = _out_dim
      _out_dim = hidden_dim

      _layer = film_layer.FiLMLayer(in_dim=_in_dim, out_dim=_out_dim, style_dim=style_dim, use_style_fc=True)

      self.network.append(_layer)
      self.style_dim_dict[f'{name_prefix}_w{idx}'] = _layer.style_dim
    # TODO style_dict divide w0 w1 用于坐标映射
    # self.network.append(
    #   film_layer.FiLMLayer(in_dim=in_dim, out_dim=hidden_dim, style_dim=style_dim, use_style_fc=True)
    # )
    # self.network.append(
    #   film_layer.FiLMLayer(in_dim=hidden_dim, out_dim=hidden_dim, style_dim=style_dim, use_style_fc=True)
    # )

    # self.final_layer = nn.Linear(hidden_dim, 1)
    # self.final_layer.apply(frequency_init(25))
    # self.module_name_list.append('final_layer')

    _in_dim= hidden_dim
    _out_dim = hidden_dim // 2
    self.color_layer_sine = film_layer.FiLMLayer(in_dim=_in_dim, out_dim=_out_dim, style_dim=style_dim, use_style_fc=True)
    # todo style_dict divide color_layer用于rgb
    # self.color_layer_sine = film_layer.FiLMLayer(in_dim=in_dim, out_dim=hidden_dim, style_dim=style_dim, use_style_fc=True)
    # self.color_layer_sine = FiLMLayer(in_dim=_in_dim, out_dim=_out_dim, style_dim=style_dim, use_style_fc=True)
    self.style_dim_dict[f'{name_prefix}_rgb'] = self.color_layer_sine.style_dim
    self.module_name_list.append('color_layer_sine')

    self.color_layer_linear = nn.Sequential(
      # nn.Linear(_out_dim, rgb_dim),
      # todo style_dict divide
      nn.Linear(hidden_dim, rgb_dim),
      # nn.LeakyReLU(0.2, inplace=True),
      # nn.Sigmoid()
    )
    # self.color_layer_linear.apply(frequency_init(25))
    self.color_layer_linear.apply(init_func.kaiming_leaky_init)
    self.module_name_list.append('color_layer_linear')

    self.dim_styles = sum(self.style_dim_dict.values())

    # Don't worry about this, it was added to ensure compatibility with another model.
    # Shouldn't affect performance.
    self.gridwarper = nerf_network.UniformBoxWarp(0.24)

    logger = logging.getLogger('tl')
    models_dict = {}
    for name in self.module_name_list:
      models_dict[name] = getattr(self, name)
    models_dict['nerf'] = self
    torch_utils.print_number_params(models_dict=models_dict, logger=logger)
    logger.info(self)
    pass

  def forward_with_frequencies_phase_shifts(self,
                                            input,
                                            style_dict,
                                            # is_rgb=False,  # todo style_dict divide
                                            **kwargs):
    """

    :param input: (b, n, 3)
    :param style_dict:
    :param ray_directions:
    :param kwargs:
    :return:
    """

    # if global_cfg.tl_debug:
    #   VerboseModel.forward_verbose(nn.Sequential(
    #     OrderedDict([
    #       ('gridwarper', self.gridwarper),
    #       # ('xyz_emb', self.xyz_emb),
    #     ])),
    #     inputs_args=(input,),
    #     name_prefix="xyz.")
    input = self.gridwarper(input)
    # xyz_emb = self.xyz_emb(input)
    # x = xyz_emb
    x = input
    for index, layer in enumerate(self.network):
      style = style_dict[f'{self.name_prefix}_w{index}']

      # if global_cfg.tl_debug:
      #   VerboseModel.forward_verbose(layer,
      #                                inputs_args=(x, style),
      #                                name_prefix=f"network.{index}.")
      x = layer(x, style)

    # if global_cfg.tl_debug:
    #   VerboseModel.forward_verbose(self.final_layer,
    #                                inputs_args=(x,),
    #                                name_prefix="final_layer")
    # sigma = self.final_layer(x)

    # todo rgb branch
    # else:
    #   style = style_dict[f'{self.name_prefix}_rgb']
    #   # style_0 = style_dict[f'{self.name_prefix}_rgb_0']  # todo 增加style_dict
    #   # style_1 = style_dict[f'{self.name_prefix}_rgb_1']  # todo 增加style_dict
    #   # if global_cfg.tl_debug:
    #   #   VerboseModel.forward_verbose(self.color_layer_sine,
    #   #                                inputs_args=(x, style),
    #   #                                name_prefix=f"color_layer_sine.")
    #   x = self.color_layer_sine(x, style_0)
    #   x = self.color_layer_sine(x, style_1)

    # if global_cfg.tl_debug:
    #   VerboseModel.forward_verbose(self.color_layer_linear,
    #                                inputs_args=(x,),
    #                                name_prefix='color_layer_linear.')
    # rbg = torch.sigmoid(self.color_layer_linear(rbg))
    x = self.color_layer_linear(x)
    return x
    # out = torch.cat([rbg, sigma], dim=-1)
    # return out

  def forward(self,
              input,
              params=None,  # todo 原参数
              # params_v=None,  # todo 分潜码，共网络 参数
              # params_rgb=None,  # todo 分潜码，共网络 参数
              ray_directions=None,
              **kwargs):
    """

    :param input: points xyz, (b, num_points, 3)
    :param style_dict:
    :param ray_directions: (b, num_points, 3)
    :param kwargs:
    :return:
    - out: (b, num_points, 4), rgb(3) + sigma(1)
    """
    style_dict = params  # todo 原参数 代码
    # style_dict_v, style_dict_rgb = params_v, params_rgb
    if self.is_rgb:  # todo rgb
      rgb_input_data = rearrange(input['img'], 'b h w c -> b c h w')
      # rgb_input_data = self.rgb_net(input_data)
      rgb_input_data = self.rgb_conv(rgb_input_data)
      rgb_input_data = self.rgb_pool(rgb_input_data)
      rgb_input_data = rearrange(rgb_input_data, 'b c h w -> b c (h w)')
      rgb_input_data = self.rgb_linear(rgb_input_data)
      input_data = rearrange(rgb_input_data, 'b c i -> b i c')
    else:
      input_data = input['coords']
    if len(input_data.shape) == 2:  # todo 训练和验证时的输入不一样
      input_data = input_data.reshape([1, *input_data.shape])
    # todo 训练和验证时的输入不一样
    # input_data = torch.cat([rgb_input_data[:, :v_input_data.shape[1]], v_input_data], dim=2)
    # input_data = input['img'] if self.is_rgb else input['coords']  # todo rgb or not
    out = self.forward_with_frequencies_phase_shifts(
      input=input_data,  # todo rgb or not
      style_dict=style_dict,
      # style_dict=style_dict_v,
      ray_directions=ray_directions,
      **kwargs)
    # out_rbg = self.forward_with_frequencies_phase_shifts(
    #   input=rgb_input_data[:, :v_input_data.shape[1]],
    #   style_dict=style_dict,
    #   # style_dict=style_dict_rgb,
    #   ray_directions=ray_directions,
    #   **kwargs)
    # out = torch.cat([out_v, out_rbg], dim=2)

    if self.is_rgb:
      output = {  # todo add rgb_model_in
        'rgb_model_in': input['img'],
        'rgb_model_out': out,
      }
    else:
      output = {
        # 'rgb_model_in': input['img'],  # todo rgb_
        'model_in': input['coords'],
        'model_out': out,
      }

    return output

  def print_number_params(self):
    print()

    pass

  def get_freq_phase(self, style_dict, name):
    styles = style_dict[name]
    styles = rearrange(styles, "b (n d) -> b d n", n=2)
    frequencies, phase_shifts = styles.unbind(-1)
    frequencies = frequencies * 15 + 30
    return frequencies, phase_shifts

  def staged_forward(self,
                     transformed_points,
                     transformed_ray_directions_expanded,
                     style_dict,
                     max_points,
                     num_steps,
                     ):

    batch_size, num_points, _ = transformed_points.shape

    rgb_sigma_output = torch.zeros((batch_size, num_points, self.rgb_dim + 1),
                                   device=self.device)
    for b in range(batch_size):
      head = 0
      while head < num_points:
        tail = head + max_points
        rgb_sigma_output[b:b + 1, head:tail] = self(
          input=transformed_points[b:b + 1, head:tail],  # (b, h x w x s, 3)
          style_dict={name: style[b:b + 1] for name, style in style_dict.items()},
          ray_directions=transformed_ray_directions_expanded[b:b + 1, head:tail])
        head += max_points
    rgb_sigma_output = rearrange(rgb_sigma_output, "b (hw s) rgb_sigma -> b hw s rgb_sigma", s=num_steps)
    return rgb_sigma_output