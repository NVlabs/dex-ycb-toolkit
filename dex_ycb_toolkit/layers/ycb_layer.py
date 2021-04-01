# DexYCB Toolkit
# Copyright (C) 2021 NVIDIA Corporation
# Licensed under the GNU General Public License v3.0 [see LICENSE for details]

"""Layer to transform YCB mesh vertices with SE3 transformation."""

import os
import torch

from torch.nn import Module

from ..obj import OBJ


class YCBLayer(Module):
  """Layer to transform YCB mesh vertices with SE3 transformation."""

  def __init__(self, i):
    """Constructor.

    Args:
      i: YCB object index.
    """
    super(YCBLayer, self).__init__()

    assert 'DEX_YCB_DIR' in os.environ, "environment variable 'DEX_YCB_DIR' is not set"
    self._path = os.environ['DEX_YCB_DIR'] + "/models"
    self._classes = ('__background__', '002_master_chef_can', '003_cracker_box',
                     '004_sugar_box', '005_tomato_soup_can',
                     '006_mustard_bottle', '007_tuna_fish_can',
                     '008_pudding_box', '009_gelatin_box',
                     '010_potted_meat_can', '011_banana', '019_pitcher_base',
                     '021_bleach_cleanser', '024_bowl', '025_mug',
                     '035_power_drill', '036_wood_block', '037_scissors',
                     '040_large_marker', '051_large_clamp',
                     '052_extra_large_clamp', '061_foam_brick')
    self._class_name = self._classes[i]
    self._obj_file = self._path + '/' + self._class_name + "/textured_simple.obj"
    self._obj = OBJ(self._obj_file)
    assert len(self._obj.mesh_list) == 1
    assert len(self._obj.mesh_list[0].groups) == 1
    g = self._obj.mesh_list[0].groups[0]

    self._material = g.material
    self._tex_coords = self._obj.t[g.f_t]

    v = torch.from_numpy(self._obj.v).t()
    n = torch.from_numpy(self._obj.n).t()
    assert (g.f_v == g.f_n).all()
    f = torch.from_numpy(g.f_v).view((-1, 3))
    self.register_buffer('v', v)
    self.register_buffer('n', n)
    self.register_buffer('f', f)

  @property
  def obj_file(self):
    return self._obj_file

  @property
  def material(self):
    return self._material

  @property
  def tex_coords(self):
    return self._tex_coords

  def forward(self, r, t):
    """Forward function.

    Args:
      r: A tensor of shape [B, 3] containing the rotation in axis-angle.
      t: A tensor of shape [B, 3] containing the translation.

    Returns:
      v: A tensor of shape [B, N, 3] containing the transformed vertices.
      n: A tensor of shape [B, N, 3] containing the transformed normals.
    """
    R = rv2dcm(r)
    v = torch.matmul(R, self.v).permute(0, 2, 1) + t.unsqueeze(1)
    n = torch.matmul(R, self.n).permute(0, 2, 1)
    return v, n


# https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula#Matrix_notation
def rv2dcm(rv):
  """Converts rotation vectors to direction cosine matrices.

  Args:
    rv: A tensor of shape [B, 3] containing the rotation vectors.

  Returns:
    A tensor of shape [B, 3, 3] containing the direction cosine matrices.
  """
  angle = torch.norm(rv + 1e-8, p=2, dim=1)
  axis = rv / angle.unsqueeze(1)
  s = torch.sin(angle).unsqueeze(1).unsqueeze(2)
  c = torch.cos(angle).unsqueeze(1).unsqueeze(2)
  I = torch.eye(3, device=rv.device).expand(rv.size(0), -1, -1)
  z = torch.zeros_like(angle)
  K = torch.stack(
      (torch.stack((z, -axis[:, 2], axis[:, 1]),
                   dim=1), torch.stack((axis[:, 2], z, -axis[:, 0]), dim=1),
       torch.stack((-axis[:, 1], axis[:, 0], z), dim=1)),
      dim=1)
  dcm = I + s * K + (1 - c) * torch.bmm(K, K)
  return dcm


# https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Rotation_matrix_%E2%86%94_Euler_axis/angle
# https://github.com/kashif/ceres-solver/blob/087462a90dd1c23ac443501f3314d0fcedaea5f7/include/ceres/rotation.h#L178
# S. Sarabandi and F. Thomas. A Survey on the Computation of Quaternions from Rotation Matrices. J MECH ROBOT, 2019.
# https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
def dcm2rv(dcm):
  """Converts direction cosine matrices to rotation vectors.

  Args:
    dcm: A tensor of shape [B, 3, 3] containing the direction cosine matrices.

  Returns:
    A tensor of shape [B, 3] containing the rotation vectors.
  """
  X = torch.stack((dcm[:, 2, 1] - dcm[:, 1, 2], dcm[:, 0, 2] - dcm[:, 2, 0],
                   dcm[:, 1, 0] - dcm[:, 0, 1]),
                  dim=1)
  s = torch.norm(X, p=2, dim=1) / 2
  c = (dcm[:, 0, 0] + dcm[:, 1, 1] + dcm[:, 2, 2] - 1) / 2
  c = torch.clamp(c, -1, 1)
  angle = torch.atan2(s, c)
  Y = torch.stack((dcm[:, 0, 0], dcm[:, 1, 1], dcm[:, 2, 2]), dim=1)
  Y = torch.sqrt((Y - c.unsqueeze(1)) / (1 - c.unsqueeze(1)))
  rv = torch.zeros((dcm.size(0), 3), device=dcm.device)
  i1 = s > 1e-3
  i2 = (s <= 1e-3) & (c > 0)
  i3 = (s <= 1e-3) & (c < 0)
  rv[i1] = angle[i1].unsqueeze(1) * X[i1] / (2 * s[i1].unsqueeze(1))
  rv[i2] = X[i2] / 2
  rv[i3] = angle[i3].unsqueeze(1) * torch.sign(X[i3]) * Y[i3]
  return rv
