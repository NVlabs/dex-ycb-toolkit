# DexYCB Toolkit
# Copyright (C) 2021 NVIDIA Corporation
# Licensed under the GNU General Public License v3.0 [see LICENSE for details]

"""Wrapper layer to hold a group of YCBLayers."""

import torch

from torch.nn import Module, ModuleList

from .ycb_layer import YCBLayer


class YCBGroupLayer(Module):
  """Wrapper layer to hold a group of YCBLayers."""

  def __init__(self, ids):
    """Constructor.

    Args:
      ids: A list of YCB object ids.
    """
    super(YCBGroupLayer, self).__init__()

    self._ids = ids
    self._layers = ModuleList([YCBLayer(i) for i in self._ids])
    self._num_obj = len(self._ids)

    f = []
    offset = 0
    for i in range(self._num_obj):
      if i > 0:
        offset += self._layers[i - 1].v.size(1)
      f.append(self._layers[i].f + offset)
    f = torch.cat(f)
    self.register_buffer('f', f)

  @property
  def num_obj(self):
    return self._num_obj

  @property
  def obj_file(self):
    return [l.obj_file for l in self._layers]

  @property
  def count(self):
    return [l.f.numel() for l in self._layers]

  @property
  def material(self):
    return [l.material for l in self._layers]

  @property
  def tex_coords(self):
    return [l.tex_coords for l in self._layers]

  def forward(self, p, inds=None):
    """Forward function.

    Args:
      p: A tensor of shape [B, D] containing the pose vectors.
      inds: A list of sub-layer indices.

    Returns:
      v: A tensor of shape [B, N, 3] containing the transformed vertices.
      n: A tensor of shape [B, N, 3] containing the transformed normals.
    """
    if inds is None:
      inds = range(self._num_obj)
    v = [
        torch.zeros((p.size(0), 0, 3),
                    dtype=torch.float32,
                    device=self.f.device)
    ]
    n = [
        torch.zeros((p.size(0), 0, 3),
                    dtype=torch.float32,
                    device=self.f.device)
    ]
    r, t = self._pose2rt(p)
    for i in inds:
      y = self._layers[i](r[:, i], t[:, i])
      v.append(y[0])
      n.append(y[1])
    v = torch.cat(v, dim=1)
    n = torch.cat(n, dim=1)
    return v, n

  def _pose2rt(self, pose):
    """Extracts rotations and translations from pose vectors.

    Args:
      pose: A tensor of shape [B, D] containing the pose vectors.

    Returns:
      r: A tensor of shape [B, O, 3] containing the rotation vectors.
      t: A tensor of shape [B, O, 3] containing the translations.
    """
    r = torch.stack(
        [pose[:, 6 * i + 0:6 * i + 3] for i in range(self._num_obj)], dim=1)
    t = torch.stack(
        [pose[:, 6 * i + 3:6 * i + 6] for i in range(self._num_obj)], dim=1)
    return r, t
