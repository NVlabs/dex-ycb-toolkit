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
  def class_name(self):
    return [l.class_name for l in self._layers]

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
    r, t = self.pose2rt(p)
    for i in inds:
      y = self._layers[i](r[:, i], t[:, i])
      v.append(y[0])
      n.append(y[1])
    v = torch.cat(v, dim=1)
    n = torch.cat(n, dim=1)
    return v, n

  def pose2rt(self, pose):
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

  def get_f_from_inds(self, inds):
    """Gets faces from sub-layer indices.

    Args:
      inds: A list of sub-layer indices.

    Returns:
      f: A tensor of shape [F, 3] containing the faces.
      m: A tensor of shape [F] containing the face to index mapping.
    """
    f = [torch.zeros((0, 3), dtype=self.f.dtype, device=self.f.device)]
    m = [torch.zeros((0,), dtype=torch.int64, device=self.f.device)]
    offset = 0
    for i, x in enumerate(inds):
      if i > 0:
        offset += self._layers[inds[i - 1]].v.size(1)
      f.append(self._layers[x].f + offset)
      m.append(x * torch.ones(
          self._layers[x].f.size(0), dtype=torch.int64, device=self.f.device))
    f = torch.cat(f)
    m = torch.cat(m)
    return f, m

  def get_num_verts_from_inds(self, inds):
    """Gets number of vertices from sub-layer indices.

    Args:
      inds: A non-empty list of sub-layer indices.

    Returns:
      num_verts: The number of vertices.
    """
    return sum([self._layers[i].v.size(1) for i in inds])
