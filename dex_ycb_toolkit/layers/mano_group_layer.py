import torch

from torch.nn import Module, ModuleList

from .mano_layer import MANOLayer


class MANOGroupLayer(Module):
  """Wrapper layer to hold a group of MANOLayers."""

  def __init__(self, sides, betas):
    """Constructor.

    Args:
      sides: A list of MANO sides. 'right' or 'left'.
      betas: A list of numpy arrays of shape [10] containing the betas.
    """
    super(MANOGroupLayer, self).__init__()

    self._sides = sides
    self._betas = betas
    self._layers = ModuleList(
        [MANOLayer(s, b) for s, b in zip(self._sides, self._betas)])
    self._num_obj = len(self._sides)

    f = []
    for i in range(self._num_obj):
      f.append(self._layers[i].f + 778 * i)
    f = torch.cat(f)
    self.register_buffer('f', f)

    r = torch.cat([l.root_trans for l in self._layers])
    self.register_buffer('root_trans', r)

  @property
  def num_obj(self):
    return self._num_obj

  def forward(self, p, inds=None):
    """Forward function.

    Args:
      p: A tensor of shape [B, D] containing the pose vectors.
      inds: A list of sub-layer indices.

    Returns:
      v: A tensor of shape [B, N, 3] containing the vertices.
      j: A tensor of shape [B, J, 3] containing the joints.
    """
    if inds is None:
      inds = range(self._num_obj)
    v = [
        torch.zeros((p.size(0), 0, 3),
                    dtype=torch.float32,
                    device=self.f.device)
    ]
    j = [
        torch.zeros((p.size(0), 0, 3),
                    dtype=torch.float32,
                    device=self.f.device)
    ]
    p, t = self.pose2pt(p)
    for i in inds:
      y = self._layers[i](p[:, i], t[:, i])
      v.append(y[0])
      j.append(y[1])
    v = torch.cat(v, dim=1)
    j = torch.cat(j, dim=1)
    return v, j

  def pose2pt(self, pose):
    """Extracts pose and trans from pose vectors.

    Args:
      pose: A tensor of shape [B, D] containing the pose vectors.

    Returns:
      p: A tensor of shape [B, O, 48] containing the pose.
      t: A tensor of shape [B, O, 3] containing the trans.
    """
    p = torch.stack(
        [pose[:, 51 * i + 0:51 * i + 48] for i in range(self._num_obj)], dim=1)
    t = torch.stack(
        [pose[:, 51 * i + 48:51 * i + 51] for i in range(self._num_obj)], dim=1)
    return p, t

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
    for i, x in enumerate(inds):
      f.append(self._layers[x].f + 778 * i)
      m.append(x * torch.ones(1538, dtype=torch.int64, device=self.f.device))
    f = torch.cat(f)
    m = torch.cat(m)
    return f, m
