"""Wavefront OBJ file loader.

Functions and classes are largely derived from
https://github.com/pyglet/pyglet/blob/f762169c9dd88c22c8d6d2399a129cc23654d99c/contrib/model/model/obj_batch.py
"""

import os
import logging
import numpy as np


class Material:
  diffuse = [.8, .8, .8]
  ambient = [.2, .2, .2]
  specular = [0., 0., 0.]
  emission = [0., 0., 0.]
  shininess = 0.
  opacity = 1.
  texture_path = None

  def __init__(self, name):
    self.name = name


class MaterialGroup:

  def __init__(self, material):
    self.material = material

    self.f_v = []
    self.f_n = []
    self.f_t = []


class Mesh:

  def __init__(self, name):
    self.name = name
    self.groups = []


class OBJ:

  def __init__(self, filename, file=None, path=None):
    self.materials = {}
    self.meshes = {}
    self.mesh_list = []

    if file is None:
      file = open(filename, 'r')

    if path is None:
      path = os.path.dirname(filename)
    self.path = path

    mesh = None
    group = None
    material = None

    self.v = []
    self.n = []
    self.t = []

    for line in file:
      if line.startswith('#'):
        continue
      values = line.split()
      if not values:
        continue

      if values[0] == 'v':
        self.v.append(list(map(float, values[1:4])))
      elif values[0] == 'vn':
        self.n.append(list(map(float, values[1:4])))
      elif values[0] == 'vt':
        self.t.append(list(map(float, values[1:3])))
      elif values[0] == 'mtllib':
        self._load_material_library(values[1])
      elif values[0] in ('usemtl', 'usemat'):
        material = self.materials.get(values[1], None)
        if material is None:
          logging.warn('Unknown material: %s' % values[1])
        if mesh is not None:
          group = MaterialGroup(material)
          mesh.groups.append(group)
      elif values[0] == 'o':
        mesh = Mesh(values[1])
        self.meshes[mesh.name] = mesh
        self.mesh_list.append(mesh)
        group = None
      elif values[0] == 'f':
        if mesh is None:
          mesh = Mesh('')
          self.mesh_list.append(mesh)
        if material is None:
          material = Material("<unknown>")
        if group is None:
          group = MaterialGroup(material)
          mesh.groups.append(group)

        for i, v in enumerate(values[1:]):
          v_index, t_index, n_index = \
              (list(map(int, [j or 0 for j in v.split('/')])) + [0, 0])[:3]
          if v_index < 0:
            v_index += len(vertices)
          if t_index < 0:
            t_index += len(tex_coords)
          if n_index < 0:
            n_index += len(normals)
          if i < 3:
            group.f_v.append(v_index - 1)
            group.f_n.append(n_index - 1)
            group.f_t.append(t_index - 1)
          else:
            # Triangulate.
            group.f_v += [group.f_v[-3 * (i - 2)], group.f_v[-1], v_index - 1]
            group.f_n += [group.f_n[-3 * (i - 2)], group.f_n[-1], n_index - 1]
            group.f_t += [group.f_t[-3 * (i - 2)], group.f_t[-1], t_index - 1]

    self.v = np.array(self.v, dtype=np.float32)
    self.n = np.array(self.n, dtype=np.float32)
    self.t = np.array(self.t, dtype=np.float32)

    for mesh in self.mesh_list:
      for group in mesh.groups:
        group.f_v = np.array(group.f_v, dtype=np.int64).reshape(-1, 3)
        group.f_n = np.array(group.f_n, dtype=np.int64).reshape(-1, 3)
        group.f_t = np.array(group.f_t, dtype=np.int64).reshape(-1, 3)

  def _open_material_file(self, filename):
    """Overrides for loading from archive/network etc."""
    return open(os.path.join(self.path, filename), 'r')

  def _load_material_library(self, filename):
    material = None
    file = self._open_material_file(filename)

    for line in file:
      if line.startswith('#'):
        continue
      values = line.split()
      if not values:
        continue

      if values[0] == 'newmtl':
        material = Material(values[1])
        self.materials[material.name] = material
      elif material is None:
        logging.warn('Expected "newmtl" in %s' % filename)
        continue

      try:
        if values[0] == 'Kd':
          material.diffuse = list(map(float, values[1:]))
        elif values[0] == 'Ka':
          material.ambient = list(map(float, values[1:]))
        elif values[0] == 'Ks':
          material.specular = list(map(float, values[1:]))
        elif values[0] == 'Ke':
          material.emissive = list(map(float, values[1:]))
        elif values[0] == 'Ns':
          material.shininess = float(values[1])
        elif values[0] == 'd':
          material.opacity = float(values[1])
        elif values[0] == 'map_Kd':
          material.texture_path = os.path.abspath(self.path + '/' + values[1])
      except BaseException as ex:
        logging.warning('Parse error in %s.' % (filename, ex))
