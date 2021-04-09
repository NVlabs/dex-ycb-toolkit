# DexYCB Toolkit
# Copyright (C) 2021 NVIDIA Corporation
# Licensed under the GNU General Public License v3.0 [see LICENSE for details]

"""Interactive 3D scene viewer using pyglet.

Functions and classes are largely derived from
https://github.com/IntelRealSense/librealsense/blob/81d469db173dd682d3bada9bd7c7570db0f7cf76/wrappers/python/examples/pyglet_pointcloud_viewer.py

Usage of class Window:
------
Mouse:
    Drag with left button to rotate around pivot (thick small axes),
    with right button to translate and the wheel to zoom.

Keyboard:
    [p]         Pause
    [r]         Reset View
    [z]         Toggle point scaling
    [x]         Toggle point distance attenuation
    [l]         Toggle lighting
    [1/2/3/...] Toggle camera switch
    [k]         Toggle point mask
    [m]         Toggle YCB/MANO mesh
    [SPACE]     Step frame during pause
    [s]         Save PNG (./out.png)
    [q/ESC]     Quit
"""

import numpy as np
import math
import pyglet
import os
import logging

from pyglet.gl import *


# https://stackoverflow.com/a/6802723
def rotation_matrix(axis, theta):
  """Returns the rotation matrix associated with counterclockwise rotation about
  the given axis by theta radians.

  Args:
    axis: Axis represented by a tuple (x, y, z).
    theta: Theta in radians.

  Returns:
    A float64 numpy array of shape [3, 3] containing the rotation matrix.
  """
  axis = np.asarray(axis)
  axis = axis / math.sqrt(np.dot(axis, axis))
  a = math.cos(theta / 2.0)
  b, c, d = -axis * math.sin(theta / 2.0)
  aa, bb, cc, dd = a * a, b * b, c * c, d * d
  bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
  return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                   [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                   [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


class AppState:
  """Viewer window app state."""

  def __init__(self, *args, **kwargs):
    """Constructor.

    Args:
      args: Variable length argument list.
      kwargs: Arbitrary keyword arguments.
    """
    self.pitch, self.yaw = math.radians(-10), math.radians(-15)
    self.translation = np.array([0, 0, 1], np.float32)
    self.distance = 2
    self.mouse_btns = [False, False, False]
    self.paused = False
    self.scale = True
    self.attenuation = False
    self.lighting = False
    self.camera_off = [False] * kwargs['num_cameras']
    self.mask = 0
    self.model_off = False

  def reset(self):
    """Resets the app to an initial state."""
    self.pitch, self.yaw, self.distance = 0, 0, 2
    self.translation[:] = 0, 0, 1

  @property
  def rotation(self):
    Rx = rotation_matrix((1, 0, 0), math.radians(-self.pitch))
    Ry = rotation_matrix((0, 1, 0), math.radians(-self.yaw))
    return np.dot(Ry, Rx).astype(np.float32)


def reset_pyglet_resource_path(path):
  """Resets pyglet's resource path.

  Args:
    path: Path to be reset to.
  """
  if not os.path.isabs(path):
    path = os.path.abspath(path)
  pyglet.resource.path = [path]
  pyglet.resource.reindex()


class Material(pyglet.graphics.Group):
  """Material."""

  def __init__(self, material, **kwargs):
    """Constructor.

    Args:
      material: A material object loaded from an OBJ file.
      kwargs: Arbitrary keyword arguments.
    """
    super(Material, self).__init__(**kwargs)
    self.material = material
    self.texture = None
    self.texture_name = None

    if material.texture_path is not None:
      texture_name = os.path.relpath(self.material.texture_path,
                                     pyglet.resource.path[0])
      try:
        self.texture = pyglet.resource.image(texture_name)
        self.texture_name = texture_name
      except BaseException as ex:
        logging.warn('Could not load texture %s: %s' % (texture_name, ex))

  def set_state(self, face=GL_FRONT_AND_BACK):
    if self.texture:
      glEnable(self.texture.target)
      glBindTexture(self.texture.target, self.texture.id)
    else:
      glDisable(GL_TEXTURE_2D)

    glMaterialfv(face, GL_DIFFUSE, (GLfloat * 4)(*(self.material.diffuse +
                                                   [self.material.opacity])))
    glMaterialfv(face, GL_AMBIENT, (GLfloat * 4)(*(self.material.ambient +
                                                   [self.material.opacity])))
    glMaterialfv(face, GL_SPECULAR, (GLfloat * 4)(*(self.material.specular +
                                                    [self.material.opacity])))
    glMaterialfv(face, GL_EMISSION, (GLfloat * 4)(*(self.material.emission +
                                                    [self.material.opacity])))
    glMaterialf(face, GL_SHININESS, self.material.shininess)

  def unset_state(self):
    if self.texture:
      glDisable(self.texture.target)
    glDisable(GL_COLOR_MATERIAL)

  def __eq__(self, other):
    if self.texture is None:
      return super(Material, self).__eq__(other)
    return (self.__class__ is other.__class__ and
            self.texture.id == other.texture.id and
            self.texture.target == other.texture.target and
            self.parent == other.parent)

  def __hash__(self):
    if self.texture is None:
      return super(Material, self).__hash__()
    return hash((self.texture.id, self.texture.target))

  def set_alpha(self, alpha):
    """Sets the alpha value.

    Args:
      alpha: Alpha value.
    """
    if self.texture is None and self.texture_name is None:
      logging.warn('Texture was not loaded successfully')
      return
    assert 0.0 <= alpha <= 1.0
    a_val = round(255 * alpha)

    f = pyglet.resource.file(self.texture_name)
    image = pyglet.image.load(self.texture_name, file=f)
    f.close()
    data = image.get_data(image.format, image.pitch)
    if image.format == 'RGB':
      rgb = np.array(data, dtype=np.uint8).reshape(
          (image.height, image.width, 3))
      a = a_val * np.ones((image.height, image.width, 1), dtype=np.uint8)
      rgba = np.concatenate((rgb, a), axis=2)
    elif image.format == 'RGBA':
      rgba = np.array(data, dtype=np.uint8).reshape(
          (image.height, image.width, 4))
      rgba[:, :, 3] = a_val
    # Somehow need to flip the height dimension.
    new_data = rgba[::-1, :, :]
    new_data = new_data.ravel().tobytes()
    image = pyglet.image.ImageData(image.width,
                                   image.height,
                                   'RGBA',
                                   new_data,
                                   pitch=image.width * 4)
    self.texture = image.get_texture(True)


def axes(size=1, width=1):
  """Draws 3D axes.

  Args:
    size: Axes length.
    width: Axes width.
  """
  glLineWidth(width)
  pyglet.graphics.draw(6, GL_LINES,
                       ('v3f', (0, 0, 0, size, 0, 0,
                                0, 0, 0, 0, size, 0,
                                0, 0, 0, 0, 0, size)),
                       ('c3f', (1, 0, 0, 1, 0, 0,
                                0, 1, 0, 0, 1, 0,
                                0, 0, 1, 0, 0, 1,
                                ))
                       )


def frustum(dimensions, intrinsics):
  """Draws the camera's frustum.

  Args:
    dimensions: A tuple (w, h) containing the image width and height.
    intrinsics: A float32 numpy array of size [3, 3] containing the intrinsic
      matrix.
  """
  w, h = dimensions[0], dimensions[1]
  batch = pyglet.graphics.Batch()

  for d in range(1, 6, 2):

    def get_point(x, y):
      p = list(np.linalg.inv(intrinsics).dot([x, y, 1]) * d)
      batch.add(2, GL_LINES, None, ('v3f', [0, 0, 0] + p))
      return p

    top_left = get_point(0, 0)
    top_right = get_point(w, 0)
    bottom_right = get_point(w, h)
    bottom_left = get_point(0, h)

    batch.add(2, GL_LINES, None, ('v3f', top_left + top_right))
    batch.add(2, GL_LINES, None, ('v3f', top_right + bottom_right))
    batch.add(2, GL_LINES, None, ('v3f', bottom_right + bottom_left))
    batch.add(2, GL_LINES, None, ('v3f', bottom_left + top_left))

  batch.draw()


def grid(size=1, n=10, width=1):
  """Draws a grid on XZ plane.

  Args:
    size: Grid line length in X and Z direction.
    n: Grid number.
    width: Grid line width.
  """
  glLineWidth(width)
  s = size / float(n)
  s2 = 0.5 * size
  batch = pyglet.graphics.Batch()

  for i in range(0, n + 1):
    x = -s2 + i * s
    batch.add(2, GL_LINES, None, ('v3f', (x, 0, -s2, x, 0, s2)))
  for i in range(0, n + 1):
    z = -s2 + i * s
    batch.add(2, GL_LINES, None, ('v3f', (-s2, 0, z, s2, 0, z)))

  batch.draw()


class Window():
  """Viewer window."""

  def __init__(self, dataloader):
    """Constructor:

    Args:
      dataloader: A SequenceLoader object.
    """
    self.dataloader = dataloader

    self.config = Config(double_buffer=True, samples=8)  # MSAA
    self.window = pyglet.window.Window(config=self.config, resizable=True)

    self.state = AppState(num_cameras=self.dataloader.num_cameras)

    self.pcd_vlist = []
    self.pcd_image = []
    w, h = self.dataloader.dimensions
    for _ in range(self.dataloader.num_cameras):
      self.pcd_vlist.append(
          pyglet.graphics.vertex_list(w * h, 'v3f/stream', 't2f/stream',
                                      'n3f/stream'))
      self.pcd_image.append(
          pyglet.image.ImageData(w, h, 'RGB', (GLubyte * (w * h * 3))()))

    reset_pyglet_resource_path(self.dataloader.ycb_model_dir)
    self.ycb_batch = pyglet.graphics.Batch()
    self.ycb_vlist = []
    for o in range(self.dataloader.num_ycb):
      n = self.dataloader.ycb_count[o]
      m = self.dataloader.ycb_material[o]
      g = Material(m)
      g.set_alpha(0.7)
      self.ycb_vlist.append(
          self.ycb_batch.add(n, GL_TRIANGLES, g, 'v3f/stream', 't2f/stream',
                             'n3f/stream'))

    self.ycb_per_view = isinstance(self.dataloader.ycb_vert[0],
                                   list) and isinstance(
                                       self.dataloader.ycb_norm[0], list)

    self.mano_batch = pyglet.graphics.Batch()
    self.mano_vlist = []
    self.mano_llist = []
    for _ in range(self.dataloader.num_mano):
      self.mano_vlist.append(
          self.mano_batch.add(4614, GL_TRIANGLES, None, 'v3f/stream',
                              'n3f/stream',
                              ('c4f/static', [0.7, 0.7, 0.7, 0.7] * 4614)))
      self.mano_llist.append(
          self.mano_batch.add(9228, GL_LINES, None, 'v3f/stream',
                              ('c3f/static', [0, 0, 0] * 9228)))

    self.fps_display = pyglet.window.FPSDisplay(self.window)

    @self.window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
      w, h = map(float, self.window.get_size())

      if buttons & pyglet.window.mouse.LEFT:
        self.state.yaw -= dx * 0.5
        self.state.pitch -= dy * 0.5

      if buttons & pyglet.window.mouse.RIGHT:
        dp = np.array((dx / w, -dy / h, 0), np.float32)
        self.state.translation += np.dot(self.state.rotation, dp)

      if buttons & pyglet.window.mouse.MIDDLE:
        dz = dy * 0.01
        self.state.translation -= (0, 0, dz)
        self.state.distance -= dz

    @self.window.event
    def handle_mouse_btns(x, y, button, modifiers):
      self.state.mouse_btns[0] ^= (button & pyglet.window.mouse.LEFT)
      self.state.mouse_btns[1] ^= (button & pyglet.window.mouse.RIGHT)
      self.state.mouse_btns[2] ^= (button & pyglet.window.mouse.MIDDLE)

    self.window.on_mouse_press = self.window.on_mouse_release = handle_mouse_btns

    @self.window.event
    def on_mouse_scroll(x, y, scroll_x, scroll_y):
      dz = scroll_y * 0.1
      self.state.translation -= (0, 0, dz)
      self.state.distance -= dz

    @self.window.event
    def on_key_press(symbol, modifiers):
      if symbol == pyglet.window.key.R:
        self.state.reset()

      if symbol == pyglet.window.key.P:
        self.state.paused ^= True

      if symbol == pyglet.window.key.Z:
        self.state.scale ^= True

      if symbol == pyglet.window.key.X:
        self.state.attenuation ^= True

      if symbol == pyglet.window.key.L:
        self.state.lighting ^= True
        self._update_pcd_normals()

      # _1, _2, _3, ...
      if 49 <= symbol < 49 + len(self.state.camera_off):
        self.state.camera_off[symbol - 49] ^= True
        self._update_ycb()

      if symbol == pyglet.window.key.K:
        self.state.mask ^= 1
        self._update_pcd()

      if symbol == pyglet.window.key.M:
        self.state.model_off ^= True

      if symbol == pyglet.window.key.SPACE and self.state.paused:
        self.update(ignore_pause=True)

      if symbol == pyglet.window.key.S:
        pyglet.image.get_buffer_manager().get_color_buffer().save('out.png')

      if symbol == pyglet.window.key.Q:
        self.window.close()

    @self.window.event
    def on_draw():
      self.window.clear()

      glEnable(GL_DEPTH_TEST)
      glEnable(GL_LINE_SMOOTH)

      width, height = self.window.get_size()
      glViewport(0, 0, width, height)

      # Set projection matrix stack.
      glMatrixMode(GL_PROJECTION)
      glLoadIdentity()
      gluPerspective(60, width / float(height), 0.01, 20)

      # Set modelview matrix stack.
      glMatrixMode(GL_MODELVIEW)
      glLoadIdentity()

      gluLookAt(0, 0, 0, 0, 0, 1, 0, -1, 0)

      glTranslatef(0, 0, self.state.distance)
      glRotated(self.state.pitch, 1, 0, 0)
      glRotated(self.state.yaw, 0, 1, 0)

      if any(self.state.mouse_btns):
        axes(0.1, 4)

      glTranslatef(0, 0, -self.state.distance)
      glTranslatef(*self.state.translation)

      # Draw grid.
      glColor3f(0.5, 0.5, 0.5)
      glPushMatrix()
      glTranslatef(0, 0.5, 0.5)
      grid()
      glPopMatrix()

      # Set point size.
      w, h = self.dataloader.dimensions
      psz = max(self.window.get_size()) / float(max(
          w, h)) if self.state.scale else 1
      glPointSize(psz)
      distance = (0, 0, 1) if self.state.attenuation else (1, 0, 0)
      glPointParameterfv(GL_POINT_DISTANCE_ATTENUATION,
                         (GLfloat * 3)(*distance))

      # Set lighting.
      if self.state.lighting:
        if not self.state.model_off:
          ldir = [0.0, 0.0, -1.0]
        else:
          ldir = np.dot(self.state.rotation, (0, 0, 1))
        ldir = list(ldir) + [0]
        glLightfv(GL_LIGHT0, GL_POSITION, (GLfloat * 4)(*ldir))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (GLfloat * 3)(1.0, 1.0, 1.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (GLfloat * 3)(0.75, 0.75, 0.75))
        glLightfv(GL_LIGHT0, GL_SPECULAR, (GLfloat * 3)(0, 0, 0))
        glEnable(GL_LIGHT0)
        glEnable(GL_NORMALIZE)
        glEnable(GL_LIGHTING)

      glColor3f(1, 1, 1)

      # Draw point cloud for each camera.
      for c in range(len(self.pcd_image)):
        if self.state.camera_off[c]:
          continue

        # Set texture matrix stack.
        glMatrixMode(GL_TEXTURE)
        glLoadIdentity()
        glTranslatef(0.5 / self.pcd_image[c].width,
                     0.5 / self.pcd_image[c].height, 0)
        image_texture = self.pcd_image[c].get_texture()
        tw, th = image_texture.owner.width, image_texture.owner.height
        glScalef(self.pcd_image[c].width / float(tw),
                 self.pcd_image[c].height / float(th), 1)

        # Draw vertices and textures.
        texture = self.pcd_image[c].get_texture()
        glEnable(texture.target)
        glBindTexture(texture.target, texture.id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

        glEnable(GL_POINT_SPRITE)

        if not self.state.scale and not self.state.attenuation:
          glDisable(GL_MULTISAMPLE)
        self.pcd_vlist[c].draw(GL_POINTS)
        glDisable(texture.target)
        if not self.state.scale and not self.state.attenuation:
          glEnable(GL_MULTISAMPLE)

      # Draw YCB mesh.
      if not self.state.model_off:
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glMatrixMode(GL_TEXTURE)
        glLoadIdentity()

        self.ycb_batch.draw()

      # Draw MANO mesh.
      if not self.state.model_off:
        self.mano_batch.draw()

      glDisable(GL_LIGHTING)

      # Draw frustum and axes.
      glColor3f(0.25, 0.25, 0.25)
      frustum(self.dataloader.dimensions, self.dataloader.master_intrinsics)
      axes()

      # Reset matrix stacks.
      glMatrixMode(GL_PROJECTION)
      glLoadIdentity()
      glOrtho(0, width, 0, height, -1, 1)
      glMatrixMode(GL_MODELVIEW)
      glLoadIdentity()
      glMatrixMode(GL_TEXTURE)
      glLoadIdentity()

      glDisable(GL_DEPTH_TEST)

      self.fps_display.draw()

  def update(self, ignore_pause=False):
    """Updates the viewer window.

    Args:
      ignore_pause: Whether to update under a pause.
    """
    if not ignore_pause and self.state.paused:
      return

    # Need to call `step()` at the start of `update()`. If called at the end
    # `self.pcd_image` will change as `pcd_rgb` changes since they share data.
    # This will cause incorrect texture in drawing.
    self.dataloader.step()

    self._update_pcd()
    self._update_pcd_normals()
    self._update_ycb()
    self._update_mano()

  def _copy(self, dst, src):
    """Copies a numpy array to a pyglet array.

    Args:
      dst: The pyglet array to copy to.
      src: The numpy array to copy from.
    """
    np.array(dst, copy=False)[:] = src.ravel()

  def _update_pcd(self):
    """Updates point clouds."""
    pcd_rgb = self.dataloader.pcd_rgb
    pcd_vert = self.dataloader.pcd_vert
    pcd_tex_coord = self.dataloader.pcd_tex_coord
    pcd_mask = self.dataloader.pcd_mask
    for c in range(len(self.pcd_image)):
      self.pcd_image[c].set_data('RGB', pcd_rgb[c].strides[0],
                                 pcd_rgb[c].ctypes.data)
      self._copy(self.pcd_vlist[c].vertices, pcd_vert[c])
      self._copy(self.pcd_vlist[c].tex_coords, pcd_tex_coord[c])
      if self.state.mask == 1:
        vertices = np.array(self.pcd_vlist[c].vertices, copy=False)
        for i in range(3):
          vertices[i::3][np.logical_not(pcd_mask[c]).ravel()] = 0

  def _update_pcd_normals(self):
    """Updates point cloud normals."""
    if self.state.lighting:
      pcd_vert = self.dataloader.pcd_vert
      for c in range(len(self.pcd_image)):
        dy, dx = np.gradient(pcd_vert[c], axis=(0, 1))
        n = np.cross(dx, dy)
        self._copy(self.pcd_vlist[c].normals, n)

  def _update_ycb(self):
    """Updates YCB objects."""
    ycb_vert = self.dataloader.ycb_vert
    ycb_norm = self.dataloader.ycb_norm
    ycb_tex_coords = self.dataloader.ycb_tex_coords
    if self.ycb_per_view:
      for c, v in enumerate(self.state.camera_off):
        if not v:
          ycb_vert = ycb_vert[c]
          ycb_norm = ycb_norm[c]
          break
    for o in range(self.dataloader.num_ycb):
      self._copy(self.ycb_vlist[o].vertices, ycb_vert[o])
      self._copy(self.ycb_vlist[o].normals, ycb_norm[o])
      self._copy(self.ycb_vlist[o].tex_coords, ycb_tex_coords[o])

  def _update_mano(self):
    """Updates MANO hands."""
    mano_vert = self.dataloader.mano_vert
    mano_norm = self.dataloader.mano_norm
    mano_line = self.dataloader.mano_line
    for o in range(self.dataloader.num_mano):
      self._copy(self.mano_vlist[o].vertices, mano_vert[o])
      self._copy(self.mano_vlist[o].normals, mano_norm[o])
      self._copy(self.mano_llist[o].vertices, mano_line[o])
