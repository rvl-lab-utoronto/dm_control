# Copyright 2019 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Quadruped Domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from dm_control import mujoco
from dm_control.mujoco.wrapper import mjbindings
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.utils import xml_tools

from lxml import etree
import numpy as np
from scipy import ndimage

enums = mjbindings.enums
mjlib = mjbindings.mjlib


_DEFAULT_TIME_LIMIT = 20/2
# _DEFAULT_TIME_LIMIT = 20
_CONTROL_TIMESTEP = .02

_N_PILLARS = 5

# Horizontal speeds above which the move reward is 1.
_RUN_SPEED = 5
_JOG_SPEED = 2.0
_WALK_SPEED = 0.5

# Constants related to terrain generation.
_HEIGHTFIELD_ID = 0
_TERRAIN_SMOOTHNESS = 0.15  # 0.0: maximally bumpy; 1.0: completely smooth.
_TERRAIN_BUMP_SCALE = 2  # Spatial scale of terrain bumps (in meters).

# Named model elements.
_TOES = ['toe_front_left', 'toe_back_left', 'toe_back_right', 'toe_front_right']
_WALLS = ['wall_px', 'wall_py', 'wall_nx', 'wall_ny']

SUITE = containers.TaggedTasks()


def make_model(floor_size=None, terrain=False, rangefinders=False,
               walls=False, ball=False, pillars=True):
  """Returns the model XML string."""
  xml_string = common.read_model('quadruped2_pillars.xml')
  parser = etree.XMLParser(remove_blank_text=True)
  mjcf = etree.XML(xml_string, parser)

  # Set floor size.
  if floor_size is not None:
    floor_geom = mjcf.find('.//geom[@name={!r}]'.format('floor'))
    floor_geom.attrib['size'] = '{} {} .5'.format(floor_size, floor_size)

  # Remove pillars
  if not pillars:
    for b in range(_N_PILLARS):
      cyl = xml_tools.find_element(mjcf, 'body', 'pillar' + str(b))
      cyl.getparent().remove(cyl)

  # Remove walls and target.
  if not walls:
    for wall in _WALLS:
      wall_geom = xml_tools.find_element(mjcf, 'geom', wall)
      wall_geom.getparent().remove(wall_geom)

    # Remove target.
    target_site = xml_tools.find_element(mjcf, 'site', 'target')
    target_site.getparent().remove(target_site)

  if not ball:
    # Remove ball.
    ball_body = xml_tools.find_element(mjcf, 'body', 'ball')
    ball_body.getparent().remove(ball_body)

  # Remove terrain.
  if not terrain:
    terrain_geom = xml_tools.find_element(mjcf, 'geom', 'terrain')
    terrain_geom.getparent().remove(terrain_geom)

  # Remove rangefinders if they're not used, as range computations can be
  # expensive, especially in a scene with heightfields.
  if not rangefinders:
    rangefinder_sensors = mjcf.findall('.//rangefinder')
    for rf in rangefinder_sensors:
      rf.getparent().remove(rf)

  return etree.tostring(mjcf, pretty_print=True)


@SUITE.add()
def reachtransferpre(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Fetch task."""
  xml_string = make_model(walls=True, ball=False)
  physics = Physics.from_xml_string(xml_string, common.ASSETS)
  task = ReachTransferPillars(pretransfer=True, desired_speed=_WALK_SPEED,random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit,
                             control_timestep=_CONTROL_TIMESTEP,
                             **environment_kwargs)

@SUITE.add()
def reachtransferpost(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Fetch task."""
  xml_string = make_model(walls=True, ball=False)
  physics = Physics.from_xml_string(xml_string, common.ASSETS)
  task = ReachTransferPillars(pretransfer=False, desired_speed=_WALK_SPEED,random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit,
                             control_timestep=_CONTROL_TIMESTEP,
                             **environment_kwargs)

@SUITE.add()
def reachtransferpostsparse(time_limit=_DEFAULT_TIME_LIMIT*2, random=None, environment_kwargs=None):
  """Returns the Fetch task."""
  xml_string = make_model(walls=True, ball=False)
  physics = Physics.from_xml_string(xml_string, common.ASSETS)
  task = ReachTransferPillars(pretransfer=False, desired_speed=_WALK_SPEED,random=random, sparse=True)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit,
                             control_timestep=_CONTROL_TIMESTEP,
                             **environment_kwargs)

class Physics(mujoco.Physics):
  """Physics simulation with additional features for the Quadruped domain."""

  def _reload_from_data(self, data):
    super(Physics, self)._reload_from_data(data)
    # Clear cached sensor names when the physics is reloaded.
    self._sensor_types_to_names = {}
    self._hinge_names = []

  def _get_sensor_names(self, *sensor_types):
    try:
      sensor_names = self._sensor_types_to_names[sensor_types]
    except KeyError:
      [sensor_ids] = np.where(np.in1d(self.model.sensor_type, sensor_types))
      sensor_names = [self.model.id2name(s_id, 'sensor') for s_id in sensor_ids]
      self._sensor_types_to_names[sensor_types] = sensor_names
    return sensor_names

  def torso_pos(self):
    torso_pos = self.named.data.xpos['torso']
    return torso_pos

  def global_forward_vector(self):
    return self.named.data.xmat['torso', ['xx', 'yx', 'zx']]
    # torso_pos = self.named.data.xmat['torso']
    # return torso_pos

  def torso_upright(self):
    """Returns the dot-product of the torso z-axis and the global z-axis."""
    return np.asarray(self.named.data.xmat['torso', 'zz'])

  def torso_velocity(self):
    """Returns the velocity of the torso, in the local frame."""
    return self.named.data.sensordata['velocimeter'].copy()

  def egocentric_state(self):
    """Returns the state without global orientation or position."""
    if not self._hinge_names:
      [hinge_ids] = np.nonzero(self.model.jnt_type ==
                               enums.mjtJoint.mjJNT_HINGE)
      self._hinge_names = [self.model.id2name(j_id, 'joint')
                           for j_id in hinge_ids]
    return np.hstack((self.named.data.qpos[self._hinge_names],
                      self.named.data.qvel[self._hinge_names],
                      self.data.act))

  def toe_positions(self):
    """Returns toe positions in egocentric frame."""
    torso_frame = self.named.data.xmat['torso'].reshape(3, 3)
    torso_pos = self.named.data.xpos['torso']
    torso_to_toe = self.named.data.xpos[_TOES] - torso_pos
    return torso_to_toe.dot(torso_frame)

  def force_torque(self):
    """Returns scaled force/torque sensor readings at the toes."""
    force_torque_sensors = self._get_sensor_names(enums.mjtSensor.mjSENS_FORCE,
                                                  enums.mjtSensor.mjSENS_TORQUE)
    return np.arcsinh(self.named.data.sensordata[force_torque_sensors])

  def imu(self):
    """Returns IMU-like sensor readings."""
    imu_sensors = self._get_sensor_names(enums.mjtSensor.mjSENS_GYRO,
                                         enums.mjtSensor.mjSENS_ACCELEROMETER)
    return self.named.data.sensordata[imu_sensors]

  def rangefinder(self):
    """Returns scaled rangefinder sensor readings."""
    rf_sensors = self._get_sensor_names(enums.mjtSensor.mjSENS_RANGEFINDER)
    rf_readings = self.named.data.sensordata[rf_sensors]
    no_intersection = -1.0
    return np.where(rf_readings == no_intersection, 1.0, np.tanh(rf_readings))

  def origin_distance(self):
    """Returns the distance from the origin to the workspace."""
    return np.asarray(np.linalg.norm(self.named.data.site_xpos['workspace']))

  def origin(self):
    """Returns origin position in the torso frame."""
    torso_frame = self.named.data.xmat['torso'].reshape(3, 3)
    torso_pos = self.named.data.xpos['torso']
    return -torso_pos.dot(torso_frame)

  def target_position(self):
    """Returns target position in torso frame."""
    torso_frame = self.named.data.xmat['torso'].reshape(3, 3)
    torso_pos = self.named.data.xpos['torso']
    torso_to_target = self.named.data.site_xpos['target'] - torso_pos
    return torso_to_target.dot(torso_frame)


  def body_2d_pose(self, body_names, orientation=True):
    """Returns positions and/or orientations of bodies."""
    # TODO this is not 2d right now ...
    if not isinstance(body_names, str):
      body_names = np.array(body_names).reshape(-1)
      # body_names = np.array(body_names).reshape(-1, 1)  # Broadcast indices.
    body_pos = self.named.data.xpos[body_names]
    # print(body_pos.shape)

    n_bodies = body_pos.shape[0]

    torso_frame = self.named.data.xmat['torso'].reshape(3, 3)
    itorso_frame = np.transpose(torso_frame).reshape(1,3,3)
    itorso_frame = np.broadcast_to(itorso_frame, (n_bodies,3,3))
    torso_pos = self.named.data.xpos['torso']
    torso_to_body = body_pos - torso_pos
    # batch matrix vector
    tframe_pos = np.einsum('bij,bj->bi', itorso_frame, torso_to_body)
    return tframe_pos

  def self_to_target_distance(self):
    """Returns horizontal distance from the quadruped torso to the target."""
    torso_pos = self.named.data.xpos['torso']
    self_to_target = (self.named.data.site_xpos['target'] -
                      torso_pos)
    return np.linalg.norm(self_to_target[:2])


def _find_non_contacting_height(physics, orientation, x_pos=0.0, y_pos=0.0):
  """Find a height with no contacts given a body orientation.

  Args:
    physics: An instance of `Physics`.
    orientation: A quaternion.
    x_pos: A float. Position along global x-axis.
    y_pos: A float. Position along global y-axis.
  Raises:
    RuntimeError: If a non-contacting configuration has not been found after
    10,000 attempts.
  """
  z_pos = 0.0  # Start embedded in the floor.
  num_contacts = 1
  num_attempts = 0
  # Move up in 1cm increments until no contacts.
  while num_contacts > 0:
    try:
      with physics.reset_context():
        physics.named.data.qpos['root'][:3] = x_pos, y_pos, z_pos
        physics.named.data.qpos['root'][3:] = orientation
    except control.PhysicsError:
      # We may encounter a PhysicsError here due to filling the contact
      # buffer, in which case we simply increment the height and continue.
      pass
    num_contacts = physics.data.ncon
    z_pos += 0.01
    num_attempts += 1
    if num_attempts > 10000:
      raise RuntimeError('Failed to find a non-contacting configuration.')


def _common_observations(physics):
  """Returns the observations common to all tasks."""
  obs = collections.OrderedDict()
  obs['egocentric_state'] = physics.egocentric_state()
  obs['torso_velocity'] = physics.torso_velocity()
  obs['torso_upright'] = physics.torso_upright()
  obs['imu'] = physics.imu()
  obs['force_torque'] = physics.force_torque()
  return obs


def _upright_reward(physics, deviation_angle=0):
  """Returns a reward proportional to how upright the torso is.

  Args:
    physics: an instance of `Physics`.
    deviation_angle: A float, in degrees. The reward is 0 when the torso is
      exactly upside-down and 1 when the torso's z-axis is less than
      `deviation_angle` away from the global z-axis.
  """
  deviation = np.cos(np.deg2rad(deviation_angle))
  return rewards.tolerance(
      physics.torso_upright(),
      bounds=(deviation, float('inf')),
      sigmoid='linear',
      margin=1 + deviation,
      value_at_margin=0)

def dist(a,b):
  return np.linalg.norm(a-b)

class ReachTransferPillars(base.Task):
  """
  Transfer version of fetch, where it first learns to walk at given speed, then learns fetch.
  A quadruped task solved by bringing a ball to the origin."""

  def __init__(self, pretransfer, desired_speed, random=None, sparse=False):
    """Initializes an instance of `Move`.

    Args:
      pretransfer: A bool. If true, solve the velocity task. If false, solve the fetch task
      desired_speed: A float. If this value is zero, reward is given simply
        for standing upright. Otherwise this specifies the velocity norm
        at which the velocity-dependent reward component is maximized.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
      sparse: Default false. Will give sparse reward in reach task instead.
    """
    self._pretransfer = pretransfer
    self._desired_speed = desired_speed
    self._sparse = sparse

    n_pillars = _N_PILLARS
    self._n_pillars = n_pillars
    self._pillar_names = ['pillar' + str(b) for b in range(n_pillars)]
    super(ReachTransferPillars, self).__init__(random=random)

  def _initialize_with_pos(self, physics, targ_pos, quad_pos, pillar_pos):
    """
    Attempt to initialize the objects in the env to certain positions.
    Assumes that there are no collisions in this position or will fail.
    """
    model = physics.named.model

    physics.named.model.site_pos['target', 'x'] = targ_pos[0]
    physics.named.model.site_pos['target', 'y'] = targ_pos[1]

    azimuth = self.random.uniform(0, 2*np.pi)
    orientation = np.array((np.cos(azimuth/2), 0, 0, np.sin(azimuth/2)))

    z_pos = 0.0  # Start embedded in the floor.
    num_contacts = 1
    num_attempts = 0
    # Move up in 1cm increments until no contacts.
    while num_contacts > 0:
      try:
        with physics.reset_context():
          x_pos = quad_pos[0]
          y_pos = quad_pos[1]
          physics.named.data.qpos['root'][:3] = x_pos, y_pos, z_pos
          physics.named.data.qpos['root'][3:] = orientation
          for i, name in enumerate(self._pillar_names):
            # data.qpos[name + '_x'] = pillar_pos[i][0]
            # data.qpos[name + '_y'] = pillar_pos[i][1]
            model.body_pos[name, 'x'] = pillar_pos[i][0]
            model.body_pos[name, 'y'] = pillar_pos[i][1]
      except control.PhysicsError:
        # We may encounter a PhysicsError here due to filling the contact
        # buffer, in which case we simply increment the height and continue.
        pass
      num_contacts = physics.data.ncon
      z_pos += 0.01
      num_attempts += 1
      if num_attempts > 10000:
        raise RuntimeError('Failed to find a non-contacting configuration.')

  def intialize_pillars_fixed(self, physics):
    data = physics.named.data

    arena_radius = physics.named.model.geom_size['floor', 0]

    spawn_radius = 0.75 * arena_radius
    targ_spawn_radius = 0.8 * arena_radius

    targ_pos = self.random.uniform(-targ_spawn_radius, targ_spawn_radius, size=(2,))
    #HACK!!!
    targ_pos = np.zeros((2,))
    targ_pos[1] = targ_spawn_radius

    quad_pos = self.random.uniform(-spawn_radius, spawn_radius, size=(2,))
    #HACK!!!
    quad_pos = np.zeros((2,))
    quad_pos[1] = -spawn_radius

    # Generate pillar pattern
    offset = 0.75
    pillar_pos = [np.array([offset*x,-0.2*abs(x)],dtype=np.float) for x in range(-2,2+1) ]

    mid_pos = 0.5*(targ_pos + quad_pos)
    # Clip if out of arena bounds
    mid_pos_bound = arena_radius-(2*offset+0.25)
    if mid_pos[0] > mid_pos_bound:
      mid_pos[0] = mid_pos_bound
    elif mid_pos[0] < -mid_pos_bound:
      mid_pos[0] = -mid_pos_bound

    pillar_pos = [x + mid_pos for x in pillar_pos]
    # print('pill', np.stack(pillar_pos))
    # print('targ', targ_pos)

    self._initialize_with_pos(physics, targ_pos, quad_pos, pillar_pos)

  def intialize_pillars_random(self, physics):
    data = physics.named.data

    spawn_radius = 0.75 * physics.named.model.geom_size['floor', 0]
    # Avoid range
    pill2quad = 1.2
    pill2pill = 0.6
    quad2targ = physics.named.model.geom_size['floor', 0]

    quad_pos = None
    pillar_pos = None
    penetrating = True
    retry_count = 0
    while penetrating:
      retry_count += 1
      if retry_count > 20:
        print("Warning!!! Retrying over 20 times")
      penetrating = False

      # Sample target and quadruped
      targ_spawn_radius = 0.8 * physics.named.model.geom_size['floor', 0]
      targ_pos = self.random.uniform(-targ_spawn_radius, targ_spawn_radius, size=(2,))
      quad_pos = self.random.uniform(-spawn_radius, spawn_radius, size=(2,))

      if dist(targ_pos, quad_pos) < quad2targ:
        penetrating=True
        continue

      pillar_pos = []
      for i in range(self._n_pillars):
        middle_pos = (0.5*quad_pos+0.5*targ_pos)
        if i == 0:
          # Put a pillar in the middle
          pillar_pos.append(middle_pos)
        else:
          # average random and the middle
          # pillar_pos.append(0.4*(0.5*quad_pos+0.5*targ_pos) + 0.6*self.random.uniform(-spawn_radius, spawn_radius, size=(2,)))

          # Spawn other pillars randomly
          pillar_pos.append(self.random.uniform(-spawn_radius, spawn_radius, size=(2,)))

          # add noise on top of middle
          # random_pos = middle_pos + 0.5*self.random.uniform(-spawn_radius, spawn_radius, size=(2,))
          # pillar_pos.append(np.clip(random_pos, -spawn_radius, spawn_radius))

        if dist(pillar_pos[i], quad_pos) < pill2quad:
          penetrating=True
          continue
        # combinatorial
        for j in range(i):
          if dist(pillar_pos[i], pillar_pos[j]) < pill2pill:
            penetrating = True
            continue
    self._initialize_with_pos(physics, targ_pos, quad_pos, pillar_pos)

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode.

    Args:
      physics: An instance of `Physics`.

    """
    if self._pretransfer:
      self.intialize_pillars_random(physics)
    else:
      self.intialize_pillars_fixed(physics)

    super(ReachTransferPillars, self).initialize_episode(physics)

  def get_observation(self, physics):
    """Returns an observation to the agent."""
    obs = _common_observations(physics)
    #obs['ball_state'] = physics.ball_state()
    obs['torso_pos'] = physics.torso_pos()
    obs['glob_fwd_vec'] = physics.global_forward_vector()
    obs['target_position'] = physics.target_position()

    obs['pillar_pos'] = physics.body_2d_pose(self._pillar_names)
    # print(obs['pillar_pos'])
    return obs

  def _reach_reward(self, physics):
    arena_radius = physics.named.model.geom_size['floor', 0] * np.sqrt(2)
    target_radius = physics.named.model.site_size['target', 0]
    #workspace_radius = physics.named.model.site_size['workspace', 0]
    #ball_radius = physics.named.model.geom_size['ball', 0]
    if self._sparse:
      # reach_reward = rewards.tolerance(
      #     physics.self_to_target_distance(),
      #     bounds=(0, 2*target_radius), sigmoid='linear',margin=4*target_radius, value_at_margin=0)
      reach_reward = rewards.tolerance(
          physics.self_to_target_distance(),
          bounds=(0, 4.*target_radius), sigmoid='linear',margin=0., value_at_margin=0)
    else:
      reach_reward = rewards.tolerance(
          physics.self_to_target_distance(),
          bounds=(0, target_radius),
          sigmoid='linear',
          margin=arena_radius, value_at_margin=0)
    # return _upright_reward(physics) * (0.1+0.9*reach_reward)
    return _upright_reward(physics) * reach_reward

  def _move_reward(self, physics):
    move_reward = rewards.tolerance(
        physics.torso_velocity()[0],
        bounds=(self._desired_speed, float('inf')),
        margin=self._desired_speed,
        value_at_margin=0.5,
        sigmoid='linear')

    # just have some velocity (any direction)
    # move_reward = rewards.tolerance(
    #     np.linalg.norm(physics.torso_velocity()[0:2]),
    #     bounds=(self._desired_speed, float('inf')),
    #     margin=self._desired_speed,
    #     value_at_margin=0.5,
    #     sigmoid='linear')

    return _upright_reward(physics) * move_reward

  def get_transfer_reward(self, physics):
    # Return transfer reward if we are in pretransfer stage
    if self._pretransfer:
      return self._reach_reward(physics)
    else:
      raise NotImplementedError()

  def get_reward(self, physics):
    """Returns a reward to the agent."""
    if self._pretransfer:
      return self._move_reward(physics)
    else:
      return self._reach_reward(physics)
