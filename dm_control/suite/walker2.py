# Copyright 2017 The dm_control Authors.
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

"""Planar Walker Domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards


_DEFAULT_TIME_LIMIT = 25
_CONTROL_TIMESTEP = .025

# Minimal height of torso over foot above which stand reward is 1.
_STAND_HEIGHT = 1.2

# Horizontal speeds (meters/second) above which move reward is 1.
_BACK_SPEED = -1
_BACKJOG_SPEED = -4
_WALK_SPEED = 1
_JOG_SPEED = 4
_RUN_SPEED = 8


SUITE = containers.TaggedTasks()


def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  return common.read_model('walker.xml'), common.ASSETS

@SUITE.add()
def back(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Run task."""
  return objective_speed_env(_BACK_SPEED, time_limit=time_limit, random=random, environment_kwargs=environment_kwargs)

@SUITE.add()
def backjog(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Run task."""
  return objective_speed_env(_BACKJOG_SPEED, time_limit=time_limit, random=random, environment_kwargs=environment_kwargs)

@SUITE.add()
def walk(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Run task."""
  return objective_speed_env(_WALK_SPEED, time_limit=time_limit, random=random, environment_kwargs=environment_kwargs)

@SUITE.add()
def jog(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Jog task."""
  return objective_speed_env(_JOG_SPEED, time_limit=time_limit, random=random, environment_kwargs=environment_kwargs)

@SUITE.add()
def run(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Run task."""
  return objective_speed_env(_RUN_SPEED, time_limit=time_limit, random=random, environment_kwargs=environment_kwargs)

def objective_speed_env(objective_speed, time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Run task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = PlanarWalkerObjectiveSpeed(objective_speed=objective_speed, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)


class Physics(mujoco.Physics):
  """Physics simulation with additional features for the Walker domain."""

  def torso_upright(self):
    """Returns projection from z-axes of torso to the z-axes of world."""
    return self.named.data.xmat['torso', 'zz']

  def torso_height(self):
    """Returns the height of the torso."""
    return self.named.data.xpos['torso', 'z']

  def horizontal_velocity(self):
    """Returns the horizontal velocity of the center-of-mass."""
    return self.named.data.sensordata['torso_subtreelinvel'][0]

  def orientations(self):
    """Returns planar orientations of all bodies."""
    return self.named.data.xmat[1:, ['xx', 'xz']].ravel()


class PlanarWalkerObjectiveSpeed(base.Task):
  """A planar walker task."""

  def __init__(self, objective_speed, random=None):
    """Initializes an instance of `PlanarWalker`.

    Args:
      objective_speed: A float. This specifies a target horizontal velocity for
        the walking task.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    self._objective_speed = objective_speed
    super(PlanarWalkerObjectiveSpeed, self).__init__(random=random)

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode.

    In 'standing' mode, use initial orientation and small velocities.
    In 'random' mode, randomize joint angles and let fall to the floor.

    Args:
      physics: An instance of `Physics`.

    """
    randomizers.randomize_limited_and_rotational_joints(physics, self.random)
    super(PlanarWalkerObjectiveSpeed, self).initialize_episode(physics)

  def get_observation(self, physics):
    """Returns an observation of body orientations, height and velocites."""
    obs = collections.OrderedDict()
    obs['orientations'] = physics.orientations()
    obs['height'] = physics.torso_height()
    obs['velocity'] = physics.velocity()
    return obs

  def get_reward(self, physics):
    """Returns a reward to the agent."""
    standing = rewards.tolerance(physics.torso_height(),
                                 bounds=(_STAND_HEIGHT, float('inf')),
                                 margin=_STAND_HEIGHT/2)
    upright = (1 + physics.torso_upright()) / 2
    stand_reward = (3*standing + upright) / 4
    move_reward = rewards.tolerance(physics.horizontal_velocity(),
                                    bounds=(self._objective_speed, self._objective_speed+0.01),
                                    margin=abs(self._objective_speed)/2,
                                    value_at_margin=0.5,
                                    sigmoid='linear')
    return stand_reward * (5*move_reward + 1) / 6
