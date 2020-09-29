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

"""Point-mass domain."""

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
import numpy as np

_DEFAULT_TIME_LIMIT = 5
SUITE = containers.TaggedTasks()


def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  return common.read_model('pm2.xml'), common.ASSETS


#@SUITE.add('benchmarking', 'easy')
@SUITE.add()
def origin(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the easy point_mass task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = PointMass(randomize_target=False, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)

@SUITE.add()
def target(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the easy point_mass task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = PointMass(randomize_target=True, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


# @SUITE.add()
# def hard(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
#   """Returns the hard point_mass task."""
#   physics = Physics.from_xml_string(*get_model_and_assets())
#   task = PointMass(randomize_gains=True, random=random)
#   environment_kwargs = environment_kwargs or {}
#   return control.Environment(
#       physics, task, time_limit=time_limit, **environment_kwargs)


class Physics(mujoco.Physics):
  """physics for the point_mass domain."""

  def mass_to_target(self):
    """Returns the vector from mass to target in global coordinate."""
    return (self.named.data.geom_xpos['target'] -
            self.named.data.geom_xpos['pointmass'])

  def target_position(self):
    return self.named.data.geom_xpos['target'][:2]

  def mass_to_target_dist(self):
    """Returns the distance from mass to the target."""
    return np.linalg.norm(self.mass_to_target()[:2])


class PointMass(base.Task):
  """A point_mass `Task` to reach target with smooth reward."""

  def __init__(self, randomize_target, random=None):
    """Initialize an instance of `PointMass`.

    Args:
      randomize_target: A `bool`, whether to randomize the target.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    self._randomize_target = randomize_target
    super(PointMass, self).__init__(random=random)

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode.

       If _randomize_gains is True, the relationship between the controls and
       the joints is randomized, so that each control actuates a random linear
       combination of joints.

    Args:
      physics: An instance of `mujoco.Physics`.
    """
    randomizers.randomize_limited_and_rotational_joints(physics, self.random)
    pm_pos = physics.position()
    min_dist = 0.
    if self._randomize_target:
      target_pos = None
      found = False
      for i in range(20):
        target_pos = self.random.uniform(-0.24,0.24,(2))
        if np.linalg.norm(pm_pos - target_pos) > min_dist:
          found = True
          break
      if not found:
        raise ValueError("Did not find valid configuration.")

      physics.named.model.geom_pos['target', 'x'] = target_pos[0]
      physics.named.model.geom_pos['target', 'y'] = target_pos[1]
      #physics.named.model.geom_pos['target', 'y'] = radius * np.cos(angle)
    super(PointMass, self).initialize_episode(physics)

  def get_observation(self, physics):
    """Returns an observation of the state."""
    obs = collections.OrderedDict()
    obs['position'] = physics.position()
    obs['velocity'] = physics.velocity()
    obs['target'] = physics.target_position()
    return obs

  def get_reward(self, physics):
    """Returns a reward to the agent."""
    target_size = physics.named.model.geom_size['target', 0]
    near_target = rewards.tolerance(physics.mass_to_target_dist(),
                                    bounds=(0, target_size), margin=target_size*3, value_at_margin=0.2, sigmoid='long_tail')
    # control_reward = rewards.tolerance(physics.control(), margin=1,
    #                                    value_at_margin=0,
    #                                    sigmoid='quadratic').mean()
    # small_control = (control_reward + 4) / 5
    return near_target #* small_control
