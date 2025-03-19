import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from typing import Callable, List, Tuple, Dict, Union, Optional
from multiagent.core import Debris, Satellite, SatWorld
from multiagent.multi_discrete import MultiDiscrete
import random
import math

# update bounds to center around agent
cam_range = 2

# vectorized wrapper 
# assumes all environments have the same observation and action space
class BatchMultiAgentEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n, time):
        obs_n = []
        shared_obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, shared_obs, reward, done, _ = env.step(action_n[i:(i+env.n)], time)
            i += env.n
            obs_n += obs
            shared_obs_n += shared_obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, shared_obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        shared_obs_n = []
        for env in self.env_batch:
            obs, shared_obs = env.reset()
            obs_n += obs
            shared_obs_n += shared_obs
        return obs_n, shared_obs

    # render environment
    def render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n

############## Satellite Environments #######################

class SatelliteMultiAgentBaseEnv(gym.Env):
    """
        Base environment for all multi-agent environments
    """
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, world:SatWorld, reset_callback:Callable=None, 
                    reward_callback:Callable=None,
                    observation_callback:Callable=None, 
                    info_callback:Callable=None,
                    done_callback:Callable=None, 
                    shared_viewer:bool=True, 
                    discrete_action:bool=True,
                    local_obs:bool=False) -> None:
        self.world = world
        self.world_length = self.world.world_length
        self.current_step = 0
        self.agents = self.world.policy_agents

        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        self.num_agents = len(world.policy_agents)  # for compatibility with offpolicy baseline envs
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        
        self.local_obs = local_obs # whether we want just local information to be included in the observation or not

        self.discrete_action_space = discrete_action
        self.discrete_action_input = False# if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        
        self.force_discrete_action = world.discrete_action if hasattr(world, 
                                                'discrete_action') else False # if true, even the action is continuous,  action will be performed discretely
        self.shared_reward = world.collaborative if hasattr(world, 
                                                    'collaborative') else False  # if true, every agent has the same reward
        self.time = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []   # adding this for compatibility with MAPPO code
        share_obs_dim = 0
        for agent in self.agents:
            total_action_space = []

            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            else:
                u_action_space = spaces.Box(low=-agent.u_range, 
                                            high=+agent.u_range, 
                                            shape=(world.dim_p,), 
                                            dtype=np.float32)
            if agent.movable:
                total_action_space.append(u_action_space)

            # communication action space
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)
            else:
                c_action_space = spaces.Box(low=0.0, 
                                            high=1.0, 
                                            shape=(world.dim_c,), 
                                            dtype=np.float32)

            if not agent.silent:
                total_action_space.append(c_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, 
                # so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) 
                        for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] 
                                        for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])

            # observation space
            obs_dim = len(observation_callback(agent=agent, world=self.world, 
                                                local_obs=self.local_obs))
            share_obs_dim += obs_dim
            self.observation_space.append(spaces.Box(low=-np.inf, 
                                                    high=+np.inf, 
                                                    shape=(obs_dim,), 
                                                    dtype=np.float32))

            agent.action.c = np.zeros(self.world.dim_c)
        
        self.share_observation_space = [spaces.Box(low=-np.inf, 
                                                    high=+np.inf, 
                                                    shape=(share_obs_dim,), 
                                                    dtype=np.float32) 
                                                    for _ in range(self.n)]
        

        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)
    
    def step(self, action_n:List):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    # get info used for benchmarking
    def _get_info(self, agent:Agent) -> Dict:
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent:Agent) -> np.ndarray:
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent=agent, world=self.world, 
                                        local_obs=self.local_obs)

    
    def _get_shared_obs(self) -> np.ndarray:  # get shared observation for the environment
        if self.shared_obs_callback is None:
            return None
        return self.shared_obs_callback(self.world)
        
    
    def _get_done(self, agent:Agent) -> bool:   # get dones for a particular agent
        if self.done_callback is None:
            if self.current_step >= self.world_length:
                return True
            else:
                return False
        return self.done_callback(agent, self.world)

    
    def _get_reward(self, agent:Agent) -> float: # get reward for a particular agent
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

   
    def _set_action(self, action, agent:Agent, action_space, 
                    time:Optional=None) -> None:  # set env action for a particular agent
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index+s)])
                index += s
            action = act
        else:
            action = [action]

        # actions: [None, ←, →, ↓, ↑, comm1, comm2]
        if agent.movable:

            if self.discrete_action_input:

                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete action
                if action[0] == 1: agent.action.u[0] = -1.0
                if action[0] == 2: agent.action.u[0] = +1.0
                if action[0] == 3: agent.action.u[1] = -1.0
                if action[0] == 4: agent.action.u[1] = +1.0
            else:
                if self.force_discrete_action:		
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:

                    agent.action.u[0] += action[0][1] - action[0][2]
                    agent.action.u[1] += action[0][3] - action[0][4]

                else:

                    agent.action.u = action[0]
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity # NOTE: refer offpolicy/envs/mpe/environment.py -> MultiAgentEnv._set_action() for non-silent agent
            action = action[1:]
        
        assert len(action) == 0# make sure we used all elements of action

    
    def _reset_render(self) -> None:  # reset rendering assets
        self.render_geoms = None
        self.render_geoms_xform = None


    def render(self, mode:str='human', close:bool=False) -> List:     # render environment


        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent: continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:

                from multiagent import rendering
                self.viewers[i] = rendering.Viewer(700,700)

        # create rendering geometry
        if self.render_geoms is None:

            from multiagent import rendering
            self.render_geoms = []
            self.render_geoms_xform = []

            self.comm_geoms = []

            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()

                entity_comm_geoms = []

                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)

                    if not entity.silent:
                        dim_c = self.world.dim_c
                        # make circles to represent communication
                        for ci in range(dim_c):
                            comm = rendering.make_circle(entity.size / dim_c)
                            comm.set_color(1, 1, 1)
                            comm.add_attr(xform)
                            offset = rendering.Transform()
                            comm_size = (entity.size / dim_c)
                            offset.set_translation(ci * comm_size * 2 -
                                                    entity.size + comm_size, 0)
                            comm.add_attr(offset)
                            entity_comm_geoms.append(comm)

                else:
                    geom.set_color(*entity.color)
                    if entity.channel is not None:
                        dim_c = self.world.dim_c
                        # make circles to represent communication
                        for ci in range(dim_c):
                            comm = rendering.make_circle(entity.size / dim_c)
                            comm.set_color(1, 1, 1)
                            comm.add_attr(xform)
                            offset = rendering.Transform()
                            comm_size = (entity.size / dim_c)
                            offset.set_translation(ci * comm_size * 2 -
                                                    entity.size + comm_size, 0)
                            comm.add_attr(offset)
                            entity_comm_geoms.append(comm)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)
                self.comm_geoms.append(entity_comm_geoms)

        


            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)
                for entity_comm_geoms in self.comm_geoms:
                    for geom in entity_comm_geoms:
                        viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from multiagent import rendering

            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0]-cam_range,
                                        pos[0]+cam_range,
                                        pos[1]-cam_range,
                                        pos[1]+cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
                if 'agent' in entity.name:
                    self.render_geoms[e].set_color(*entity.color, alpha=0.5)

                    if not entity.silent:
                        for ci in range(self.world.dim_c):
                            color = 1 - entity.state.c[ci]
                            self.comm_geoms[e][ci].set_color(
                                color, color, color)
                else:
                    self.render_geoms[e].set_color(*entity.color)
                    if entity.channel is not None:
                        for ci in range(self.world.dim_c):
                            color = 1 - entity.channel[ci]
                            self.comm_geoms[e][ci].set_color(
                                color, color, color)

            # render the graph connections
            if hasattr(self.world, 'graph_mode'):
                if self.world.graph_mode:
                    edge_list = self.world.edge_list.T
                    assert edge_list is not None, ("Edge list should not be None")
                    for entity1 in self.world.entities:
                        for entity2 in self.world.entities:
                            e1_id, e2_id = entity1.global_id, entity2.global_id
                            if e1_id == e2_id:
                                continue
                            # if edge exists draw a line
                            if [e1_id, e2_id] in edge_list.tolist():
                                src = entity1.state.p_pos
                                dest = entity2.state.p_pos
                                self.viewers[i].draw_line(start=src, end=dest)

            # render to display or array
            results.append(self.viewers[i].render(
                        return_rgb_array = mode=='rgb_array'))

        return results

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent:Agent) -> List:

        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []

        for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
            for distance in np.linspace(range_min, range_max, 3):
                dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
        # add origin
        dx.append(np.array([0.0, 0.0]))
        
        return dx


