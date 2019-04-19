# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 14:55:44 2019

@author: Jack Lee
"""

import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        num_agents = 2
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.05
            agent.density = 1
            
        world.landmarks = [Landmark() for i in range(4)]

        for i,landmark in enumerate(world.landmarks):
            if i % 2 == 1:#1 3 5 ... landmarks 球门
                landmark.name = 'landmark{}'.format(i)
                landmark.collide = False
                landmark.movable = False
            else:#0 2 4 target 球
                landmark.name = 'target{}'.format(i)
                landmark.collide = True
                landmark.movable = True
                landmark.density = 25
                landmark.size = 0.1
                landmark.initial_mass = 15
            
            
#        world.landmarks[0].name = 'landmark1'
#        world.landmarks[0].collide = False
#        world.landmarks[0].movable = False
#
#        world.landmarks[1].name = 'target1'
#        world.landmarks[1].collide = True
#        world.landmarks[1].movable = True
#        world.landmarks[1].density = 25
#        world.landmarks[1].size = 0.1
#        world.landmarks[1].initial_mass = 15
#
#        world.landmarks[0].name = 'landmark2'
#        world.landmarks[0].collide = False
#        world.landmarks[0].movable = False
#
#        world.landmarks[1].name = 'target2'
#        world.landmarks[1].collide = True
#        world.landmarks[1].movable = True
#        world.landmarks[1].density = 25
#        world.landmarks[1].size = 0.1
#        world.landmarks[1].initial_mass = 15


        self.reset_world(world)   
        return world

    def reset_world(self, world):
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        world.landmarks[0].color = np.array([0.9, 0.9, 0.9])
        world.landmarks[1].color = np.array([0.1, 0.1, 0.1])
        world.landmarks[2].color = np.array([0.9, 0.9, 0.9])
        world.landmarks[3].color = np.array([0.1, 0.1, 0.1])
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        world.landmarks[1].state.p_pos = np.array([0, 0.2])
        world.landmarks[0].state.p_pos = np.array([0.4, 0.1])
        world.landmarks[3].state.p_pos = np.array([0, -0.2])
        world.landmarks[2].state.p_pos = np.array([0.4, -0.1])
        
        
        world.agents[0].state.p_pos = np.array([0.8, -0.1])#下面那个
        world.agents[1].state.p_pos = np.array([0.8, 0.1])#上面

        
        

    def reward(self, agent, world):
        indx = int(agent.name[-1])
        target = world.landmarks[2 * indx]
        landmark = world.landmarks[2 * indx + 1]
        dis_agent2target = np.sum(np.square(agent.state.p_pos - target.state.p_pos))
        dis_target2landmark = np.sum(np.square(target.state.p_pos - landmark.state.p_pos))
        r = - 2*dis_target2landmark
#        if self.is_collision(agent, target):
#            r += 0.5
        return - dis_target2landmark 
#        print()
#        return -dis_agent2target 
#        return r
        
        

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False



    def observation(self, agent, world):
        indx = int(agent.name[-1])
#        target = world.landmarks[2 * indx]
#        landmark = world.landmarks[2 * indx + 1]
#        other_pos = [other.state.p_pos - agent.state.p_pos for other in world.agents if other is not agent]
#        target_pos = [landmark.state.p_pos for landmark in world.landmarks if 'target' in landmark.name]
#        return  np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + other_pos + target_pos)
        return  np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + [world.landmarks[2 * indx].state.p_pos])

    def done(self, agent, world):
        
        return False