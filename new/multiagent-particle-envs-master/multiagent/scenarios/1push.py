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
        num_agents = 1
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.05
            agent.density = 1
            
        world.landmarks = [Landmark() for i in range(2)]

        world.landmarks[0].name = 'landmark'
        world.landmarks[0].collide = False
        world.landmarks[0].movable = False

        world.landmarks[1].name = 'target'
        world.landmarks[1].collide = True
        world.landmarks[1].movable = True
        world.landmarks[1].density = 25
        world.landmarks[1].size = 0.1
        world.landmarks[1].initial_mass = 15


        self.reset_world(world)   
        return world

    def reset_world(self, world):
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        world.landmarks[0].color = np.array([0.1, 0.1, 0.1])
        world.landmarks[1].color = np.array([0.9, 0.9, 0.9])
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        world.landmarks[0].state.p_pos = np.array([0, 0])
        world.landmarks[1].state.p_pos = np.array([0.4, 0])
        world.agents[0].state.p_pos = np.array([0.8, 0])

        
        

    def reward(self, agent, world):
        dis_2landmark = np.sum(np.square(world.landmarks[1].state.p_pos - world.landmarks[0].state.p_pos))
        dis_2target = np.sum(np.square(agent.state.p_pos - world.landmarks[1].state.p_pos))
#        print(dis_2landmark)
#        return - dis_2landmark
#        return -dis_2target
        return - dis_2landmark - 0.2 * dis_2target

    def observation(self, agent, world):
#        other_pos = []
#        for other in world.agents:
#            if other is agent: continue
#            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return  np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + 
                               [world.landmarks[1].state.p_pos])

    def done(self, agent, world):
        dis_2landmark = np.sum(np.square(world.landmarks[1].state.p_pos - world.landmarks[0].state.p_pos))
        return True if dis_2landmark <= 0.015 else False
        
        
        
        