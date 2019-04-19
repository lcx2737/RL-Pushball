# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 14:55:44 2019

@author: Jack Lee
"""

import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario 
import random 
from multiagent.football import Pair

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
            
        world.pairs = [Pair(i) for i in range(2)]


        self.reset_world(world)   
        return world

    def reset_world(self, world):
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        # random properties for landmarks
        
        for pair in world.pairs:
            pair.reset()
            
        world.pairs[0].set_ball_loc([0.4, 0.3])
        world.pairs[0].set_goal_loc([0, 0.2])
        world.pairs[1].set_ball_loc([0.4, -0.3])
        world.pairs[1].set_goal_loc([0, -0.2])
        
        for i, landmark in enumerate(world.landmarks):
#            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
#        world.landmarks[0].state.p_pos = np.array([0, 0.01])
#        world.landmarks[0].state.p_pos[1] += random.uniform(0.5,-0.5)
#        world.landmarks[1].state.p_pos = np.array([0.4, 0])
#        
#        world.landmarks[1].state.p_pos[1] += random.uniform(0.3,-0.3)
        
#        world.agents[0].state.p_pos = np.array([0.8, -0.1])
#        world.agents[1].state.p_pos = np.array([0.8, 0.1])

        
        

    def reward(self, agent, world):
#        dis_2landmark = np.sum(np.square(world.landmarks[1].state.p_pos - world.landmarks[0].state.p_pos))
#        dis_2target = np.sum(np.square(agent.state.p_pos - world.landmarks[1].state.p_pos))
##        print(dis_2landmark)
##        return - dis_2landmark
##        return -dis_2target
#        dis = [np.sum(np.square(entity.state.p_pos - world.landmarks[1].state.p_pos)) for entity in world.agents]
#        index = dis.index(min(dis))
#        if world.agents[index] is not agent:
#            return - 5 - 5 * dis_2target
#        else:   
#            return - 5 * dis_2landmark
        return 0

    def observation(self, agent, world, goal):
        for each in world.pairs:
            each.in_goal = each.check_goal()
#        other_pos = []
#        for other in world.agents:
#            if other is agent: continue
#            other_pos.append(other.state.p_pos - agent.state.p_pos)
        other_pos = [other.state.p_pos - agent.state.p_pos for other in world.agents if other is not agent]
#        return  np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + other_pos + 
#                               [world.landmarks[0].state.p_pos - world.landmarks[1].state.p_pos] + [world.landmarks[1].state.p_pos - agent.state.p_pos])
        return  np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + other_pos + 
                               [world.pairs[goal].goal.state.p_pos] + [world.pairs[goal].ball.state.p_pos])

    def done(self, agent, world):
        return [each.in_goal for each in world.pairs]
        
        
        
        
    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False