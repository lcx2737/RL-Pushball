from multiagent.core import Landmark
import numpy as np


class Goal(Landmark):
    def __init__(self):
        super(Landmark, self).__init__()
        self.reset_goal()
        
    def reset_goal(self):
        self.collide = False
        self.movable = False
        self.color = np.array([0.1, 0.1, 0.1])
        self.state.p_vel = np.zeros([2,])


class Ball(Landmark):
    def __init__(self):
        super(Landmark, self).__init__()
        self.reset_ball()
        
    def reset_ball(self):
        self.collide = True
        self.movable = True
        self.color = np.array([0.9, 0.9, 0.9])
        self.density = 25
        self.size = 0.1
        self.initial_mass = 15
        self.state.p_vel = np.zeros([2,])


class Pair(object):
    def __init__(self, number:int):
        self.ball = Ball()
        self.goal = Goal()
        self.number = number
        self.in_goal = False
    
    def set_goal_loc(self, a):
        self.goal.state.p_pos = np.array(a)
        
    def set_ball_loc(self,a):
        self.ball.state.p_pos = np.array(a)
        
    def check_goal(self):
        delta_pos = self.ball.state.p_pos - self.goal.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = self.ball.size + self.goal.size
        if dist < dist_min:
            self.ball.color = np.array([0.1, 0.1, 0.1])
            self.ball.collide = False
            return True  
        else: 
            return False
        
    def reset(self):
        self.ball.reset_ball()
        self.goal.reset_goal()
        self.in_goal = False
        
    def get_squared_dis(self):
        return np.sum(np.square(self.ball.state.p_pos - self.goal.state.p_pos))
        
        
        
