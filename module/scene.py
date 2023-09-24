import os
import sys
sys.path.insert(0,os.path.join(os.getcwd(), 'module'))
import numpy as np
from museum import *
    
class SceneState(object):
    def __init__(self, state: MuseumScene):
        self.scene = state
        self.terminal = False
        self.threshold = 0.1

    def get_legal_actions(self):
        legal_actions = self.scene.get_legal_actions()
        return legal_actions
    
    @property
    def get_reward(self):
        total_cost, costs = self.scene.evaluation()
        reward = (1-total_cost) if (1-total_cost) > 0 else 0
        # reward = self.scene.evaluation()
        return reward, costs
    
    def move(self, action_tup):
        #action (Action, art_id, wall_id)
        new_scene = self.scene.do_action(*action_tup)
        # self.scene.update_scene(new_scene)
        new_state = MuseumScene()
        new_state.update_scene(new_scene)
        return SceneState(new_state)
    

    def is_terminal(self):
        return self.terminal