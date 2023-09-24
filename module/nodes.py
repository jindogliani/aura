import os
import sys
sys.path.insert(0,os.path.join(os.getcwd(), 'module'))
import numpy as np
from collections import defaultdict
from scene import SceneState
from tqdm import tqdm
import time

class MCTSNode(object):
    def __init__(self, state: SceneState, parent=None):
        self.state = state
        self.parent = parent 
        self.children = []
        self.is_leap = True
        self.self_score = 0.
        self.potential_score = 0.
        self.depth = 0

    @property
    def all_actions(self):
        if not hasattr(self, '_all_actions'):
            self._all_actions = self.state.get_legal_actions()
        return self._all_actions
    
    @property
    def q(self):
        return self.potential_score

    def expand(self):
        action = self.all_actions.pop()
        next_state = self.state.move(action)
        child_node = MCTSNode(next_state, parent=self)
        self.is_leap = False
        child_node.self_score = child_node.state.get_reward
        child_node.depth = self.depth + 1
        self.children.append(child_node)
        return child_node
    
    def selection(self):
        if self.is_leaf_node():
            return self
        else:
            return self.best_child()
        
    def best_child(self, c_param=1.4):
        choices_weights = [
            c.q
            for c in self.children
        ]
        return self.children[np.argmax(choices_weights)]
    
    def is_leaf_node(self):
        return self.is_leap
    
    def is_terminate(self):
        if self.depth > 20:
            return True
        else:
            return False
    
    def rollout(self):
        current_rollout_state = self.state
        max_reward = 0.
        max_state = None
        for idx in range(10):
            # print("Rollout: ", idx)
            possible_moves = current_rollout_state.get_legal_actions()
            action = self.rollout_policy(possible_moves)
            next_rollout_state = current_rollout_state.move(action)
            reward = next_rollout_state.get_reward
            current_rollout_state = next_rollout_state
            if reward > max_reward:
                max_reward = reward
                max_state = next_rollout_state
        
        return max_reward, max_state


    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]
    
    def backpropagate(self, max_reward):
        self.potential_score = max_reward
        if self.parent:
            self.parent.backpropagate(max_reward)

    def is_fully_expanded(self):
        return len(self.all_actions) == 0

