import os
import sys
sys.path.insert(0,os.path.join(os.getcwd(), 'module'))
import numpy as np
from collections import defaultdict
from scene import SceneState
from tqdm import tqdm
import time
import random


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
            legal_moves_dict = self.state.get_legal_actions()
            self._all_actions = []
            for v in legal_moves_dict.values():
                self._all_actions += v
        return self._all_actions
    
    @property
    def q(self):
        return self.potential_score

    def expand(self):
        action = self.all_actions.pop()
        next_state = self.state.move(action)
        child_node = MCTSNode(next_state, parent=self)
        self.is_leap = False
        child_node.self_score, _ = child_node.state.get_reward
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
        return False
    
    def rollout(self):
        current_rollout_state = self.state
        max_reward = 0.
        max_state = None
        max_costs = [0, 0, 0]
        action_path = []
        for idx in range(10):
            possible_moves = current_rollout_state.get_legal_actions()
            action = self.rollout_policy(possible_moves)
            next_rollout_state = current_rollout_state.move(action)
            action_path.append(action)
            reward, costs = next_rollout_state.get_reward
            current_rollout_state = next_rollout_state
            if reward >= max_reward:
                max_reward = reward
                max_state = next_rollout_state
                max_costs = costs
                best_action_path = action_path
        
        return max_reward, max_state, max_costs, best_action_path


    def rollout_policy(self, possible_moves):
        action_list = ['Forward', 'Flip', 'Swap', 'Add', 'Delete']
        action_weight = [0.1, 0.0, 0.5, 0.2, 0.2]
        selected_action = random.choices(action_list, weights=action_weight, k=1)[0]
        while len(possible_moves[selected_action]) == 0:
            selected_action = random.choices(action_list, weights=action_weight, k=1)[0]
        selected_moves = possible_moves[selected_action]
        return selected_moves[np.random.randint(len(selected_moves))]
    
    def backpropagate(self, max_reward):
        self.potential_score = max_reward
        if self.parent:
            self.parent.backpropagate(max_reward)

    def is_fully_expanded(self):
        return len(self.all_actions) == 0

