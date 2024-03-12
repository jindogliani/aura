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
        self.is_leaf = True
        self.self_score = 0.
        self.self_costs = []
        self.potential_score = 0.
        self.depth = 0
        self.history = []

    @property 
    def all_actions(self):
        if not hasattr(self, '_all_actions'):
            legal_moves_dict = self.state.get_legal_actions()
            self._all_actions = []
            for k, v in legal_moves_dict.items():
                self._all_actions += v
                # if k == 'Forward' or k == 'Flip' or k == 'Swap':
                #     self._all_actions += v
        return self._all_actions
    
    @property
    def q(self):
        return self.potential_score

    def expand(self):
        action = self.all_actions.pop()
        next_state = self.state.move(action)
        child_node = MCTSNode(next_state, parent=self)
        self.is_leaf = False
        child_node.self_score, child_node.self_costs = child_node.state.get_reward
        child_node.depth = self.depth + 1
        child_node.history = self.history + [action]
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
        return self.is_leaf
    
    def is_terminate(self):
        return False
    
    def rollout(self):
        current_rollout_state = self.state
        max_reward = 0.
        max_state = None
        max_costs = [0, 0, 0]
        action_path = []
        best_action_path = []
        rollout_depth = 20 #롤아웃 뎁스 
        for idx in range(rollout_depth):
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
        # self.history = best_action_path
        # parent = self.parent
        # all_action_path = best_action_path
        # while parent is not None:
        #     all_action_path = parent.history + all_action_path
        #     parent = parent.parent
    
        return max_reward, max_state, max_costs, best_action_path


    def rollout_policy(self, possible_moves): 
        action_list = ['Forward', 'Flip', 'Swap', 'Add', 'Delete', 'EmptySwap'] #TODO #HTW
        # action_list = ['Forward', 'Flip', 'Swap', 'Add', 'EmptySwap']
        selected_action = random.choices(action_list)[0]
        if selected_action == 'Swap' and len(possible_moves['EmptySwap']) != 0: #TODO #HTW
            selected_action = 'EmptySwap'
        else: 
            while len(possible_moves[selected_action]) == 0:
                selected_action = random.choices(action_list)[0]
        
        selected_moves = possible_moves[selected_action]
        return selected_moves[np.random.randint(len(selected_moves))]
    
    def backpropagate(self, max_reward):
        gamma = 0.8
        self.potential_score += max_reward
        if self.parent:
            self.parent.backpropagate(gamma * max_reward)

    def is_fully_expanded(self):
        return len(self.all_actions) == 0

