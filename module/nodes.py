import os
import sys
sys.path.insert(0,os.path.join(os.getcwd(), 'module'))
import numpy as np
from collections import defaultdict
from scene import SceneState

class MCTSNode(object):
    def __init__(self, state: SceneState, parent=None):
        self._number_of_visits = 0.
        self._results = defaultdict(int)
        self.cost_value = 0.
        self.state = state #Museum Scene what do i need?
        self.parent = parent 
        self.children = []

    @property
    def all_actions(self):
        if not hasattr(self, '_all_actions'):
            self._all_actions = self.state.get_legal_actions()
        return self._all_actions
    
    @property
    def q(self):
        #q-value
        pass

    @property
    def n(self):
        return self._number_of_visits

    def expand(self):
        action = self.all_actions.pop()
        next_state = self.state.move(action)
        child_node = MCTSNode(next_state, parent=self)
        self.children.append(child_node)
        return child_node
    
    def is_terminal_node(self):
        return self.state.is_terminal()
    
    def rollout(self):
        current_rollout_state = self.state
        while not current_rollout_state.is_terminal():
            possible_moves = current_rollout_state.get_legal_actions()
            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.move(action)
        return current_rollout_state.get_reward

    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]
    
    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expanded(self):
        return len(self.all_actions) == 0
    
    def best_child(self, c_param=1.4):
        choices_weights = [
            (c.q / (c.n)) + c_param * np.sqrt((2 * np.log(self.n) / (c.n)))
            for c in self.children
        ]
        return self.children[np.argmax(choices_weights)]

