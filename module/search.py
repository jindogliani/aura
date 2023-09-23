import os
import sys
sys.path.insert(0,os.path.join(os.getcwd(), 'module'))
from nodes import MCTSNode
from scene import SceneState
from museum import MuseumScene
import time
from tqdm import tqdm


class MonteCarloTreeSearch:
    def __init__(self, node: MCTSNode):
        self.root = node

    def best_action(self):
        simulations_number = len(self.root.all_actions)
        pbar = tqdm(range(0, simulations_number))
        for idx in pbar:
            v = self.tree_policy()
            reward = v.rollout()
            pbar.set_description("Reward: %f"%reward)
            v.backpropagate(reward)
        # exploitation only
        return self.root.best_child(c_param=0.0)

    def tree_policy(self):
        current_node = self.root
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node
    
if __name__ == "__main__":
    for epoch in range(100):
        print("Epoch: %d / %d "%(epoch, 200))
        currnet_node = MCTSNode(state=SceneState(MuseumScene()))
        mcts = MonteCarloTreeSearch(currnet_node)
        best_node = mcts.best_action()
        currnet_node = best_node