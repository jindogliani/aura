import os
import sys
sys.path.insert(0,os.path.join(os.getcwd(), 'module'))
from nodes import MCTSNode
from scene import SceneState
from museum import MuseumScene


class MonteCarloTreeSearch:
    def __init__(self, node: MCTSNode):
        self.root = node

    def best_action(self, simulations_number):
        for idx in range(0, simulations_number):
            print("%d/%d"%(idx, simulations_number))
            v = self.tree_policy()
            reward = v.rollout()
            print("Reward :", reward)
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
    for _ in range(1000):
        currnet_node = MCTSNode(state=SceneState(MuseumScene()))
        mcts = MonteCarloTreeSearch(currnet_node)
        best_node = mcts.best_action(100)
        currnet_node = best_node