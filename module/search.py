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
        if not current_node.is_fully_expanded():
            return current_node.expand()
        else:
            raise "Fully expanded Tree"
    
if __name__ == "__main__":
    best_node = None
    pbar = tqdm(range(200))
    for epoch in pbar:
        pbar.set_description("Epoch: %d / %d "%(epoch, 200))
        currnet_node = MCTSNode(state=SceneState(MuseumScene()))
        mcts = MonteCarloTreeSearch(currnet_node)
        potential_node = mcts.best_action() #potentia score 가 가장 큰 노드
        if best_node == None:
            best_node = potential_node
        else:
            if potential_node.self_score > best_node.self_score:
                best_node = potential_node
        pbar.set_description("Best Score: %f"%best_node.self_score)
        currnet_node = potential_node