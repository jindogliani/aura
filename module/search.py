import os
import sys
from time import localtime, time
sys.path.insert(0,os.path.join(os.getcwd(), 'module'))
from nodes import MCTSNode
from scene import SceneState
from museum import MuseumScene
# import time
from tqdm import tqdm
import pickle

date = '+' + '(' + str(localtime(time()).tm_mon) +'-'+ str(localtime(time()).tm_mday) +'-'+ str(localtime(time()).tm_hour) + '-'+ str(localtime(time()).tm_min) + ')'

print(date)

class MonteCarloTreeSearch:
    def __init__(self, node: MCTSNode):
        self.root = node
        self.root.is_leap = False

    def best_action(self, simulations_number):
        pbar = tqdm(range(0, simulations_number))
        best_state = None
        best_reward = 0.
        for idx in pbar:
            v = self.tree_policy()
            cur_depth = v.depth
            reward, max_state, max_costs = v.rollout()
            if reward > best_reward:
                print("Update Best Reward: ", reward)
                best_reward = reward
                best_state = max_state
            pbar.set_description("Depth : %d, Reward: %f [Goal : %f, Regularization : %f, Similarity : %f]"%(cur_depth, reward, max_costs[0], max_costs[1], max_costs[2]))
            v.backpropagate(reward)
        # exploitation only
        return best_state

    def tree_policy(self):
        current_node = self.root
        while not current_node.is_terminate():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node
    
if __name__ == "__main__":
    Tree = MonteCarloTreeSearch(MCTSNode(SceneState(MuseumScene())))
    best_state = Tree.best_action(30000)
    best_scene = best_state.scene.scene_data
    #dictionary to pickle data
    with open('best_scene_901_30000' + date +'.pickle', 'wb') as f:
        pickle.dump(best_scene, f, pickle.HIGHEST_PROTOCOL)