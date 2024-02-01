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
import numpy as np
date = '2023_0+' + '(' + str(localtime(time()).tm_mon) +'-'+ str(localtime(time()).tm_mday) +'-'+ str(localtime(time()).tm_hour) + '-'+ str(localtime(time()).tm_min) + ')'
# date = time.time()

print(date)

class MonteCarloTreeSearch:
    def __init__(self, node: MCTSNode):
        self.root = node
        self.root.is_leaf = False

    def best_action(self, simulations_number):
        # pbar = tqdm(range(0, simulations_number))
        best_state = None
        best_reward = -np.inf
        idx = 0
        while True:
            idx += 1
            v = self.tree_policy()
            cur_depth = v.depth
            reward, max_state, max_costs_, best_action_path = v.rollout()
            art_num = len(v.state.scene.scene_data)
            if idx % 100 == 0:
                print(idx)
            if v.self_score > best_reward:
                best_reward = v.self_score
                best_state = v.state
                best_art_num = len(best_state.scene.scene_data)
                max_costs = v.self_costs
                best_history = v.history
                print("Update Best Reward: %f  Number: %d [Goal : %f, Regularization : %f, Similarity : %f, Num : %f, Artists : %f]"%(best_reward, best_art_num, max_costs[0], max_costs[1], max_costs[2], max_costs[3], max_costs[4]))
                print("Saved path: " + date +'.pickle') 
                with open(date +'.pickle', 'wb') as f:
                    pickle.dump(best_state, f, pickle.HIGHEST_PROTOCOL)

                for tup in v.history:
                    print(tup)

                    if len(v.history) > 40:
                        return best_state, v.history

            # print("Depth : %d, Reward: %f, Art Num: %d [Goal : %f, Regularization : %f, Similarity : %f, Num : %f, Artists : %f]"%(cur_depth, reward, art_num, max_costs[0], max_costs[1], max_costs[2], max_costs[3], max_costs[4]))
            # pbar.set_description("Depth : %d, Reward: %f, Art Num: %d [Goal : %f, Regularization : %f, Similarity : %f, Num : %f, Artists : %f]"%(cur_depth, reward, art_num, max_costs[0], max_costs[1], max_costs[2], max_costs[3], max_costs[4]))
            if best_reward > 0.9:
                return best_state, v.history
            v.backpropagate(reward)

        return best_state, best_history 

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
    best_state, history = Tree.best_action(2000000)
    best_scene = best_state.scene.scene_data
    #dictionary to pickle data
    print("Saved path: " + date +'.pickle') 
    for tup in history:
        print(tup)
    with open(date +'.pickle', 'wb') as f:
        pickle.dump(best_scene, f, pickle.HIGHEST_PROTOCOL)