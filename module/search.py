import os
import sys
from time import localtime, time
sys.path.insert(0,os.path.join(os.getcwd(), 'module'))
from nodes import MCTSNode
from scene import SceneState
from museum import MuseumScene
from museum import ver
# import time
from tqdm import tqdm
import pickle
import numpy as np
import argparse
import logging
date = '+' + '(' + str(localtime(time()).tm_mon) +'-'+ str(localtime(time()).tm_mday) +'-'+ str(localtime(time()).tm_hour) + '-'+ str(localtime(time()).tm_min) + ')'
# date = time.time()

class MonteCarloTreeSearch:
    def __init__(self, node: MCTSNode, logger, save_dir,t_depth=40):
        self.root = node
        self.root.is_leaf = False
        self.t_depth = t_depth
        self.logger = logger
        self.save_dir = save_dir

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
            if v.self_score > best_reward:
                best_reward = v.self_score
                best_state = v.state
                best_art_num = len(best_state.scene.scene_data)
                max_costs = v.self_costs
                best_history = v.history
                self.logger.info("[%d] Update Best Reward: %f  Number: %d [Goal : %f, Regularization : %f, Similarity : %f, Num : %f, Artists : %f]"%(idx, best_reward, best_art_num, max_costs[0], max_costs[1], max_costs[2], max_costs[3], max_costs[4]))
                print("[%d] Update Best Reward: %f  Number: %d [Goal : %f, Regularization : %f, Similarity : %f, Num : %f, Artists : %f]"%(idx, best_reward, best_art_num, max_costs[0], max_costs[1], max_costs[2], max_costs[3], max_costs[4]))
                self.logger.info("Saved path: " + self.save_dir + "/reward_" + str(best_reward) + '.pickle') 
                with open(os.path.join(self.save_dir, "reward_" + str(best_reward) +'.pickle'), 'wb') as f:
                    pickle.dump(best_state.scene.scene_data, f, pickle.HIGHEST_PROTOCOL)

                for tup in v.history:
                    self.logger.info(tup)

                    if len(v.history) > self.t_depth:
                        return best_state, v.history
                self.logger.info("==========================================================")
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
    parser = argparse.ArgumentParser(description='MCTS')
    parser.add_argument('--trials', type=int, required=True, help='number of trials')
    parser.add_argument('--r_depth', type=int, required=True, help='rollout depth')
    parser.add_argument('--t_depth', type=int, default=40, help='tree depth')
    args = parser.parse_args()

    log_dir = os.path.join(os.getcwd(), ver+"_"+str(args.trials)+date)
    if (os.path.exists(log_dir) == False):
        os.makedirs(log_dir, exist_ok=False)
    file_dir = os.path.join(log_dir, 'trials_' + str(args.trials) + "_" + str(args.r_depth) + "_" + str(args.t_depth))
    if (os.path.exists(file_dir) == False):
        os.makedirs(file_dir, exist_ok=False)

    logging.basicConfig(filename=os.path.join(log_dir, 'trials_' + str(args.trials) + "_" + str(args.r_depth) + "_" + str(args.t_depth) + '_log.txt'), 
                        level=logging.INFO,
                        format='[%(asctime)s - %(name)s - %(levelname)s] %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    museumScene = MuseumScene()
    weights = museumScene.get_weights()
    Tree = MonteCarloTreeSearch(MCTSNode(SceneState(museumScene)), logger, file_dir, args.t_depth)

    logger.info(f"Weights: Goal {weights['g']}, Regularization {weights['r']}, Similarity {weights['s']}, Number {weights['n']}, Artists {weights['an']}")
    logger.info("==========================================================")

    best_state, history = Tree.best_action(2000000)
    best_scene = best_state.scene.scene_data
    #dictionary to pickle data
    logger.info("End of MCTS")
    logger.info("Final Saved path: " + os.path.join(file_dir, 'final.pickle')) 
    for tup in history:
        logger.info(tup)
    with open(os.path.join(file_dir, 'final.pickle'), 'wb') as f:
        pickle.dump(best_scene, f, pickle.HIGHEST_PROTOCOL)