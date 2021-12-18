import numpy as np

from GridGraphAPI import RoutingAPI

import time

from copy import deepcopy

def test1():
    #a unit test to check functionality: resuming of states, generating feasible actions, 
    for i in range(10):
        filename = f"benchmark_reduced/test_benchmark_{i+1}.gr"
        env = RoutingAPI(filename = filename)
        env.gridgraph.reset()
        # print(len(env.saveCheckpoint()))
        # print(env.getValidMoves())

        #iterate for 10 actions
        for i in range(20):
            actions = env.getValidMoves()
            i = np.random.randint(len(actions))
            env.getNextState(action = actions[i])
        
        #new state reached
        print("Valid actions in the checkpoint state", env.getValidMoves(), env.gridgraph.instantreward, env.gridgraph.route)
        checkpoint = env.saveCheckpoint()

        env.reset_to(checkpoint)


        is_terminal = False
        while not is_terminal:
            actions = env.getValidMoves()
            try:
                i = np.random.randint(0, len(actions))
            except:
                print(actions)
                import pdb; pdb.set_trace()
            _, reward, is_terminal, _ = env.getNextState(action = actions[i])

        print("valid actions in the new state now", env.getValidMoves(), env.gridgraph.instantreward, env.gridgraph.route)

        print("attempting to reset the environment")
        env.reset_to(checkpoint)

        new_checkpoint = env.saveCheckpoint()
        print("valid actions in the resumed state", env.getValidMoves(), env.gridgraph.instantreward, env.gridgraph.route, np.all(checkpoint[6] == new_checkpoint[6]))

        print("Checking if everything has been copied from the checkpoint correctly")
        same = True
        for item, new_item in zip(checkpoint, new_checkpoint):
            same*np.all(item == new_item)
        print("Result is", same)

def test2(lookahead = True):
    #a unit test to check functionality: resuming of states, generating feasible actions, 
    ##also includes a clear algo for carrying out an episode
    print("Lookahead", lookahead)
    for i in range(10):
        filename = f"benchmark_reduced/test_benchmark_{i+1}.gr"
        env = RoutingAPI(filename = filename)
        # copy_env = RoutingAPI(filename = filename)
        episodes = 3

        reward_log = []
        test_reward_log = []
        test_episode = []
        solution_combo = []

        reward_plot_combo = []
        reward_plot_combo_pure = []

        start_ep = time.time()

        for episode in range(len(env.gridgraph.twopin_combo)*episodes):
            solution_combo.append(env.gridgraph.route)
            state, reward_plot, is_best = env.gridgraph.reset()
            # state, reward_plot, is_best = copy_env.gridgraph.reset()

            reward_plot_pure = reward_plot-env.gridgraph.posTwoPinNum*100

            # print(episode)
            if episode% len(env.gridgraph.twopin_combo) == 0:
                print(filename, env.gridgraph.instantrewardcombo)
                reward_plot_combo.append(reward_plot)
                reward_plot_combo_pure.append(reward_plot_pure)
                print("one main episode takes", time.time() - start_ep)
                start_ep = time.time()
            
            times = []
            rewardi = 0.0
            rewardfortwopin = 0
            is_terminal = False

            ctr = 0
            while not is_terminal:
                # observation = env.gridgraph.state2obsv()
                start = time.time()
                actions = env.getValidMoves()

                selected_action = None
                checkpoint = None
                if lookahead:
                    checkpoint = env.saveCheckpoint()
                    # np.save("check.npy", checkpoint)
                    # print("length of checkpoint", len(checkpoint))
                    rewards = []
                    for take_action in actions:
                        # _, reward, is_terminal, _ = copy_env.getNextState(compState = checkpoint, action = action)
                        _, reward, is_terminal, _ = copy_env.getNextState(compState = checkpoint, action = take_action)
                        rewards.append(reward)

                    i = np.argmax(rewards)
                    # env.reset_to(checkpoint_state=checkpoint)
                    # nextstate1, reward, is_terminal, _ = env.getNextState(action = actions[i])
                    nextstate1, reward, is_terminal, _ = env.getNextState(compState = checkpoint, action = actions[i])
                    selected_action = actions[i]
                    # print("TOOK ACTION", selected_action)

                else:
                    # checkpoint = env.saveCheckpoint()
                    i = np.random.randint(0, len(actions))
                    nextstate1, reward, is_terminal, _ = env.getNextState(action = actions[i])
                    # _, reward, is_terminal, _ = env.getNextState(compState = checkpoint, action = actions[i])
                    selected_action = actions[i]

                # print("Time taken for one action", time.time()-start)
                times.append(time.time()-start)
                rewardi = rewardi+reward
                rewardfortwopin = rewardfortwopin + reward

                ctr+=1
            # checkpoint1 = env.saveCheckpoint()
            # # print("DEBUGG",selected_action)
            # nextstate2, reward2, is_terminal, _ = copy_env.getNextState(compState = checkpoint, action = selected_action)
            # checkpoint2 = copy_env.saveCheckpoint()

            # if checkpoint1 != checkpoint2:#np.random.rand()<.2:
            #     where_unequal = np.array([a==b for a,b in zip(checkpoint1, checkpoint2)])
            #     print(np.arange(len(where_unequal))[where_unequal == True])
            #     # checkpoint = env.saveCheckpoint()
            #     # env.reset_to(checkpoint, debug = True)
            #     # print("COUNTER IS", ctr, episode)
            #     # checkpoint1 = env.saveCheckpoint()
            #     # # print(checkpoint1)
            #     # # copy_env.reset_to(checkpoint)
            #     # nextstate2, reward2, is_terminal, _ = copy_env.getNextState(compState = checkpoint, action = selected_action)
            #     # checkpoint2 = copy_env.saveCheckpoint()
                
            #     # print("checkpoint1")
            #     # # print(checkpoint1)
                
            #     # print("Checkpoint 2")
            #     # print(checkpoint1 == checkpoint2)
            #     import pdb; pdb.set_trace()
            #     # xx


            reward_log.append(rewardi)
            env.gridgraph.instantrewardcombo.append(rewardfortwopin)
            
            # print(env.gridgraph.instantrewardcombo)
            
            # print(np.sum(times))

def test3():
    """testing alternate method of deep copying by creating a string representation of the state"""
    #a unit test to check functionality of string version of deep copy 
    for i in range(10):
        filename = f"benchmark_reduced/test_benchmark_{i+1}.gr"
        env = RoutingAPI(filename = filename)
        episodes = 3

        for episode in range(len(env.gridgraph.twopin_combo)*episodes):
            # print(episode)
            if episode% len(env.gridgraph.twopin_combo) == 0:
                print(filename)

            env.gridgraph.reset()
            
            times = []
            is_terminal = False
            while not is_terminal:
                observation = env.gridgraph.state2obsv()

                start = time.time()
                checkpoint = env.saveCheckpoint()
                times.append(time.time()-start)

                actions = env.getValidMoves()
                i = np.random.randint(0, len(actions))
                # _, reward, is_terminal, _ = env.getNextState(compState = checkpoint, action = actions[i])
                _, reward, is_terminal, _ = env.getNextState(action = actions[i])
                # print("Time taken for one action", time.time()-start)
                del checkpoint
            print(np.sum(times))

def test4():
    """Similar to test 2 but with one-step lookahead implemented"""
    #a unit test to check functionality: resuming of states, generating feasible actions, 
    for i in range(10):
        filename = f"benchmark_reduced/test_benchmark_{i+1}.gr"
        env = RoutingAPI(filename = filename)
        episodes = 3

        start_ep = time.time()
        for episode in range(len(env.gridgraph.twopin_combo)*episodes):

            if episode% len(env.gridgraph.twopin_combo) == 0:
                print(filename)
                print("one main episode takes", time.time() - start_ep)
                start_ep = time.time()
            env.gridgraph.reset()
            
            times = []
            is_terminal = False
            while not is_terminal:

                start = time.time()
                checkpoint = env.saveCheckpoint()

                actions = env.getValidMoves()
                rewards = []
                for action in actions:
                    _, reward, is_terminal, _ = env.getNextState(compState = checkpoint, action = action)
                    rewards.append(reward)
                
                # i = np.random.randint(0, len(actions))
                i = np.argmax(rewards)
                _, reward, is_terminal, _ = env.getNextState(compState = checkpoint, action = actions[i])
                # _, reward, is_terminal, _ = env.getNextState(action = actions[i])
                
                # print("Time taken for one action", time.time()-start)
                times.append(time.time()-start)
            
            # print(np.sum(times))

def _areidentical(list1, list2):
    correct = True
    for a, b in zip(list1, list2):
        correct *= np.all(a==b)
    return correct

def test5():
    "check if resuming works for newly initialized env objects"
    for i in range(10):
        filename = f"benchmark_reduced/test_benchmark_{i+1}.gr"
        print(filename)
        env = RoutingAPI(filename = filename)
        env.gridgraph.reset()
        print(len(env.saveCheckpoint()))
        print(env.getValidMoves())

        #iterate for 10 actions
        for i in range(10):
            actions = env.getValidMoves()
            if len(actions) == 0:
                is_terminal = True
                break
            else:
                ind = np.random.randint(0, len(actions))
            env.getNextState(action = actions[ind])

        print(env.gridgraph.current_step, i)
        #new state reached
        actions = env.getValidMoves()
        # print("Valid actions in the checkpoint state", actions, env.gridgraph.instantreward, env.gridgraph.route)
        checkpoint = env.saveCheckpoint()

        new_env = RoutingAPI(filename=filename)
        new_actions = new_env.getValidMoves(compState = checkpoint)

        action = actions[np.random.randint(0, len(actions))]
        nextstate,  reward, _, _ = env.getNextState(action = action)
        # action = new_actions[np.random.randint(0, len(new_actions))]
        new_nextstate,  reward, _, _ = new_env.getNextState(action = action)

        print("Are the set of actions same", _areidentical(new_actions, actions))
        print("Are states same", _areidentical(nextstate, new_nextstate))




if __name__ == '__main__':
    
    # test3()

    #compare a random episode vs one-step lookahead epsiode
    test2(lookahead = False)


    
