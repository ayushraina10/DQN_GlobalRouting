# import matplotlib
# matplotlib.use('TkAgg')
import numpy as np


import Initializer as init
import GridGraph as graph
from copy import deepcopy

"""Create grid graph API that wraps the original GridGraph enviuoirnment to enable 3 things:
    - Complex action space
    - Plot visual state including global capacity
    - Ability to checkpoint and resume states
"""

#creating the wrapper function for therouting environment:
import routing_utils as ru

class RoutingAPI():

    def __init__(self, filename = "benchmark_reduced/test_benchmark_7.gr", max_step = 100):
    #initialization of variables that control the wrapper function and the original environment
    
        """Functions in utils:
        _feasibility()
        _feasibilityComplex()
        _processTwoPinData()
        """
        #main variables associated with the problem and environment
        grid_info = init.read(filename)
        self.gridParameters = init.gridParameters(grid_info)

        # # GridGraph environment
        self.gridgraph = graph.GridGraph(init.gridParameters(grid_info))
        self.capacity = self.gridgraph.generate_capacity()

        twopinlist_nonet, twoPinNumEachNet = ru._processTwoPinData(self.gridParameters, grid_info, self.capacity)

        # DRL Module from here
        self.gridgraph.max_step = max_step #20
        self.gridgraph.twopin_combo = twopinlist_nonet
        # print('twopinlist_nonet',twopinlist_nonet)
        self.gridgraph.net_pair = twoPinNumEachNet

        self.gridgraph.reset()
    
    
    def _generate_image(self, compState = None):
        """
        Input:
        compState = [observation, [list of state variables]] 

        Output:
        observation = [3 channel image, [state variables(gx, gy, gz, ax, ay, az)]]
        """
        ##generate a 2d image of horizontal, vertical and via capacity
        ##some indicator for current location of agent

        ## variables for distance frorm goal, agent location
        if compState is not None:
            self.reset_to(compState = compState)
    
        return np.sum(self.gridgraph.capacity, axis = 2)
    
    def saveCheckpoint(self):
        """
        Returns a resumable checkpoint including all state variables
        
        returns [gridParameters, max_step, current_step, goal_state, init_state, current_state, capacity, route, 
        twopin_combo, twopint_pt, twopin_rdn, loop, reward, instantreward, instantrewardcombo, best_reward, 
        best_route, route_combo, net_pair, net_ind, pair_ind, passby, action, posTwoPinNum, episode] """
        
        return deepcopy([self.gridParameters, self.gridgraph.max_step, self.gridgraph.current_step, self.gridgraph.goal_state, \
                self.gridgraph.init_state, self.gridgraph.current_state, self.gridgraph.capacity, self.gridgraph.route, \
                self.gridgraph.twopin_combo, self.gridgraph.twopin_pt, self.gridgraph.twopin_rdn, self.gridgraph.loop, \
                self.gridgraph.reward, self.gridgraph.instantreward, self.gridgraph.instantrewardcombo, self.gridgraph.best_reward, \
                self.gridgraph.best_route, self.gridgraph.route_combo, self.gridgraph.net_pair, self.gridgraph.net_ind, self.gridgraph.pair_ind, \
                self.gridgraph.passby, self.gridgraph.previous_action, self.gridgraph.posTwoPinNum, self.gridgraph.episode])        
        
    
    def reset_to(self, checkpoint_state):
        """define the environment snapshot which can enable resuming the environment trajectory
        state info: 
        [gridParameters, max_step, current_step, goal_state, init_state, current_state, capacity, route, 
        twopin_combo, twopint_pt, twopin_rdn, loop, reward, instantreward, instantrewardcombo, best_reward, 
        best_route, route_combo, net_pair, net_ind, pair_ind, passby, action, posTwoPinNum, episode]
        
        """
        #update the state variables to the given set of values
        print("This RESET_TO function is being called")
        self.gridParameters, self.gridgraph.max_step, self.gridgraph.current_step, self.gridgraph.goal_state, \
                self.gridgraph.init_state, self.gridgraph.current_state, self.gridgraph.capacity, self.gridgraph.route, \
                self.gridgraph.twopin_combo, self.gridgraph.twopin_pt, self.gridgraph.twopin_rdn, self.gridgraph.loop, \
                self.gridgraph.reward, self.gridgraph.instantreward, self.gridgraph.instantrewardcombo, self.gridgraph.best_reward, \
                self.gridgraph.best_route, self.gridgraph.route_combo, self.gridgraph.net_pair, self.gridgraph.net_ind, self.gridgraph.pair_ind, \
                self.gridgraph.passby, self.gridgraph.previous_action, self.gridgraph.posTwoPinNum, self.gridgraph.episode = deepcopy(checkpoint_state) 

        
    def _mapAction(self, action = None):
        """This function maps complex action to the basic functions in the older environment"""
        
        direction, distance = action
        if distance<1:
            return self.gridgraph.step(direction)

        while distance >= 1:
            self.gridgraph.step(direction)
            distance-=1

        return self.gridgraph.step(direction)
        
    def getNextState(self, compState = None, action = None):
        """
        Input:
        compState = [observation, [list of state variables]] 
        action = [direction: one-hot, distance: one-hot] (0-5, 0-N)
        
        Output:
        nextCompState = [nextObservation, [updated list of state variables]]
        reward
        done
        _
        
        Algo:
        - reset the environment to the input compState
        - renormalize the state and action input
        - map complex action to basic actions (repeat action)
        - return the update the step output
        """
        
        #get next state
        ##mapping function that converts the complex action 
        ##into a sequence of unit actions
        
        if compState is not None:
            self.reset_to(checkpoint_state = compState)
        
        # direction = np.argmax(action[:6])
        # distance = np.argmax(action[6:])
        direction, distance = action
            
        return self._mapAction([direction, distance])
        
    
    def getValidMoves(self, compState = None, spatial_region = None):
        #get a set of valid functions given a spatial prior
        ##generate a list of all actions in increasing distance
        ##loop them through the feasibility of those actions, discard higher distances if small distance is infeasible
        if compState is not None:
            self.reset_to(compState = compState)
        
        feasible = []
        # #for debugging
        # limits = {}
        
        #loop through all directions
        for d in range(6):
            lim_x, lim_y, lim_z = self.gridParameters['gridSize']
            if d in [0, 1]:     
                distance = abs((1-d)*(lim_x-1) - self.gridgraph.current_state[0])-1
            elif d in [2, 3]:
                distance = abs((1-d+2)*(lim_y-1) - self.gridgraph.current_state[1])-1
            else:
                distance = abs((1-d+4)*(lim_z-1) - self.gridgraph.current_state[2]+1)-1
            #limits[d] = [limit, distance]
                
            while distance >= 0:
                if distance == 0:
                    if ru._feasibility(state = self.gridgraph.current_state, action = d, capacity = self.gridgraph.capacity) and \
                    self.gridgraph.current_step + distance < self.gridgraph.max_step:
                        feasible.append([d, distance])
                        break
                else:
                    if ru._feasibilityComplex(state = self.gridgraph.current_state, action = [d, distance], capacity = self.gridgraph.capacity) and \
                        self.gridgraph.current_step + distance < self.gridgraph.max_step:
                        for s in range(distance+1):
                            feasible.append([d, s])
                        break
                distance -= 1
        return feasible
        
    def getReward(self, compState = None):
        #get reward
        if compState is not None:
            self.reset_to(compState = compState)
        return self.gridgraph.reward
            



if __name__ == '__main__':
    # Filename corresponds to benchmark to route
    # filename = 'small.gr'
    # filename = '4by4small.gr'
    # filename = 'adaptec1.capo70.2d.35.50.90.gr'
    # filename = 'sampleBenchmark'

    # Getting Net Info
    # grid_info = init.read(filename)
    # print(grid_info)

    # # # print(init.gridParameters(grid_info)['netInfo'])
    #
    # for item in init.gridParameters(grid_info).items():
    #     print(item)

    # # for net in init.gridParameters(grid_info)['netInfo']:
    # #     print (net)
    # init.GridGraph(init.gridParameters(grid_info)).show_grid()
    # init.GridGraph(init.gridParameters(grid_info)).pin_density_plot()
    #
    # capacity = GridGraph(init.gridParameters(grid_info)).generate_capacity()
    # print(capacity[:,:,0,1])
    # gridX, gridY, gridZ= GridGraph(init.gridParameters(grid_info)).generate_grid()
    # print(gridX[1,1,0])
    # print(gridY[1,1,0])
    # print(gridZ[1,1,0])

    # print('capacity[1,0,0,:]',capacity[1,0,0,:])
    # print('capacity[2,0,0,:]',capacity[2,0,0,:])
    # print('capacity[1,1,1,:]',capacity[1,1,1,:])
    # print('capacity[0,1,1,:]',capacity[0,1,1,:])
    # print('capacity[2,2,1,:]',capacity[2,2,1,:])


    # # Check capacity update
    # print("Check capacity update")
    # print(capacity[1, 2, 0, 4])
    # RouteListMerged = [(1,2,1,12,23),(1,2,2,12,23)] # Coordinates rule follows (xGrid, yGrid,Layer(1,2),xLength,yLength)
    # capacity = updateCapacity(capacity,RouteListMerged)
    # print(capacity[1, 2, 0, 4])

    # # # Check capacity update
    # print("Check updateCapacityRL")
    # print(capacity[1,2,0,3])
    # state = (1,2,1,13,23); action = 3;
    # capacity = updateCapacityRL(capacity,state,action)
    # print(capacity[1,2,0,3])
    # print(capacity[1,1,0,2])

    # # # Check get action
    # position = (20, 60, 2, 2, 6)
    # nextposition = (20, 50, 2, 2, 5)
    # actiontest = get_action(position,nextposition)
    # print('Action',actiontest)


    ##a unit test to check functionality: resuming of states, generating feasible actions, 
    for i in range(10):
        filename = f"benchmark_reduced/test_benchmark_{i+1}.gr"
        env = RoutingAPI(filename = filename)
        print(len(env.saveCheckpoint()))
        print(env.getValidMoves())

        #iterate for 10 actions
        for i in range(10):
            actions = env.getValidMoves()
            i = np.random.randint(len(actions))
            env.getNextState(action = actions[i])
        
        #new state reached
        print("Valid actions in the checkpoint state", env.getValidMoves(), env.gridgraph.instantreward, env.gridgraph.route)
        checkpoint = env.saveCheckpoint()

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


    
