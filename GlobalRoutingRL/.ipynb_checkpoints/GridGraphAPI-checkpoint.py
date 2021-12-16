import matplotlib
# matplotlib.use('TkAgg')
import numpy as np


import Initializer as init

"""Create grid graph API that wraps the original GridGraph enviuoirnment to enable 3 things:
    - Complex action space
    - Plot visual state including global capacity
    - Ability to checkpoint and resume states
"""

#creating the wrapper function for therouting environment:
import routing_utils as ru

class RoutingAPI():

    def __init__(self, filename = "benchmark_reduced/test_benchmark_5.gr", max_step = 100):
    #initialization of variables that control the wrapper function and the original environment
    
    """Functions in utils:
        _feasibility()
        _feasibilityComplex()
        _processTwoPinData()
        
        """
        #main variables associated with the problem and environment
        self.grid_info = init.read(filename)
        self.gridParameters = init.gridParameters(grid_info)

        # # GridGraph environment
        self.gridgraph = graph.GridGraph(init.gridParameters(self.grid_info))
        self.capacity = self.gridgraph.generate_capacity()

        twopinlist_nonet, twoPinNumEachNet = ru._processTwoPinData(gridParameters)

        # DRL Module from here
        self.gridgraph.max_step = max_step #20
        self.gridgraph.twopin_combo = twopinlist_nonet
        # print('twopinlist_nonet',twopinlist_nonet)
        self.gridgraph.net_pair = twoPinNumEachNet
    
    
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
        if compState not None:
            self.reset_to(compState = compState)
    
        return np.sum(self.gridgraph.capacity, axis = 2)
    
    def saveCheckpoint(self):
        """
        Returns a resumable checkpoint including all state variables
        
        returns [gridParameters, max_step, current_step, goal_state, init_state, current_state, capacity, route, 
        twopin_combo, twopint_pt, twopin_rdn, loop, reward, instantreward, instantrewardcombo, best_reward, 
        best_route, route_combo, net_pair, net_ind, pair_ind, passby, action, posTwoPinNum, episode] """
    
        return [self.gridParameters, self.gridgraph.max_step, self.gridgraph.current_step, self.gridgraph.goal_state, \
                self.gridgraph.init_state, self.gridgraph.current_state, self.gridgraph.capacity, self.gridgraph.route, \
                self.gridgraph.twopin_combo, self.gridgraph.twopint_pt, self.gridgraph.twopin_rdn, self.gridgraph.loop, \
                self.gridgraph.reward, self.gridgraph.instantreward, self.gridgraph.instantrewardcombo, self.gridgraph.best_reward, \
                self.gridgraph.best_route, self.gridgraph.route_combo, self.gridgraph.net_pair, self.gridgraph.net_ind, self.gridgraph.pair_ind, \
                self.gridgraph.passby, self.gridgraph.action, self.gridgraph.posTwoPinNum, self.gridgraph.episode]        
        
    
    def reset_to(self, checkpoint_state):
        """define the environment snapshot which can enable resuming the environment trajectory
        state info: 
        [gridParameters, max_step, current_step, goal_state, init_state, current_state, capacity, route, 
        twopin_combo, twopint_pt, twopin_rdn, loop, reward, instantreward, instantrewardcombo, best_reward, 
        best_route, route_combo, net_pair, net_ind, pair_ind, passby, action, posTwoPinNum, episode]
        
        """
        #update the state variables to the given set of values
        
        self.gridParameters, self.gridgraph.max_step, self.gridgraph.current_step, self.gridgraph.goal_state, \
                self.gridgraph.init_state, self.gridgraph.current_state, self.gridgraph.capacity, self.gridgraph.route, \
                self.gridgraph.twopin_combo, self.gridgraph.twopint_pt, self.gridgraph.twopin_rdn, self.gridgraph.loop, \
                self.gridgraph.reward, self.gridgraph.instantreward, self.gridgraph.instantrewardcombo, self.gridgraph.best_reward, \
                self.gridgraph.best_route, self.gridgraph.route_combo, self.gridgraph.net_pair, self.gridgraph.net_ind, self.gridgraph.pair_ind, \
                self.gridgraph.passby, self.gridgraph.action, self.gridgraph.posTwoPinNum, self.gridgraph.episode = checkpoint_state 

        
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
        
        if compState not None:
            self.reset_to(compState = compState)
        
        direction = np.argmax(action[:6])
        distance = np.argmax(action[6:])
            
        return self._mapAction([direction, distance])
        
    
    def getValidMoves(state = None, spatial_region = None):
        #get a set of valid functions given a spatial prior
        ##generate a list of all actions in increasing distance
        ##loop them through the feasibility of those actions, discard higher distances if small distance is infeasible
        if compState not None:
            self.reset_to(compState = compState)
        
        feasible = []
        #for debugging
        limits = {}
        
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
                    self.gridgraph.current_step + distance+1 < self.gridgraph.max_step:
                        feasible1.append([d, distance])
                        break
                else:
                    if ru._feasibilityComplex(state = self.gridgraph.current_state, action = [d, distance], capacity = self.gridgraph.capacity) and \ 
                    self.gridgraph.current_step + distance+1 < self.gridgraph.max_step:
                        for s in range(distance+1):
                            feasible1.append([d, s])
                        break
                distance -= 1
        return feasible1
        
    def getReward(self, compState = None):
        #get reward
        if compState not None:
            self.reset_to(compState = compState)
        return self.gridgraph.reward
            
