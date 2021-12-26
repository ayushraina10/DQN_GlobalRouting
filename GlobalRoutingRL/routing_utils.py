import operator
import Env.DQN_GlobalRouting.GlobalRoutingRL.TwoPinRouterASearch as twoPinASearch
import Env.DQN_GlobalRouting.GlobalRoutingRL.Initializer as init
import Env.DQN_GlobalRouting.GlobalRoutingRL.MST as tree


import numpy as np

def _processTwoPinData(gridParameters, grid_info, capacity):
    # Real Router for Multiple Net
    # Note: pinCoord input as absolute length coordinates
    gridGraphSearch = twoPinASearch.AStarSearchGraph(gridParameters, capacity)

    # Sort net
    halfWireLength = init.VisualGraph(init.gridParameters(grid_info)).bounding_length()
    #    print('Half Wire Length:',halfWireLength)

    sortedHalfWireLength = sorted(halfWireLength.items(),key=operator.itemgetter(1),reverse=True) # Large2Small
    # sortedHalfWireLength = sorted(halfWireLength.items(),key=operator.itemgetter(1),reverse=False) # Small2Large

    netSort = []
    for i in range(gridParameters['numNet']):
        order = int(sortedHalfWireLength[i][0])
        netSort.append(order)
    # random order the nets
    # print('netSort Before',netSort)
    # random.shuffle(netSort)
    # print('netSort After',netSort)

    routeListMerged = []
    routeListNotMerged = []

    # print('gridParameters',gridParameters)
    # Getting two pin list combo (For RL)
    twopinListCombo = []
    twopinListComboCleared = []
    for i in range(len(init.gridParameters(grid_info)['netInfo'])):
        netNum = i
        netPinList = []; netPinCoord = []
        for j in range(0, gridParameters['netInfo'][netNum]['numPins']):
            pin = tuple([int((gridParameters['netInfo'][netNum][str(j+1)][0]-gridParameters['Origin'][0])/gridParameters['tileWidth']),
                             int((gridParameters['netInfo'][netNum][str(j+1)][1]-gridParameters['Origin'][1])/gridParameters['tileHeight']),
                             int(gridParameters['netInfo'][netNum][str(j+1)][2]),
                              int(gridParameters['netInfo'][netNum][str(j+1)][0]),
                              int(gridParameters['netInfo'][netNum][str(j+1)][1])])
            if pin[0:3] in netPinCoord:
                continue
            else:
                netPinList.append(pin)
                netPinCoord.append(pin[0:3])
        twoPinList = []
        for i in range(len(netPinList)-1):
            pinStart = netPinList[i]
            pinEnd = netPinList[i+1]
            twoPinList.append([pinStart,pinEnd])

        twoPinListVanilla = twoPinList

        # Insert Tree method to decompose two pin problems here
        twoPinList = tree.generateMST(twoPinList)
    #        print('Two pin list after:', twoPinList, '\n')

        # Remove pin pairs that are in the same grid 
        nullPairList = []
        for i in range(len(twoPinListVanilla)):
            if twoPinListVanilla[i][0][:3] == twoPinListVanilla[i][1][:3]:
                nullPairList.append(twoPinListVanilla[i])
        for i in range(len(nullPairList)):
            twoPinListVanilla.reomove(nullPairList[i])

        # Remove pin pairs that are in the same grid 
        nullPairList = []
        for i in range(len(twoPinList)):
            if twoPinList[i][0][:3] == twoPinList[i][1][:3]:
                nullPairList.append(twoPinList[i])
        for i in range(len(nullPairList)):
            twoPinList.reomove(nullPairList[i])

        # Key: use original sequence of two pin pairs
        twopinListComboCleared.append(twoPinListVanilla)

    # print('twopinListComboCleared',twopinListComboCleared)
    twoPinEachNetClear = []
    for i in twopinListComboCleared:
        num = 0
        for j in i:
            num = num + 1
        twoPinEachNetClear.append(num)

    # print('twoPinEachNetClear',twoPinEachNetClear)

    for i in range(len(init.gridParameters(grid_info)['netInfo'])):
        netNum = int(sortedHalfWireLength[i][0]) # i 
        # Sort the pins by a heuristic such as Min Spanning Tree or Rectilinear Steiner Tree
        netPinList = []
        netPinCoord = []
        for j in range(0, gridParameters['netInfo'][netNum]['numPins']):
            pin = tuple([int((gridParameters['netInfo'][netNum][str(j+1)][0]-gridParameters['Origin'][0])/gridParameters['tileWidth']),
                             int((gridParameters['netInfo'][netNum][str(j+1)][1]-gridParameters['Origin'][1])/gridParameters['tileHeight']),
                             int(gridParameters['netInfo'][netNum][str(j+1)][2]),
                              int(gridParameters['netInfo'][netNum][str(j+1)][0]),
                              int(gridParameters['netInfo'][netNum][str(j+1)][1])])
            if pin[0:3] in netPinCoord:
                continue
            else:
                netPinList.append(pin)
                netPinCoord.append(pin[0:3])

        twoPinList = []
        for i in range(len(netPinList)-1):
            pinStart = netPinList[i]
            pinEnd = netPinList[i+1]
            twoPinList.append([pinStart,pinEnd])

        # Insert Tree method to decompose two pin problems here
        twoPinList = tree.generateMST(twoPinList)
    #        print('Two pin list after:', twoPinList, '\n')

        # Remove pin pairs that are in the same grid 
        nullPairList = []
        for i in range(len(twoPinList)):
            if twoPinList[i][0][:3] == twoPinList[i][1][:3]:
                nullPairList.append(twoPinList[i])

        for i in range(len(nullPairList)):
            twoPinList.reomove(nullPairList[i])

        # Key: Use MST sorted pin pair sequence under half wirelength sorted nets
        twopinListCombo.append(twoPinList)

    # print('twopinListCombo',twopinListCombo)

    # for i in range(1):
    for i in range(len(init.gridParameters(grid_info)['netInfo'])):

        # Determine nets to wire based on sorted nets (stored in list sortedHalfWireLength)
    #        print('*********************')
        # print('Routing net No.',init.gridParameters(grid_info)['netInfo'][int(sortedHalfWireLength[i][0])]['netName'])
        # (above output is to get actual netName)
    #        print('Routing net No.',sortedHalfWireLength[i][0])

        netNum = int(sortedHalfWireLength[i][0])

        # Sort the pins by a heuristic such as Min Spanning Tree or Rectilinear Steiner Tree
        netPinList = []
        netPinCoord = []
        for j in range(0, gridParameters['netInfo'][netNum]['numPins']):
            pin = tuple([int((gridParameters['netInfo'][netNum][str(j+1)][0]-gridParameters['Origin'][0])/gridParameters['tileWidth']),
                             int((gridParameters['netInfo'][netNum][str(j+1)][1]-gridParameters['Origin'][1])/gridParameters['tileHeight']),
                             int(gridParameters['netInfo'][netNum][str(j+1)][2]),
                              int(gridParameters['netInfo'][netNum][str(j+1)][0]),
                              int(gridParameters['netInfo'][netNum][str(j+1)][1])])
            if pin[0:3] in netPinCoord:
                continue
            else:
                netPinList.append(pin)
                netPinCoord.append(pin[0:3])
        twoPinList = []
        for i in range(len(netPinList)-1):
            pinStart = netPinList[i]
            pinEnd = netPinList[i+1]
            twoPinList.append([pinStart,pinEnd])

        # Insert Tree method to decompose two pin problems here
        twoPinList = tree.generateMST(twoPinList)

        # Remove pin pairs that are in the same grid 
        nullPairList = []
        for i in range(len(twoPinList)):
            if twoPinList[i][0][:3] == twoPinList[i][1][:3]:
                nullPairList.append(twoPinList[i])
        for i in range(len(nullPairList)):
            twoPinList.reomove(nullPairList[i])

        i = 1
        routeListSingleNet = []
        for twoPinPair in twoPinList:
            pinStart = twoPinPair[0]; pinEnd =  twoPinPair[1]
            route, cost = twoPinASearch.AStarSearchRouter(pinStart, pinEnd, gridGraphSearch)
            routeListSingleNet.append(route)
            i += 1

        mergedrouteListSingleNet = []

        for list in routeListSingleNet:
            # if len(routeListSingleNet[0]) == 2:
            #     mergedrouteListSingleNet.append(list[0])
            #     mergedrouteListSingleNet.append(list[1])
            # else:
            for loc in list:
                    if loc not in mergedrouteListSingleNet:
                        mergedrouteListSingleNet.append(loc)

        routeListMerged.append(mergedrouteListSingleNet)
        routeListNotMerged.append(routeListSingleNet)

        # Update capacity and grid graph after routing one pin pair
        # # WARNING: there are some bugs in capacity update
        # # # print(route)
        # capacity = graph.updateCapacity(capacity, mergedrouteListSingleNet)
        # gridGraph = twoPinASearch.AStarSearchGraph(gridParameters, capacity)

    # print('\nRoute List Merged:',routeListMerged)

    twopinlist_nonet = []
    for net in twopinListCombo:
    # for net in twopinListComboCleared:
        for pinpair in net:
            twopinlist_nonet.append(pinpair)

    # Get two pin numbers
    twoPinNum = 0
    twoPinNumEachNet = []
    for i in range(len(init.gridParameters(grid_info)['netInfo'])):
        netNum = int(sortedHalfWireLength[i][0]) # i
        twoPinNum = twoPinNum + (init.gridParameters(grid_info)['netInfo'][netNum]['numPins'] - 1)
        twoPinNumEachNet.append(init.gridParameters(grid_info)['netInfo'][netNum]['numPins'] - 1)

    return twopinlist_nonet, twoPinNumEachNet, twoPinEachNetClear, netSort


def _feasibility(state, action, capacity):
    if action == 0 and capacity[state[0], state[1], state[2]-1, 0] > 0 :
        return True
    elif action == 1 and capacity[state[0], state[1], state[2]-1, 1] > 0 :
        return True
    elif action == 2 and capacity[state[0], state[1], state[2]-1, 2] > 0 :
        return True
    elif action == 3 and capacity[state[0], state[1], state[2]-1, 3] > 0 :
        return True
    elif action == 4 and capacity[state[0], state[1], state[2]-1, 4] > 0 :
        return True
    elif action == 5 and capacity[state[0], state[1], state[2]-1, 5] > 0 :
        return True
    else:
        return False
    
def _feasibilityComplex(state, action, capacity):
    direction, distance = action
    if direction == 0 and np.all(capacity[state[0]:state[0]+distance+1, state[1], state[2]-1, 0]) > 0 :
        return True
    elif direction == 1 and np.all(capacity[state[0]-distance:state[0]+1, state[1], state[2]-1, 1]) > 0 :
        return True
    elif direction == 2 and np.all(capacity[state[0], state[1]:state[1]+distance+1, state[2]-1, 2]) > 0 :
        return True
    elif direction == 3 and np.all(capacity[state[0], state[1]-distance:state[1]+1, state[2]-1, 3]) > 0 :
        return True
    elif direction == 4 and np.all(capacity[state[0], state[1], state[2]-1:state[2]-1+distance+1, 4]) > 0 :
        return True
    elif direction == 5 and np.all(capacity[state[0], state[1], state[2]-1-distance:state[2]-1+1, 5]) > 0 :
        return True
    else:
        return False
    