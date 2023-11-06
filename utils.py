import numpy as np


# Load data
def loadData(path):
    data = []
    with open(path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(",")
            temp = []
            for i in line:
                temp.append(int(i))
            print(temp)
            data.append(temp)
    return np.array(data)

# insert memory node
def insertMemNode(data):
    k = 5
    row = len(data)
    for i in range(row):
        flag = False
        for j in range(1,9,2):
            if data[i][j] == 0:
                continue
            if (data[i][-3]-data[i][-4]>=k-1) or (data[data[i][j]-1][-3]-data[i][-4] >= k and data[data[i][j]-1][-4]-data[i][-4]>1):
                flag = True
                temp = [0]*len(data[i])
                temp[0] = len(data)+1
                temp[1] = data[i][j]
                temp[-1] = 1
                temp[-2] = -1
                temp[-3] = data[data[i][j]-1][-3]-1
                temp[-4] = data[data[i][j]-1][-3]-1
                data[data[i][j]-1][-4] = data[data[i][j]-1][-3]
                data[i][j]=0
                list_data = list(data)
                list_data.append(temp)
                data = np.array(list_data)
        if flag:
            temp = [0]*len(data[i])
            temp[0] = len(data)+1
            temp[-1] = 1
            temp[-2] = 1
            temp[-3] = data[i][-4]+1
            temp[-4] = data[i][-4]+1
            data[i][-3] = data[i][-4]
            for j in range(1,9,2):
                if data[i][j] == 0:
                    data[i][j] = temp[0]
                    break
            for x in range(row):
                for y in range(1,9,2):
                    if data[x][y] == data[i][0]:
                        data[x][-3] = data[i][-3]-1
                        data[x][-4] = data[i][-4]-1
            list_data = list(data)
            list_data.append(temp)
            data = np.array(list_data)
            flag = False
    return data

# recomputation
def recomp(data):
    row = len(data)
    data_re=[]
    k = 3
    for i in range(row):
        temp_c = 0
        for j in range(1,9,2):
            if(data[i][j] != 0):
                temp_c += 1
        if temp_c > k:
            temp_data = data[i].copy()
            temp_data[0]=len(data)+1
            temp_data[1:5] = data[i][5:9].copy()
            temp_data[-2] = i+1
            for col in range(5,9):
                temp_data[col]=0
                data[i][col]=0
            for x in range(row):
                for y in range(1,9,2):
                    if data[x][y] == i+1:
                        for m in range(y+2,9,2):
                            if data[x][m] == 0:
                                data[x][m] = temp_data[0]
                                break
            list_data = list(data)
            list_data.append(temp_data)
            data = np.array(list_data)
            data_re.append((temp_data,i+1))
    return data,data_re
            
# Obtain nodes that can be mapped for each layer
def get_layer_infeasible(each_time_node, action_dims, ii):
    layer_infeasible = [[_ for _ in range(1,action_dims)] for _ in range(ii)]
    for row in range(len(each_time_node)):
        for col in range(len(each_time_node[0])):
            if  (each_time_node[row][col] == 1) and col in layer_infeasible[row%ii]:

                layer_infeasible[row%ii].remove(col)

    return layer_infeasible

def topLogical(adjacency_list):
    in_degree = dict((u[0],0) for u in adjacency_list)
    #print(in_degree)
    for i in range(len(adjacency_list)):
        for j in range(1,len(adjacency_list[0])):
            if adjacency_list[i][j] != 0:
                in_degree[adjacency_list[i][j]] += 1
    Q = [u[0] for u in adjacency_list if in_degree[u[0]] == 0]  ##All vertices with a degree of zero
    res=[]
    while Q:
        u = Q.pop()
        res.append(u)
        for j in range(1,len(adjacency_list[0])):
            if adjacency_list[u-1][j] != 0:
                in_degree[adjacency_list[u-1][j]] -= 1
                #print(in_degree)
                if in_degree[adjacency_list[u-1][j]] == 0:
                    Q.append(adjacency_list[u-1][j])
    return res

# Node activity scope    
def save_asap_alap(data,pea_width,ii):
    num_pe = pea_width*pea_width*ii
    # Save asap and alap of nodes 
    range_lp = np.concatenate([data[:,:1], data[:,-4:-2]], axis=1)
    # Possible nodes for each time step
    each_time_node = np.zeros([np.max(range_lp[:,-1]+1), len(range_lp)+1], dtype=int)
    #print(each_time_node)
    for i in range(len(range_lp)):
        #print(range_lp[i][1])
        #print(range_lp[i][2])
        for j in range(range_lp[i][1],range_lp[i][2]+1):
            each_time_node[j][range_lp[i][0]] = 1
            #each_time_node[j].append(range_lp[i][0])
    
    # Modification of activity scope
    # If there is a node in the child node whose asap-1 is not equal to its parent node's asap-1. The range of the parent node is not from asap to alap, but from asap to the maximum alap-1 among the changing child nodes
    count_all = np.sum(each_time_node == 1)
    k = 4
    range_lp2 = range_lp.copy()
    for i in range(len(range_lp2)):
        if range_lp2[i][2] - range_lp2[i][1] >= k:

            max_lp = -1
            for j in range(1,len(data[i])-4,2):
                if data[i][j] != 0 and data[data[i][j]-1][-3]-1-range_lp2[i][1] < k:
                    max_lp = max(max_lp,data[data[i][j]-1][-3]-1)

            if max_lp == -1:
                if count_all+1-(range_lp2[i][2] - range_lp2[i][1]) <= num_pe:
                    count_all += 1-(range_lp2[i][2] - range_lp2[i][1])
                    range_lp2[i][2] = range_lp2[i][1]+1
                else:
                    range_lp2[i][2] = range_lp2[i][1]
            else:
                if count_all+max_lp-range_lp2[i][2] <= num_pe:
                    count_all += max_lp-range_lp2[i][2]
                    range_lp2[i][2] = max_lp
                else:
                    range_lp2[i][2] = range_lp2[i][1]
    
    for i in range(len(range_lp2)):

        min_lp = -1
        for j in range(1,len(data[i])-4,2):
            if data[i][j] != 0 and data[data[i][j]-1][-4] != data[i][-4]+1:
                if data[data[i][j]-1][-3]-range_lp2[i][1]-1 < k:
                    min_lp = max(min_lp,data[data[i][j]-1][-3]-1)
        if min_lp != -1:
            if count_all+min_lp-range_lp2[i][2] <= num_pe:
                count_all += min_lp-range_lp2[i][2]
                range_lp2[i][2] = min_lp

        if range_lp2[i][2] - range_lp2[i][1] >= k:
            max_lp = -1
            for j in range(1,len(data[i])-4,2):
                if data[i][j] != 0 and data[data[i][j]-1][-3]-1-range_lp2[i][1] < k:
                    max_lp = max(max_lp,data[data[i][j]-1][-3]-1)

            if max_lp == -1:
                if count_all+1-(range_lp2[i][2] - range_lp2[i][1]) <= num_pe:
                    count_all += 1-(range_lp2[i][2] - range_lp2[i][1])
                    range_lp2[i][2] = range_lp2[i][1]+1
                else:
                    range_lp2[i][2] = range_lp2[i][1]
            else:
                if count_all+max_lp-range_lp2[i][2] <= num_pe:
                    count_all += max_lp-range_lp2[i][2]
                    range_lp2[i][2] = max_lp
                else:
                    range_lp2[i][2] = range_lp2[i][1]


    # Possible nodes for each time step
    each_time_node2 = np.zeros([np.max(range_lp2[:,-1]+1), len(range_lp2)+1], dtype=int)

    for i in range(len(range_lp2)):
        for j in range(range_lp2[i][1],range_lp2[i][2]+1):
            each_time_node2[j][range_lp2[i][0]] = 1

    """
    count = [0 for _ in range(ii)]
    for i in range(len(range_lp2)):
        #data[i][-3] = range_lp2[i][2]
        for j in range(range_lp2[i][1],range_lp2[i][2]+1):
            each_time_node2[j][range_lp2[i][0]] = 1
            count[j%ii] += 1

    count_1 = np.sum(each_time_node2 == 1)
    if count_1 < pea_width*pea_width*ii:
        for i in range(len(range_lp2)):
            # Nodes with a degree of zero
            if(data[i][-1] == 0 and range_lp2[i][2] != 0 and range_lp2[i][1] == range_lp2[i][2]):
                if(count[(range_lp2[i][1]-1)%ii] >= pea_width*pea_width):
                    continue
                each_time_node2[range_lp2[i][1]-1][i+1] = 1
                count_1 += 1
                count[(range_lp2[i][1]-1)%ii] += 1
            #data[i][-4] = range_lp2[i][1]-1
    """
    return range_lp,each_time_node,each_time_node2
