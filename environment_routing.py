import numpy as np
import hashlib
import copy


# This variable represents a point where mapping cannot be performed
absolute_infeasible_prob = -np.inf
# The following variable represents the tolerable repeated mapping
infeasible_prob = -100000


def abs_infeasible_update(batch_index, i, nodes, logits, mask):
    # This function is used to adjust unmapped nodes to negative infinity and change the corresponding mask
    # batch_index mean:Which batch
    # i           mean:Which node's logits need to be modified
    for node in nodes:
        logits[node] = absolute_infeasible_prob
        mask[batch_index][i][node] = 1


def infeasible_update(batch_index, i, nodes, logits, mask):
    # This function is used to adjust unmapped nodes to negative infinity and change the corresponding mask
    # batch_index mean:Which batch
    # i           mean:Which node's logits need to be modified
    for node in nodes:
        logits[node] = infeasible_prob
        mask[batch_index][i][node] = 1

class Environment:

    def __init__(self, action_dims, memory_mode, max_memory, max_GRF, max_LRF, C, temperature, ii, pea_width, total_adj,
                 total_embedding, total_graph, total_dict, total_net_input, reward_mode, each_time_node, source_dict, mapping_ver, layer_infeasible, top_log, beta=0.2):
        self.batch_index = None
        self.max_LRF = max_LRF
        self.max_GRF = max_GRF
        self.max_memory = max_memory
        # Whether to start memory, True to start, False to not start
        self.memory_mode = memory_mode
        self.C = C
        self.temperature = temperature
        self.beta = beta
        self.ii = ii
        self.pea_width = pea_width
        self.pea_size = pea_width * pea_width
        self.action_dims = action_dims
        self.total_net_input = total_net_input
        self.total_adj = total_adj
        self.total_embedding = total_embedding
        self.total_graph = total_graph
        self.total_dict = total_dict
        self.mapping_table = dict()
        self.reward_mode = reward_mode
        self.each_time_node = each_time_node
        self.source_dict = source_dict
        self.mapping_ver = mapping_ver
        self.layer_infeasible = layer_infeasible
        self.top_log = top_log
        self.max_grf_read = max_GRF
        self.grf_dict = [0 for _ in range(ii)]        # Record grf usage
        self.lrf_dict = [[0]*self.pea_size for _ in range(ii)]
        self.degree_matrix = self.degree(total_graph[0])
        self.init_grf_write(max_GRF)

    def init_grf_write(self,max_GRF):
        if max_GRF%2 == 0:
            self.max_grf_write = max_GRF//2
        else:
            self.max_grf_write = max_GRF

    def degree(self, source_adj_list):
        source_adj_list = source_adj_list.copy()
        degree_dict = {}
        for node in source_adj_list:
            node_degree = 0
            node_idx = int(node[0])
            # print("node_id:", node_idx)
            for i in node[1:]:
                if i == 0:
                    continue
                else:
                    node_degree += 1
            wnode = np.argwhere(source_adj_list == node_idx)
            node_degree += len(wnode) - 1
            degree_dict[node_idx] = node_degree
        degree_matrix = np.zeros([len(source_adj_list), len(source_adj_list)])

        min_degree = 10000
        for node in source_adj_list:
            node_idx = node[0]
            for neighbour_idx in node[1:]:
                if neighbour_idx == 0:
                    continue
                ngb_idx = int(neighbour_idx)
                total_degree = degree_dict[node_idx] + degree_dict[ngb_idx]
                degree_matrix[node_idx-1][ngb_idx-1] = total_degree
                degree_matrix[ngb_idx-1][node_idx-1] = total_degree
                if total_degree < min_degree:
                    min_degree = total_degree
        # degree_matrix = np.where(degree_matrix==0, min_degree, degree_matrix)
        # normalized_degree_matrix = degree_matrix
        normalized_degree_matrix = (degree_matrix - np.min(degree_matrix))/(np.max(degree_matrix)-np.min(degree_matrix))
        # print(normalized_degree_matrix)
        return normalized_degree_matrix

    def generate_batch(self, batch_size):
        # This function is used to generate a batch_size dataset
        total_nums, _, _ = self.total_adj.shape
        # Using the method of choice, select batch_size data from them and repeat the selection to ensure the diversity of input data
        batch_index = np.random.choice(total_nums, batch_size, True)
        
        self.batch_index = batch_index
        return self.total_adj[batch_index], self.total_dict[batch_index], self.total_embedding[batch_index], \
               self.total_net_input[batch_index]

    def action(self, actor_logits, train):
        # This function is used to provide feedback on what action to take after having logits
        node_size = self.action_dims - 1
        none_pos = 0

        source_dict = self.source_dict
        #print("source_dict:")
        #print(source_dict)

        # print(src_node_num)
        # This variable is used to read in the time step of each node
        actor_logits = actor_logits.numpy().copy()
        node_IDs = self.total_embedding[self.batch_index, :, 0]
        time_layer = self.total_embedding[self.batch_index, :, 1]
        total_embeddings = self.total_embedding[self.batch_index, :, :]
        #print("time_layer:")
        #print(time_layer)
        #pe_IDs = self.total_embedding[self.batch_index, :, 2]
        routing = self.get_routing_count(self.source_dict)
        layer_infeasible = self.layer_infeasible
        batch_size, node_nums, _ = actor_logits.shape
        # print(node_nums)
        predicted_ids = np.ones((batch_size, node_nums), dtype=np.int)

        mask = np.zeros(shape=[batch_size, node_nums, self.action_dims])
        for batch in range(batch_size):
            #cur_graph = self.total_graph[self.batch_index][batch]
            #cur_embedding = self.total_embedding[self.batch_index][batch]
            routing_count = copy.deepcopy(routing)
            #print("routing_count:")
            #print(routing_count)
            new_predicted_ids = []
            
            #true_position = []
            infeasible = [[] for _ in range(self.ii)]
            # Record the number of mapped nodes per layer
            have_map_num = np.zeros(shape=[self.ii])

            # Record whether each pe * ii is mapped
            map_node = np.zeros(shape=[self.ii*self.pea_size+1])

            # The time mapping range of each node, with time as the horizontal axis and nodes as the vertical axis
            each_time_node = copy.deepcopy(self.each_time_node)

            # Record the current dependency situation that needs to be met for each time step. The first dimension is time, the second dimension:0 is the PE node, and 1 is the original graph node, indicating that these original graph nodes can only be mapped on these PE nodes
            rely_on = [[[] for _ in range(2)] for _ in range(len(each_time_node))]
            

            for i in range(node_nums):
                
                # set the weight on all infeasible actions to -inf
                cur_node_id = node_IDs[batch, i]

                
                #cur_time_step = time_layer[batch, i]
                cur_time_layer = time_layer[batch, i] % self.ii
                
                cur_embedding = total_embeddings[batch, i]

                #cur_infeasible = infeasible[cur_time_layer].copy()


                new_logits = np.array(actor_logits[batch][i])/self.temperature
                # You can set the tanh layer here
                new_logits = self.C * np.tanh(new_logits)
                # print(new_logits)


                # TODO If each_layer_infeasible does not make all nodes full, perform the update operation. If all nodes are already occupied, randomly map
                
                # Determine how many layers are there, based on which nodes can be mapped in each layer and whether they meet dependency relationships
                if self.action_dims <= 41:
                    can_map = []
                    can_not_map = []
                    for time in range(len(each_time_node)):
                        if time%self.ii == cur_time_layer :
                            for node in range(1,len(each_time_node[0])):
                                if each_time_node[time][node] == 1 and node not in infeasible[cur_time_layer] and node not in layer_infeasible[cur_time_layer]:
                                    pos_temp = []
                                    for m in range(len(rely_on[time][1])):
                                        if node in rely_on[time][1][m]:
                                            pos_temp += rely_on[time][0][m]
                                        if cur_node_id not in rely_on[time][0][m]:
                                            for t_n in rely_on[time][1][m]:
                                                if t_n not in can_not_map:
                                                    can_not_map.append(t_n)
                                    if pos_temp.count(cur_node_id) != 0:
                                        flag = True
                                        for pos in pos_temp:
                                            if pos_temp.count(pos) > pos_temp.count(cur_node_id) and pos not in infeasible[cur_time_layer] and pos not in layer_infeasible[cur_time_layer]:
                                                flag = False
                                        if flag and node not in can_map:
                                            can_map.append(node)
                                                
                    if(len(can_map) == 0):
                        each_layer_infeasible = []
                        each_layer_infeasible = copy.deepcopy(infeasible[cur_time_layer])
                        for temp_node2 in layer_infeasible[cur_time_layer]:
                            if temp_node2 not in each_layer_infeasible:
                                each_layer_infeasible.append(temp_node2)
                        for temp_node3 in can_not_map:
                            if temp_node3 not in each_layer_infeasible:
                                each_layer_infeasible.append(temp_node3)
                    else:
                        each_layer_infeasible = [_ for _ in range(1,self.action_dims)]
                        for can_node in can_map:
                            each_layer_infeasible.remove(can_node)
                else:
                    # According to each_time_node, rely_on and map_node to determine which nodes can be mapped
                    can_map = []
                    for time in range(len(each_time_node)):
                        if time%self.ii == cur_time_layer :
                            for node in range(1,len(each_time_node[0])):
                                if each_time_node[time][node] == 1:
                                    flag = True
                                    for m in range(len(rely_on[time][1])):
                                        if node in rely_on[time][1][m] and cur_node_id not in rely_on[time][0][m]:
                                            free_count = 0
                                            for node1 in rely_on[time][0][m]:
                                                if map_node[node1] != 1:
                                                    free_count += 1
                                            if free_count >= len(rely_on[time][1][m]):
                                                flag = False
                                    if flag and node not in can_map:
                                        can_map.append(node)
                    if can_map == []:
                        temp_count = 0
                        for time in range(len(each_time_node)):
                            if time%self.ii == cur_time_layer :
                                temp_count += np.sum(self.each_time_node[time] == 1)
                        if(temp_count >= self.pea_size):
                            each_layer_infeasible = []
                            each_layer_infeasible = copy.deepcopy(infeasible[cur_time_layer])
                            for temp_node2 in layer_infeasible[cur_time_layer]:
                                if temp_node2 not in each_layer_infeasible:
                                    each_layer_infeasible.append(temp_node2)
                        else:
                            each_layer_infeasible = [_ for _ in range(1,self.action_dims)]
                        #print(1/0)
                    else:
                        each_layer_infeasible = [_ for _ in range(1,self.action_dims)]
                    for node in can_map:
                        each_layer_infeasible.remove(node)                
                #"""
                
                #print("each_layer_infeasible:")
                #print(each_layer_infeasible)

                # Record the number of nodes that can be mapped per layer
                each_layer_num = self.action_dims - 1 - len(each_layer_infeasible)
                
                infeasible_update(batch, i, nodes=each_layer_infeasible, logits=new_logits, mask=mask)

                new_logits = (new_logits - np.max(new_logits))
                probs = np.exp(new_logits) / np.sum(np.exp(new_logits))
                # print(probs)
                # only train
                if train:
                    action = np.random.choice(self.action_dims, 1, p=probs)
                    #print("action:")
                    #print(action[0])

                    # It only ensures that all nodes are mapped in the end
                    if action[0] == none_pos and 1.0 not in probs and self.pea_size-have_map_num[cur_time_layer] <= each_layer_num:

                    #Or priority Mapping Node
                    #if action[0] == none_pos and 1.0 not in probs:
                        action = np.random.choice(self.action_dims, 2, replace=False, p=probs)
                        if action[0] == none_pos:
                            action[0] = action[1]

                    #true_position.append(action[0])
                    if action[0] != none_pos:
                        if routing_count[cur_time_layer][action[0]-1] > 0:
                            routing_count[cur_time_layer][action[0]-1] -= 1
                        else:
                            infeasible[cur_time_layer].append((action[0]))
                            
                        
                        total_embedding = total_embeddings[batch]
                        #"""
                        # Next time step PE sequence number adjacent to cgra
                        temp_pos = cur_embedding[3:][cur_embedding[3:]!=0].tolist()
                        # Previous time step PE sequence number adjacent to cgra
                        last_temp_pos = []
                        for cur_ in total_embedding:
                            if cur_embedding[0] in cur_[3:]:
                                last_temp_pos.append(cur_[0])
                        # Next time step node number adjacent to dfg
                        temp_next_node = source_dict[action[0]-1][1:-4][source_dict[action[0]-1][1:-4]!=0].tolist()
                        # Previous time step node number adjacent to dfg
                        last_temp_next_node = []
                        for cur_ in source_dict:
                            for t_j in range(1,9,2):
                                if cur_[t_j] == action[0]:
                                    last_temp_next_node.append(cur_[0])
                        #print("action[0]:")
                        #print(action[0])
                        #print("rely_on:")
                        #print(rely_on)
                        time_action = cur_time_layer
                        for time in range(len(each_time_node)):
                            if time%self.ii == cur_time_layer and each_time_node[time][action[0]] == 1:
                                time_action = time
                                each_time_node[time][action[0]] = 0
                                if time_action+1 < len(each_time_node) and each_time_node[time_action+1][action[0]] == 1:
                                    temp_next_node.append(action[0])

                                if time_action-1 >= 0 and each_time_node[time_action-1][action[0]] == 1:
                                    last_temp_next_node.append(action[0])
                                
                                break
                        for node in temp_next_node:
                            if each_time_node[time_action+1][node] != 1:
                                temp_next_node.remove(node)

                        for node in last_temp_next_node:
                            if time_action-1 >=0 and each_time_node[time_action-1][node] != 1:
                                last_temp_next_node.remove(node)
                        #del_j = []
                        #print("cur_rely_on:")
                        #print(cur_rely_on)

                        for k in range(len(rely_on[time_action][1])-1,-1,-1):

                            if action[0] in rely_on[time_action][1][k]:
                                #del_j.append(j)
                                rely_on[time_action][1][k].remove(action[0])
                                if len(rely_on[time_action][1][k]) == 0:
                                    rely_on[time_action][1].remove(rely_on[time_action][1][k])
                                    rely_on[time_action][0].remove(rely_on[time_action][0][k])

                        if len(temp_next_node) != 0 and time_action+1 < len(each_time_node):
                            rely_on[(time_action+1)][0].append(temp_pos)
                            rely_on[(time_action+1)][1].append(temp_next_node)
                        
                        if len(last_temp_next_node) != 0 and time_action-1 >= 0:
                            rely_on[(time_action-1)][0].append(last_temp_pos)
                            rely_on[(time_action-1)][1].append(last_temp_next_node)
                            

                        #"""
                    map_node[cur_node_id] = 1
                    new_predicted_ids.append(action[0])
                    have_map_num[cur_time_layer] += 1
                else:

                    action = np.argmax(probs)
                    if action == none_pos and 1.0 not in probs:
                        action = np.argsort(probs)[-2]
                    new_predicted_ids.append(action)
                    if routing_count[action] > 0:
                        routing_count[action] -= 1
                    else:
                        infeasible[cur_time_layer].append(action)
                # print(new_predicted_ids)
            predicted_ids[batch] = new_predicted_ids


        return predicted_ids, mask

    # Obtain the maximum number of nodes per time step 
    def get_routing_count(self,source_dict):
        each_time_node = self.each_time_node
        routing_count = [[0]*(len(source_dict)) for _ in range(self.ii)]
        for node in source_dict:
            start = node[-4]
            end = node[-3]
            for i in range(start+1,len(each_time_node)):
                if each_time_node[i][node[0]] == 0:
                    end = i-1
                    break
                if i == len(each_time_node)-1:
                    end = i
            #print(action[value-1])
            # Nodes with a degree of zero
            #"""
            if source_dict[node[0]-1][-1] == 0:
                for i in range(end,-1,-1):
                    if each_time_node[i][node[0]] == 0:
                        start = i + 1
                        break
                    if i == 0:
                        start = 0
            #"""
            if end-start >= self.ii:
                for i in range(start+self.ii,end+1):
                    routing_count[i%self.ii][node[0]-1] += 1
        #print("routing_count:")
        #print(routing_count)
        #print(1/0)
        return routing_count


    def rewards(self, actions, dict_, embedding, source_dict):

        # This function is used to record the output of the entire batch's actions
        batch_size, node_nums = actions.shape
        rewards = np.zeros(batch_size, dtype=float)
        penaltys = np.zeros(batch_size, dtype=float)
        #lrf_dicts = [[] for _ in range(batch_size)]
        lrf_uses = [[] for _ in range(batch_size)]
        grf_uses = [[] for _ in range(batch_size)]
        action_dicts = [[] for _ in range(batch_size)]
        for index, action in enumerate(actions):
            
            action_dicts[index],mapping_dict = self.get_map_dict(action=action,dict_=dict_[index],embedding=embedding[index],source_dict_all = source_dict)

            penaltys[index],rewards[index],lrf_uses[index],grf_uses[index],action_dicts[index] = self.reward(action=action,dict_=dict_[index],source_dict_all=source_dict,mapping_dict=mapping_dict, embedding=embedding[index], action_dict=action_dicts[index], lrf_dict=copy.deepcopy(self.lrf_dict), grf_dict=copy.deepcopy(self.grf_dict))

        return penaltys,rewards,lrf_uses,grf_uses,action_dicts

    def reward(self, action, dict_, source_dict_all, mapping_dict, embedding, action_dict, lrf_dict, grf_dict):
        lrf_use = []       #Record the usage of lrf（string）
        grf_use = []       #Record the usage of grf（string）

        grf_use_table = []
        grf_write = [0]*self.ii
        grf_read = [0]*self.ii

        flag_valid = [0 for _ in range(len(action_dict))] #用于show place
        
        source_dict = source_dict_all[:,:-4]
        distance_sum = 0
        wrong_mapping = 0
        each_time_node = self.each_time_node
        conn_max = [0 for _ in range(len(source_dict))]
        count_pe=0 

        re_lrf=0
        re_grf=0
        count_grf=0
        count_lrf=0

        action_index_dict=[[] for _ in range(len(self.top_log))]
        

        # For 8X8 reward feedback, the closer the PE is, the higher the reward
        pe_left = self.pea_width
        pe_right = -1
        pe_up = self.pea_width
        pe_down = -1
        count_action = 0
        for index_ in range(len(action_dict)):
            if action_dict[index_] != 0:
                action_index_dict[action_dict[index_]-1].append(index_)
                pe_id = (index_-1)%self.pea_size
                pe_row = pe_id//self.pea_width
                pe_col = pe_id%self.pea_width
                pe_left = min(pe_left,pe_col)
                pe_right = max(pe_right,pe_col)
                pe_up = min(pe_up,pe_row)
                pe_down = max(pe_down,pe_row)
                count_action += 1
        rewards_8 = (pe_down-pe_up+1)*(pe_right-pe_left+1)


        for start in self.top_log:
            #start = source_dict[i][0]

            for k in range(1,len(source_dict[0])):
                if source_dict[start-1][k] == 0:
                    continue
                end = source_dict[start-1][k]
                """
                start_index = []
                end_index = []
                for index in range(len(action_dict)):
                    if action_dict[index] == start:
                        start_index.append(index)
                    if action_dict[index] == end:
                        end_index.append(index)
                """
                start_index = action_index_dict[start-1]
                end_index = action_index_dict[end-1]

                end_line = self.get_start_line(lrf_use, grf_use, end)
                if len(start_index) <= 1 and len(end_index) <= 1:
                    if len(start_index) == 0 or len(end_index) == 0:
                        wrong_mapping += 1
                        # distance_sum += self.pea_width+self.ii
                        # Increase the penalty for unmapped nodes
                        distance_sum += (self.pea_width*2)
                        continue
                    if self.ishave(start, end, mapping_dict):
                        continue
                    else:
                        start_time = (start_index[0]-1)//self.pea_size
                        start_pe = (start_index[0]-1)%self.pea_size
                        end_time = (end_index[0]-1)//self.pea_size
                        end_pe = (end_index[0]-1)%self.pea_size
                        
                        if end_time-start_time < 2 or (end_line != -1 and end_line != end_time):
                            wrong_mapping_temp,distance_sum_temp = self.getdistance([start_index[0],end_index[0]],max(source_dict_all[:,-3]))
                            distance_sum += distance_sum_temp
                            wrong_mapping += wrong_mapping_temp
                            continue

                        if start_pe != end_pe:
                            flag3 = True
                            te_fl = False
                            temp_max_st = 100000
                            temp_max_end = -1

                            # Is grf available
                            temp_max_st, temp_max_end, te_fl, flag3 = self.can_grf(grf_use_table, action_dict[start_index[0]], temp_max_st, temp_max_end, start_time, end_time, grf_dict, grf_read, grf_write, flag3, te_fl)
                            if flag3:

                                if te_fl == False:
                                    grf_write[start_time%self.ii] += 1
                                    grf_read[(end_time-1)%self.ii] += 1
                                    for j in range(start_time,end_time):
                                        grf_dict[j%self.ii] += 1
                                else:
                                    grf_read[(end_time-1)%self.ii] += 1
                                    if start_time < temp_max_st:
                                        grf_write[start_time%self.ii] += 1
                                        for j in range(start_time,temp_max_st):
                                            grf_dict[j%self.ii] += 1
                                    if end_time > temp_max_end:
                                        for j in range(temp_max_end,end_time):
                                            grf_dict[j%self.ii] += 1

                                grf_use.append("pe"+str(start_pe)+"->"+"pe"+str(end_pe)+":"+str(action_dict[start_index[0]])+"->"+str(action_dict[end_index[0]])+"("+str(start_time)+"->"+str(end_time)+")")
                                grf_use_table.append([action_dict[start_index[0]],start_time,end_time])
                                #re_grf += (2/(end_time-start_time-1))*2
                                #count_grf += 1
                            else:
                                wrong_mapping_temp,distance_sum_temp = self.getdistance([start_index[0],end_index[0]],max(source_dict_all[:,-3]))
                                distance_sum += distance_sum_temp
                                wrong_mapping += wrong_mapping_temp
                        else:
                            flag = True     # Is there an available lrf
                            temp_lrf_use = copy.deepcopy(lrf_dict)
                            for j in range(start_time,end_time):
                                temp_lrf_use[j%self.ii][start_pe] += 1
                                if temp_lrf_use[j%self.ii][start_pe] > self.max_LRF:
                                    flag = False

                            if flag:
                                
                                for j in range(start_time,end_time):
                                    lrf_dict[j%self.ii][start_pe] += 1
                                
                                lrf_use.append("pe"+str(start_pe)+":"+str(action_dict[start_index[0]])+"->"+str(action_dict[end_index[0]])+"("+str(start_time)+"->"+str(end_time)+")")
                                #re_lrf += 2/(end_time-start_time-1)*1
                                #count_lrf += 1
                            else:
                                flag2 = True
                                te_fl = False
                                temp_max_st = 100000
                                temp_max_end = -1

                                # Is grf available
                                temp_max_st, temp_max_end, te_fl, flag2 = self.can_grf(grf_use_table, action_dict[start_index[0]], temp_max_st, temp_max_end, start_time, end_time, grf_dict, grf_read, grf_write, flag2, te_fl)
                                if flag2:

                                    if te_fl == False:
                                        grf_write[start_time%self.ii] += 1
                                        grf_read[(end_time-1)%self.ii] += 1
                                        for j in range(start_time,end_time):
                                            grf_dict[j%self.ii] += 1
                                    else:
                                        grf_read[(end_time-1)%self.ii] += 1
                                        if start_time < temp_max_st:
                                            grf_write[start_time%self.ii] += 1
                                            for j in range(start_time,temp_max_st):
                                                grf_dict[j%self.ii] += 1
                                        if end_time > temp_max_end:
                                            for j in range(temp_max_end,end_time):
                                                grf_dict[j%self.ii] += 1

                                    grf_use.append("pe"+str(start_pe)+"->"+"pe"+str(end_pe)+":"+str(action_dict[start_index[0]])+"->"+str(action_dict[end_index[0]])+"("+str(start_time)+"->"+str(end_time)+")")
                                    grf_use_table.append([action_dict[start_index[0]],start_time,end_time])
                                    #re_grf += (2/(end_time-start_time-1))*2
                                    #count_grf += 1
                                else:
                                    wrong_mapping_temp,distance_sum_temp = self.getdistance([start_index[0],end_index[0]],max(source_dict_all[:,-3]))
                                    distance_sum += distance_sum_temp
                                    wrong_mapping += wrong_mapping_temp

                else:
                    if len(start_index) == 0 or len(end_index) == 0:
                        wrong_mapping += 1
                        #distance_sum += self.pea_width+self.ii
                        # Increase the penalty for unmapped nodes
                        distance_sum += (self.pea_width*2)
                        continue
                    res_start = 0
                    res_end = 0
                    res_start_time = 0
                    res_end_time = 0
                    res_temp_start_time = 0
                    res_temp_end_time = 0
                    res_start_pe = 0
                    res_end_pe = 0
                    res_way = 3     # 0 represents PE routing, 1 represents lrf routing, and 2 represents grf routing
                    min_dis = max(source_dict_all[:,-3])+1
                    start_line_xian = self.get_start_line(lrf_use, grf_use, start)
                    start_line_hou = self.get_start_line(lrf_use, grf_use, end)
                    #print("start_line_xian:")
                    #print(start_line_xian)
                    #print("start_line_hou:")
                    #print(start_line_hou)

                    flag_ = False
                    end_end = len(source_dict)+1
                    for col in range(1,len(source_dict[0])):
                        if source_dict[end-1][col] != 0:
                            end_end = min(source_dict_all[source_dict[end-1][col]-1][-3]-1,end_end)
                    
                    for st in start_index:
                        for en in end_index:
                            temp_max_conn = conn_max[start-1]
                            start_time = (st-1)//self.pea_size
                            start_pe = (st-1)%self.pea_size
                            end_time = (en-1)//self.pea_size
                            end_pe = (en-1)%self.pea_size
                            if start_time < start_line_xian:
                                continue
                            if end_time < start_line_hou:
                                continue
                            if end_time-start_time == 1:
                                if end_time > end_end:
                                    continue
                                wrong_mapping_temp,distance_sum_temp = self.getdistance([st,en],max(source_dict_all[:,-3]))
                                #print("distance_sum_temp:")
                                #print(distance_sum_temp)
                                if distance_sum_temp == 0:
                                    #"""
                                    # Nodes with a degree of zero
                                    #if source_dict_all[start-1][-1] == 0 and source_dict_all[start-1][-3] != 0 and each_time_node[source_dict_all[start-1][-4]-1][start] == 1:
                              
                                    if source_dict_all[start-1][-1] == 0 and source_dict_all[start-1][-3] != 0:
                                        conn_max[end-1] = max(conn_max[end-1],end_time)
                                        flag_ = True
                                        res_way = 0
                                        break
                                    #"""
                                    same_map1 = copy.deepcopy(start_index)
                                    action_start = source_dict_all[start-1][-4]
                                    if start_line_xian != -1:
                                        same_map1 = same_map1[start_line_xian-action_start:start_time+1-action_start]
                                    else:
                                        same_map1 = same_map1[:start_time+1-action_start]
                                    while 0 in same_map1:
                                        same_map1.remove(0)
                                    wrong_mapping_temp2,distance_sum_temp2 = self.getdistance(same_map1,max(source_dict_all[:,-3]))
                                    #print("same_map:")
                                    #print(same_map1)
                                    #print("distance_sum_temp2:")
                                    #print(distance_sum_temp2)

                                  
                                    same_map2 = copy.deepcopy(end_index)
                                    action_start = source_dict_all[end-1][-4]
                                    same_map2 = same_map2[end_time-action_start:]
                                    while 0 in same_map2:
                                        same_map2.remove(0)
                                    wrong_mapping_temp3,distance_sum_temp3 = self.getdistance(same_map2,max(source_dict_all[:,-3]))
                                    #print("same_map2:")
                                    #print(same_map2)
                                    #print("distance_sum_temp3:")
                                    #print(distance_sum_temp3)
                                    if distance_sum_temp2 == 0 and distance_sum_temp3 == 0:
                                        conn_max[end-1] = max(conn_max[end-1],end_time)
                                        flag_ = True
                                        res_way = 0

                                        flag_valid[st] = 1
                                        flag_valid[en] = 1
                                        for x in same_map1:
                                            flag_valid[x] = 1
                                        if(len(same_map2) > 1):
                                            count_pe += len(same_map2)-1
                                        if(len(same_map1) > 1):
                                            count_pe += len(same_map1)-1
                                        break
                            if end_line != -1 and end_line != end_time:
                                continue
                            if temp_max_conn != 0 and start_time < temp_max_conn:
                                continue
                            if end_time-start_time < 2:
                                continue

                            if start_pe == end_pe:

                                temp_way = 1
                                flag = True    # Is there an available lrf
                                temp_lrf_use2 = copy.deepcopy(lrf_dict)
                                for j in range(start_time,end_time):
                                    temp_lrf_use2[j%self.ii][start_pe] += 1
                                    if temp_lrf_use2[j%self.ii][start_pe] > self.max_LRF:
                                        flag = False


                                if flag and end_time-start_time < min_dis and temp_way <= res_way:
                                    #same_map = self.get_same_map(action, start, embedding, source_dict_all)
                                    same_map1 = copy.deepcopy(start_index)
                                    action_start = source_dict_all[start-1][-4]
                                    if start_line_xian != -1:
                                        same_map1 = same_map1[start_line_xian-action_start:start_time+1-action_start]
                                    else:
                                        same_map1 = same_map1[:start_time+1-action_start]
                                    while 0 in same_map1:
                                        same_map1.remove(0)
                                    wrong_mapping_temp2,distance_sum_temp2 = self.getdistance(same_map1,max(source_dict_all[:,-3]))
                                    
                                    #same_map = self.get_same_map(action, end, embedding, source_dict_all)
                                    same_map2 = copy.deepcopy(end_index)
                                    action_start = source_dict_all[end-1][-4]
                                    while 0 in same_map2:
                                        same_map2.remove(0)
                                    wrong_mapping_temp3,distance_sum_temp3 = self.getdistance(same_map2,max(source_dict_all[:,-3]))
                                    if distance_sum_temp2 == 0 and distance_sum_temp3 == 0:
                                        flag_valid[st] = 1
                                        flag_valid[en] = 1

                                        for x in same_map1:
                                            flag_valid[x] = 1
                                        res_start = st
                                        res_end = en
                                        res_start_time = start_time
                                        res_end_time = end_time
                                        res_start_pe = start_pe
                                        res_end_pe = end_pe
                                        min_dis = end_time-start_time
                                        res_way = temp_way
                                        if(len(same_map2) > 1):
                                            count_pe += len(same_map2)-1
                                        if(len(same_map1) > 1):
                                            count_pe += len(same_map1)-1
                                
                            temp_way2 = 2
                            flag = True    # Is there an available grf 

                            te_fl = False
                            temp_max_st = 100000
                            temp_max_end = -1

                            # Is grf available
                            temp_max_st, temp_max_end, te_fl, flag = self.can_grf(grf_use_table, action_dict[st], temp_max_st, temp_max_end, start_time, end_time, grf_dict, grf_read, grf_write, flag, te_fl)

                            if flag and end_time-start_time < min_dis and temp_way2 <= res_way:
                                same_map3 = copy.deepcopy(start_index)
                                action_start = source_dict_all[start-1][-4]
                                if start_line_xian != -1:
                                    same_map3 = same_map3[start_line_xian-action_start:start_time+1-action_start]
                                else:
                                    same_map3 = same_map3[:start_time+1-action_start]
                                while 0 in same_map3:
                                    same_map3.remove(0)
                                wrong_mapping_temp2,distance_sum_temp2 = self.getdistance(same_map3,
                                max(source_dict_all[:,-3]))
                                
                                same_map4 = copy.deepcopy(end_index)
                                action_start = source_dict_all[end-1][-4]
              
                                while 0 in same_map4:
                                    same_map4.remove(0)
                                wrong_mapping_temp3,distance_sum_temp3 = self.getdistance(same_map4,max(source_dict_all[:,-3]))
                                if distance_sum_temp2 == 0 and distance_sum_temp3 == 0:
                                    flag_valid[st] = 1
                                    flag_valid[en] = 1

                                    for x in same_map3:
                                        flag_valid[x] = 1
                                    res_start = st
                                    res_end = en
                                    res_start_time = start_time
                                    res_end_time = end_time
                                    res_temp_start_time = temp_max_st
                                    res_temp_end_time = temp_max_end
                                    res_start_pe= start_pe
                                    res_end_pe = end_pe
                                    min_dis = end_time-start_time
                                    res_way = temp_way2
                                    if(len(same_map4) > 1):
                                        count_pe += len(same_map4)-1
                                    if(len(same_map3) > 1):
                                        count_pe += len(same_map3)-1

                        if flag_:
                            break
                    #print("res_way:")
                    #print(res_way)
                    if res_way == 3:
                        same_map5 = copy.deepcopy(start_index)
                        action_start = source_dict_all[start-1][-4]
                        #"""
                        # Nodes with a degree of zero
                        if source_dict_all[start-1][-1] == 0:
                            for i in range(action_start,-1,-1):
                                if each_time_node[i][start] == 0:
                                    action_start = i + 1
                                    break
                                if i == 0:
                                    action_start = 0
                        #"""
                        if start_line_xian != -1:
                            #start_line_xian = action_start
                            wrong_mapping_temp4,distance_sum_temp4 = self.getdistance(same_map5[start_line_xian-action_start:],max(source_dict_all[:,-3]))
                        else:
                            wrong_mapping_temp4,distance_sum_temp4 = self.getdistance(same_map5,max(source_dict_all[:,-3]))

                        wrong_mapping += wrong_mapping_temp4
                        distance_sum += distance_sum_temp4
                    
                        #wrong_mapping_temp5,distance_sum_temp5 = self.get_diff_dis(start_index,end_index,source_dict_all,end_end)
                        wrong_mapping_temp5,distance_sum_temp5 = self.get_diff_dis2(start_index,end_index,source_dict_all,end_end)
                        #wrong_mapping_temp5,distance_sum_temp5 = self.get_diff_dis3(start_index,end_index,source_dict_all,end_end)
                        
                        wrong_mapping += wrong_mapping_temp5
                        distance_sum += distance_sum_temp5
                        
                                
                    elif res_way == 1:
                        
                        for j in range(res_start_time,res_end_time):
                            lrf_dict[j%self.ii][res_start_pe] += 1

                        lrf_use.append("pe"+str(res_start_pe)+":"+str(action_dict[res_start])+"->"+str(action_dict[res_end])+"("+str(res_start_time)+"->"+str(res_end_time)+")")
                        #re_lrf += 2/(res_end_time-res_start_time-1)*1
                        #count_lrf += 1
                    elif res_way == 2:
                        
                        if te_fl == False:
                            grf_write[res_start_time%self.ii] += 1
                            grf_read[(res_end_time-1)%self.ii] += 1
                            for j in range(res_start_time,res_end_time):
                                grf_dict[j%self.ii] += 1
                        else:
                            grf_read[(res_end_time-1)%self.ii] += 1
                            if res_start_time < res_temp_start_time:
                                grf_write[res_start_time%self.ii] += 1
                                for j in range(res_start_time,res_temp_start_time):
                                    grf_dict[j%self.ii] += 1
                            if res_end_time > res_temp_end_time:
                                for j in range(res_temp_end_time,res_end_time):
                                    grf_dict[j%self.ii] += 1

                        grf_use.append("pe"+str(res_start_pe)+"->"+"pe"+str(res_end_pe)+":"+str(action_dict[res_start])+"->"+str(action_dict[res_end])+"("+str(res_start_time)+"->"+str(res_end_time)+")")
                        grf_use_table.append([action_dict[res_start],res_start_time,res_end_time])
                        #re_grf += (2/(res_end_time-res_start_time-1))*2
                        #count_grf += 1
            #"""
            s_index = copy.deepcopy(action_index_dict[start-1])
            # the edge which points to itself
            if len(s_index)>0 and source_dict_all[start-1][-2] == 666 and self.ii > 1:
                
                same_map_self = copy.deepcopy(s_index)
                
                #print("same_map_self_before:")
                #print(same_map_self)
                same_map_self.append(same_map_self[0]+self.ii*self.pea_size)
                #print("same_map_self_end:")
                #print(same_map_self)
                wrong_mapping_temp_self,distance_sum_temp_self = self.getdistance(same_map_self,max(source_dict_all[:,-3]))
                #print("distance_sum_temp_self:")
                #print(distance_sum_temp_self)
                if distance_sum_temp_self != 0:
                    start_t = (s_index[0]-1)//self.pea_size
                    start_p = (s_index[0]-1)%self.pea_size
                    end_p = start_p
                    end_t = start_t+self.ii
                    flag_l = True     # Is there an available lrf
                    temp_lrf_use_ = copy.deepcopy(lrf_dict)
                    for y in range(self.ii):
                        temp_lrf_use_[y%self.ii][start_p] += 1
                        if temp_lrf_use_[y%self.ii][start_p] > self.max_LRF:
                            
                            flag_l = False
                    if flag_l:
                        for y in range(self.ii):
                            lrf_dict[y%self.ii][start_p] += 1
                        lrf_use.append("pe"+str(start_p)+":"+str(action_dict[s_index[0]])+"->"+str(action_dict[s_index[0]])+"("+str(start_t)+"->"+str(start_t+self.ii)+")")
                    else:
                        
                        flag_g = True
                        te_fl_ = False
                        temp_max_st_ = 100000
                        temp_max_end_ = -1

                        # Is grf available
                        temp_max_st_, temp_max_end_, te_fl_, flag_g = self.can_grf(grf_use_table, action_dict[s_index[0]], temp_max_st_, temp_max_end_, start_t, end_t, grf_dict, grf_read, grf_write, flag_g, te_fl_)
                        if flag_g:
                            if te_fl_ == False:
                                grf_write[start_t%self.ii] += 1
                                grf_read[(end_t-1)%self.ii] += 1
                                for y in range(start_t,end_t):
                                    grf_dict[y%self.ii] += 1
                            else:
                                grf_read[(end_t-1)%self.ii] += 1
                                if start_t < temp_max_st_:
                                    grf_write[start_t%self.ii] += 1
                                    for y in range(start_t,temp_max_st_):
                                        grf_dict[y%self.ii] += 1
                                if end_t > temp_max_end_:
                                    for y in range(temp_max_end_,end_t):
                                        grf_dict[y%self.ii] += 1

                            grf_use.append("pe"+str(start_p)+"->"+"pe"+str(end_p)+":"+str(action_dict[s_index[0]])+"->"+str(action_dict[s_index[0]])+"("+str(start_t)+"->"+str(end_t)+")")
                            grf_use_table.append([action_dict[s_index[0]],start_t,end_t])
                            
                        else:
                            
                            distance_sum += distance_sum_temp_self
                            wrong_mapping += wrong_mapping_temp_self


        penalty = (distance_sum + wrong_mapping*4)
        rewards = - ((penalty) ** (1 / 3))
        if(rewards == 0):
            for i in range(len(action_index_dict)):
                if(len(action_index_dict[i])>1):
                    #print(i+1)
                    tmp_start,tmp_end = self.get_start_end_line(lrf_use, grf_use, 1, i+1, i+1)
                    if source_dict_all[i][-1] == 0:
                        if tmp_end == -1:
                            action_dict[action_index_dict[i][0]] = 0
                        
                    if tmp_end != -1:
                        for tmp_index in action_index_dict[i]:
                            if (tmp_index-1)//self.pea_size > tmp_end and flag_valid[tmp_index] != 1:
                                action_dict[tmp_index] = 0

                    if tmp_start <= max(source_dict_all[:,-3]):
                        for tmp_index in action_index_dict[i]:
                            if (tmp_index-1)//self.pea_size < tmp_start and flag_valid[tmp_start] != 1:
                                action_dict[tmp_index] = 0

        return penalty,rewards,lrf_use,grf_use,action_dict

    def can_grf(self, grf_use_table, action, temp_max_st, temp_max_end, start_time, end_time, grf_dict, grf_read, grf_write, flag3, te_fl):
        for use in grf_use_table:
            if use[0] == action:
                temp_max_st = min(temp_max_st,use[1])
                temp_max_end = max(temp_max_end,use[2])
        if temp_max_end != -1:
            te_fl = True
            if grf_read[(end_time-1)%self.ii] >= self.max_grf_read:
                flag3 = False
            else:
                temp_grf = copy.deepcopy(grf_dict)
                if start_time < temp_max_st:
                    if grf_write[start_time%self.ii] >= self.max_grf_write:
                        flag3 = False
                    else:
                        for j in range(start_time,temp_max_st):
                            temp_grf[j%self.ii] += 1
                            if temp_grf[j%self.ii] > self.max_GRF:
                                flag3 = False
                if end_time > temp_max_end:
                    for j in range(temp_max_end,end_time):
                        temp_grf[j%self.ii] += 1
                        if temp_grf[j%self.ii] > self.max_GRF:
                            flag3 = False
        
        if te_fl == False:
            temp_grf = copy.deepcopy(grf_dict)
            if grf_write[start_time%self.ii] >= self.max_grf_write or grf_read[(end_time-1)%self.ii] >= self.max_grf_read:
                flag3 = False
            else:
                for j in range(start_time,end_time):
                    temp_grf[j%self.ii] += 1
                    if temp_grf[j%self.ii] > self.max_GRF:
                        flag3 = False
        return temp_max_st, temp_max_end, te_fl, flag3

    def get_diff_dis(self, start_index, end_index, source_dict_all, end_end):
        start_ = start_index[0]
        end_ = end_index[0]
        min_dis = max(source_dict_all[:,-3])

        for index_i in range(len(start_index)):
            start_temp = start_index[index_i]
            start_ii = (start_temp-1)//(self.pea_width**2)
            for index_j in range(len(end_index)):
                end_temp = end_index[index_j]
                end_ii = (end_temp-1)//(self.pea_width**2)
                if end_ii <= start_ii and end_ii > end_end:
                    continue
                if (end_ii-start_ii) <= min_dis:
                    min_dis = (end_ii-start_ii)
                    start_ = start_temp
                    end_ = end_temp
        wrong_mapping_temp,distance_sum_temp = self.getdistance([start_,end_],max(source_dict_all[:,-3]))
        return wrong_mapping_temp,distance_sum_temp

    def get_diff_dis2(self, start_index, end_index, source_dict_all, end_end):
        wrong_mapping_,distance_sum_ = 0,0
        for index_i in range(len(start_index)):
            start_temp = start_index[index_i]
            start_ii = (start_temp-1)//(self.pea_width**2)
            for index_j in range(len(end_index)):
                end_temp = end_index[index_j]
                end_ii = (end_temp-1)//(self.pea_width**2)
                if end_ii <= start_ii and end_ii > end_end:
                    continue
                start_id = (start_temp-1)%(self.pea_width**2) #pe id
                end_id = (end_temp-1)%(self.pea_width**2)
                distance_sum_temp = self.get_pe_distance(start_id,end_id)
                if distance_sum_temp:
                    wrong_mapping_ += 1
                distance_sum_ += distance_sum_temp
        return wrong_mapping_,distance_sum_

    def get_diff_dis3(self, start_index, end_index, source_dict_all, end_end):
        wrong_mapping_,distance_sum_ = 0,0
        for index_i in range(len(start_index)):
            start_temp = start_index[index_i]
            start_ii = (start_temp-1)//(self.pea_width**2)
            for index_j in range(len(end_index)):
                end_temp = end_index[index_j]
                end_ii = (end_temp-1)//(self.pea_width**2)
                if end_ii <= start_ii and end_ii > end_end:
                    continue
                start_id = (start_temp-1)%(self.pea_width**2) #pe id
                end_id = (end_temp-1)%(self.pea_width**2)
                distance_sum_temp = self.get_pe_distance(start_id,end_id)
                if distance_sum_temp:
                    wrong_mapping_ += 1
                    distance_sum_ += distance_sum_temp
                    break
        return wrong_mapping_,distance_sum_

    def get_start_line(self, lrf_use, grf_use, action):
        start_line = -1
        for each_lrf_use in lrf_use:
            split_right = each_lrf_use.split(":")[1]
            node = split_right.split("(")[0].split("->")[1]
            if node == str(action):
                """
                start_line = max(int(split_right.split("(")[1].split("->")[1].split(")")[0]),start_line)
                """
                start_line = int(split_right.split("(")[1].split("->")[1].split(")")[0])
                break
        if start_line == -1:
            for each_grf_use in grf_use:
                split_right = each_grf_use.split(":")[1]
                node = split_right.split("(")[0].split("->")[1]
                if node == str(action):
                    """
                    start_line = max(int(split_right.split("(")[1].split("->")[1].split(")")[0]),start_line)
                    """
                    start_line = int(split_right.split("(")[1].split("->")[1].split(")")[0])
                    break
        return start_line

    def get_start_end_line(self, lrf_use, grf_use, flag, action_start, action_end):
        #flag=0, used to determine grf and lrf routes, can be activated_start= action_end; flag=1, used to determine which part of the same node needs to be connected, action_start=action_end
        start_line = max(self.source_dict[:,-3])+1
        end_line = -1
        if flag:
  
            for each_lrf_use in lrf_use:
                split_right = each_lrf_use.split(":")[1]
                node = split_right.split("(")[0].split("->")[1]
                if node == str(action_start):
                    start_line = min(int(split_right.split("(")[1].split("->")[1].split(")")[0]),start_line)
            for each_grf_use in grf_use:
                split_right = each_grf_use.split(":")[1]
                node = split_right.split("(")[0].split("->")[1]
                if node == str(action_start):
                    start_line = min(int(split_right.split("(")[1].split("->")[1].split(")")[0]),start_line)
            
            for each_lrf_use in lrf_use:
                split_right = each_lrf_use.split(":")[1]
                node = split_right.split("->")[0]
                if node == str(action_start):
                    end_line = max(int(split_right.split("(")[1].split("->")[0]),end_line)
            for each_grf_use in grf_use:
                split_right = each_grf_use.split(":")[1]
                node = split_right.split("->")[0]
                if node == str(action_start):
                    end_line = max(int(split_right.split("(")[1].split("->")[0]),end_line)
        else:
            
            for each_lrf_use in lrf_use:
                split_right = each_lrf_use.split(":")[1]
                node = split_right.split("->")[0]
                if node == str(action_end):
                    start_line = min(int(split_right.split("(")[1].split("->")[0]),start_line)
            for each_grf_use in grf_use:
                split_right = each_grf_use.split(":")[1]
                node = split_right.split("->")[0]
                if node == str(action_end):
                    start_line = min(int(split_right.split("(")[1].split("->")[0]),start_line)
            
            for each_lrf_use in lrf_use:
                split_right = each_lrf_use.split(":")[1]
                node = split_right.split("(")[0].split("->")[1]
                if node == str(action_start):
                    end_line = max(int(split_right.split("(")[1].split("->")[1].split(")")[0]),end_line)
            for each_grf_use in grf_use:
                split_right = each_grf_use.split(":")[1]
                node = split_right.split("(")[0].split("->")[1]
                if node == str(action_start):
                    end_line = max(int(split_right.split("(")[1].split("->")[1].split(")")[0]),end_line)

        return start_line,end_line

    # Check whether GRF routing is possible
    def grf(self, start_index, end_index, grf_dict, grf_use, action_dict, lrf_use):
        if len(start_index) == 0 or len(end_index) == 0:
            return False
        res_start = 0
        res_end = 0
        res_start_time = 0
        res_end_time = 0
        res_start_pe = 0
        res_end_pe = 0
        min_dis = len(grf_dict)

        start_line,end_line = self.get_start_end_line(lrf_use, grf_use, 0, action_dict[start_index[0]], action_dict[end_index[0]])
        for start in start_index:
            for end in end_index:
                start_time = (start-1)//self.pea_size
                start_pe = (start-1)%self.pea_size
                end_time = (end-1)//self.pea_size
                end_pe = (end-1)%self.pea_size
                if end_time > start_line:
                    continue
                if start_time < end_line:
                    continue
                if end_time-start_time < 2:
                    continue
                flag = True    
                for j in range(start_time+1,end_time):
                    if grf_dict[j] >= self.max_GRF:
                        flag = False
                if flag and end_time-start_time < min_dis:
                    res_start = start
                    res_end = end
                    res_start_time = start_time
                    res_end_time = end_time
                    res_start_pe = start_pe
                    res_end_pe = end_pe
                    min_dis = end_time-start_time
        if min_dis == len(grf_dict):
            return False
        else:
            for j in range(res_start_time+1,res_end_time):
                grf_dict[j] += 1
            grf_use.append("pe"+str(res_start_pe)+"->"+"pe"+str(res_end_pe)+":"+str(action_dict[res_start])+"->"+str(action_dict[res_end])+"("+str(res_start_time)+"->"+str(res_end_time)+")")
            return True

    # Check whether LRF routing is possible
    def lrf(self, start_index, end_index, lrf_dict, lrf_use, action_dict, grf_use):
        if len(start_index) == 0 or len(end_index) == 0:
            return False
        res_start = 0
        res_end = 0
        res_start_time = 0
        res_end_time = 0
        res_pe = 0
        min_dis = len(lrf_dict)
        start_line,end_line = self.get_start_end_line(lrf_use, grf_use, 0, action_dict[start_index[0]], action_dict[end_index[0]])
        
        for start in start_index:
            for end in end_index:
                start_time = (start-1)//self.pea_size
                start_pe = (start-1)%self.pea_size
                end_time = (end-1)//self.pea_size
                end_pe = (end-1)%self.pea_size
                if end_time > start_line:
                    continue
                if start_time < end_line:
                    continue
                if end_time-start_time < 2:
                    continue
                if start_pe != end_pe:
                    continue
                flag = True     
                for j in range(start_time+1,end_time):
                    if lrf_dict[j][start_pe] >= self.max_LRF:
                        flag = False
                if flag and end_time-start_time < min_dis:
                    res_start = start
                    res_end = end
                    res_start_time = start_time
                    res_end_time = end_time
                    res_pe = start_pe
                    min_dis = end_time-start_time
        if min_dis == len(lrf_dict):
            return False
        else:
            for j in range(res_start_time+1,res_end_time):
                lrf_dict[j][res_pe] += 1
            lrf_use.append("pe"+str(res_pe)+":"+str(action_dict[res_start])+"->"+str(action_dict[res_end])+"("+str(res_start_time)+"->"+str(res_end_time)+")")
            return True

    def get_same_map(self, action, cur_action, embedding, source_dict_all):
        each_time_node = self.each_time_node
        start = source_dict_all[cur_action-1][-4]
        end = source_dict_all[cur_action-1][-3]
        
        for i in range(start+1,len(each_time_node)):
            if each_time_node[i][cur_action] == 0:
                end = i-1
                break
            if i == len(each_time_node)-1:
                end = i
        #"""
        # Nodes with a degree of zero
        if source_dict_all[cur_action-1][-1] == 0:
            for i in range(start,-1,-1):
                if each_time_node[i][cur_action] == 0:
                    start = i + 1
                    break
                if i == 0:
                    start = 0
        #"""
        #same_map = self.get_same_map(start,end,embedding,action)
        same_map = [0]*(end-start+1)
        #print("same_map:")
        #print(same_map)
        for i in range(len(action)):
            if action[i] == cur_action:
                c_time = embedding[i][1]
                index = c_time-start
                #print("check:")
                #print(start)
                #print(end)
                #print(c_time)
                while index < 0:
                    index = index + self.ii
                index = index % self.ii
                while same_map[index] != 0:
                    index = index + self.ii
                same_map[index] = embedding[i][0]

        return same_map

    # Check whether nodes mapped to the same node are adjacent
    def sameRightRewards(self, action, embedding, source_dict_all, lrf_use, grf_use):
        #print("action:")
        #print(action)
        #print("embedding:")
        #print(embedding)
        each_time_node = self.each_time_node
        action_ = action.tolist()
        chfu = []
        count = 0
        rewards = 0
        while len(action_)>0:
            if action_[0] != 0 and action_.count(action_[0])>1 and action_[0] not in chfu:
                #print("action_[0]:")
                #print(action_[0])
                start = source_dict_all[action_[0]-1][-4]
                end = source_dict_all[action_[0]-1][-3]
                
                for i in range(start+1,len(each_time_node)):
                    if each_time_node[i][action_[0]] == 0:
                        end = i-1
                        break
                
                #same_map = self.get_same_map(start,end,embedding,action)
                same_map = [0]*(end-start+1)
                for i in range(len(action)):
                    if action[i] == action_[0]:
                        c_time = embedding[i][1]
                        index = c_time-start
                        #print("check:")
                        #print(start)
                        #print(end)
                        #print(c_time)
                        while index < 0:
                            index = index + self.ii
                        index = index % self.ii
                        while same_map[index] != 0:
                            index = index + self.ii
                        same_map[index] = embedding[i][0]
                start_line,end_line = self.get_start_end_line(lrf_use, grf_use, 1, action_[0], action_[0])
               
                chfu.append(action_[0])

                if start_line != max(self.source_dict[:,-3])+1 and end_line != -1:
                    if start_line >= end_line:
                        continue
                    same_map = same_map[start_line-start:end_line+1-start]
                if start_line != max(self.source_dict[:,-3])+1 and end_line == -1:
                    same_map = same_map[start_line-start:]
                if start_line == max(self.source_dict[:,-3])+1 and end_line != -1:
                    same_map = same_map[:end_line+1-start]
                while 0 in same_map:
                    same_map.remove(0) 
                #print("same_map:")
                #print(same_map)
                temp_count, temp_rewards = self.getdistance(same_map,self.ii)
                count = count + temp_count
                rewards = rewards + temp_rewards
                #print("rewards:")
                #print(rewards)
                #print(1/0)
            action_.remove(action_[0])
        return count,rewards

    # Obtain the number and distance of routing node mapping failures
    def getdistance(self, same_map, ii):
        if len(same_map) <= 1:
            return 0,0
        pea_width = self.pea_width
        same_map = np.array(same_map)
        c_time = (same_map-1)//(pea_width**2)
        pe_id = (same_map-1)%(pea_width**2)
        count = 0
        rewards = 0
        for i in range(len(c_time)-1):
            time_dis = c_time[i+1]-c_time[i]
            if time_dis == 0:
                time_dis = ii
            
            if (time_dis+ii)%ii != 1 or self.get_pe_distance(pe_id[i],pe_id[i+1]) > 1:
                count = count + 1
                rewards = rewards + (time_dis+ii)%ii + self.get_pe_distance(pe_id[i],pe_id[i+1])
                #rewards = rewards + self.get_pe_distance(pe_id[i],pe_id[i+1])
            
        return count,rewards

    # Obtain the distance between PE
    def get_pe_distance(self,pe_id1,pe_id2):
        distance = 0
        if self.pea_width == 2:
            distance = abs(pe_id1-pe_id2)
            if self.reward_mode == 2:
                if distance == 3:
                    distance = 1
        else:
            pe1_x = pe_id1%self.pea_width
            pe1_y = pe_id1//self.pea_width
            pe2_x = pe_id2%self.pea_width
            pe2_y = pe_id2//self.pea_width
            distance_x = abs(pe1_x-pe2_x)
            distance_y = abs(pe1_y-pe2_y)
            if self.reward_mode == 2:
                distance_x = min(distance_x,self.pea_width-distance_x)
                distance_y = min(distance_y,self.pea_width-distance_y)
            elif self.reward_mode == 4:
                distance_x = (distance_x+1)//2
                distance_y = (distance_y+1)//2
            elif self.reward_mode == 6:
                distance_x = min((distance_x+1)//2,((min(pe1_x,pe2_x)-0+1)//2+(self.pea_width-max(pe1_x,pe2_x))//2)+1)
                distance_y = min((distance_y+1)//2,((min(pe1_y,pe2_y)-0+1)//2+(self.pea_width-max(pe1_y,pe2_y))//2)+1)
            distance = distance_x+distance_y
            if self.reward_mode == 3:
                distance = max(distance_x,distance_y)
            if self.reward_mode == 5:
                distance1 = (distance_x+distance_y+1)//2
                distance_x2 = min(distance_x,self.pea_width-distance_x)
                distance_y2 = min(distance_y,self.pea_width-distance_y)
                if(distance_x2<distance_x and distance_y2<distance_y):
                    distance2 = (distance_x2+distance_y2+1)//2+1
                elif(distance_x2==distance_x and distance_y2==distance_y):
                    distance2 = distance1
                else:
                    distance2 = (distance_x2+distance_y2)//2+1
                if(distance_x==0):
                    #distance2 = ((min(pe1_y,pe2_y)-0)//2+(self.pea_width-max(pe1_y,pe2_y))//2)
                    distance2 = ((min(pe1_y,pe2_y)-0+1)//2+(self.pea_width-max(pe1_y,pe2_y))//2)+1
                if(distance_y==0):
                    #distance2 = ((min(pe1_x,pe2_x)-0)//2+(self.pea_width-max(pe1_x,pe2_x))//2)
                    distance2 = ((min(pe1_x,pe2_x)-0+1)//2+(self.pea_width-max(pe1_x,pe2_x))//2)+1
                distance = min(distance1,distance2)
        return distance

    # Check whether it is an adjacent edge 
    def ishave(self ,start_, end_, mapping_dict):

        for i in range(len(mapping_dict)):
            if mapping_dict[i][0] != start_:
                continue
            for j in range(1,len(mapping_dict[0])):
                if mapping_dict[i][j] == end_:
                    return True

        return False

    # Obtain a mapping of c_type rows instead of ii rows
    def get_map_dict(self, action, dict_, embedding, source_dict_all):
        each_time_node = self.each_time_node
        action_ = action.tolist()
        mapping_ver = self.mapping_ver
        source_dict = source_dict_all.copy()
        action_dict = [0]*(len(mapping_ver)+1)      # Subscript represents node, 0 is empty
        for key,value in dict_.items():
            if action[value-1] == 0:
                continue
            start = source_dict[action[value-1]-1][-4]       #asap
            end = source_dict[action[value-1]-1][-3]         #alap
            for i in range(start+1,len(each_time_node)):
                if each_time_node[i][action[value-1]] == 0:
                    end = i-1
                    break
                if i == len(each_time_node)-1:
                    end = i
            #print(action[value-1])
            #"""
            # Nodes with a degree of zero
            if source_dict[action[value-1]-1][-1] == 0:
                for i in range(start,-1,-1):
                    if each_time_node[i][action[value-1]] == 0:
                        start = i + 1
                        break
                    if i == 0:
                        start = 0
            #"""
            if end-start == 0:
                action_dict[key+(start//self.ii)*(self.ii*self.pea_size)] = action[value-1]
            elif action_.count(action[value-1]) > 1:
                c_time = embedding[value-1][1]
                while c_time < start:
                    c_time += self.ii
                    key = key+(self.ii*self.pea_size)
                #print(c_time)
                while action[value-1] in action_dict[c_time*self.pea_size+1:(c_time+1)*self.pea_size+1] and c_time <= end:
                    #print("action_dict:")
                    #print(action_dict)
                    #print(action_dict[c_time*self.pea_size+1:(c_time+1)*self.pea_size+1])
                    c_time += self.ii
                    key += (self.ii*self.pea_size)
                    #print("key:")
                    #print(key)
                action_dict[key] = action[value-1]

        map_dict = np.zeros([len(mapping_ver), 6], dtype=int)

        for i in range(len(mapping_ver)):
            for j in range(len(mapping_ver[0])):
                if j == 1 or j == 2:
                    continue
                if j == 0:
                    map_dict[i][j] = action_dict[mapping_ver[i][j]]
                elif mapping_ver[i][j] != 0:
                    k = 1
                    while k<6:
                        if map_dict[i][k] == 0:
                            map_dict[i][k] = action_dict[mapping_ver[i][j]]
                            break
                        k = k+1

        #print("map_dict:")
        #print(map_dict)       
        return action_dict,map_dict

    def show_placer(self, action_dict, lrf_use, grf_use):
        #"""
        print("answer:")
        # c_type rows
        for i in range(1,len(action_dict),self.pea_size):
            print("    ",end="")
            for num in range(self.pea_size):
                if num == self.pea_size-1:
                    print("%2d"%action_dict[i+num])
                else:
                    print("%2d"%action_dict[i+num],end=" ")
                    if (num+1)%self.pea_width == 0:
                        print("\n    ",end="")
            print("--------------------")
        #"""
        """
        # ii rows
        answer = [[0 for i in range(self.pea_size)] for j in range(self.ii)]
        for i in range(1,len(action_dict)):
            if action_dict[i] != 0:
                tmp = (i-1)%(self.pea_size*self.ii)
                answer[tmp//self.pea_size][tmp%self.pea_size]=action_dict[i]
        print("answer:")
        for i in range(len(answer)):
            print("    ",end="")
            for j in range(len(answer[i])):
                print("%2d"%answer[i][j],end=" ")
                if (j+1)%self.pea_width == 0:
                        print("\n    ",end="")
            print("--------------------")
        """

        if len(lrf_use) != 0:
            print("    lrf_use:")
            for use in lrf_use:
                print("    ",end="")
                print(use)
        if len(grf_use) != 0:
            print("    grf_use:")
            for use in grf_use:
                print("    ",end="")
                print(use)
        