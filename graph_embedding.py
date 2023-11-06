import numpy as np

class Graph_cgra:
    def __init__(self, pea_width, ii, dfg_data, reward_mode):
        self.total_node = (pea_width**2)*ii
        self.graph = None
        self.adj_m = None
        self.normalized_adj = None
        self.feature_m = None
        self.net_input = None
        self.dfg_data = dfg_data
        self.ii = ii
        self.pea_width = pea_width
        self.reward_mode = reward_mode
        self.gen_graph(pea_width, ii)
        self.gen_adj()
        self.gen_feature_m(self.total_node, pea_width, ii)
        self.gen_net_input()
        

    def gen_graph(self,pea_width, ii):
        graph = np.zeros([self.total_node, 6], dtype=int)
        for i in range(self.total_node):
            graph[i][0] = i+1
        if pea_width == 2:
            for i in range(ii):
                for j in range(4):
                    temp = [m for m in range(((i+1)*4+1)%self.total_node,((i+1)*4+1)%self.total_node+4)]
                    graph[i*4+j][1] = temp[j-1]
                    graph[i*4+j][2] = temp[j]
                    graph[i*4+j][3] = temp[(j+1)%4]
        else:
            pea_size = pea_width*pea_width
            if self.reward_mode == 2:
                # torus
                for i in range(ii):
                    for j in range(pea_size):
                        temp = [m for m in range(((i+1)*pea_size+1)%self.total_node,((i+1)*pea_size+1)%self.total_node+pea_size)]
                        #print("temp:")
                        #print(temp)
                        graph[i*pea_size+j][1] = temp[(j-pea_width+pea_size)%pea_size]
                        graph[i*pea_size+j][2] = temp[(j-1+pea_size)%pea_size if j%pea_width != 0 else (j-1+pea_width+pea_size)%pea_size]
                        graph[i*pea_size+j][3] = temp[j]
                        graph[i*pea_size+j][4] = temp[(j+1+pea_size)%pea_size if (j+1)%pea_width != 0 else (j+1-pea_width+pea_size)%pea_size]
                        graph[i*pea_size+j][5] = temp[(j+pea_width+pea_size)%pea_size]
                        #print(graph[i*pea_size+j])
                #print(1/0)
            elif self.reward_mode == 1:
                # mesh
                for i in range(ii):
                    for j in range(pea_size):
                        temp = [m for m in range(((i+1)*pea_size+1)%self.total_node,((i+1)*pea_size+1)%self.total_node+pea_size)]
                        if j not in [_ for _ in range(pea_width)]:
                            graph[i*pea_size+j][1] = temp[j-pea_width]
                        if j not in [pea_width*_ for _ in range(pea_width)]:
                            graph[i*pea_size+j][2] = temp[j-1] 
                        graph[i*pea_size+j][3] = temp[j]
                        if j not in [pea_width*(_+1)-1 for _ in range(pea_width)]:
                            graph[i*pea_size+j][4] = temp[j+1]
                        if j not in [(pea_width-1)*pea_width+_ for _ in range(pea_width)]:
                            graph[i*pea_size+j][5] = temp[j+pea_width]
            elif self.reward_mode == 3:
                # diagonal+mesh
                graph = np.zeros([self.total_node, 10], dtype=int)
                for i in range(self.total_node):
                    graph[i][0] = i+1
                for i in range(ii):
                    temp = [m for m in range(((i+1)*pea_size+1)%self.total_node,((i+1)*pea_size+1)%self.total_node+pea_size)]
                    #print("temp:")
                    #print(temp)
                    for j in range(pea_size):
                        if j not in [_ for _ in range(pea_width)] and j not in [pea_width*_ for _ in range(pea_width)]:
                            graph[i*pea_size+j][1] = temp[j-pea_width-1]
                        if j not in [_ for _ in range(pea_width)]:
                            graph[i*pea_size+j][2] = temp[j-pea_width]
                        if j not in [_ for _ in range(pea_width)] and j not in [pea_width*(_+1)-1 for _ in range(pea_width)]:
                            graph[i*pea_size+j][3] = temp[j-pea_width+1]
                        if j not in [pea_width*_ for _ in range(pea_width)]:
                            graph[i*pea_size+j][4] = temp[j-1] 
                        graph[i*pea_size+j][5] = temp[j]
                        if j not in [pea_width*(_+1)-1 for _ in range(pea_width)]:
                            graph[i*pea_size+j][6] = temp[j+1]
                        if j not in [pea_width*_ for _ in range(pea_width)] and j not in [(pea_width-1)*pea_width+_ for _ in range(pea_width)]:
                            graph[i*pea_size+j][7] = temp[j+pea_width-1]
                        if j not in [(pea_width-1)*pea_width+_ for _ in range(pea_width)]:
                            graph[i*pea_size+j][8] = temp[j+pea_width]
                        if j not in [pea_width*(_+1)-1 for _ in range(pea_width)] and j not in [(pea_width-1)*pea_width+_ for _ in range(pea_width)]:
                            graph[i*pea_size+j][9] = temp[j+pea_width+1]
                        #print(graph[i*pea_size+j])
                #print(1/0)
            elif self.reward_mode == 4:
                # 1-hop+mesh
                graph = np.zeros([self.total_node, 10], dtype=int)
                for i in range(self.total_node):
                    graph[i][0] = i+1
                for i in range(ii):
                    temp = [m for m in range(((i+1)*pea_size+1)%self.total_node,((i+1)*pea_size+1)%self.total_node+pea_size)]
                    #print("temp:")
                    #print(temp)
                    #print(1/0)
                    for j in range(pea_size):
                        if j not in [_ for _ in range(pea_width*2)]:
                            graph[i*pea_size+j][1] = temp[j-pea_width*2]
                        if j not in [_ for _ in range(pea_width)]:
                            graph[i*pea_size+j][2] = temp[j-pea_width]
                        if j not in [pea_width*_ for _ in range(pea_width)] and j not in [pea_width*_+1 for _ in range(pea_width)]:
                            graph[i*pea_size+j][3] = temp[j-2]
                        if j not in [pea_width*_ for _ in range(pea_width)]:
                            graph[i*pea_size+j][4] = temp[j-1] 
                        graph[i*pea_size+j][5] = temp[j]
                        if j not in [pea_width*(_+1)-1 for _ in range(pea_width)]:
                            graph[i*pea_size+j][6] = temp[j+1]
                        if j not in [pea_width*(_+1)-1 for _ in range(pea_width)] and j not in [pea_width*(_+1)-2 for _ in range(pea_width)]:
                            graph[i*pea_size+j][7] = temp[j+2]
                        if j not in [(pea_width-1)*pea_width+_ for _ in range(pea_width)]:
                            graph[i*pea_size+j][8] = temp[j+pea_width]
                        if j not in [(pea_width-1)*pea_width+_ for _ in range(pea_width)] and j not in [(pea_width-2)*pea_width+_ for _ in range(pea_width)]:
                            graph[i*pea_size+j][9] = temp[j+pea_width*2]  
                        #print(graph[i*pea_size+j])
                #print(1/0)
            elif self.reward_mode == 5:
                
                graph = np.zeros([self.total_node, 14], dtype=int)
                for i in range(self.total_node):
                    graph[i][0] = i+1
                for i in range(ii):
                    temp = [m for m in range(((i+1)*pea_size+1)%self.total_node,((i+1)*pea_size+1)%self.total_node+pea_size)]
                    #print("temp:")
                    #print(temp)
                    for j in range(pea_size):
                        # 1-hop+mesh
                        if j not in [_ for _ in range(pea_width*2)]:
                            graph[i*pea_size+j][1] = temp[j-pea_width*2]
                        if j not in [_ for _ in range(pea_width)]:
                            graph[i*pea_size+j][2] = temp[j-pea_width]
                        if j not in [pea_width*_ for _ in range(pea_width)] and j not in [pea_width*_+1 for _ in range(pea_width)]:
                            graph[i*pea_size+j][3] = temp[j-2]
                        if j not in [pea_width*_ for _ in range(pea_width)]:
                            graph[i*pea_size+j][4] = temp[j-1] 
                        graph[i*pea_size+j][5] = temp[j]
                        if j not in [pea_width*(_+1)-1 for _ in range(pea_width)]:
                            graph[i*pea_size+j][6] = temp[j+1]
                        if j not in [pea_width*(_+1)-1 for _ in range(pea_width)] and j not in [pea_width*(_+1)-2 for _ in range(pea_width)]:
                            graph[i*pea_size+j][7] = temp[j+2]
                        if j not in [(pea_width-1)*pea_width+_ for _ in range(pea_width)]:
                            graph[i*pea_size+j][8] = temp[j+pea_width]
                        if j not in [(pea_width-1)*pea_width+_ for _ in range(pea_width)] and j not in [(pea_width-2)*pea_width+_ for _ in range(pea_width)]:
                            graph[i*pea_size+j][9] = temp[j+pea_width*2]   
                        # dia
                        if j not in [_ for _ in range(pea_width)] and j not in [pea_width*_ for _ in range(pea_width)]:
                            graph[i*pea_size+j][10] = temp[j-pea_width-1]
                        if j not in [_ for _ in range(pea_width)] and j not in [pea_width*(_+1)-1 for _ in range(pea_width)]:
                            graph[i*pea_size+j][11] = temp[j-pea_width+1]
                        if j not in [pea_width*_ for _ in range(pea_width)] and j not in [(pea_width-1)*pea_width+_ for _ in range(pea_width)]:
                            graph[i*pea_size+j][12] = temp[j+pea_width-1]
                        if j not in [pea_width*(_+1)-1 for _ in range(pea_width)] and j not in [(pea_width-1)*pea_width+_ for _ in range(pea_width)]:
                            graph[i*pea_size+j][13] = temp[j+pea_width+1]
                        # torus
                        if j%pea_width == 0:
                            graph[i*pea_size+j][4] = temp[(j-1+pea_width+pea_size)%pea_size]
                        if (j+1)%pea_width == 0:
                            graph[i*pea_size+j][6] = temp[(j+1-pea_width+pea_size)%pea_size]
                        if j in range(pea_width):
                            graph[i*pea_size+j][2] = temp[(j-pea_width+pea_size)%pea_size]
                        if j in range(pea_width*(pea_width-1),pea_width*pea_width):
                            graph[i*pea_size+j][8] = temp[(j+pea_width+pea_size)%pea_size]

                        #print(graph[i*pea_size+j])
                #print(1/0)

            elif self.reward_mode == 6:
                # 1-hop+mesh+tours
                graph = np.zeros([self.total_node, 10], dtype=int)
                for i in range(self.total_node):
                    graph[i][0] = i+1
                for i in range(ii):
                    temp = [m for m in range(((i+1)*pea_size+1)%self.total_node,((i+1)*pea_size+1)%self.total_node+pea_size)]
                    #print("temp:")
                    #print(temp)
                    #print(1/0)
                    for j in range(pea_size):
                        # 1-hop+mesh
                        if j not in [_ for _ in range(pea_width*2)]:
                            graph[i*pea_size+j][1] = temp[j-pea_width*2]
                        if j not in [_ for _ in range(pea_width)]:
                            graph[i*pea_size+j][2] = temp[j-pea_width]
                        if j not in [pea_width*_ for _ in range(pea_width)] and j not in [pea_width*_+1 for _ in range(pea_width)]:
                            graph[i*pea_size+j][3] = temp[j-2]
                        if j not in [pea_width*_ for _ in range(pea_width)]:
                            graph[i*pea_size+j][4] = temp[j-1] 
                        graph[i*pea_size+j][5] = temp[j]
                        if j not in [pea_width*(_+1)-1 for _ in range(pea_width)]:
                            graph[i*pea_size+j][6] = temp[j+1]
                        if j not in [pea_width*(_+1)-1 for _ in range(pea_width)] and j not in [pea_width*(_+1)-2 for _ in range(pea_width)]:
                            graph[i*pea_size+j][7] = temp[j+2]
                        if j not in [(pea_width-1)*pea_width+_ for _ in range(pea_width)]:
                            graph[i*pea_size+j][8] = temp[j+pea_width]
                        if j not in [(pea_width-1)*pea_width+_ for _ in range(pea_width)] and j not in [(pea_width-2)*pea_width+_ for _ in range(pea_width)]:
                            graph[i*pea_size+j][9] = temp[j+pea_width*2]
                        
                        # torus
                        if j%pea_width == 0:
                            graph[i*pea_size+j][4] = temp[(j-1+pea_width+pea_size)%pea_size]
                        if (j+1)%pea_width == 0:
                            graph[i*pea_size+j][6] = temp[(j+1-pea_width+pea_size)%pea_size]
                        if j in range(pea_width):
                            graph[i*pea_size+j][2] = temp[(j-pea_width+pea_size)%pea_size]
                        if j in range(pea_width*(pea_width-1),pea_width*pea_width):
                            graph[i*pea_size+j][8] = temp[(j+pea_width+pea_size)%pea_size]
            else:
                print("not support this structure")    
                #print("graph:")
                #print(graph)
                #print(1/0)
       
        self.graph = graph

    def gen_net_input(self):
        embedding = self.feature_m.copy()

        #"""
        # The first part of net_input uses one-hot to represent the node number of each node
        #node_number = np.identity(len(embedding))
        node_number = np.identity(self.pea_width**2)
        node_number = np.tile(node_number,(len(embedding)//(self.pea_width**2),1))
        # The second part of net_input uses one-hot to represent the time steps of each node
        #node_timestep = np.zeros([len(embedding), max(self.dfg_data[:, -3]) + 1], dtype=np.int)
        node_timestep = np.zeros([len(embedding), self.ii], dtype=np.int)
        node_timestep[range(len(embedding)), embedding[range(len(embedding)), 1]] = 1
        # The third part of net_input uses one-hot to represent the PE number of each node
        node_pe_num = np.zeros([len(embedding), max(embedding[:, 2]) + 1], dtype=np.int)
        node_pe_num[range(len(embedding)), embedding[range(len(embedding)), 2]] = 1
        net_input = np.concatenate([node_number, node_timestep, node_pe_num], axis=1)
        #"""
        #print("net_input:")
        #print(net_input.shape)
        """
        # The first part of net_input represents the node number of each node
        node_number = np.zeros([len(embedding),1], dtype=np.int)
        node_number[range(len(embedding)),0] = embedding[range(len(embedding)), 0]
        # The second part of net_input uses one-hot to represent the time steps of each node
        node_timestep = np.zeros([len(embedding), 1], dtype=np.int)
        node_timestep[range(len(embedding)),0] = embedding[range(len(embedding)), 1]
        # The third part of net_input uses one-hot to represent the time steps of each node
        node_timestep2 = np.zeros([len(embedding), 1], dtype=np.int)
        node_timestep2 = node_timestep
        # The fourth part of net_input represents the number of adjacent nodes for each node
        neighbor_number = np.zeros([len(embedding), 1], dtype=np.int)
        neighbor_number[range(len(embedding)),0] = np.sum(np.count_nonzero([embedding[:,3:]], axis=0),axis=1)[range(len(embedding))]

        net_input = np.concatenate([node_number, node_timestep, node_timestep2, neighbor_number], axis=1)
        """

        self.net_input = net_input


    def gen_adj(self):
        graph = self.graph
        node_num = len(graph)
        adj = np.zeros([node_num, node_num], dtype=int)
        for i in range(node_num):
            for j in range(node_num):
                if i == j:
                    adj[i][j] = 0
                elif j + 1 in graph[i] or i + 1 in graph[j]:
                    adj[i][j] = 1
                """
                elif j + 1 in graph[i]:
                    adj[i][j] = 1
                elif i + 1 in graph[j]:
                    adj[j][i] = 1
                """
        self.adj_m = adj
        #print("self.adj_m:")
        #print(self.adj_m)
        # normalization
        adj = adj + np.identity(len(adj))
        rowsum = np.sum(adj, axis=1)
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        normalized_adj = np.dot(np.dot(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        self.normalized_adj = normalized_adj

    def gen_feature_m(self, node_num, pea_width, ii):
        #print("self.graph:")
        #print(self.graph)
        fea = np.zeros([node_num, 2+len(self.graph[0])], dtype=int)
        for i in range(node_num):
          # The first three columns are node serial number, time step, and PE number, respectively
            fea[i][0] = i+1
            fea[i][1] = i//(pea_width**2)
            fea[i][2] = i%(pea_width**2)
          # The last few columns are child nodes
            for j in range(3,len(fea[0])):
                fea[i][j] = self.graph[i][j-2]
        self.feature_m = fea
        #print("self.feature_m:")
        #print(self.feature_m)

    def get_graph_adj_feature_input(self):
        return self.graph,self.normalized_adj,self.feature_m,self.net_input

    def get_grf_size(self):
        # This function is used to provide feedback on how many nodes there are
        return len(self.graph)

    def get_grf_input_size(self):
        # This function is used to provide feedback on the feature_size of each node
        _, feature_size = self.net_input.shape
        return feature_size
        

class Graph_dfg:
    def __init__(self, origin_embedding, pea_width, ii):
        self.total_node = len(origin_embedding)
        self.net_input = None
        self.graph = None
        self.adj_m = None
        self.normalized_adj = None
        self.pea_width = pea_width
        self.ii = ii
        self.gen_graph(origin_embedding)
        self.gen_net_input(origin_embedding)
        self.gen_adj()
        self.normalize_adj(self.adj_m)

    def gen_graph(self,origin_embedding):
        self.graph = origin_embedding[:,:-4]

    def gen_net_input(self,origin_embedding):

        embedding = origin_embedding.copy()

        #"""
        # The first part of net_input uses one-hot to represent the node number of each node
        #node_number = np.identity(self.pea_width*self.pea_width*self.ii)[:len(embedding)]
        node_number = np.identity(self.pea_width*self.pea_width*self.ii)[:len(embedding)]
        # The second part of net_input uses one-hot to represent the asap of each node
        #node_timestep = np.zeros([len(embedding), self.ii], dtype=np.int)
        #node_timestep[range(len(embedding)), (embedding[range(len(embedding)), -4])%self.ii] = 1
        node_timestep = np.zeros([len(embedding), max(embedding[:, -4]) + 1], dtype=np.int)
        node_timestep[range(len(embedding)), embedding[range(len(embedding)), -4]] = 1
        # The third part of net_input uses one-hot to represent the alap of each node
        #node_timestep2 = np.zeros([len(embedding), self.ii], dtype=np.int)
        #node_timestep2[range(len(embedding)), (embedding[range(len(embedding)), -3])%self.ii] = 1
        node_timestep2 = np.zeros([len(embedding), max(embedding[:, -3]) + 1], dtype=np.int)
        node_timestep2[range(len(embedding)), embedding[range(len(embedding)), -3]] = 1
        
        net_input = np.concatenate([node_number, node_timestep, node_timestep2], axis=1)
        #print(1/0)
        #"""
        #print("embedding:")
        #print(embedding)
        """
        node_number = np.zeros([len(embedding),1], dtype=np.int)
        node_number[range(len(embedding)),0] = embedding[range(len(embedding)), 0]
        # The second part of net_input represent the asap of each node
        node_timestep = np.zeros([len(embedding), 1], dtype=np.int)
        node_timestep[range(len(embedding)),0] = embedding[range(len(embedding)), -4]
        # The third part of net_input represent the alap of each node
        node_timestep2 = np.zeros([len(embedding), 1], dtype=np.int)
        node_timestep2[range(len(embedding)),0] = embedding[range(len(embedding)), -3]
        # The fourth part of net_input represents the number of adjacent nodes for each node
        neighbor_number = np.zeros([len(embedding), 1], dtype=np.int)
        neighbor_number[range(len(embedding)),0] = np.sum(np.count_nonzero([embedding[:,1:-4]], axis=0),axis=1)[range(len(embedding))]

        net_input = np.concatenate([node_number, node_timestep, node_timestep2, neighbor_number], axis=1)
        """
        #print("net_input:")
        #print(net_input)
        #print(1/0)
        self.net_input = net_input
        

    def gen_adj(self):
        graph = self.graph
        node_num = len(graph)
        adj = np.zeros([node_num, node_num], dtype=int)
        for i in range(node_num):
            for j in range(node_num):
                if i == j:
                    adj[i][j] = 0
                elif j + 1 in graph[i] or i + 1 in graph[j]:
                    adj[i][j] = 1
        self.adj_m = adj

    def normalize_adj(self, adj):
        # normalization
        adj = adj + np.identity(len(adj))
        rowsum = np.sum(adj, axis=1)
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        normalized_adj = np.dot(np.dot(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        self.normalized_adj = normalized_adj

    def get_grf_size(self):
        # This function is used to provide feedback on how many nodes there are
        return len(self.graph)

    def get_grf_input_size(self):
        # This function is used to provide feedback on the feature_size of each node
        _, feature_size = self.net_input.shape
        return feature_size
