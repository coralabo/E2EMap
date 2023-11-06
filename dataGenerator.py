from scipy import sparse
import numpy as np


class DataGenerator1:                      
    def __init__(self, graph):
        self.adj_dict = None
        self.graph = graph
        self.graph_ini()
    
    def graph_ini(self):
        adj = self.graph.adj_m
        refine_matrix = []
        node_num = np.size(adj, 1)
        for i in range(node_num):
            row_array = []
            node_idx = 1
            row_array.append(i + 1)
            for j in adj[i]:
                if j == 1:
                    row_array.append(node_idx)
                    node_idx += 1
                if j == 0:
                    node_idx += 1
            refine_matrix.append(row_array)
        adj_dict = {}
        # Convert Adjacency List to Dictionary of graph
        for i in refine_matrix:
            a = i[1:]
            b = []
            for n in a:
                if n != 0:
                    b.append(n)
            adj_dict[i[0]] = b
        self.adj_dict = adj_dict

    def normalize_adj(self, adj):
        # normalization
        adj = adj + np.identity(len(adj))
        rowsum = np.sum(adj, axis=1)
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        normalized_adj = np.dot(np.dot(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return normalized_adj

    # Depth First Search
    def DFS(self, s):
        adj_dict = self.adj_dict
        stack = []
        stack.append(s)
        seen = []
        seen.append(s)
        sort = []
        while stack:
            vertex = stack.pop()
            nodes = adj_dict[vertex]
            for w in nodes:
                if w not in seen:
                    stack.append(w)
                    seen.append(w)
            sort.append(vertex)
        return sort

    # breadth-first search
    def BFS(self, s):
        adj_dict = self.adj_dict
        queue = []
        queue.append(s)
        seen = []
        seen.append(s)
        sort = []
        while queue:
            vertex = queue.pop(0)
            nodes = adj_dict[vertex]
            for w in nodes:
                if w not in seen:
                    queue.append(w)
                    seen.append(w)
            sort.append(vertex)
        return sort


    def generate_original(self):
        adj_matrix_list = []
        adj_list = []
        embedding_list = []
        dict_list = []

        adj_matrix_list.append(self.normalize_adj(self.graph.adj_m))
        embedding_list.append(self.graph.feature_m)
        adj_list.append(self.graph.graph)
        dict_list.append(self.adj_dict)

        return np.array(adj_matrix_list, dtype=np.float), np.array(embedding_list), np.array(adj_list), np.array(dict_list)
    
    def sort2embedding(self, sort):
        embedding_new = self.graph.feature_m.copy()
        adj_new = self.graph.adj_m.copy()  # + np.identity(len(self.graph.adj))
        net_input = self.graph.net_input.copy()
        source_adj_list = np.delete(embedding_new, 1, axis=1)
        adj_graph = np.delete(source_adj_list, -1, axis=1)
        # Input data processing
        adj_graph_new = []
        net_input_new = []
        dict = {}
        timestep = {}
        route = {}
        # Establish time steps, routing, and mapping of old and new nodes
        t = embedding_new[:, 1]
        a = np.arange(1, len(sort) + 1, 1)
        r = embedding_new[:, -1]
        j = 0
        for i in sort:
            timestep[i] = t[i - 1]
            dict[i] = a[j]
            route[i] = r[i - 1]
            j += 1

        # generate new net_input
        for i in sort:
            net = net_input[i - 1]
            net_input_new.append(net)
        net_input_new = np.array(net_input_new)
        
        # generate new adj_list
        for i in sort:
            node = adj_graph[i - 1]
            adj_graph_new.append(node)
        adj_graph_new = np.array(adj_graph_new)
        # generate new time_step
        time_new = {}
        for i in sort:
            time_new[i] = t[i - 1]
        time_new = np.array(list(time_new.values()))
        # generate new routing
        route_new = {}
        for i in sort:
            route_new[i] = r[i - 1]
        route_new = np.array(list(route_new.values()))
        embedding_new = np.insert(adj_graph_new, 1, time_new, axis=1)
        embedding_new = np.c_[embedding_new, route_new]
        # The following section goes from the adjacency table to the adjacency matrix
        dict_new = {}
        j = 0
        for i in sort:
            dict_new[i] = j
            j += 1
        adj_row = []
        adj_last = []
        for key in dict_new.keys():
            adj_row.append(list(adj_new[key - 1]))
        adj_row = np.array(adj_row)
        for key in dict_new.keys():
            adj_last.append(list(adj_row[:, key - 1]))
        adj_last = np.array(adj_last)
        # Delete the last recomp column
        adj_graph_new = np.delete(adj_graph_new, -1, axis=1)
        return adj_last, adj_graph_new, embedding_new, dict, net_input_new

    def generate(self):
        # This function is used to generate adjacency matrices and corresponding embedding information (including node number, time step information, and adjacent nodes) from all neighboring matrices that are breadth first and depth first, respectively

        node_nums = self.graph.get_grf_size()
        feature_size = self.graph.get_grf_input_size()
        adj_matrix_list = []
        adj_list = []
        embedding_list = []
        dict_list = []
        net_input_list = []

        for starting_node in range(self.graph.get_grf_size()):
            # Perform DFS and BFS on each node as the starting point, and then store it in a file
            # Because starting_node starts from 0, so you need to do+1 once
            #"""
            # CGRA dataset changes, DFG dataset remains unchanged
            sort = self.BFS(starting_node+1)

            # DFG dataset changes, CGRA dataset remains unchanged
            #sort = [i+1 for i in range(node_nums)]

            adj_matrix, adj, embedding, dict, net_input = self.sort2embedding(sort)

            #if not any((self.normalize_adj(adj_matrix) == x).all() for x in adj_matrix_list):
            adj_matrix_list.append(self.normalize_adj(adj_matrix))
            # adj_matrix_list.append(adj_matrix)
            adj_list.append(adj)
            embedding_list.append(embedding)
            dict_list.append(dict)
            #net_input_list.append(net_input)
            net_input_list.append(sparse.csr_matrix(np.reshape(net_input, [1, node_nums*feature_size])))

            #"""
            # CGRA dataset changes, DFG dataset remains unchanged
            sort = self.DFS(starting_node+1)
            
            adj_matrix, adj, embedding, dict, net_input = self.sort2embedding(sort)
            
            #if not any((self.normalize_adj(adj_matrix) == x).all() for x in adj_matrix_list):
            adj_matrix_list.append(self.normalize_adj(adj_matrix))
            # adj_matrix_list.append(adj_matrix)
            adj_list.append(adj)
            embedding_list.append(embedding)
            dict_list.append(dict)
            # net_input_list.append(net_input)
            net_input_list.append(sparse.csr_matrix(np.reshape(net_input, [1, node_nums*feature_size])))
            #print(1/0)
            

        return np.array(adj_matrix_list, dtype=np.float), np.array(embedding_list), np.array(adj_list), np.array(dict_list), np.array(net_input_list)

# DFG dataset changes, CGRA dataset remains unchanged
class DataGenerator2:                      
    def __init__(self, graph):
        self.dfg_adj = graph.normalized_adj
        self.dfg_net_input = graph.net_input
        self.adj_dict = None
        self.graph = graph
        self.graph_ini()
    
    def graph_ini(self):
        adj = self.graph.adj_m
        refine_matrix = []
        node_num = np.size(adj, 1)
        for i in range(node_num):
            row_array = []
            node_idx = 1
            row_array.append(i + 1)
            for j in adj[i]:
                if j == 1:
                    row_array.append(node_idx)
                    node_idx += 1
                if j == 0:
                    node_idx += 1
            refine_matrix.append(row_array)
        adj_dict = {}

        for i in refine_matrix:
            a = i[1:]
            b = []
            for n in a:
                if n != 0:
                    b.append(n)
            adj_dict[i[0]] = b
        self.adj_dict = adj_dict

    def normalize_adj(self, adj):
        # normalization
        adj = adj + np.identity(len(adj))
        rowsum = np.sum(adj, axis=1)
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        normalized_adj = np.dot(np.dot(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return normalized_adj

    def DFS(self, s):
        adj_dict = self.adj_dict
        stack = []
        stack.append(s)
        seen = []
        seen.append(s)
        sort = []
        while stack:
            vertex = stack.pop() 
            nodes = adj_dict[vertex]
            for w in nodes:
                if w not in seen:
                    stack.append(w)
                    seen.append(w)
            sort.append(vertex)
        return sort


    def BFS(self, s):
        adj_dict = self.adj_dict
        queue = []
        queue.append(s)
        seen = []
        seen.append(s)
        sort = []
        while queue:
            vertex = queue.pop(0)
            nodes = adj_dict[vertex]
            for w in nodes:
                if w not in seen:
                    queue.append(w)
                    seen.append(w)
            sort.append(vertex)
        return sort


    def generate_original(self):
        adj_matrix_list = []
        dict_list = []

        adj_matrix_list.append(self.normalize_adj(self.graph.adj_m))
        dict_list.append(self.adj_dict)

        return np.array(adj_matrix_list, dtype=np.float), np.array(dict_list)
    
    def sort2embedding(self, sort):

        adj_new = self.graph.adj_m.copy()  # + np.identity(len(self.graph.adj))
        net_input = self.graph.net_input.copy()

        net_input_new = []

        for i in sort:
            net = net_input[i - 1]
            net_input_new.append(net)
        net_input_new = np.array(net_input_new)

        adj_list = []

        for i in sort:
            adj = adj_new[i - 1]
            adj_list.append(adj)
        adj_list = np.array(adj_list)

        return adj_list, net_input_new

    def generate(self):

        node_nums = self.graph.get_grf_size()
        feature_size = self.graph.get_grf_input_size()
        adj_matrix_list = []

        net_input_list = []

        for starting_node in range(self.graph.get_grf_size()):

            sort = self.BFS(starting_node+1)
            adj_matrix, net_input = self.sort2embedding(sort)
            
            #if not any((self.normalize_adj(adj_matrix) == x).all() for x in adj_matrix_list):
            adj_matrix_list.append(self.normalize_adj(adj_matrix))
            # adj_matrix_list.append(adj_matrix)
            #net_input_list.append(net_input)
            net_input_list.append(sparse.csr_matrix(np.reshape(net_input, [1, node_nums*feature_size])))
            
            sort = self.DFS(starting_node+1)
            adj_matrix, net_input = self.sort2embedding(sort)
            
            #if not any((self.normalize_adj(adj_matrix) == x).all() for x in adj_matrix_list):
            adj_matrix_list.append(self.normalize_adj(adj_matrix))

            net_input_list.append(sparse.csr_matrix(np.reshape(net_input, [1, node_nums*feature_size])))
            #print(1/0)
            

        return np.array(adj_matrix_list, dtype=np.float), np.array(net_input_list)