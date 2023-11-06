import os
import time
import sys
import math
import numpy as np
import tensorflow as tf

from utils import loadData, save_asap_alap, get_layer_infeasible, topLogical, recomp, insertMemNode
from Agent import Agent
from environment_routing import Environment
from graph_embedding import Graph_cgra,Graph_dfg
from config import get_config
from dataGenerator import DataGenerator1,DataGenerator2
#from save_input import Data
"""
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
np.set_printoptions(threshold=sys.maxsize)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
      tf.config.experimental.set_virtual_device_configuration(
              gpu,
              [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*0.5)])
"""
#"""
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
#"""
def main():
    # Define the parameters through the input of the command line, which is also convenient to modify the parameters
    config = get_config()
    # the path of dfg
    source_file_path = config.src_file_path

    #get parameters
    batch_size = int(config.batch_size)
    max_iteration = int(config.max_iteration)
    head_nums = int(config.head_nums)
    ckpt_dir = config.ckpt_dir
    actor_lr = config.actor_lr
    load_model = config.load_model
    pea_width = config.pea_width
    temperature = config.temperature
    beta = config.beta
    layer_nums = config.layer_nums
    C = config.c
    max_LRF = config.max_LRF
    max_GRF = config.max_GRF
    max_memory = config.max_memory
    memory_mode = config.memory_mode
    #memory_mode = True
    reward_mode = config.reward_mode
    min_ii = config.mii
    hidden_dims = int(config.gcn_dims)

    # Create the root cause of the storage weight
    ckpt_dir = os.path.join(os.getcwd(), ckpt_dir)
    if not os.path.exists(path=ckpt_dir):
        os.mkdir(ckpt_dir)
    

    start_time = time.time()
    raw_data = loadData(source_file_path)
    print("输入DDG：")
    print(raw_data)

#Pre -processing       
    raw_data = insertMemNode(raw_data)
    print("insert_DDG：")
    print(raw_data)
    if reward_mode <= 2:
        raw_data,data_re = recomp(raw_data)
        print("recomp_DDG：")
        print(raw_data)

    ddg_range, each_time_node1, each_time_node2 = save_asap_alap(raw_data,pea_width,min_ii)
    print("Node activity scope：")
    print(each_time_node2)

    action_dims2 = len(ddg_range)+1
    layer_infeasible = get_layer_infeasible(each_time_node2,action_dims2,min_ii)

    source_dict = raw_data
    top_log = topLogical(source_dict[:,:-4])

    #dfg                                    
    #source_graph2 ,adj_m2 = Graph2(pea_width=pea_width, ii=min_ii)
    dfg_graph = Graph_dfg(raw_data, pea_width=pea_width, ii=min_ii)

    #print("dfg_graph:")
    #print(dfg_graph.graph)
    dfg_adj = dfg_graph.normalized_adj
    dfg_net_input = dfg_graph.net_input

    #cgra
    source_graph2 = Graph_cgra(pea_width=pea_width, ii=min_ii, dfg_data=raw_data, reward_mode=reward_mode)
    mapping_ver = Graph_cgra(pea_width=pea_width, ii=max(raw_data[:,-3])+1, dfg_data=raw_data, reward_mode=reward_mode).feature_m
    #print("mapping_ver:")
    #print(mapping_ver)

    cgra_adj = source_graph2.normalized_adj
    cgra_embedding = source_graph2.net_input
    for _ in range(layer_nums):
        cgra_embedding = tf.matmul(cgra_adj,cgra_embedding)+cgra_embedding

    #print("cgra_adj:")
    #print(source_graph2.adj_m)

    # get Dataset  
    generator2 = DataGenerator1(graph=source_graph2)
    adj_matrix_list, embedding_list, adj_list, dict_list, net_input_list = generator2.generate()

    generator3 = DataGenerator2(graph=dfg_graph)
    dfg_adj_list,dfg_net_input_list = generator3.generate()

    # init environment and Agent       
    agent = Agent(total_adj=adj_matrix_list, total_embedding=embedding_list, total_graph=adj_list, layer_nums=layer_nums,
                      pea_width=pea_width, actor_lr=actor_lr, batch_size=batch_size, hidden_dims=hidden_dims,                                    
                      beta=beta, ii=min_ii, total_dict=dict_list, temperature=temperature, C=C, max_LRF=max_LRF,
                      max_GRF=max_GRF, action_dims=action_dims2, memory_mode=memory_mode, max_memory=max_memory, reward_mode=reward_mode,
                      transfer_learning=False, total_net_input=net_input_list, source_graph=source_graph2,each_time_node=each_time_node2, source_dict=source_dict, mapping_ver=mapping_ver, layer_infeasible=layer_infeasible, top_log=top_log, dfg_net_input=dfg_net_input, dfg_adj=dfg_adj, cgra_embedding=cgra_embedding, dfg_adj_list=dfg_adj_list, dfg_net_input_list=dfg_net_input_list, dfg_graph=dfg_graph)
                         

  
    for episode in range(max_iteration):
        train = True
        load_model = False

        result_flag = agent.learn(episode=episode, load_model=load_model)

        if result_flag:
            end_time = time.time()
            print("total time:",(end_time-start_time))
            break


if __name__ == '__main__':
    main()
