import os
import datetime
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense,Conv1D,MaxPooling1D
import math
import pygmtools as pygm

pygm.BACKEND = 'tensorflow'

class ActorNetwork(keras.Model):
    def __init__(self, gcn_dims, action_dims, layer_nums, name="Actor", chkpt_dir='tmp'):
        super(ActorNetwork, self).__init__()
        self.gcn_dims = gcn_dims
        self.action_dims = action_dims
        print("ActorNetwork")
        self.model_name = name
        current_time = datetime.datetime.now().strftime("%d-%H%M")
        self.checkpoint_dir = os.path.join(chkpt_dir, current_time+"_"+str(action_dims))
        self.checkpoint_file = os.path.join(self.checkpoint_dir,
                                            self.model_name+'_ddpg.h5')

        # Determine how many layers are rolled up based on the number of layers read from outside
        self.layer_list = []
        for _ in range(layer_nums):
            self.layer_list.append(Dense(self.gcn_dims, activation='relu'))
        # The last layer is the output layer, so there is no need to add activation
        self.fc1 = Dense(self.gcn_dims, activation='relu')
        self.fc2 = Dense(self.action_dims, activation=None)

    def call(self, adj, state):

        #if train:
        input_drop_rate = 0.1
        drop_rate = 0.5
        #else:
        #input_drop_rate = 0.
        #drop_rate = 0.

        adj = tf.convert_to_tensor(adj, dtype=tf.float32)
        # state = tf.nn.dropout(state, rate=input_drop_rate)
        embedding = self.fc1(state)
        
        for layer_index, layer in enumerate(self.layer_list):
            
            embedding = layer(tf.matmul(adj, embedding)) + embedding
            # print("============================================")

        logits = self.fc2(embedding)

        return logits


#"""
class ActorNetwork2(keras.Model):
    def __init__(self, gcn_dims, action_dims, layer_nums, name="Actor", chkpt_dir='tmp'):
        super(ActorNetwork2, self).__init__()
        print("gcn_dims:")
        print(gcn_dims)
        print("ActorNetwork2")
        self.gcn_dims = gcn_dims
        self.action_dims = action_dims

        self.model_name = name
        current_time = datetime.datetime.now().strftime("%d-%H%M")
        self.checkpoint_dir = os.path.join(chkpt_dir, current_time+"_"+str(action_dims))
        self.checkpoint_file = os.path.join(self.checkpoint_dir,
                                            self.model_name+'_ddpg.h5')

        # Determine how many layers are rolled up based on the number of layers read from outside
        self.layer_list = []
        for _ in range(layer_nums):
            self.layer_list.append(Dense(self.gcn_dims, activation='relu'))
            self.layer_list.append(Dense(self.gcn_dims, activation='relu'))
        # The last layer is the output layer, so there is no need to add activation
        self.fc1 = Dense(self.gcn_dims, activation='relu')
        self.fc2 = Dense(self.action_dims, activation=None)
        self.fc3 = Dense(self.gcn_dims, activation='relu')
        self.fc4 = Dense(self.gcn_dims, activation='relu')
        self.fc5 = Dense(self.gcn_dims, activation='relu')
 
        self.conv1d = Conv1D(1, 3, strides=1 ,activation='relu',padding="same")
        self.conv1d2 = Conv1D(1, 3, strides=1 ,activation='relu',padding="same")
        self.max_pool_1d = MaxPooling1D(pool_size=2,strides=2)
        self.max_pool_1d2 = MaxPooling1D(pool_size=2,strides=2)

    def call(self, adj, state, dfg_adj, dfg_net_input):

        #if train:
        input_drop_rate = 0.1
        drop_rate = 0.5
        #else:
        #input_drop_rate = 0.
        #drop_rate = 0.

        # state = tf.nn.dropout(state, rate=input_drop_rate)

        embedding = self.fc1(state)

        embedding2 = self.fc3(dfg_net_input)

        for layer_index, layer in enumerate(self.layer_list):

            if layer_index%2 == 0:
                embedding = layer(tf.matmul(adj, embedding)) + embedding
            else:
                embedding2 = layer(tf.matmul(dfg_adj, embedding2)) + embedding2
            # print("============================================")

        dfg_net_input2 = tf.transpose(embedding2,[0,2,1])
        global_embedding2 = self.conv1d(dfg_net_input2)      
        global_embedding2 = self.max_pool_1d(global_embedding2)
        global_embedding2 = tf.transpose(global_embedding2,[0,2,1])

        net_input = tf.transpose(embedding,[0,2,1])
        global_embedding = self.conv1d2(net_input) 
        global_embedding = self.max_pool_1d2(global_embedding)
        global_embedding = tf.transpose(global_embedding,[0,2,1])
        
        global_embedding = tf.concat([global_embedding,global_embedding2],2)
        embedding = embedding+global_embedding
        
        embedding = self.fc4(embedding)
        embedding = self.fc5(embedding)
        logits = self.fc2(embedding)
        return logits
