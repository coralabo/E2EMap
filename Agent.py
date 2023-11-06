import datetime
import numpy as np
import tensorflow as tf
# import tensorflow_addons as tfa
from tensorflow.keras.optimizers import Adam
import random
import datetime
import os

from scipy import sparse
from Networks import ActorNetwork, ActorNetwork2
from environment_routing import Environment


class Agent:
    def __init__(self, action_dims, max_memory, memory_mode, reward_mode, max_GRF, max_LRF, total_adj, total_embedding, total_graph,
                 C, temperature, source_graph, total_dict, total_net_input, ii, each_time_node, source_dict, mapping_ver, layer_infeasible, top_log, dfg_net_input, dfg_adj, cgra_embedding, dfg_adj_list, dfg_net_input_list,  dfg_graph, transfer_learning=False, beta=0.2, layer_nums=7, pea_width=4, chkpt_dir="tmp/ddpg",
                 actor_lr=1e-3, hidden_dims=32, batch_size=64):
        self.baseline = None
        self.graph = source_graph
        self.dfg_graph = dfg_graph
        self.batch_size = batch_size
        self.C = C
        self.ii = ii
        self.temperature = temperature
        self.source_dict = source_dict
        self.dfg_net_input = dfg_net_input
        self.dfg_adj = dfg_adj
        self.cgra_embedding = cgra_embedding
        self.dfg_adj_list = dfg_adj_list
        self.dfg_net_input_list = dfg_net_input_list
        self.load = False
        self.environment = Environment(C=C, temperature=temperature, total_adj=total_adj, max_LRF=max_LRF, total_net_input=total_net_input,
                                       total_embedding=total_embedding, total_graph=total_graph, action_dims=action_dims,
                                       total_dict=total_dict, pea_width=pea_width, beta=beta, ii=ii, max_GRF=max_GRF,
                                       memory_mode=memory_mode, max_memory=max_memory, reward_mode=reward_mode,each_time_node=each_time_node, source_dict=source_dict, mapping_ver=mapping_ver, layer_infeasible=layer_infeasible, top_log=top_log)


        # Dual GNN
        self.actor = ActorNetwork2(gcn_dims=hidden_dims, action_dims=action_dims, name="actor",  chkpt_dir=chkpt_dir, layer_nums=layer_nums)
        # Single GNN
        #self.actor = ActorNetwork(gcn_dims=hidden_dims, action_dims=action_dims,name="actor", chkpt_dir=chkpt_dir, layer_nums=layer_nums)

        self.actor.compile(optimizer=Adam(learning_rate=actor_lr))
                                                                                                    
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    def generate_batch_dfg(self, batch_size):
        # This function is used to generate a batch_size dataset
        total_nums, _, _ = self.dfg_adj_list.shape
        # Using the method of choice, select batch_size data from them and repeat the selection to ensure the diversity of input data
        batch_index = np.random.choice(total_nums, batch_size, True)

        #self.environment.batch_index = batch_index
        return self.dfg_adj_list[batch_index], self.dfg_net_input_list[batch_index]

    def save_models(self):
        print(".... saving models ....")
        #tf.saved_model.save(self.actor, self.actor.checkpoint_dir)
        tf.saved_model.save(self.actor, "tmp/ddpg/actor_ddpg.h5")
        #self.actor.save("tmp/ddpg/actor_ddpg2", save_format="tf")

    def load_part_models(self):
        if self.load:
            return
        print(".... loading part models ....")
        #"""
        # This function is used to load the model parameters
        #pretrain_model_weights = tf.saved_model.load(self.actor.checkpoint_dir)
        pretrain_model_weights = tf.saved_model.load("tmp/ddpg/actor_ddpg.h5")
        params_dict = {}
        for v in pretrain_model_weights.trainable_variables:
            params_dict[v.name] = v.read_value()
        #print(params_dict[0])
        if(len(self.actor.variables)>0):
            self.load=True

        for idx, layer in enumerate(self.actor.variables[:]):
            if idx > 11:
                continue
            layer.assign(pretrain_model_weights.variables[idx])


    def load_models(self):
        print(".... loading models ....")
        #self.actor.load_weights(self.actor.checkpoint_file)
        self.actor.load_weights("tmp/ddpg/actor_ddpg.h5")

    def learn(self, episode, load_model):
        train = True
        #self.load_part_models()
        with tf.GradientTape() as tape:

            batch_adj, batch_dict, batch_embedding, batch_net_input = self.environment.generate_batch(batch_size=self.batch_size)
            # CGRA dataset changes, DFG dataset remains unchanged
            batch_dfg_adj = np.array([self.dfg_adj for _ in range(self.batch_size)])
            batch_dfg_net_input = np.array([self.dfg_net_input for _ in range(self.batch_size)])
            
            # DFG dataset changes, CGRA dataset remains unchanged
            #batch_dfg_adj, batch_dfg_net_input = self.generate_batch_dfg(batch_size=self.batch_size)

            sparse_batch_net_input = sparse.vstack(batch_net_input)
            indices = list(zip(*sparse_batch_net_input.nonzero()))
            sparse_batch_net_input = tf.SparseTensor(indices=indices, values=np.float32(sparse_batch_net_input.data),
                                                    dense_shape=sparse_batch_net_input.get_shape())
            sparse_batch_net_input = tf.sparse.reshape(sparse_batch_net_input, [-1, self.graph.get_grf_size(), self.graph.get_grf_input_size()])
            states = tf.sparse.to_dense(sparse_batch_net_input)

            # states = tf.convert_to_tensor(batch_net_input, dtype=tf.float32)
            #batch_cgra_embeddings = np.array([[self.cgra_embedding[batch_embedding[x][i][0]-1] for i in range(len(self.cgra_embedding))] for x in range(self.batch_size)])
            
            #Dual GNN
            batch_adj = tf.convert_to_tensor(batch_adj, dtype=tf.float32)
            states = tf.convert_to_tensor(states, dtype=tf.float32)
            batch_dfg_adj = tf.convert_to_tensor(batch_dfg_adj, dtype=tf.float32)
            batch_dfg_net_input = tf.convert_to_tensor(batch_dfg_net_input, dtype=tf.float32)
            new_policy_logits = self.actor(batch_adj, states, batch_dfg_adj, batch_dfg_net_input)

            #Single GNN
            #new_policy_logits = self.actor(batch_adj, states)

            #new_policy_logits = self.actor(batch_adj, states, batch_dfg_adj, batch_dfg_net_input,batch_cgra_embeddings)
            new_action, mask = self.environment.action(actor_logits=new_policy_logits, train=True)

            penaltys,new_rewards,lrf_uses,grf_uses,action_dicts = self.environment.rewards(new_action, batch_dict, batch_embedding, self.source_dict)

            total_rewards = new_rewards 
            # init baseline:
            if self.baseline is None:
                self.baseline = np.mean(total_rewards)
            else:
                self.baseline = self.baseline * 0.99 + np.mean(total_rewards) * 0.01
            # Because the variable types required by the two Operations below are different, there are two types of ACTION here
            new_actions_i = tf.convert_to_tensor(new_action, dtype=tf.int32)

            # refine_policy_logits = new_policy_logits/self.temperature
            refine_policy_logits = self.C * tf.tanh(new_policy_logits/self.temperature)

            # print("Output after using tanh", refine_policy_logits)
            refine_policy_logits = tf.where(mask,
                                            tf.ones_like(new_policy_logits) * (-np.inf),
                                            refine_policy_logits)

            # Adding the probability of log
            neg_log_prob = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=refine_policy_logits,labels=new_actions_i),axis=1)
            advantage = new_rewards - self.baseline
            
            actor_loss = tf.reduce_mean(advantage * neg_log_prob)


        actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))

        if (episode+1) % 500 == 0:
            with self.train_summary_writer.as_default():
                tf.summary.scalar("actor_loss", actor_loss, step=episode)
                tf.summary.scalar("baseline", self.baseline, step=episode)
                tf.summary.scalar("mean_penaltys", np.mean(penaltys), step=episode)
                tf.summary.scalar("best_reward", np.max(new_rewards), step=episode)
                tf.summary.scalar("mean_reward", np.mean(new_rewards), step=episode)
                # tf.summary.scalar("neg_log_prob", tf.reduce_mean(neg_log_prob), step=episode)
            print("Episode: ", (episode+1), end=" ")
            print("II: ", self.ii)
            #print("current action_dicts: \n", repr(action_dicts[0:5]))
            print("current action_dicts: \n", repr(action_dicts[0]))
            print(np.mean(total_rewards))
            print("----------------------------------")
        if np.max(total_rewards) == 0:
            index = np.argmax(total_rewards)
            correct = new_action[index]
            print("Success Mapping!")
            #print("Parameters:")
            #print(self.actor.summary())
            print("total episode: ", episode)
            print("Result: \n", correct)
            print("II: ", self.ii)
            #def get_map_dict(self, action, dict_, embedding, source_dict_all):
            #action_dict,map_dict = self.environment.get_map_dict(correct, batch_dict[index], batch_embedding[index], self.source_dict)
            #self.environment.show_placer(correct, batch_embedding[index])
            self.environment.show_placer(action_dicts[index],lrf_uses[index],grf_uses[index])
            #self.save_models()
            return True
        else:
            return False
