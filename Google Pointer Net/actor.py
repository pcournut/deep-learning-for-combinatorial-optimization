import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, GRUCell, MultiRNNCell, DropoutWrapper
from tqdm import tqdm

import numpy as np

from dataset import DataGenerator
#from tsp_with_ortools import Solver
from decoder import pointer_decoder

import matplotlib.pyplot as plt

from config import get_config, print_config


# Tensor summaries for TensorBoard visualization
def variable_summaries(name,var, with_max_min=False):
  with tf.name_scope(name):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    if with_max_min == True:
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))



class Actor(object):


    def __init__(self, config):
        self.config=config
        self.batch_size = config.batch_size # batch size
        self.max_length = config.max_length # input sequence length (number of cities)
        self.input_dimension = config.input_dimension # dimension of a city (coordinates)

        self.input_embed = config.input_embed # dimension of embedding space (actor)
        self.num_neurons = config.hidden_dim # dimension of hidden states (actor LSTM cell)
        self.hidden_dim = config.hidden_dim # same thing...
        self.initializer = tf.random_uniform_initializer(config.init_min_val, config.init_max_val) # variables initializer

        self.inference_mode = config.inference_mode

        self.global_step= tf.Variable(0, trainable=False, name="global_step") #0 # int to reuse_variable in scope
        self.lr1_start = config.lr1_start
        self.lr1_decay_rate= config.lr1_decay_rate
        self.lr1_decay_step= config.lr1_decay_step

        self.init_temperature = config.init_temperature
        self.T_decay_step = config.T_decay_step
        self.T_decay_rate = config.T_decay_rate

        self.C = config.C # tanh clipping

        self.avg_baseline = tf.Variable(15.0, trainable=False, name="moving_avg_baseline") #0 # int to reuse_variable in scope
        self.alpha = config.alpha # for avg baseline update

        # Tensor blocks holding the input sequences [Batch Size, Sequence Length, Features]
        self.input_coordinates = tf.placeholder(tf.float32, [self.batch_size, self.max_length, self.input_dimension], name="input_coordinates")

        with tf.name_scope('actor'):
            self.build_permutation()
        with tf.name_scope('environment'):
            self.build_reward()
        with tf.name_scope('optimizer'):
            self.build_optim()

        self.merged = tf.summary.merge_all()


    def build_permutation(self):

        with tf.variable_scope("encoder") as encoder:
            # Embed input sequence
            with tf.variable_scope("embedding") as embedding:
                W_embed =tf.get_variable("weights",[1,self.input_dimension,self.input_embed],initializer=self.initializer)
                embeded_input = tf.nn.conv1d(self.input_coordinates, W_embed, 1, "VALID", name="encoder_input_forward")
            # Encode input sequence
            with tf.variable_scope("cell_fw") as cell_fw:
                cell1 = LSTMCell(self.num_neurons,initializer=self.initializer)   # cell = DropoutWrapper(cell, output_keep_prob=dropout) or MultiRNNCell([cell] * num_layers)
            # Return the output activations [Batch size, Sequence Length, Num_neurons] and last hidden state as tensors.
            encoder_output, encoder_state = tf.nn.dynamic_rnn(cell1, embeded_input, dtype=tf.float32, scope=encoder) ### NOTE: encoder_output is the ref for attention ###

        with tf.variable_scope('decoder') as decoder:
            # Decoder initial state is the last relevant state from encoder
            decoder_initial_state = encoder_state ### NOTE: if state_tuple=True, self.decoder_initial_state = (c,h) ###
            # Decoder initial input is 'GO', a variable tensor
            first_input = tf.get_variable("GO",[1,self.hidden_dim],initializer=self.initializer) #2*self.hidden_dim because of bidirectional RNN
            decoder_first_input=tf.tile(first_input,[self.batch_size,1])
            # Decoder LSTM cell        
            cell2 = LSTMCell(self.num_neurons,initializer=self.initializer) #2*self.num_neurons because of bidirectional RNN
            # Ptr-net returns the permutations (self.positions), with their probability and log_softmax for backprop
            with tf.variable_scope('ptr_net') as ptr_net:
                self.T = tf.train.exponential_decay(self.init_temperature, self.global_step, self.T_decay_step,self.T_decay_rate, staircase=True, name="temperature")
                tf.summary.scalar('temperature', self.T)
                self.ptr = pointer_decoder(encoder_output, cell2, self.T,self.C, self.inference_mode, self.initializer)
                self.positions, self.log_softmax = self.ptr.loop_decode(decoder_initial_state, decoder_first_input)
            

    def build_reward(self):

        # Get Agent's trip and shift it
        self.trip, self.shifted_trip = [], []
        for cities, path in zip(tf.unstack(self.input_coordinates,axis=0), tf.unstack(self.positions,axis=0)): # Unstack % batch axis
            # Get tour for batch_b
            tour = tf.gather_nd(cities,tf.expand_dims(path,1))
            self.trip.append(tour)
            # Shift tour for batch_b
            shifted_tour = [tour[i] for i in range(1,self.max_length)]
            shifted_tour.append(tour[0])
            self.shifted_trip.append(tf.stack(shifted_tour,0))
        # Stack % batch
        self.trip = tf.stack(self.trip,0)
        self.shifted_trip = tf.stack(self.shifted_trip,0)

        # Get tour length (euclidean distance)
        inter_city_distances = tf.sqrt(tf.reduce_sum(tf.square(self.shifted_trip-self.trip),axis=2)) # sqrt(delta_x**2 + delta_y**2) this is the euclidean distance between each city
        self.distances = tf.reduce_sum(inter_city_distances,axis=1)

        # Minimum of the batch (for inference on single input)
        #self.minimal_distance = tf.reduce_min(self.distances)

        # Define reward from improvement
        self.reward = tf.cast(self.distances,tf.float32)
        variable_summaries('reward',self.reward, with_max_min = True)


    def build_optim(self):

        with tf.name_scope('baseline'):
            # Update baseline
            self.base_op = tf.assign(self.avg_baseline, self.alpha*self.avg_baseline+(1.0-self.alpha)*tf.reduce_mean(self.reward))
            tf.summary.scalar('average baseline',self.avg_baseline)


        with tf.name_scope('actor'):
            # Actor learning rate
            self.lr1 = tf.train.exponential_decay(self.lr1_start, self.global_step, self.lr1_decay_step,self.lr1_decay_rate, staircase=False, name="learning_rate1")
            # Optimizer
            self.opt1 = tf.train.AdamOptimizer(learning_rate=self.lr1,beta1=0.9,beta2=0.99, epsilon=0.0000001)
            # Discounted reward
            self.reward_baseline=tf.stop_gradient(self.reward-self.avg_baseline) # [Batch size, 1] 
            # Loss
            self.loss1=tf.reduce_mean(self.reward_baseline*self.log_softmax,0)
            tf.summary.scalar('loss1', self.loss1)
            # Minimize step
            gvs = self.opt1.compute_gradients(self.loss1)
            capped_gvs = [(tf.clip_by_norm(grad, 1.), var) for grad, var in gvs if grad is not None] # L2 clip
            self.train_step1 = self.opt1.apply_gradients(capped_gvs, global_step=self.global_step)









if __name__ == "__main__":
    # get config
    config, _ = get_config()

    # Build Model and Reward from config
    actor = Actor(config)

    print("Starting training...")
    with tf.Session() as sess: #tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        tf.global_variables_initializer().run() #tf.initialize_all_variables().run()
        print_config()

        solver = Solver(actor.max_length)
        training_set = DataGenerator(solver)

        nb_epoch=2
        for i in tqdm(range(nb_epoch)): # epoch i

            # Get feed_dict
            coord_batch = training_set.next_batch(actor.batch_size, actor.max_length, actor.input_dimension, seed=1)
            feed = {actor.input_coordinates: coord_batch}

            #permutation = sess.run(actor.positions,feed_dict=feed)
            #print('\n Permutation \n',permutation)
            distances, reward = sess.run([actor.distances, actor.reward],feed_dict=feed)
            permutation = sess.run(actor.positions,feed_dict=feed)
            lp = sess.run(actor.log_softmax,feed_dict=feed)
            loss1 = sess.run(actor.loss1,feed_dict=feed)

            #ask = sess.run(actor.ptr.masked_distribution,feed_dict=feed)
            #print(' Masked distribution (last) \n',mask)
            print(' Permutation \n',permutation)
            print(' Reward \n',reward)
            print(' LP \n',lp)
            print(' Loss1 \n',loss1)


        print('\n Trainable variables')
        for v in tf.trainable_variables():
            print(v.name)