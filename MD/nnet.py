import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, GRUCell, MultiRNNCell, DropoutWrapper
from tqdm import tqdm

from decoder import pointer_decoder
import dataset

import matplotlib.pyplot as plt


# TODO

"""
Michel:
_____________________________________________________________________

- Plot tour with networkx

- Try Google's parameters

- Normalize input (scale distances) + reorder sequence

- Add bidirectional encoder, Dropout, Layers...

- Improve baseline estimate (CNN vs. FFN)
_____________________________________________________________________

- Variable seq length (padding) ****

- TSP with soft time windows (add exp loss) ***
  (+ use Graph Theory : Coreness, Connectivity (or Katz_beta), Closeness, Betweeness, Cluster... & Topological descriptors - Reeb filtration ?)
_____________________________________________________________________

- Gumbel Softmax ?

- Monte Carlo Tree Search ?
_____________________________________________________________________

Pierre
- GAN (Discriminator CNN, Wasserstein...): pretrain ****
- Supervised setting: Cplex, Concorde... ***
- Back prop ***
_____________________________________________________________________

- Save & load model

- Use varscope, graph...

- import argparse (config)
- Summary writer (log)...

- Parallelize (GPU), C++...
- Nice plots, interface...
"""



class EncDecModel(object):



    def __init__(self,args):
        self.batch_size=args['batch_size'] # batch size
        self.max_length = args['max_length'] # input sequence length (number of cities)
        self.input_dimension = args['input_dimension'] # dimension of a city (coordinates)
        self.K = self.input_dimension+1 # for KNN
        self.input_new_dimension  = self.input_dimension+2*self.K+1 # x,y + kNN index + kNN dist + indegree_ft (>0 for a hub)
        self.input_embed=args['input_embed'] # dimension of embedding space (actor)
        self.input_embed_c=args['input_embed'] # dimension of embedding space (critic)
        self.num_neurons = args['num_neurons'] # dimension of hidden states (actor LSTM cell)
        self.num_neurons_c = args['num_neurons'] # dimension of hidden states (critic LSTM cell)
        self.hidden_dim= args['num_neurons'] # same thing...
        self.initializer = tf.random_uniform_initializer(-args['init_range'], args['init_range']) # variables initializer
        self.step=0 # int to reuse_variable in scope
        self.init_bias_c = args['init_bias_c'] # initial bias for critic
        self.temperature_decay = args['temperature_decay'] # temperature decay rate

        self.build_actor()
        self.build_critic()
        self.build_reward()
        self.build_optim()



    def build_actor(self):

        # Tensor blocks holding the input sequences [Batch Size, Sequence Length, Features]
        self.input_coordinates = tf.placeholder(tf.float32, [self.batch_size, self.max_length, self.input_dimension], name="Input")
        self.input_description = tf.placeholder(tf.float32, [self.batch_size, self.max_length, self.input_new_dimension], name="Input")

        # Embed input sequence
        W_embed = tf.Variable(tf.truncated_normal([1,self.input_new_dimension,self.input_embed]), name="W_embed")
        with tf.variable_scope("Embed"):
            if self.step>0:
                tf.get_variable_scope().reuse_variables()
            embeded_input = tf.nn.conv1d(self.input_description, W_embed, 1, "VALID", name="EncoderInput")

        # ENCODER LSTM cell
        cell1 = LSTMCell(self.num_neurons,initializer=self.initializer)   # cell = DropoutWrapper(cell, output_keep_prob=dropout) or MultiRNNCell([cell] * num_layers)
        # RNN-ENCODER returns the output activations [Batch size, Sequence Length, Num_neurons] and last hidden state as tensors.
        encoder_output, encoder_state = tf.nn.dynamic_rnn(cell1, embeded_input, dtype=tf.float32) ### NOTE: encoder_output is the ref for attention ###

        # DECODER initial state is the last relevant state from encoder
        decoder_initial_state = encoder_state ### NOTE: if state_tuple=True, self.decoder_initial_state = (c,h) ###
        # DECODER initial input is 'GO', a variable tensor
        decoder_first_input = tf.Variable(tf.truncated_normal([self.batch_size,self.hidden_dim]), name="GO")
        # DECODER LSTM cell        
        cell2 = LSTMCell(self.num_neurons,initializer=self.initializer)

        # POINTER-DECODER returns the output activations, hidden states, hard attention and decoder inputs as tensors.
        self.ptr = pointer_decoder(encoder_output, cell2)
        self.positions, self.proba, self.log_softmax = self.ptr.loop_decode(decoder_initial_state, decoder_first_input)



    def build_critic(self):

        # Embed input sequence (for critic)
        W_embed_c = tf.Variable(tf.truncated_normal([1,self.input_new_dimension,self.input_embed_c]), name="critic_W_embed")
        with tf.variable_scope("Critic"):
            if self.step>0:
                tf.get_variable_scope().reuse_variables()
            embeded_input_c = tf.nn.conv1d(self.input_description, W_embed_c, 1, "VALID", name="Critic_EncoderInput")

            # ENCODER LSTM cell
            cell_c = LSTMCell(self.num_neurons_c,initializer=self.initializer)   # cell = DropoutWrapper(cell, output_keep_prob=dropout) or MultiRNNCell([cell] * num_layers)

            # RNN-ENCODER returns the output activations [Batch size, Sequence Length, Num_neurons] and last hidden state as tensors.
            encoder_output_c, encoder_state_c = tf.nn.dynamic_rnn(cell_c, embeded_input_c, dtype=tf.float32)
            encoder_output_c = tf.transpose(encoder_output_c, [1, 0, 2]) # transpose time axis first [time steps x Batch size x num_neurons]
            last_c = tf.gather(encoder_output_c, int(encoder_output_c.get_shape()[0]) - 1) # select last frame [Batch size x num_neurons]

        ### DO A CONVOLUTION HERE INSTEAD OF A FFN !!! ###
        weight_c = tf.Variable(tf.truncated_normal([self.num_neurons_c, 1], stddev=0.1))
        bias_c = tf.Variable(tf.constant(self.init_bias_c, shape=[1]))
        self.prediction_c = tf.matmul(last_c, weight_c) + bias_c



    def build_reward(self):

        # From input sequence and hard attention, get coordinates of the agent's trip
        tours=[]
        shifted_tours=[]
        for k in range(self.batch_size):
            strip=tf.gather_nd(self.input_coordinates,[k])
            tour=tf.gather_nd(strip,tf.expand_dims(self.positions[k],1))
            tours.append(tour)
            # Shift tour to calculate tour length
            shifted_tour=[tour[k] for k in range(1,self.max_length)]
            shifted_tour.append(tour[0])
            shifted_tours.append(tf.stack(shifted_tour,0))
        self.trip=tf.stack(tours,0)
        self.shifted_trip=tf.stack(shifted_tours,0)

        # Get delta_x**2 and delta_y**2 for shifting from a city to another
        sqdeltax=tf.transpose(tf.square(tf.transpose((self.shifted_trip-self.trip),[2,1,0]))[0],[1,0]) # [batch,length,(x,y)] to (x)[length,batch] back to [batch,length]
        sqdeltay=tf.transpose(tf.square(tf.transpose((self.shifted_trip-self.trip),[2,1,0]))[1],[1,0])
        # Get distances separating cities at each step
        euclidean_distances=tf.sqrt(sqdeltax+sqdeltay)
        # Reduce to obtain tour length
        self.distances=tf.expand_dims(tf.reduce_sum(euclidean_distances,axis=1),1)

        # Define reward from objective and penalty
        self.reward = -tf.cast(self.distances,tf.float32)



    def build_optim(self):

        # ACTOR Optimizer
        self.opt1 = tf.train.AdamOptimizer(learning_rate=0.01,beta1=0.9,beta2=0.9,epsilon=0.1) 
        # Discounted reward
        self.reward_baseline=tf.stop_gradient(self.reward-self.prediction_c) # [Batch size, 1]
        # Loss
        self.loss1=tf.reduce_sum(tf.multiply(self.log_softmax,self.reward_baseline),0)/self.batch_size
        # Minimize step
        self.train_step1 = self.opt1.minimize(self.loss1)

        # Critic Optimizer
        self.opt2 = tf.train.AdamOptimizer(learning_rate=0.01,beta1=0.9,beta2=0.9,epsilon=0.1)
        # Loss
        self.loss2=tf.losses.mean_squared_error(self.reward,self.prediction_c)
        # Minimize step
        self.train_step2 = self.opt2.minimize(self.loss2)
        


    def run_episode(self,sess):

        # Get feed_dict
        training_set = dataset.DataGenerator()
        coord_batch, dist_batch, input_batch = training_set.next_batch(self.batch_size, self.max_length, self.input_dimension)
        feed = {self.input_coordinates: coord_batch, self.input_description: input_batch}

        # Actor Forward pass
        seq_input, permutation, seq_proba = sess.run([self.input_coordinates,self.positions,self.proba],feed_dict=feed)

        # Critic Forward pass
        b_s = sess.run(self.prediction_c,feed_dict=feed)

        # Environment response
        trip, circuit_length, reward = sess.run([self.trip,self.distances,self.reward], feed_dict=feed)

        # Train step
        if self.step==0:
            loss1, train_step1 = sess.run([self.loss1,self.train_step1],feed_dict=feed)
        else:
            loss1, train_step1, loss2, train_step2= sess.run([self.loss1,self.train_step1,self.loss2,self.train_step2],feed_dict=feed)

        self.step+=1

        if self.step%100==0:
            self.ptr.temperature*=self.temperature_decay

        return seq_input, permutation, seq_proba, b_s, trip, circuit_length, reward








def train():

    # Config
    args={}
    args['batch_size']=32
    args['max_length']=5
    args['input_dimension']=2
    args['input_embed']=16

    args['init_bias_c']=-args['max_length']/2

    args['num_neurons']=256
    args['init_range']=1

    args['temperature_decay']=1


    # Build Model and Reward
    print("Initializing the Model...")
    model = EncDecModel(args)

    print("Starting training...")
    with tf.Session() as sess:
        tf.global_variables_initializer().run() #tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())

        print('\n')
        print('Config:')
        print('* Batch size:',model.batch_size)
        print('* Sequence length:',model.max_length)
        print('* City coordinates:',model.input_dimension)
        print('* City dimension:',model.input_new_dimension)
        print('* Input embedding:',model.input_embed)
        print('* Num neurons (Actor & critic):',model.num_neurons)
        print('\n')

        avg_ac_deviation = []
        avg_seq_proba = []

        for i in tqdm(range(100)): # epoch i

            seq_input, permutation, seq_proba, b_s, trip, circuit_length, reward = model.run_episode(sess)

            # Store Actor-Critic deviation & seq proba
            avg_ac_deviation.append(sess.run(tf.reduce_mean(100*(reward-b_s)/circuit_length)))
            avg_seq_proba.append(sess.run(tf.reduce_mean(seq_proba)))


            if i % 10 == 0:
                #print('\n Input: \n', seq_input)
                #print('\n Permutation: \n', permutation)
                #print('\n Seq proba: \n', seq_proba)
                #print('\n Critic baseline: \n', b_s)
                #print('\n Trip : \n', trip)
                #print('\n Circuit length: \n',circuit_length)
                #print('\n Reward : \n', reward)

                #print('  Average seq proba :',sess.run(tf.reduce_mean(seq_proba,0)))
                print('  Average seq proba :',sess.run(tf.reduce_mean(seq_proba)))
                print(' Average circuit length :',sess.run(tf.reduce_mean(circuit_length)))
                print(' Average baseline :', sess.run(-tf.reduce_mean(b_s)))
                print(' Average deviation:', sess.run(tf.reduce_mean(100*(reward-b_s)/circuit_length)))
                print('\n')

            if i % 1000 == 0 and not(i == 0):
                saver.save(sess,"save/" +str(i) +".ckpt")

        plt.figure(1)
        plt.subplot(211)
        plt.plot(avg_ac_deviation)
        plt.ylabel('Critic average deviation (%)')
        plt.xlabel('Epoch')

        plt.subplot(212)
        plt.plot(avg_seq_proba)
        plt.ylabel('Actor average seq proba')
        plt.xlabel('Epoch')
        plt.show()


        print('\n Trainable variables')
        for v in tf.trainable_variables():
            print(v.name)

        print("Training is COMPLETE!")
        saver.save(sess,"save/model.ckpt")




if __name__ == "__main__":
    train()