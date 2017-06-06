import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tensorflow.contrib.rnn import LSTMCell


from dataset import DataGenerator
from solver import Solver
from config import get_config, print_config



class Critic(object):

    def __init__(self,config):
        self.batch_size = config.batch_size # batch size
        self.max_length = config.max_length # input sequence length (number of cities)
        self.n_components = config.n_components

        self.input_embed_c = config.input_embed_c # dimension of embedding space (critic)

        self.num_neurons_c = config.hidden_dim
        self.filter_sizes = config.filter_sizes
        self.num_filters = config.num_filters
        self.init_bias_c = config.init_bias_c # initial bias for critic
        self.initializer = tf.random_uniform_initializer(config.init_min_val, config.init_max_val)
        self.global_step= tf.Variable(0, trainable=False, name="global_step")

        self.lr_start = config.lr2_start
        self.lr_decay_step = config.lr2_decay_step
        self.lr_decay_rate = config.lr2_decay_rate


        self.input_description = tf.placeholder(tf.float32, [self.batch_size, self.max_length, self.n_components], name="input_description")
        self.initial_tour_length = tf.placeholder(tf.float32, [self.batch_size, 1], name="initial_tour_length")
        self.target = tf.placeholder(tf.float32, [self.batch_size, 1], name="target")

        with tf.name_scope('critic'):
            self.build_prediction('cnn')
        with tf.name_scope('optimizer'):
            self.build_optim()


    def build_prediction(self,nn_type='cnn'):
        
        W_embed_c = tf.Variable(tf.truncated_normal([1,self.n_components,self.input_embed_c],stddev=1.0), name="critic_W_embed")
        with tf.variable_scope("Critic"):
            self.embedded_input = tf.nn.conv1d(self.input_description, W_embed_c, 1, "VALID", name="Critic_EncoderInput")


        if nn_type == 'cnn':
            
            self.embedded_input_expanded = tf.reshape(self.embedded_input, [-1, self.max_length, 1, self.input_embed_c]) # [None, max_length, 1, embedded_dim]
            #self.embedded_input_expanded = tf.reshape(self.embedded_input, [-1, self.max_length, self.input_embed_c, 1])

            # (3 x 1) convolution layer, from depth input_embed_c to depth 32
            n_filters1 = 32
            a1=(6/(3*1*self.input_embed_c+n_filters1))**0.5
            W_conv1 = tf.Variable(tf.random_uniform([3,1,self.input_embed_c,n_filters1],-a1,a1))  # TODO
            b_conv1 = tf.Variable(tf.zeros([n_filters1])) # TODO 5*5*32 ou 28*28*32
            # Perform the convolution and then apply a ReLU
            self.h_conv1 = tf.nn.relu(tf.add(tf.nn.conv2d(self.embedded_input_expanded, W_conv1, strides=[1, 1, 1, 1], padding='SAME'),b_conv1))  # TODO
            self.h_conv1 = tf.reshape(self.h_conv1,[-1, self.max_length, n_filters1]) #[batch_size, max_length, n_filters]

            # Perform a 2x2 max pool, see tf.nn.max_pool
            # h_pool1 = max_pool_2x2(h_conv1)  # TODO

            a2=(2/(self.max_length*n_filters1))**0.5
            self.h_conv1 = tf.reshape(self.h_conv1,[-1, self.max_length*n_filters1])
            W_fc2 = tf.Variable(tf.truncated_normal([self.max_length*n_filters1,1024], stddev=a2))  # TODO # Weights of the fully connected layer
            b_fc2 = tf.Variable(tf.zeros(shape=[1024])) # TODO # Biases of the fully connected layer
            self.hidden_layer1 = tf.sigmoid(tf.matmul(self.h_conv1, W_fc2) + b_fc2) # [batch_size * 1024]


            a3=(2/(1024))**0.5
            W_fc3 = tf.Variable(tf.truncated_normal([1024,1], stddev=a3))  # TODO # Weights of the fully connected layer
            b_fc3 = tf.Variable(tf.constant(self.init_bias_c, shape=[1])) # TODO # Biases of the fully connected layer
            #self.predicted_tour_length = tf.matmul(self.hidden_layer1, W_fc3) + b_fc3 # [batch_size * 1]
            self.predicted_reward = 2*tf.sigmoid(tf.matmul(self.hidden_layer1, W_fc3) + b_fc3) - 1

            """
            a4=(2/(32))**0.5
            W_fc4 = tf.Variable(tf.truncated_normal([32,32], stddev=a4))  # TODO # Weights of the fully connected layer
            b_fc4 = tf.Variable(tf.zeros(shape=[1])) # TODO # Biases of the fully connected layer
            #self.predicted_tour_length = tf.matmul(self.hidden_layer1, W_fc3) + b_fc3 # [batch_size * 1]
            self.hidden_layer3 = tf.sigmoid(tf.matmul(self.hidden_layer2, W_fc4) + b_fc4)
            

            a5=(2/(32))**0.5
            W_fc5 = tf.Variable(tf.truncated_normal([32,1], stddev=a5))  # TODO # Weights of the fully connected layer
            b_fc5 = tf.Variable(tf.constant(self.init_bias_c, shape=[1])) # TODO # Biases of the fully connected layer
            #self.predicted_tour_length = tf.matmul(self.hidden_layer1, W_fc3) + b_fc3 # [batch_size * 1]
            self.predicted_improvement = 2*tf.sigmoid(tf.matmul(self.hidden_layer2, W_fc5) + b_fc5) - 1
            """


        if nn_type == 'cnn_2':
            
            # Expand dim to meet conv2d shape requirements
            self.embedded_input_expanded = tf.reshape(self.embedded_input, [-1, self.max_length, self.input_embed_c, 1])

            # Create a layer for each convolution of different filter sizes and then merge
            pooled_outputs = []
            for i, filter_size in enumerate(self.filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):

                    # Convolution Layer
                    filter_shape = [filter_size, self.input_embed_c, 1, self.num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                    conv = tf.nn.conv2d(
                        self.embedded_input_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")

                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                    # Max-pooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, self.max_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)
 
            # Combine all the pooled features
            num_filters_total = self.num_filters * len(self.filter_sizes)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

            # Add dropout
            #with tf.name_scope("dropout"):
                #self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
            #self.h_drop = self.h_pool_flat

            # Predictions
            with tf.name_scope("output"):
                W = tf.Variable(tf.truncated_normal([num_filters_total, 1], stddev=0.1), name="W")
                b = tf.Variable(tf.constant(self.init_bias_c, shape=[1]), name="b")
                self.predicted_tour_length = tf.matmul(self.h_pool_flat, W) + b
                
                #self.predicted_tour_length = tf.argmax(self.scores, 1, name="predictions")
                #self.predicted_tour_length = tf.cast(self.predicted_tour_length, tf.float32)

            #self.predicted_tour_length=tf.stack([self.predicted_tour_length,1-self.predicted_tour_length],1)

        
        if nn_type == 'rnn':  

            # ENCODER LSTM cell
            #cell_c = LSTMCell(self.num_neurons_c,initializer=self.initializer)   # cell = DropoutWrapper(cell, output_keep_prob=dropout) or MultiRNNCell([cell] * num_layers)

            # RNN-ENCODER returns the output activations [Batch size, Sequence Length, Num_neurons] and last hidden state as tensors.
            #encoder_output_c, encoder_state_c = tf.nn.dynamic_rnn(cell_c, embedded_input, dtype=tf.float32)
            
            cell1_f = LSTMCell(self.num_neurons_c,initializer=self.initializer)   # cell = DropoutWrapper(cell, output_keep_prob=dropout) or MultiRNNCell([cell] * num_layers)
            cell1_b = LSTMCell(self.num_neurons_c,initializer=self.initializer)   # cell = DropoutWrapper(cell, output_keep_prob=dropout) or MultiRNNCell([cell] * num_layers)
            # Return the output activations [Batch size, Sequence Length, 2*Num_neurons] and last hidden state as tensors.
            sequence_length = tf.tile([self.max_length], [self.batch_size])
            encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell1_f, cell1_b, self.embedded_input, sequence_length=sequence_length, dtype=tf.float32) ### NOTE: encoder_output is the ref for attention ###
            encoder_output_c = tf.concat(encoder_output,2) # [Batch size x time steps x num_neurons]
            encoder_state = tf.concat(encoder_state,2)
            encoder_state = encoder_state[0], encoder_state[1]
            encoder_output_c = tf.transpose(encoder_output_c, [1, 0, 2]) # transpose time axis first [time steps x Batch size x num_neurons]
            last_c = tf.gather(encoder_output_c, int(encoder_output_c.get_shape()[0]) - 1) # select last frame [Batch size x num_neurons]

            ### DO A CONVOLUTION HERE INSTEAD OF A FFN !!! ###
            weight_c1 = tf.Variable(tf.truncated_normal([2*self.num_neurons_c, 512], stddev=1.0)) #####################################################
            bias_c1 = tf.Variable(tf.truncated_normal([self.batch_size, 512], name="bc1")) #####################################################
            hidden_layer1 = tf.sigmoid(tf.matmul(last_c, weight_c1) + bias_c1)

            weight_c2 = tf.Variable(tf.truncated_normal([512, 1], stddev=1.0)) #####################################################
            bias_c2 = tf.Variable(tf.constant(0.0, shape=[1])) #####################################################
            logits = tf.sigmoid(tf.matmul(hidden_layer1, weight_c2) + bias_c2)

            bias_c3 = tf.Variable(tf.constant(self.init_bias_c, shape=[1])) #####################################################
            ampli_c3 = tf.Variable(tf.constant(10.0, shape=[1])) #####################################################

            self.predicted_tour_length = ampli_c3*logits+bias_c3 # initialize in [-100,0]


        #self.predicted_improvement = (self.initial_tour_length - self.predicted_tour_length)/self.initial_tour_length   
        self.optimal_improvement = (self.initial_tour_length - self.target)/self.initial_tour_length
        self.optimal_reward = 2*tf.sigmoid(10*self.optimal_improvement)-1


    def build_optim(self):

        # Learning rate
        self.lr = tf.train.exponential_decay(self.lr_start, self.global_step, self.lr_decay_step,self.lr_decay_rate, staircase=True, name="learning_rate")

        # Optimizer
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr,beta1=0.9,beta2=0.99, epsilon=0.00001) #####################################################
        
        # Loss
        self.loss = tf.reduce_sum(tf.square(self.predicted_reward - self.optimal_reward),0)/self.batch_size
        #self.loss = tf.reduce_sum(tf.square(self.predicted_improvement - self.optimal_improvement),0)/self.batch_size
        
        # Minimize step
        self.train_step = self.opt.minimize(self.loss, global_step=self.global_step)



if __name__ == "__main__":

    # Get config
    config, _ = get_config()
    batch_size = config.batch_size
    max_length = config.max_length
    input_dimension = config.input_dimension
    scale = config.scale
    n_components = config.n_components

    # Build Model and Reward from config
    critic = Critic(config)

    # Initialize solver
    solver = Solver(config.max_length)

    print("Starting training...")
    with tf.Session() as sess:
        tf.global_variables_initializer().run() #tf.initialize_all_variables().run()
        print_config()

        nb_epoch = 1000

        average_loss = 0
        test_loss = 0

        for i in tqdm(range(nb_epoch)):

            # Generate instances
            training_set = DataGenerator(solver)
            coord_batch, dist_batch, input_batch, initial_tour_length = training_set.next_batch(batch_size, max_length, input_dimension, scale, n_components)
            target = training_set.solve_batch(coord_batch)
            #optimal_improvement = (initial_tour_length - optimal_tour_length) / initial_tour_length
            #input_description = tf.concat([input_batch,initial_tour_length],0)

            # Construct feed_dict
            # y = [[(initial_tour_length - solver.run(dist_mat))/initial_tour_length] for dist_mat in dist_batch]
            feed = {critic.input_description: input_batch, critic.initial_tour_length: initial_tour_length, critic.target: target}

            # Run session
            predicted_reward, optimal_reward, loss, train_step = sess.run([critic.predicted_reward, critic.optimal_reward, critic.loss, critic.train_step], feed_dict=feed)
            #predicted_improvement, optimal_improvement, loss, train_step = sess.run([critic.predicted_improvement, critic.optimal_improvement, critic.loss, critic.train_step], feed_dict=feed)

            average_loss += loss[0]
            if i > 898:
                test_loss += loss[0]

            if i == 0:
                #print "\n Initial tour length: ", initial_tour_length
                #print "\n Predicted tour length:", predicted_tour_length
                #print "\n Optimal tour length:", optimal_tour_length
                print "\n Predicted reward:", predicted_reward
                print "\n Optimal reward:", optimal_reward
                print "\n Initial loss: ", average_loss
            if i%50 == 0 and i != 0:
                #print "\n Initial tour length: ", initial_tour_length
                #print "\n Predicted tour length:", predicted_tour_length
                #print "\n Optimal tour length:", optimal_tour_length
                print "\n Predicted improvement:", predicted_reward
                print "\n Optimal improvement:", optimal_reward
                print "\n Loss: ", average_loss/50
                average_loss = 0

        test_loss = test_loss/100
        print "\n Test loss: ", test_loss

