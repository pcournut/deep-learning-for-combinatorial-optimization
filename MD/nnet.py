import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, GRUCell, MultiRNNCell, DropoutWrapper
from tqdm import tqdm

from decoder import pointer_decoder
import dataset


# TODO

"""
Michel:
- Variable seq length ****
- TSP with soft time windows ***
- Replace coordinates by KNN distances (K=p+1) ***

- Replace reward by -length(tour + AR non visited cities) ***
- (Actor-)Critic for reward baseline ****


Pierre
- GAN (Discriminator CNN) ****
- Supervised setting: Cplex, Concorde... ***
- Back prop ***


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
        self.input_embed=args['input_embed'] # dimension of embedding space (input)
        self.num_neurons = args['num_neurons'] # dimension of hidden states (LSTM cell)
        self.hidden_dim= args['num_neurons'] # same thing...
        self.initializer = tf.random_uniform_initializer(-args['init_range'], args['init_range']) # variables initializer
        self.step=0 # int to reuse_variable in scope
        self.baseline=args['init_baseline']

        self.build_model()
        self.build_reward()
        self.build_optim()



    def build_model(self):

        # Tensor block holding the input sequences [Batch Size, Sequence Length, Features]
        self.X = tf.placeholder(tf.float32, [self.batch_size, self.max_length, self.input_dimension], name="Input")

        # Embed input sequence
        self.W_embed = tf.Variable(tf.truncated_normal([1,self.input_dimension,self.input_embed]), name="W_embed")
        with tf.variable_scope("Embed"):
            if self.step>0:
                tf.get_variable_scope().reuse_variables()
            self.embeded_input = tf.nn.conv1d(self.X, self.W_embed, 1, "VALID", name="EncoderInput")

        # ENCODER LSTM cell
        self.cell1 = LSTMCell(self.num_neurons,initializer=self.initializer)   # Or GRUCell(num_neurons)
        #cell = DropoutWrapper(cell, output_keep_prob=dropout) or MultiRNNCell([cell] * num_layers)

        # RNN-ENCODER returns the output activations [Batch size, Sequence Length, Num_neurons] and last hidden state as tensors.
        self.encoder_output, self.encoder_state = tf.nn.dynamic_rnn(self.cell1, self.embeded_input, dtype=tf.float32) ### NOTE: encoder_output is the ref for attention ###

        # DECODER initial state is the last relevant state from encoder
        self.decoder_initial_state = self.encoder_state ### NOTE: if state_tuple=True, self.decoder_initial_state = (c,h) ###

        # DECODER initial input is 'GO', a variable tensor
        #self.decoder_first_input = tf.cast(tf.Variable(tf.truncated_normal([self.batch_size,self.hidden_dim]), name="GO"),tf.float32) #tf.constant(0.1, shape=[self.batch_size,self.hidden_dim]) #
        self.decoder_first_input = tf.Variable(tf.truncated_normal([self.batch_size,self.hidden_dim]), name="GO")

        # DECODER LSTM cell        
        self.cell2 = LSTMCell(self.num_neurons,initializer=self.initializer)

        # POINTER-DECODER returns the output activations, hidden states, hard attention and decoder inputs as tensors.
        self.ptr = pointer_decoder(self.encoder_output, self.cell2)
        self.outputs, self.states, self.positions, self.inputs, self.proba = self.ptr.loop_decode(self.decoder_initial_state, self.decoder_first_input)



    def build_reward(self):

        # From input sequence and hard attention, get coordinates of the agent's trip
        tours=[]
        shifted_tours=[]
        for k in range(self.batch_size):
            strip=tf.gather_nd(self.X,[k])
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

        # Get number of cities visited
        counter=[]
        for k in range(self.batch_size):
            y, idx, count = tf.unique_with_counts(self.positions[k])
            counter.append(tf.shape(y))
        self.penality=tf.stack(counter,0)
        
        # Define reward from objective and penalty
        alpha=10
        self.reward=alpha*tf.cast(self.penality,tf.float32)-tf.cast(self.distances,tf.float32)

        output_prob = []
        for k in range(self.batch_size):
            seq_proba=[]
            for kk in range(self.max_length):
                seq_proba.append(self.proba[k][kk][self.positions[k][kk]])
            output_prob.append(tf.stack(seq_proba,0))
        self.seq_proba=tf.stack(output_prob,0)



    def build_optim(self):
        self.opt = tf.train.AdamOptimizer() # optimizer 

        # Discounted reward
        self.reward_baseline=tf.stop_gradient(self.reward-self.baseline) # [Batch size, 1]

        # Loss
        self.loss=tf.reduce_sum(tf.multiply(self.ptr.log_softmax,self.reward_baseline),0)/self.batch_size

        # Loss=self.outputs
        """
        self.cellW2 = [v for v in tf.trainable_variables() if v.name == "Decode/lstm_cell/weights:0"][0]

        var=[self.ptr.v,self.ptr.W_q,self.ptr.W_ref]

        self.compute_grad_step = self.opt.compute_gradients(self.loss,var_list=var)

        self.grad_v, _ = self.compute_grad_step[0] # grav v, v
        self.grad_W_q, _ = self.compute_grad_step[1] 
        self.grad_W_ref, _ = self.compute_grad_step[2]

        self.train_step = self.opt.apply_gradients([(self.grad_v,self.ptr.v),(self.grad_W_q,self.ptr.W_q),(self.grad_W_ref,self.ptr.W_ref)]) # minimize operation #
        """

        self.train_step = self.opt.minimize(self.loss)
        


    def run_episode(self,sess):

        # Get feed_dict
        training_set = dataset.DataGenerator()
        inp = training_set.next_batch(self.batch_size, self.max_length, self.input_dimension)
        feed = {self.X: inp}

        # Forward pass
        seq_input, permutation, circuit_length, cities_visited, reward, seq_proba = sess.run([self.X,self.positions,self.distances,self.penality,self.reward, self.seq_proba], feed_dict=feed)

        # Train step
        loss, train_step = sess.run([self.loss,self.train_step],feed_dict=feed)
        self.step+=1

        # Update baseline
        if self.step%100==0:
            self.baseline+=1

        return seq_input, permutation, circuit_length, cities_visited, reward, seq_proba, loss, train_step








def train():

    # Config
    args={}
    args['batch_size']=32
    args['max_length']=6
    args['input_dimension']=2
    args['init_baseline']=25
    args['input_embed']=32
    args['num_neurons']=256
    args['init_range']=0.1

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
        print('* City dimension:',model.input_dimension)
        print('* Embedding:',model.input_embed)
        print('* Num neurons (Encoder & Decoder LSTM cell):',model.num_neurons)
        print('\n')

        for i in tqdm(range(500)): # epoch i

            seq_input, permutation, circuit_length, cities_visited, reward, seq_proba, loss, train_step = model.run_episode(sess)

            if i % 100 == 0:
                #print('\n Input: \n', seq_input)
                #print('\n Permutation: \n', permutation)
                #print('\n Circuit length: \n',circuit_length)
                print('\n Average number of cities visited :', sess.run(tf.reduce_mean(tf.cast(cities_visited,tf.float32))))
                print(' Baseline :', model.baseline)
                print(' Average Reward:', sess.run(tf.reduce_mean(reward)))
                print(' Loss:', loss)
                #print('\n Log softmax: \n', sess.run(self.ptr.log_softmax,feed_dict=feed))
                print('\n')

            if i % 1000 == 0 and not(i == 0):
                saver.save(sess,"save/" +str(i) +".ckpt")

        print('\n Trainable variables')
        for v in tf.trainable_variables():
            print(v.name)

        print("Training is COMPLETE!")
        saver.save(sess,"save/model.ckpt")




if __name__ == "__main__":
    train()