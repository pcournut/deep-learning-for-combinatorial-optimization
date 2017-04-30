import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, GRUCell, MultiRNNCell, DropoutWrapper
from tqdm import tqdm

from decoder import pointer_decoder
import dataset


# TODO

"""
- Optimize network (POLICY GRADIENT REINFORCE or GUMBEL SOFTMAX): W_embed, cell1, decod_fst_input, cell2, W_ref, W_q, v
    Actor - Critic
- Variable seq length
- GAN

- import argparse
- Summary writer (log)...
- Parallelize (GPU), C++...
- Nice plots, interface...
- Compare results to other solvers
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

        self.build_model()
        self.build_reward()
        self.build_optim()



    def build_model(self):

        # Tensor block holding the input sequences [Batch Size, Sequence Length, Features]
        self.X = tf.placeholder(tf.float32, [self.batch_size, self.max_length, self.input_dimension], name="Input")

        # Embed input sequence
        self.W_embed = tf.Variable(tf.truncated_normal([1,self.input_dimension,self.input_embed]), name="W_ref")
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
        self.decoder_first_input = tf.cast(tf.Variable(tf.truncated_normal([self.batch_size,self.hidden_dim]), name="GO"),tf.float32) #tf.constant(0.1, shape=[self.batch_size,self.hidden_dim]) #

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
        self.reward_baseline=tf.expand_dims((self.reward-30),1) # [Batch size, 1, 1]

        self.gradient_v=tf.matmul(self.reward_baseline,self.ptr.seq_grad_v) # [Batch size, 1, n_hidden]
        self.gradient_v=tf.reshape(tf.reduce_sum(self.gradient_v,0),[-1]) # [Batch size, 1, n_hidden] to [1, n_hidden] to [n_hidden]
        
        """
        self.gradient_W_q=[]
        for k in range(self.batch_size):
            self.gradient_W_q.append(self.reward_baseline[k]*self.ptr.seq_grad_W_q[k])
        self.gradient_W_q=tf.stack(self.gradient_W_q,0)
        self.gradient_W_q=tf.reduce_sum(self.gradient_W_q,0)
        """
        
        self.opt = tf.train.AdamOptimizer() # optimize 
        self.train_step = self.opt.apply_gradients([(self.gradient_v,self.ptr.v)]) # minimize operation #(self.gradient_W_q,self.ptr.W_q)
        


    def run_episode(self,sess):

        # Get feed_dict
        print("\n Getting Dataset...")
        training_set = dataset.DataGenerator()
        inp = training_set.next_batch(self.batch_size, self.max_length, self.input_dimension)
        feed = {self.X: inp}

        # Forward pass
        seq_input, permutation, circuit_length, penality, reward, proba, seq_proba, gradient_v = sess.run([self.X,self.positions,self.distances,self.penality,self.reward,self.proba, self.seq_proba, self.gradient_v], feed_dict=feed)
        print('\n Input: \n', seq_input)
        print('\n Permutation: \n', permutation)
        print('\n Circuit length: \n',circuit_length)
        print('\n Number of cities visited (+10 bonus / city) :\n', penality)
        print('\n Reward - baseline: \n', reward-30)

        #print('\n Attention proba: \n', proba) # [Batch size, Prob step i (over n ref), number of step (=n)]
        print('\n Seq proba: \n', seq_proba)

        #print('\n Seq_grad_v: \n', sess.run(self.ptr.seq_grad_v,feed_dict=feed))
        #print('\n Grad_v: \n', gradient_v)

        print('\n Seq_grad_W_q: \n', sess.run(self.ptr.seq_grad_W_q,feed_dict=feed)) #self.ptr.seq_grad_W_q
        #print('\n V before grad: \n', self.ptr.v.eval())
        sess.run(self.train_step, feed_dict=feed)
        #print('\n V after grad: \n', self.ptr.v.eval())


        self.step+=1
        
        print('\n')









def train():

    # Config
    args={}
    args['batch_size']=2
    args['max_length']=4
    args['input_dimension']=2
    args['input_embed']=5
    args['num_neurons']=5
    args['init_range']=3

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
        """
        print('Tensors shape:')
        print('* Input:','[',model.batch_size,',', model.max_length,',', model.input_dimension,']')
        print('* Embedded input:','[',model.batch_size,',',model.max_length,',',model.input_embed,']')
        print('* Encoder & Decoder output activations and hidden state:','[',model.batch_size,',', model.max_length,',', model.hidden_dim,']')
        print('* Decoder input:','[',model.batch_size,',', model.hidden_dim,']')
        print('* Attention matrices W_ref, W_q:','[',model.hidden_dim,',',model.hidden_dim,']')
        print('* Attention vector v:','[',model.hidden_dim,']')
        print('\n')
        """

        for i in tqdm(range(10)): # epoch i

            model.run_episode(sess)

            if i % 1000 == 0 and not(i == 0):
                saver.save(sess,"save/" +str(i) +".ckpt")

        print("Training is COMPLETE!")
        saver.save(sess,"save/model.ckpt")




if __name__ == "__main__":
    train()