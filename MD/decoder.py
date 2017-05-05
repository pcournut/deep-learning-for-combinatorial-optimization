import tensorflow as tf



class pointer_decoder(object):

    def __init__(self, encoder_output, cell):
        # RNN decoder with pointer net for the sequence-to-sequence model.
        self.encoder_output = encoder_output # Ref vectors to which attention is pointed: Tensor [Batch size x time steps x cell.state_size]
        self.cell = cell # DECODER LSTM Cell

        self.batch_size = encoder_output.get_shape()[0] # batch size
        self.seq_length = encoder_output.get_shape()[1] # sequence length
        self.n_hidden = cell.output_size # num_neurons

        self.W_ref = tf.Variable(tf.truncated_normal([1, self.n_hidden, self.n_hidden]), name="W_ref") # W_ref attention matrix 
        self.W_q = tf.Variable(tf.truncated_normal([self.n_hidden, self.n_hidden]), name="W_q") # W_q attention matrix 
        self.v = tf.Variable(tf.truncated_normal([self.n_hidden]), name="v") # v attention vector 
        self.temperature=1 # temperature parameter

        self.log_softmax = [] # store log(p_theta(pi(t)|pi(<t),s)) for backprop
        self.positions = [] # store visited cities (to apply mask)
        self.proba = [] # store sequence probability (for temperature tuning)


    # Attention mechanism takes a query (decoder output) [Batch size, n_hidden] and a set of reference (encoder_output) [Batch size, seq_length, n_hidden] 
    # to predict a distribution over next decoder input
    def attention(self,ref,query,temperature,):
        encoded_ref = tf.nn.conv1d(ref, self.W_ref, 1, "VALID", name="encoded_ref") # [Batch size, seq_length, n_hidden]
        encoded_query = tf.expand_dims(tf.matmul(query, self.W_q, name="encoded_query"), 1) # [Batch size, 1, n_hidden]
        scores = tf.reduce_sum(self.v * tf.tanh(encoded_ref + encoded_query), [-1]) # [Batch size, seq_length]
        return tf.nn.softmax(scores/temperature), scores # [Batch size, Seq_length]


    # One pass of the decode mechanism
    def decode(self,prev_state,prev_input,timestep):
        with tf.variable_scope("Decode"):
            if timestep > 0:
                tf.get_variable_scope().reuse_variables()

            # Run the cell on a combination of the previous input and state
            output,state=self.cell(prev_input,prev_state)

            # Attention mechanism
            distribution,scores=self.attention(self.encoder_output,output,self.temperature)

            # Store results for backprop
            self.log_softmax.append(tf.log(distribution+0.0000001))  ######################## GUMBEL SOFTMAX ! ##########################

            # Apply new attention mask
            masked_distribution=distribution
            if timestep>0: 
                visited_indices = tf.stack(self.positions,axis=1) # Update visited_indices Tensor [Batch size, step+1]
                mask = 1-tf.reduce_sum(tf.one_hot(visited_indices,tf.to_int32(self.seq_length)),1) # [Batch size, Seq_length]
                masked_distribution=tf.multiply(distribution,mask)

            # Take the arg_max index as new input
            position = tf.arg_max(masked_distribution,1)  
            position = tf.cast(position,tf.int32)
            self.positions.append(position)

            # Store probability
            self.proba.append(tf.gather(tf.transpose(distribution,[1,0]),position)[0])  

            # Retrieve decoder's new input
            h = tf.transpose(self.encoder_output, [1, 0, 2]) # [Batch size x time steps x cell.state_size] to [time steps x Batch size x cell.state_size]
            new_decoder_input = tf.gather(h,position)[0]

            return state,new_decoder_input


    def loop_decode(self,decoder_initial_state,decoder_first_input):
        # decoder_initial_state: Tuple Tensor (c,h) of size [batch_size x cell.state_size]
        # decoder_first_input: Tensor [batch_size x cell.state_size]

        # Loop the decoding process and collect results
        s,i = decoder_initial_state,tf.cast(decoder_first_input,tf.float32)
        for step in range(self.seq_length):
            s,i = self.decode(s,i,step)

        # Stack visited indices
        self.positions=tf.stack(self.positions,axis=1)
        self.proba=tf.stack(self.proba,axis=1)

        # Sum log_softmax over output steps
        self.log_softmax=tf.reduce_sum(self.log_softmax,0)
        
        # Return stacked lists of visited_indices, seq proba and log_softmax for backprop
        return self.positions,self.proba,self.log_softmax