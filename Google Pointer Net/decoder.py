import tensorflow as tf
distr = tf.contrib.distributions



# RNN decoder with pointer net for the sequence-to-sequence model.
class pointer_decoder(object):

    def __init__(self, encoder_output, cell, temperature, C, inference_mode, initializer):

        self.encoder_output = encoder_output # Ref vectors to which attention is pointed: Tensor [Batch size x time steps x cell.state_size]
        self.cell = cell # DECODER LSTM Cell

        self.batch_size = encoder_output.get_shape()[0] # batch size
        self.seq_length = encoder_output.get_shape()[1] # sequence length
        self.n_hidden = cell.output_size # num_neurons

        # Attending mechanism
        with tf.variable_scope("glimpse") as glimpse:
            self.W_ref_g =tf.get_variable("W_ref_g",[1,self.n_hidden,self.n_hidden],initializer=initializer)
            self.W_q_g =tf.get_variable("W_q_g",[self.n_hidden,self.n_hidden],initializer=initializer)
            self.v_g =tf.get_variable("v_g",[self.n_hidden],initializer=initializer)

        # Pointing mechanism
        with tf.variable_scope("pointer") as pointer:
            self.W_ref =tf.get_variable("W_ref",[1,self.n_hidden,self.n_hidden],initializer=initializer)
            self.W_q =tf.get_variable("W_q",[self.n_hidden,self.n_hidden],initializer=initializer)
            self.v =tf.get_variable("v",[self.n_hidden],initializer=initializer)

        self.mask = tf.zeros((self.batch_size,self.seq_length))

        self.inference_mode = inference_mode # True for inference / False for training
        self.temperature = temperature # temperature parameter
        self.C = C # logit clip

        self.log_softmax = [] # store log(p_theta(pi(t)|pi(<t),s)) for backprop
        self.positions = [] # store visited cities for reward


    # From a query (decoder output) [Batch size, n_hidden] and a set of reference (encoder_output) [Batch size, seq_length, n_hidden]
    # predict a distribution over next decoder input
    def attention(self,ref,query,temperature):

        encoded_ref_g = tf.nn.conv1d(ref, self.W_ref_g, 1, "VALID", name="encoded_ref_g") # [Batch size, seq_length, n_hidden]
        encoded_query_g = tf.expand_dims(tf.matmul(query, self.W_q_g, name="encoded_query_g"), 1) # [Batch size, 1, n_hidden]
        scores_g = tf.reduce_sum(self.v_g * tf.tanh(encoded_ref_g + encoded_query_g), [-1], name="scores_g") # [Batch size, seq_length]
        attention_g = tf.nn.softmax(scores_g,name="attention_g")

        # 1 Glimpse = Linear combination of ref weighted by attention mask (or mask) = pointing mechanism query #########################################
        glimpse = tf.multiply(ref, tf.expand_dims(attention_g,2))
        glimpse = tf.reduce_sum(glimpse,1)

        encoded_ref = tf.nn.conv1d(ref, self.W_ref, 1, "VALID", name="encoded_ref") # [Batch size, seq_length, n_hidden]
        encoded_query = tf.expand_dims(tf.matmul(glimpse, self.W_q, name="encoded_query"), 1) # [Batch size, 1, n_hidden]
        scores = tf.reduce_sum(self.v * tf.tanh(encoded_ref + encoded_query), [-1], name="scores") # [Batch size, seq_length]
        attention = tf.nn.softmax(scores,name="attention") # [Batch size, Seq_length]
        """
        if self.inference_mode == True:
            attention = tf.nn.softmax(scores/temperature, name="attention") # [Batch size, Seq_length]
        else:
            attention = tf.nn.softmax(self.C*tf.tanh(scores), name="attention") # [Batch size, Seq_length]
        """
        return attention, scores


    # One pass of the decode mechanism
    def decode(self,prev_state,prev_input,timestep):
        with tf.variable_scope("loop"):
            if timestep > 0:
                tf.get_variable_scope().reuse_variables()

            # Run the cell on a combination of the previous input and state
            output,state=self.cell(prev_input,prev_state)

            # Attention mechanism
            distribution, scores=self.attention(self.encoder_output,output,self.temperature)

            # Apply attention mask
            masked_scores = scores - 100000000.*self.mask # [Batch size, seq_length]

            # Multinomial distribution
            prob = distr.Categorical(masked_scores)

            # Sample from distribution
            position = prob.sample()
            position = tf.cast(position,tf.int32)
            self.positions.append(position)

            # Store log_prob for backprop
            self.log_softmax.append(prob.log_prob(position))

            # Update mask
            self.mask = self.mask + tf.one_hot(position, self.seq_length)

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

        # Sum log_softmax over output steps
        self.log_softmax=tf.add_n(self.log_softmax) #tf.reduce_sum(self.log_softmax,0)
        
        # Return stacked lists of visited_indices and log_softmax for backprop
        return self.positions,self.log_softmax