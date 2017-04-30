import tensorflow as tf



class pointer_decoder(object):

    def __init__(self, encoder_output, cell):
        # RNN decoder with pointer net for the sequence-to-sequence model.
        """
        Args:
          encoder_output: Tensor [Batch size x time steps x cell.state_size]   ### NOTE: These are the ref vectors to which attention is pointed ###
          cell: Decoder LSTM cell
        Returns:
            Stacked lists of decoder outputs, states, hard attention and inputs of same length as input.
        """

        self.encoder_output=encoder_output
        self.cell=cell

        self.batch_size = encoder_output.get_shape()[0]
        self.seq_length = encoder_output.get_shape()[1]
        self.n_hidden = cell.output_size

        self.W_ref = tf.Variable(tf.truncated_normal([1, self.n_hidden, self.n_hidden]), name="W_ref")
        self.W_q = tf.Variable(tf.truncated_normal([self.n_hidden, self.n_hidden]), name="W_q")
        self.v = tf.Variable(tf.truncated_normal([self.n_hidden]), name="v")

        self.seq_grad_W_ref=[]
        self.seq_grad_W_q=[]
        self.seq_grad_v=[]

        # Initialize lists to return
        self.outputs,self.states,self.positions,self.inputs,self.proba=[],[],[],[],[]

    # Attention mechanism takes a query (decoder output) [Batch size, n_hidden] and a set of reference (encoder_output) [Batch size, seq_length, n_hidden] 
    # as input to predict a distribution over next decoder input
    def attention(self,ref, query, with_softmax=True):

        encoded_ref = tf.nn.conv1d(ref, self.W_ref, 1, "VALID", name="encoded_ref") # [Batch size, seq_length, n_hidden]
        encoded_query = tf.expand_dims(tf.matmul(query, self.W_q, name="encoded_query"), 1) # [Batch size, 1, n_hidden]
        scores = tf.reduce_sum(self.v * tf.tanh(encoded_ref + encoded_query), [-1]) # [Batch size, seq_length]


        inv_soft = tf.expand_dims((1-tf.nn.softmax(scores)),1) # [Batch size, 1, seq_length]
        tan = tf.tanh(encoded_ref + encoded_query) # [Batch size, seq_length, n_hidden]
        v_batch_grad = tf.matmul(inv_soft,tan) # [Batch size, 1, n_hidden]
        self.seq_grad_v.append(v_batch_grad)

        inv_tan2=1-tf.square(tan) # [Batch size, seq_length, n_hidden]
        invsoft_x_invtan2 = tf.matmul(inv_soft,inv_tan2) # [Batch size, 1, n_hidden]
        v_t = tf.tile(tf.expand_dims(tf.expand_dims(self.v,0),0),[tf.cast(self.batch_size,tf.int32),1,1])  # [Batch size, 1, n_hidden]
        invsoft_x_invtan2_x_vt = tf.matmul(v_t,invsoft_x_invtan2,transpose_a=True) # [Batch size, n_hidden, n_hidden]
        invsoft_x_invtan2_x_vt_x_q = tf.multiply(invsoft_x_invtan2_x_vt,tf.expand_dims(query,1)) # [Batch size, n_hidden, n_hidden]
        self.seq_grad_W_q.append(invsoft_x_invtan2_x_vt_x_q)


        if with_softmax:
            return tf.nn.softmax(scores)
        else:
            return scores

    # One pass of the decode mechanism
    def decode(self,prev_input,prev_state,timestep):
        with tf.variable_scope("Decode"):
            if timestep > 0:
                tf.get_variable_scope().reuse_variables()
            # First, we run the cell on a combination of the previous input and state
            output,state=self.cell(prev_input,prev_state)
            # Then, we calculate new attention mask and take the arg_max index
            distribution=self.attention(self.encoder_output,output)
            position = tf.arg_max(distribution,1)       ######################## GUMBEL SOFTMAX ! ##########################
            position = tf.cast(position,tf.int32)
            # We select the decoder's new input
            h = tf.transpose(self.encoder_output, [1, 0, 2]) # [Batch size x time steps x cell.state_size] to [time steps x Batch size x cell.state_size]
            new_decoder_input = tf.gather(h,position)[0]

            return output,state,position,new_decoder_input,distribution

    def loop_decode(self,decoder_initial_state,decoder_first_input):
        # decoder_initial_state: Tuple Tensor (c,h) of size [batch_size x cell.state_size]
        # decoder_first_input: Tensor [batch_size x cell.state_size]

        # Loop the decoding process and collect results
        o,s,p,i= None,decoder_initial_state,None,decoder_first_input
        for step in range(self.seq_length):
            o,s,p,i,d= self.decode(i,s,step)
            self.outputs.append(o)
            self.states.append(s)
            self.positions.append(p)
            self.inputs.append(i)
            self.proba.append(d)


        self.seq_grad_v=tf.reduce_sum(self.seq_grad_v,0) # seq_length * [Batch size, 1, n_hidden] to [Batch size, 1, n_hidden]
        self.seq_grad_W_q=self.seq_grad_W_q[0]    #tf.reduce_sum(self.seq_grad_W_q,0) # seq_length * [Batch size, n_hidden, n_hidden] to [Batch size, n_hidden, n_hidden]


        # Stack lists to tensors
        self.outputs,self.states,self.positions,self.inputs,self.proba=tf.stack(self.outputs,axis=1),tf.stack(self.states,axis=1),tf.stack(self.positions,axis=1),tf.stack(self.inputs,axis=1),tf.stack(self.proba,axis=1)
        return self.outputs,self.states,self.positions,self.inputs,self.proba
