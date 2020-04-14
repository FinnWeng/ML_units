import tensorflow as tf
import math


# EMA

class VQVAE:
    def __init__(self, embedding_dim, _num_embeddings, commit_loss_coef,partition, scope):
        self.commit_loss_coef = commit_loss_coef
        self._num_embeddings = _num_embeddings  # the number of embed vectors
        # self._embedding_dim = embedding_dim  # which means how many discrete symbol for a digit(base n numerical)
        self.scope = scope
        self.gamma = 0.99
        self.start_temp = 0.00001
        self.temp_lower_bound = 0.0
        self.temp_decay_rate = 0.005
        self.partition = partition


        
        self._embedding_dim = embedding_dim # hidden_size
        self.part_embd_dim = int(self._embedding_dim/self.partition)
        # self.part_k = int(self._num_embeddings/self.partition)
        # self.part_k = 2**int(math.log(self._num_embeddings,base =2)//self.partition)
        self.part_k = 33

        #  embedding_dim: length of latent variable, in this implementation it is channel number of input tensor(how many bits)
        """
        So, it is like this:
        the num_embed is like we recongnize all experience we enconter into several kind of situations. i.e. all 2000 pics to 10 kind of classes.
        the embedding_dim, is the memory we like to spend to memorize the classed. The more we spend, the more details model can remember.


        """
        # print("_num_embeddings:", self._num_embeddings)
        # print("self._embedding_dim:", self._embedding_dim)

    def variable_def(self):
        initializer = tf.uniform_unit_scaling_initializer()

        with tf.variable_scope(self.scope, 'VQVAE', reuse=tf.AUTO_REUSE):
            # means_shape =  [num_blocks, block_v_size, block_dim]
            self._w = tf.get_variable('embedding', [self.partition, self.part_k, self.part_embd_dim],
                                      initializer=initializer, trainable=True)

            self.embedding_total_count = tf.get_variable("embedding_total_count", [1, self.partition*self.part_k],
                                                         initializer=tf.zeros_initializer(),
                                                         dtype=tf.int32)

            self.sampling_temperature = tf.Variable(self.start_temp, dtype=tf.float32, name="sampling_decay_temp")

    def input_slicing(self, inputs,input_shape):
        flat_inputs = tf.reshape(inputs, [-1, input_shape[1] * input_shape[2],self.partition ,self.part_embd_dim])
        # inputs_flat = [batch_size, latent_dim, hidden_size]
        # x_shape = [batch_size, latent_dim, num_blocks,block_dim]
        return flat_inputs



    def loop_assign_moving_avg(self, encodings, sliced_inputs):
        # print("encodings:", encodings)   [b*h*w,partion, part_k]

        flat_input = tf.reshape(sliced_inputs,[-1,self.partition,self.part_embd_dim]) # b, h*w,self.partition ,self.part_embd_dim = > b*h*w,self.partition ,self.part_embd_dim
        print("flat_input:",flat_input)

        embedding_count = tf.reshape(tf.reduce_sum(encodings, axis=0), [1, -1]) # [b*h*w,partion, part_k] = > [1, partion * part_k]
        print("embedding_count:",embedding_count)
        

        update_or_not = tf.ceil(embedding_count / tf.reduce_max(embedding_count)) # to one hot  [1, partion * part_k]
        print("update_or_not:", update_or_not)

        print("self.embedding_total_count:", self.embedding_total_count) #  [1, partion * part_k]

     
        self.embedding_total_count -= tf.cast(update_or_not * tf.floor(
            (1 - self.gamma) * (tf.cast(self.embedding_total_count, tf.float32) - embedding_count)), tf.int32) #  [1, partion * part_k]*[1, partion * part_k] =>  [1, partion * part_k]
        
        print("self.embedding_total_count:", self.embedding_total_count)

        # expand_flat_inputs = tf.transpose(flat_input, [0, 2, 1])#(batch,w*h,128)) (?, 128, ?)
        # print("expand_flat_inputs:", expand_flat_inputs) 

        input_contrib_per_embedding_value = tf.matmul(tf.expand_dims(flat_input,-1), tf.expand_dims(encodings,axis=-2)) #(b*h*w,self.partition ,self.part_embd_dim,1)*  [b*h*w, partion, 1,part_k] => (b*h*w,self.partition ,self.part_embd_dim,part_K)
        print("input_contrib_per_embedding_value:", input_contrib_per_embedding_value)
        input_contrib_per_embedding_value = tf.reduce_sum(input_contrib_per_embedding_value, axis=[0]) # (self.partition ,self.part_embd_dim,part_K)
        print("input_contrib_per_embedding_value:", input_contrib_per_embedding_value)
        input_contrib_per_embedding_value = input_contrib_per_embedding_value / tf.cast(tf.reshape(self.embedding_total_count,[self.partition,1,self.part_k]),
                                                                                        tf.float32) # Moving average, [self.partition ,self.part_embd_dim,part_K] ,[partion ,1, part_k]
        print("input_contrib_per_embedding_value:", input_contrib_per_embedding_value) # [self.partition ,self.part_embd_dim,part_K]
        input_contrib_per_embedding_value = tf.transpose(input_contrib_per_embedding_value,[0,2,1]) # [self.partition,self.part_k,self.part_embd_dim]
        print("input_contrib_per_embedding_value:", input_contrib_per_embedding_value)

        contrib = (1 - self.gamma) * (input_contrib_per_embedding_value) * tf.reshape(update_or_not,[self.partition,self.part_k,1])# [self.partition,self.part_k,self.part_embd_dim] *[self.partition,self.part_k,1] 


        self._w  = self._w*self.gamma + contrib

        return [self._w,
                tf.reduce_mean(tf.reduce_sum(self._w, axis=0)),
                tf.reduce_sum(update_or_not),
                tf.math.top_k(self.embedding_total_count, k=10),
                tf.reduce_sum(embedding_count),
                tf.math.top_k(embedding_count, k=10),
                tf.reduce_mean(tf.reduce_sum(flat_input, axis=-1)),
                self.take_num
                ]

    def temperature_decay(self):
        return tf.cond(tf.math.greater_equal(tf.subtract(self.sampling_temperature,self.temp_decay_rate), self.temp_lower_bound),
                       lambda: tf.assign_sub(self.sampling_temperature, self.temp_decay_rate),
                       lambda: self.temp_lower_bound)

    def temperature_sampler(self, distances, temperature):
        '''

        :param distance: (batch,h*w ,k)
        0: argmin
        1: update all
        :param temperature:
        :return: multilhot encoding (batch, h*w, take_num)
        '''

        floor_temp = tf.floor(temperature*100)*100

        self.take_num  = tf.cond(tf.not_equal(floor_temp,0.0),lambda :tf.floor((temperature) * (self._num_embeddings)),lambda :1.0)

        top_1_idx = tf.stack([tf.multinomial(-distances[:, i, :], num_samples=tf.cast(self.take_num,tf.int32))for i in range(self.partition)],axis=1) 
        # top_1_idx: Tensor("vea_autoencoder/top_VQVAE/stack:0", shape=(?, 8, ?), dtype=int64)

        return top_1_idx

    def quantize(self, encoding_indices):

        # self._w : [self.partition, self.part_k, self.part_embd_dim]

        # encoding_indices: [b*H*W,partition,1]

        trans_idx = tf.transpose(encoding_indices,[1,0,2])
        print("trans_idx:",trans_idx) # (partition, b*h*w, 1)


        quantize = tf.stack([tf.nn.embedding_lookup(self._w[i,:,:], trans_idx[i,:,:], validate_indices=False) for i in range(self.partition)],axis=0) 
        # [self.part_k, self.part_embd_dim] <= b*h*w, 1, search through axis 0 of w ( self.park ) and take one elementm and the element len is part_embd_dim
        print("quantize:",quantize) # (8, b*h*w, 1, 32)
        quantize = tf.reduce_mean(quantize,axis=-2)
        print("quantize:",quantize) # # (8, b*h*w, 32)
        quantize = tf.transpose(quantize,[1,0,2])
        print("quantize:",quantize) # # (b*h*w, 8, 32)

        # quantize = tf.nn.embedding_lookup(self._w, encoding_indices, validate_indices=False) 
        # print("quantize:",quantize)
        return quantize


        # return quantize  # (b,h*w,dim)

    def VQVAE_layer(self, inputs):
        # Assert last dimension is same as self._embedding_dim
        print("inputs:", inputs)

        input_shape = tf.shape(inputs)
        with tf.control_dependencies([tf.Assert(tf.equal(input_shape[-1], self._embedding_dim),[input_shape])]):
            sliced_inputs = self.input_slicing(inputs,input_shape)
            print("sliced_inputs:", sliced_inputs) # -1, input_shape[1] * input_shape[2],self.partition ,self.part_embd_dim


        self.variable_def()  # set all variable
        print("self._w:", self._w)
        self.embedding_total_count += 1

        # the _w is already qunatized: for each row, each idx(latent variable digit) have its own value to pass, value pf _w is quantized embd ouput

        x = tf.reshape(sliced_inputs,[-1,self.partition ,self.part_embd_dim]) # batch*h*w ,self.partition ,self.part_embd_dim
        print("x:", x)
        x_norm_sq = tf.reduce_sum(tf.square(x), axis=-1, keep_dims=True)
        print("x_norm_sq:",x_norm_sq)
        means_norm_sq = tf.reduce_sum(tf.square(self._w), axis=-1, keep_dims=True) # self.partition, self.part_k, self.part_embd_dim
        print("means_norm_sq:",means_norm_sq)
        scalar_prod = tf.matmul(tf.transpose(x, perm=[1, 0, 2]), tf.transpose(self._w, perm=[0, 2, 1]))# (self.partition, batch*h*w, self.part_embd_dim),(self.partition, self.part_embd_dim,self.part_k)
        print("scalar_prod:",scalar_prod) 
        scalar_prod = tf.transpose(scalar_prod, perm=[1, 0, 2])
        print("scalar_prod:",scalar_prod)
        distances = x_norm_sq + tf.transpose(means_norm_sq, perm=[2, 0, 1]) - 2 * scalar_prod # batch*h*w,self.partition, self.part_k
        print("distances:",distances)



        ####
        #### EMA Moving average(non max)
        ####

        non_max_encoding_indices = self.temperature_sampler(distances, self.sampling_temperature)
        print("non_max_encoding_indices",non_max_encoding_indices)
        # non_max_encoding_indices Tensor("vea_autoencoder/top_VQVAE/stack:0", shape=(?, 8, ?), dtype=int64)  # [b*H*W,partition,1]

        encoding_indices = tf.expand_dims(tf.argmin(distances,2),-1)  # [b*h*w,partion,1]
        print("non_max_encoding_indices(argmax)", encoding_indices)
        same_idx =tf.reduce_sum(tf.cast(tf.equal(non_max_encoding_indices,encoding_indices),tf.float32)) # [b*h*w,partion,1]
        # print("same_idx:",same_idx)

        # multi_hot_encodings = tf.map_fn(lambda x: tf.reduce_sum(tf.one_hot(x, self.part_k), axis=-2),
        #                                 tf.transpose(non_max_encoding_indices, perm=[1, 0, 2]), dtype=tf.float32)
        # multi_hot_encodings = tf.transpose(multi_hot_encodings, perm=[1, 0, 2])
        multi_hot_encodings = tf.one_hot(non_max_encoding_indices,self.part_k) # [b*h*w,partion, top K,part_k]
        print("multi_hot_encodings:", multi_hot_encodings)
        multi_hot_encodings = tf.reduce_sum(multi_hot_encodings,axis=-2) # [b*h*w,partion, part_k]
        print("multi_hot_encodings:", multi_hot_encodings)



        non_max_quantized_embd_out = self.quantize(non_max_encoding_indices) # (b*h*w, 8, 32)

        print("non_max_quantized_embd_out:",    non_max_quantized_embd_out)
        non_max_quantized_embd_out = tf.reshape(non_max_quantized_embd_out, tf.shape(inputs))# reverse partition #1
                                                                              
        # non_max_quantized_embd_out = tf.reshape(non_max_quantized_embd_out,tf.shape(inputs))# reverse partition #2


        print("non_max_quantized_embd_out:", non_max_quantized_embd_out)


        e_latent_loss = tf.reduce_mean((tf.stop_gradient(non_max_quantized_embd_out) - inputs) ** 2)  # embedding loss
        q_latent_loss = tf.reduce_mean((tf.stop_gradient(inputs) - non_max_quantized_embd_out) ** 2)
        # VQ_loss = self.commit_loss_coef*e_latent_loss
        VQ_loss = self.commit_loss_coef*e_latent_loss + 1/1000*q_latent_loss

        non_max_quantized_embd_out = inputs + tf.stop_gradient(
            non_max_quantized_embd_out - inputs)  # in order to pass value to decoder???

        assign_moving_avg_op = self.loop_assign_moving_avg(multi_hot_encodings, sliced_inputs)
        temp_decay_op = self.temperature_decay()

        return {
            # 'quantized_embd_out': quantized_embd_out,
            "quantized_embd_out": non_max_quantized_embd_out,
            'VQ_loss': VQ_loss,
            'encodings': multi_hot_encodings,
            # 'encodings': encodings,
            # 'encoding_indices': encoding_indices,
            'encoding_indices': multi_hot_encodings,
            'assign_moving_avg_op': assign_moving_avg_op,
            'temp_decay_op': temp_decay_op,
            # "top_k_idx":self.top_k_idx.shape
            'top_k_idx':same_idx
        }


    def idx_inference(self, outer_encoding_indices):
        outer_encodings = tf.one_hot(outer_encoding_indices, self._num_embeddings)

        return outer_encodings
