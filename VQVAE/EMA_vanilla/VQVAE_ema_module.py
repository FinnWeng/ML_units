import tensorflow as tf


# EMA

class VQVAE:
    def __init__(self, embedding_dim, _num_embeddings, commit_loss_coef, scope):
        self.commit_loss_coef = commit_loss_coef
        self._num_embeddings = _num_embeddings  # the number of embed vectors
        self._embedding_dim = embedding_dim  # which means how many discrete symbol for a digit(base n numerical)
        self.scope = scope
        self.gamma = 0.99
        self.start_temp = 0.5
        self.temp_lower_bound = 0.0
        self.temp_decay_rate = 0.005

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
            self._w = tf.get_variable('embedding', [self._embedding_dim, self._num_embeddings],
                                      initializer=initializer, trainable=True)

            self.embedding_total_count = tf.get_variable("embedding_total_count", [1, self._num_embeddings],
                                                         initializer=tf.zeros_initializer(),
                                                         dtype=tf.int32)

            self.sampling_temperature = tf.Variable(self.start_temp, dtype=tf.float32, name="sampling_decay_temp")

    def loop_assign_moving_avg(self, encodings, flat_inputs):
        # print("encodings:", encodings)  # (b,H*W,self._num_embeddings),float32
        embedding_count = tf.reshape(tf.reduce_sum(encodings, axis=[0, 1]), [1, -1])  # [1,1024]

        update_or_not = tf.ceil(embedding_count / tf.reduce_max(embedding_count))
        print("update_or_not:", update_or_not)

        print("self.embedding_total_count:", self.embedding_total_count)

        # embedding_total_count_temp = tf.math.floordiv(
        #     tf.cast(self.embedding_total_count, tf.float32) * (self.gamma) + embedding_count * (1 - self.gamma))
        self.embedding_total_count -= tf.cast(update_or_not * tf.floor(
            (1 - self.gamma) * (tf.cast(self.embedding_total_count, tf.float32) - embedding_count)), tf.int32)

        # self.expand_encodings = tf.expand_dims(encodings, -2)
        # expand_flat_inputs = tf.expand_dims(flat_inputs, -1)

        expand_flat_inputs = tf.transpose(flat_inputs, [0, 2, 1])
        print("expand_flat_inputs:", expand_flat_inputs)

        input_contrib_per_embedding_value = tf.matmul(expand_flat_inputs, encodings)  # (?, 128, 1024)

        input_contrib_per_embedding_value = tf.reduce_sum(input_contrib_per_embedding_value, axis=[0])
        # input_contrib_per_embedding_value = tf.transpose(input_contrib_per_embedding_value,[1,0])
        input_contrib_per_embedding_value = input_contrib_per_embedding_value / tf.cast(self.embedding_total_count,
                                                                                        tf.float32)
        # print("input_contrib_per_embedding_value:", input_contrib_per_embedding_value)

        w_defference = (1 - self.gamma) * (self._w - input_contrib_per_embedding_value) * update_or_not
        # print("w_defference:", w_defference)

        self._w -= w_defference

        return [self._w,
                tf.reduce_mean(tf.reduce_sum(self._w, axis=0)),
                tf.reduce_sum(update_or_not),
                tf.math.top_k(self.embedding_total_count, k=10),
                tf.reduce_sum(embedding_count),
                tf.math.top_k(embedding_count, k=10),
                tf.reduce_mean(tf.reduce_sum(flat_inputs, axis=-1)),
                self.take_num
                ]

    def temperature_decay(self):
        return tf.cond(tf.math.greater_equal(tf.subtract(self.sampling_temperature,self.temp_decay_rate), self.temp_lower_bound),
                       lambda: tf.assign_sub(self.sampling_temperature, self.temp_decay_rate),
                       lambda: self.temp_lower_bound)

    def temperature_sampler(self, distance, temperature):
        '''

        :param distance: (batch,h*w ,k)
        0: argmin
        1: update all
        :param temperature:
        :return: multilhot encoding (batch, h*w, take_num)
        '''

        floor_temp = tf.floor(temperature*100)*100

        self.take_num  = tf.cond(tf.not_equal(floor_temp,0.0),lambda :tf.floor((temperature) * (self._num_embeddings)),lambda :1.0)

        top_k_idx = tf.math.top_k(-distance, k=tf.cast(self.take_num,tf.int32), sorted=False, name=None)[1]
        self.top_k_idx = top_k_idx


        return top_k_idx

    def quantize(self, encoding_indices):
        w = tf.transpose(self._w, [1, 0])
        # trans_idx = tf.transpose(encoding_indices,perm=[2,0,1]) # (b, h*w, top_k_vector) => (h*w, b, top_k_vector)
        # trans_quantize = tf.map_fn(lambda x:tf.reduce_sum(tf.nn.embedding_lookup(w, x, validate_indices=False),axis=-2),trans_idx,dtype=tf.float32) #(b, top_k_vector,_dim) => (b,_dim) => (h*w,b,dim)
        # return tf.transpose(trans_quantize, perm=[1, 0, 2])  # (h*w,b,dim) = > (b,h*w,dim)
        # quantize = tf.map_fn(
        #     lambda x: tf.reduce_sum(tf.nn.embedding_lookup(w, x, validate_indices=False), axis=-2), encoding_indices,
        #     dtype=tf.float32)  # (b, h*w, top_k_vector) => (b)*(h*w, top_k_vector) => (b)*(h*w, top_k_vector,dim) => (b)*(h*w,dim) => (b, h*w,dim)
        quantize = tf.map_fn(
            lambda x: tf.reduce_mean(tf.nn.embedding_lookup(w, x, validate_indices=False), axis=-2), encoding_indices,
            dtype=tf.float32)  # (b, h*w, top_k_vector) => (b)*(h*w, top_k_vector) => (b)*(h*w, top_k_vector,dim) => (b)*(h*w,dim) => (b, h*w,dim)

        return quantize  # (b,h*w,dim)

    def VQVAE_layer(self, inputs):
        # Assert last dimension is same as self._embedding_dim
        print("inputs:", inputs)

        input_shape = tf.shape(inputs)
        with tf.control_dependencies([
            tf.Assert(tf.equal(input_shape[-1], self._embedding_dim),
                      [input_shape])]):
            flat_inputs = tf.reshape(inputs, [-1, input_shape[1] * input_shape[2], self._embedding_dim])
            print("flat_inputs:", flat_inputs)

        self.variable_def()  # set all variable
        self.embedding_total_count += 1

        # the _w is already qunatized: for each row, each idx(latent variable digit) have its own value to pass, value pf _w is quantized embd ouput

        def dist_fn(tensor_apart):
            a2 = tf.reduce_sum(tensor_apart ** 2, 1, keepdims=True)
            b2 = tf.reduce_sum(self._w ** 2, 0, keepdims=True)
            ab = tf.matmul(tensor_apart, self._w)
            # print("tensor_apart:",tensor_apart)
            # print("self._w:",self._w)
            # print("ab:", ab)
            # print("a2:", a2)
            # print("b2:", b2)
            return a2 - 2 * ab + b2

            # dist = (tf.reduce_sum(tensor_apart ** 2, 1, keepdims=True)
            # - 2 * tf.matmul(tensor_apart, self._w)
            # + tf.reduce_sum(self._w ** 2, 0, keepdims=True))  # different shape: tf.add broadcast
            # return dist

        distances = tf.map_fn(dist_fn, flat_inputs)
        print("distances:", distances)


        #####
        ##### Gradient Based update
        #####
        # # distance.shape = [b,H*W,num_embeddings]
        # encoding_indices = tf.argmin(distances,
        #                              2)  # [b,H*W]
        # encodings = tf.one_hot(encoding_indices, self._num_embeddings)
        # quantized_embd_out = self.quantize(
        #     encoding_indices)  # Actually, this quantized method find the value from corespond econding_idx from w
        # print("quantized_embd_out:", quantized_embd_out)
        # print("inputs:", inputs)
        # print("encoding_indices:", encoding_indices)
        #
        #
        # encoding_indices = tf.expand_dims(encoding_indices, axis=-1)
        #
        #
        # quantized_embd_out = self.quantize(
        #     encoding_indices)  # Actually, this quantized method find the value from corespond econding_idx from w
        # print("quantized_embd_out:", quantized_embd_out)
        # quantized_embd_out = tf.reshape(quantized_embd_out, [tf.shape(inputs)[0],
        #                                                      tf.shape(inputs)[1],
        #                                                      tf.shape(inputs)[2],
        #                                                      quantized_embd_out.get_shape().as_list()[
        #                                                          2]])
        #
        # e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized_embd_out) - inputs) ** 2)  # embedding loss
        # q_latent_loss = tf.reduce_mean((tf.stop_gradient(inputs) - quantized_embd_out) ** 2)
        # VQ_loss = e_latent_loss + self.commit_loss_coef * q_latent_loss
        #
        # quantized_embd_out = inputs + tf.stop_gradient(
        #     quantized_embd_out - inputs)  # in order to pass value to decoder???
        # assign_moving_avg_op = self.loop_assign_moving_avg(encodings, flat_inputs)
        # temp_decay_op = self.temperature_decay()
        #
        # return {
        #     'quantized_embd_out': quantized_embd_out,
        #     # "quantized_embd_out": non_max_quantized_embd_out,
        #     'VQ_loss': VQ_loss,
        #     # 'encodings': multi_hot_encodings,
        #     'encodings': encodings,
        #     'encoding_indices': encoding_indices,
        #     'assign_moving_avg_op': assign_moving_avg_op,
        #     'temp_decay_op': temp_decay_op}


        # #####
        # ##### EMA Moving average(argmin)
        # #####
        #
        # # distance.shape = [b,H*W,num_embeddings]
        # encoding_indices = tf.argmin(distances,
        #                              2)  # [b,H*W]
        # encodings = tf.one_hot(encoding_indices, self._num_embeddings)
        # quantized_embd_out = self.quantize(
        #     encoding_indices)  # Actually, this quantized method find the value from corespond econding_idx from w
        # print("quantized_embd_out:", quantized_embd_out)
        # print("inputs:", inputs)
        # print("encoding_indices:", encoding_indices)
        #
        #
        # encoding_indices = tf.expand_dims(encoding_indices, axis=-1)
        #
        #
        # quantized_embd_out = self.quantize(
        #     encoding_indices)  # Actually, this quantized method find the value from corespond econding_idx from w
        # print("quantized_embd_out:", quantized_embd_out)
        # quantized_embd_out = tf.reshape(quantized_embd_out, [tf.shape(inputs)[0],
        #                                                      tf.shape(inputs)[1],
        #                                                      tf.shape(inputs)[2],
        #                                                      quantized_embd_out.get_shape().as_list()[
        #                                                          2]])
        #
        # e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized_embd_out) - inputs) ** 2)  # embedding loss
        # # q_latent_loss = tf.reduce_mean((tf.stop_gradient(inputs) - quantized_embd_out) ** 2)
        # VQ_loss = e_latent_loss
        #
        # quantized_embd_out = inputs + tf.stop_gradient(
        #     quantized_embd_out - inputs)  # in order to pass value to decoder???
        # assign_moving_avg_op = self.loop_assign_moving_avg(encodings, flat_inputs)
        # temp_decay_op = self.temperature_decay()
        #
        # return {
        #     'quantized_embd_out': quantized_embd_out,
        #     # "quantized_embd_out": non_max_quantized_embd_out,
        #     'VQ_loss': VQ_loss,
        #     # 'encodings': multi_hot_encodings,
        #     'encodings': encodings,
        #     'encoding_indices': encoding_indices,
        #     'assign_moving_avg_op': assign_moving_avg_op,
        #     'temp_decay_op': temp_decay_op}




        ####
        #### EMA Moving average(non max)
        ####

        non_max_encoding_indices = self.temperature_sampler(distances, self.sampling_temperature)  # [b,H*W,top_k]
        print("non_max_encoding_indices",non_max_encoding_indices)


        # non_max_encoding_indices = tf.cast(tf.expand_dims(tf.argmin(distances, 2), -1),tf.int32)  # [b,H*W]
        # print("non_max_encoding_indices:",non_max_encoding_indices)



        encoding_indices = tf.expand_dims(tf.argmin(distances,2),-1)  # [b,H*W]
        print("non_max_encoding_indices(argmax)", encoding_indices)
        same_idx =tf.reduce_sum(tf.cast(tf.equal(non_max_encoding_indices,tf.cast(encoding_indices,tf.int32)),tf.float32))



        multi_hot_encodings = tf.map_fn(lambda x: tf.reduce_sum(tf.one_hot(x, self._num_embeddings), axis=-2),
                                        tf.transpose(non_max_encoding_indices, perm=[1, 0, 2]), dtype=tf.float32)
        multi_hot_encodings = tf.transpose(multi_hot_encodings, perm=[1, 0, 2])
        print("multi_hot_encodings:", multi_hot_encodings)

        non_max_quantized_embd_out = self.quantize(non_max_encoding_indices)

        # print("non_max_quantized_embd_out:",    non_max_quantized_embd_out)
        non_max_quantized_embd_out = tf.reshape(non_max_quantized_embd_out, [tf.shape(inputs)[0],
                                                                             tf.shape(inputs)[1],
                                                                             tf.shape(inputs)[2],
                                                                             non_max_quantized_embd_out.get_shape().as_list()[2]])
        # print("non_max_quantized_embd_out:", non_max_quantized_embd_out)


        e_latent_loss = tf.reduce_mean((tf.stop_gradient(non_max_quantized_embd_out) - inputs) ** 2)  # embedding loss
        # q_latent_loss = tf.reduce_mean((tf.stop_gradient(inputs) - non_max_quantized_embd_out) ** 2)
        VQ_loss = e_latent_loss
        non_max_quantized_embd_out = inputs + tf.stop_gradient(
            non_max_quantized_embd_out - inputs)  # in order to pass value to decoder???
        assign_moving_avg_op = self.loop_assign_moving_avg(multi_hot_encodings, flat_inputs)


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
