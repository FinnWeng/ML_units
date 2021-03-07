import numpy as np
import tensorflow as tf

__all__ = ["MPCNetV1p2", "MPCNetV2"]


class VQ_layer(tf.keras.Model):
    def __init__(
        self,
        vq_input_shape,
        partition,
        category,
        commit_loss_coef,
        gradient_update,
        gamma=0.99,
    ):
        """
        input ==slicing==> vector slicing input
        vector slicing input, _w ==l2 norm compute==> distance
        distance ==sampler==> indices
        indices ==onehot==> onehot_indices
        onehot_indices ==quantize==> embd_out

        assign to _w
        """

        super().__init__()

        self.gradient_update = gradient_update
        self.partition = partition  # for vector slicing
        self.part_k = category  # to how many categories
        self.commit_loss_coef = commit_loss_coef
        self.gamma = gamma
        self.vq_input_shape = vq_input_shape
        self.part_embd_dim = int(self.vq_input_shape[-1] / self.partition)
        print("self.part_embd_dim:", self.part_embd_dim)

        w_init = tf.keras.initializers.GlorotUniform()(
            shape=[self.partition, self.part_k, self.part_embd_dim]
        )
        if self.gradient_update:
            self._w = tf.Variable(
                w_init, trainable=True, name="embedding"
            )  # the code book
        else:
            self._w = tf.Variable(
                w_init, trainable=False, name="embedding"
            )  # the code book

        # print("self._w:", self._w.shape)
        embedding_total_count_init = tf.keras.initializers.Ones()(
            [1, self.partition, self.part_k]
        )  # start from one to avoid embd contribute devided by zero.
        self.embedding_total_count = tf.Variable(
            embedding_total_count_init, trainable=False, name="embedding_total_count"
        )

        # self.VQ_layer = self.get_model()

    def call(self, VQ_input):
        # print("VQ_input.shape:", VQ_input.shape)
        sliced_inputs = self.input_slicing(VQ_input)
        distances = self.l2_norm_computation(sliced_inputs, self._w)
        encoding_indices = self.sampler(distances)

        # # print("encoding_indices:", encoding_indices)  # (b, h*w, partition)
        # one_hot_encodings = tf.one_hot(encoding_indices, self.part_k, axis=-1)
        # # print("one_hot_encodings:", one_hot_encodings)  # (b, h*w, partition, part_k)

        quantized_embd_out = self.quantize(encoding_indices)  # (b, h*w, 8, 32)

        # print("quantized_embd_out:", quantized_embd_out)
        # print("VQ_input.shape[1:]:", VQ_input.shape[1:])
        quantized_embd_out = tf.keras.layers.Reshape(VQ_input.shape[1:])(
            quantized_embd_out
        )  # reverse partition # here I use layers.Reshape to reshape tensor with axis 0 shape is None.
        # print("quantized_embd_out:", quantized_embd_out)

        quantized_embd_out = VQ_input + tf.stop_gradient(
            quantized_embd_out - VQ_input
        )  # in order to pass value to decoder???
        # print("quantized_embd_out:", quantized_embd_out)

        return quantized_embd_out

    def input_slicing(self, inputs):
        # print("inputs:", inputs.shape)
        flat_inputs = tf.reshape(
            inputs,
            [
                -1,
                inputs.shape[1] * inputs.shape[2],
                self.partition,
                self.part_embd_dim,
            ],
        )
        # inputs_flat = [batch_size, latent_dim, hidden_size]
        # x_shape = [batch_size, latent_dim, num_blocks,block_dim]
        return flat_inputs

    def l2_norm_computation(self, sliced_inputs, _w):
        # x = tf.reshape(
        #     sliced_inputs, [-1, self.partition, self.part_embd_dim]
        # )  # batch*h*w ,self.partition ,self.part_embd_dim
        x = sliced_inputs  # batch, h*w ,self.partition , self.part_embd_dim
        x = tf.expand_dims(x, -2)  # batch, h*w ,self.partition , 1, self.part_embd_dim
        # print("x:", x.shape)
        reshape_w = tf.reshape(
            self._w, [1, 1, self.partition, self.part_k, self.part_embd_dim]
        )
        # print("reshape_w:", reshape_w.shape)
        diff = x - reshape_w
        # print("diff:", diff.shape)  # (None, h*w, partition, part_k, part_embd_dim)
        distances = tf.norm(diff, axis=-1)
        # print("distances:", distances.shape)  # (None, h*w, partition, part_k,1)

        return distances

    def sampler(self, distances):
        """

        :param distance: (batch,h*w ,k)
        0: argmin
        1: update all
        :param temperature:
        :return: multilhot encoding (batch, h*w, take_num)
        """

        top_1_idx = tf.argmin(distances, -1)  # (None, h*w, partition)
        return top_1_idx

    def quantize(self, encoding_indices):

        # self._w : [self.partition, self.part_k, self.part_embd_dim]

        # encoding_indices: [b, H*W,partition,1](None, h*w, partition)

        trans_idx = tf.transpose(encoding_indices, [2, 0, 1])
        # print("trans_idx:", trans_idx)  # (partition, b, h*w)

        quantize = tf.stack(
            [
                tf.nn.embedding_lookup(self._w[i, :, :], trans_idx[i, :, :])
                for i in range(self.partition)
            ],
            axis=0,
        )
        # for each partition, serarch through partition code book(partk, part_embd) by partition ids(1, b, h*w)
        # print("quantize:", quantize)  # (partition, b, h*w, part_embd)
        quantize = tf.transpose(quantize, [1, 2, 0, 3])
        # print("quantize:", quantize)  # # (b ,h*w, partition, part_embd)

        return quantize

    """
    choose gradient update or moving avg updata
    """

    def loss_computation(self, VQ_input, quantized_embd_out):
        e_latent_loss = tf.reduce_mean(
            (tf.stop_gradient(quantized_embd_out) - VQ_input) ** 2
        )  # embedding loss
        q_latent_loss = tf.reduce_mean(
            (tf.stop_gradient(VQ_input) - quantized_embd_out) ** 2
        )
        # VQ_loss = self.commit_loss_coef*e_latent_loss
        VQ_loss = self.commit_loss_coef * e_latent_loss + 1 / 1000 * q_latent_loss

        return VQ_loss

    def loop_assign_moving_avg(self, VQ_input, encodings):
        print("encodings:", encodings.shape)  # (b, h*w, partition, part_k)

        sliced_inputs = self.input_slicing(VQ_input)  # (b, h*w, partition, embd)
        # print("sliced_inputs:", sliced_inputs.shape)

        embedding_count = tf.reduce_sum(
            encodings, axis=[0, 1]
        )  # [b, h*w,partion, part_k] = > [partion, part_k]
        print("embedding_count:", embedding_count.shape)
        embedding_count = tf.expand_dims(
            embedding_count, 0
        )  # [partion, part_k] = > [1 , partion, part_k]

        print("embedding_count:", embedding_count.shape)

        update_or_not = tf.clip_by_value(
            embedding_count, clip_value_min=0, clip_value_max=1
        )
        # update_or_not = tf.ceil(embedding_count / tf.reduce_max(embedding_count)) # to one hot  [1, partion * part_k]
        print("update_or_not:", update_or_not.shape)  # [1 , partion, part_k]

        print(
            "self.embedding_total_count:", self.embedding_total_count.shape
        )  #  [1, partion , part_k]

        input_contrib_per_embedding_value = tf.linalg.matmul(
            tf.expand_dims(sliced_inputs, -1), tf.expand_dims(encodings, axis=-2)
        )  # (b, h*w,self.partition ,self.part_embd_dim,1)*  [b, h*w, partion, 1,part_k] => (b, h*w,self.partition ,self.part_embd_dim,part_K)
        print(
            "input_contrib_per_embedding_value:",
            input_contrib_per_embedding_value.shape,
        )
        input_contrib_per_embedding_value = tf.reduce_sum(
            input_contrib_per_embedding_value, axis=[0, 1]
        )  # (self.partition ,self.part_embd_dim,part_K)

        print(
            "input_contrib_per_embedding_value:",
            input_contrib_per_embedding_value.shape,
        )

        self.embedding_total_count.assign_sub(
            (1 - self.gamma)
            * (self.embedding_total_count - tf.cast(embedding_count, tf.float32))
        )
        print("self.embedding_total_count:", self.embedding_total_count.shape)

        input_contrib_per_embedding_value = input_contrib_per_embedding_value / tf.cast(
            tf.transpose(self.embedding_total_count, [1, 0, 2]),
            tf.float32,
        )  # Moving average, (self.partition ,self.part_embd_dim,part_K) /[partion ,1, part_k]
        print(
            "input_contrib_per_embedding_value:",
            input_contrib_per_embedding_value.shape,
        )  # [self.partition ,self.part_embd_dim,part_K]
        input_contrib_per_embedding_value = tf.transpose(
            input_contrib_per_embedding_value, [0, 2, 1]
        )  # [self.partition,self.part_k,self.part_embd_dim] same with w
        print(
            "input_contrib_per_embedding_value:",
            input_contrib_per_embedding_value.shape,
        )

        #
        contrib = (1 - self.gamma) * (
            input_contrib_per_embedding_value
        )  # [self.partition,self.part_k,self.part_embd_dim] *[self.partition,self.part_k,1]
        # contrib = (1 - self.gamma) * (input_contrib_per_embedding_value) * tf.reshape(update_or_not,[self.partition,self.part_k,1])# [self.partition,self.part_k,self.part_embd_dim] *[self.partition,self.part_k,1]

        self._w = self._w * self.gamma + contrib

        return [
            self._w,
            tf.reduce_mean(self._w),  # to see if self_w all decrease to zero
            tf.reduce_sum(update_or_not),
            tf.math.top_k(self.embedding_total_count, k=10),
            tf.reduce_sum(embedding_count),
            tf.math.top_k(embedding_count, k=10),
        ]

    def train_call(self, VQ_input):
        # print("VQ_input.shape:", VQ_input.shape)
        sliced_inputs = self.input_slicing(VQ_input)
        # print("sliced_inputs:", sliced_inputs.shape)
        distances = self.l2_norm_computation(sliced_inputs, self._w)
        encoding_indices = self.sampler(distances)

        # print("encoding_indices:", encoding_indices.shape)  # (b, h*w, partition)
        if self.gradient_update:
            one_hot_encodings = None
        else:
            one_hot_encodings = tf.one_hot(encoding_indices, self.part_k, axis=-1)
            # print(
            #     "one_hot_encodings:", one_hot_encodings.shape
            # )  # (b, h*w, partition, part_k)

        quantized_embd_out = self.quantize(encoding_indices)  # (b, h*w, 8, 32)

        # print("quantized_embd_out:", quantized_embd_out.shape)
        # print("VQ_input.shape[1:]:", VQ_input.shape[1:])
        quantized_embd_out = tf.keras.layers.Reshape(VQ_input.shape[1:])(
            quantized_embd_out
        )  # reverse partition # here I use layers.Reshape to reshape tensor with axis 0 shape is None.
        # print("quantized_embd_out:", quantized_embd_out.shape)

        # quantized_embd_out = VQ_input + tf.stop_gradient(
        #     quantized_embd_out - VQ_input
        # )  # in order to pass value to decoder???
        quantized_embd_out = VQ_input + (
            quantized_embd_out - VQ_input
        )  # in order to pass value to decoder???
        # print("quantized_embd_out:", quantized_embd_out.shape)

        return quantized_embd_out, one_hot_encodings
