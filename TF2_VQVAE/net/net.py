import tensorflow as tf
from net.VQ_layer import VQ_layer


class VQVAE(tf.keras.Model):
    def __init__(
        self,
        input_shape,
        latent_dim,
        partition,
        category,
        commit_loss_coef,
        gradient_update=True,
        is_training=False,
    ):

        super().__init__()
        self.is_training = is_training
        self.gradient_update = gradient_update

        self._input_shape = input_shape

        self.enc_1 = tf.keras.layers.Conv2D(
            latent_dim, 2, 2, activation="tanh"
        )  # 28, 28, 1 => 14, 14, 1
        self.enc_2 = tf.keras.layers.Conv2D(
            latent_dim * 2, 2, 2, activation="tanh"
        )  # 14, 14, 1 => 7, 7, 1
        self.dec_1 = tf.keras.layers.Conv2DTranspose(
            latent_dim * 2, 2, 2, activation="tanh"
        )  # 7, 7, 1 => 14, 14, 1
        self.dec_2 = tf.keras.layers.Conv2DTranspose(
            latent_dim, 2, 2, activation="tanh"
        )  # 14, 14, 1 => 28, 28, 1
        self.dec_3 = tf.keras.layers.Conv2D(
            1,
            1,
            1,
        )  # 14, 14, 1 => 28, 28, 1

        self.vq_layer = VQ_layer(
            [-1, 7, 7, latent_dim * 2],
            partition,
            category,
            commit_loss_coef,
            gradient_update=gradient_update,
        )

    def call(self, input):
        x = self.enc_1(input)
        vq_input = self.enc_2(x)

        if self.is_training:
            x, one_hot_encodings = self.vq_layer.train_call(vq_input)

            if self.gradient_update:
                vq_loss = self.vq_layer.loss_computation(vq_input, x)
            else:
                self.vq_layer.loop_assign_moving_avg(vq_input, one_hot_encodings)
                vq_loss = None
            x = self.dec_1(x)
            x = self.dec_2(x)
            x = self.dec_3(x)
            return {"output": x, "vq_loss": vq_loss}
        else:
            x = self.vq_layer(vq_input)
            x = self.dec_1(x)
            x = self.dec_2(x)
            x = self.dec_3(x)
            return {"output": x}
