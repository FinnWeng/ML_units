import os
import time

# import VQVAE_ema_model as model
import matplotlib.pyplot as plt
import matplotlib

# matplotlib.use("qt5agg")
import numpy as np
import tensorflow as tf
from net.net import VQVAE


def preprocess(data_dict):
    """
    img: 28,28,1
    """
    data_dict["data"] = (
        tf.expand_dims(tf.cast(data_dict["data"], tf.float32), axis=-1) / 255
    )
    data_dict["label"] = (
        tf.expand_dims(tf.cast(data_dict["label"], tf.float32), axis=-1) / 255
    )

    return data_dict


def make_dataset(data_array, label_array, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(
        {"data": data_array, "label": label_array}
    )
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(len(data_array)).repeat()
    dataset = dataset.batch(batch_size)

    return dataset


def with_moving_avg_loss_function(y_true, predict_logit):
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=predict_logit, labels=y_true
    )
    return tf.reduce_mean(tf.reduce_sum(cross_ent, axis=[1, 2, 3]))


def gradient_loss_function(y_true, predict_logit, vq_loss):
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=predict_logit, labels=y_true
    )
    return tf.reduce_sum(cross_ent) + tf.reduce_sum(vq_loss)
    return tf.reduce_mean(
        tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        + tf.reduce_sum(vq_loss, axis=[1, 2, 3])
    )


if __name__ == "__main__":

    # use which GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # hyperparameter
    EPOCH = 5000
    STEP = 60000
    time_set = "0704"
    BATCH_SIZE = 32
    LATENT_SIZE = 16
    LATENT_BASE = 128
    FILTER_NUM = 32
    LR = 1e-4
    TEST_SIZE = 2000
    TEST_IMAGE_PER_RUN = 1
    ATTENTION_HEAD_NUM = 20
    total_step = 0

    logs_path = "./tf_log/"
    model_path = "./model/"

    is_training = True

    FLAGS = None

    # VQ hyperparameter
    partition = 2
    category = 33
    commit_loss_coef = 0.25
    gradient_update = False

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    print("x_train.shape:", x_train.shape)  # (60000, 28, 28)
    print("y_train.shape:", y_train.shape)  # (60000,)

    train_dataset = make_dataset(
        x_train, x_train, BATCH_SIZE
    )  # both are x_train since it is reconstruction
    val_dataset = make_dataset(x_test, x_test, BATCH_SIZE)
    train_dataset_prefetch = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    train_iter = train_dataset_prefetch.as_numpy_iterator()
    val_dataset_prefetch = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    val_iter = val_dataset_prefetch.as_numpy_iterator()

    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

    vqvae = VQVAE(
        input_shape=list(x_train.shape)[1:],
        latent_dim=LATENT_SIZE,
        partition=partition,
        category=category,
        commit_loss_coef=commit_loss_coef,
        gradient_update=gradient_update,
        is_training=is_training,
    )

    for e in range(EPOCH):
        for i in range(STEP):
            one_data = next(val_iter)
            one_x = one_data["data"]
            one_y = one_data["label"]

            # print("one_x:", one_x.shape)
            # print("one_y:", one_y.shape)

            with tf.GradientTape() as tape:

                output_dict = vqvae(one_x)
                # print('output_dict["output"]', output_dict["output"])

                if gradient_update:
                    loss = gradient_loss_function(
                        one_y, output_dict["output"], output_dict["vq_loss"]
                    )
                    # print('output_dict["vq_loss"]:', output_dict["vq_loss"])

                else:
                    loss = with_moving_avg_loss_function(one_y, output_dict["output"])

            # print("loss:", loss, "_w:", vqvae.vq_layer._w)
            print("loss:", loss)
            vqvae_var = vqvae.trainable_variables
            vqvae_grads = tape.gradient(loss, vqvae_var)
            optimizer.apply_gradients(zip(vqvae_grads, vqvae_var))

            if total_step % 1000 == 0:
                vqvae.save_weights(model_path)

            if total_step % 20 == 0:

                reconstruct_img = tf.keras.activations.sigmoid(
                    output_dict["output"]
                ).numpy()[0]
                data_img = tf.keras.activations.sigmoid(one_y).numpy()[0]

                show_img = np.concatenate([data_img, reconstruct_img], axis=1) * 255
                show_img = show_img.astype(np.uint8)
                show_img = np.concatenate([show_img, show_img, show_img], axis=-1)
                print("show_img.shape:", show_img.shape)

                # show_img = show_img.reshape([56, -1])
                # print("show_img.shape:", show_img.shape)
                plot1 = plt.figure(1)
                plt.clf()
                plot1.suptitle("EMA_test", fontsize=20)
                plot1.suptitle(
                    "EMA_test ,epoch:{} , step:{} , loss:{}".format(
                        e, total_step, loss.numpy()
                    ),
                    fontsize=10,
                )
                plt.imsave(
                    "./img/"
                    + "epoch_"
                    + str(e)
                    + "_train_step_"
                    + str(total_step).zfill(5)
                    + ".jpg",
                    show_img,
                )
                plt.imshow(show_img)
                plt.show()
                plt.pause(0.000001)

            total_step += 1
