import os
import time

import VQVAE_ema_model as model
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

if __name__ == "__main__":

    # use which GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # hyperparameter
    EPOCH = 5000
    STEP = 2000
    time_set = "0704"
    BATCH_SIZE = 32
    LATENT_SIZE = 1024
    LATENT_BASE = 128
    FILTER_NUM = 32
    LR = 1e-4
    TEST_SIZE = 2000
    TEST_IMAGE_PER_RUN = 1
    ATTENTION_HEAD_NUM = 20

    logs_path = "./tf_log/"

    FLAGS = None

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


    # create session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False

    VaeModel = model.MODEL(LR, FILTER_NUM, BATCH_SIZE, LATENT_SIZE, LATENT_BASE, ATTENTION_HEAD_NUM)

    learning_figures = [tf.summary.scalar("loss", VaeModel.loss),
                # tf.summary.scalar("top_VQ_loss", VaeModel.top_VQ_loss),
                # tf.summary.scalar("bottom_VQ_loss", VaeModel.bottom_VQ_loss),
                tf.summary.scalar("reconstruct_loss", tf.reduce_mean(VaeModel.reconstruct_loss)),
                tf.summary.scalar("bottom_VQ_loss", tf.reduce_mean(VaeModel.bottom_VQ_loss)),
                tf.summary.scalar("top_VQ_loss", tf.reduce_mean(VaeModel.top_VQ_loss))
                ]
    merged_summary_op = tf.summary.merge(learning_figures)

    saver = tf.train.Saver(max_to_keep=10)

    with tf.Session(config = config) as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # # keep training
        # model_file = tf.train.latest_checkpoint('./model/')
        # saver.restore(sess, model_file)

        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        update_count = 0

        sess.run(VaeModel.dataset_iter.initializer,
                 feed_dict={VaeModel.ori_x: x_train})

        for e in range(EPOCH):

            for step in range(STEP):
                start = time.time()

                data_img = sess.run(
                    [VaeModel.dataset_prefetch])[0]
                print("np.max(data_img):",np.max(data_img))
                print("np.min(data_img)",np.min(data_img))


                _, train_loss, gray_data, reconstruct_img,reg_recon_out, logpx_z, top_VQ_w, \
                bottom_VQ_w, top_VQ_temp, bottom_VQ_temp,top_k_idx,summary = sess.run(
                    [VaeModel.train_op, VaeModel.loss, VaeModel.gray_data_img,
                     VaeModel.recon_output,VaeModel.reg_recon_out, VaeModel.logpx_z,
                     VaeModel.top_VQ_assign_moving_avg_op,
                     VaeModel.bottom_VQ_assign_moving_avg_op,
                     VaeModel.top_VQ_temp_decay_op,
                     VaeModel.bottom_VQ_temp_decay_op,
                     VaeModel.top_k_idx,merged_summary_op
                     ], feed_dict={VaeModel.data_img_holder: data_img})
                

                print("np.max(reconstruct_img):",np.max(reconstruct_img))
                print("np.min(reconstruct_img)",np.min(reconstruct_img))
                print("np.max(reg_recon_out)",np.max(reg_recon_out))
                print("np.min(reg_recon_out)",np.min(reg_recon_out))

                





                ### Gradient Based update
                # _, train_loss, gray_data, reconstruct_img, logpx_z= sess.run(
                #     [VaeModel.train_op, VaeModel.loss, VaeModel.gray_data_img,
                #      VaeModel.recon_output, VaeModel.logpx_z,
                #      ], feed_dict={VaeModel.data_img_holder: data_img})

                # print("top_VQ_w:",top_VQ_w[0])
                # print("reconstruct_img:",reconstruct_img.shape)

                print("tf.reduce_mean(tf.reduce_sum(flat_inputs,axis=-1)):", top_VQ_w[6])
                print("tf.reduce_sum(update_or_not):", top_VQ_w[2])
                print("self.take_num",top_VQ_w[7])
                print("tf.math.top_k(self.embedding_total_count:", top_VQ_w[3])
                print("tf.reduce_sum(embedding_count):", top_VQ_w[4])
                # print("tf.math.top_k(embedding_count,k=10):", top_VQ_w[5])
                print("top_VQ_temp;", top_VQ_temp)
                print("bottom_VQ_temp:", bottom_VQ_temp)
                print("top_k_idx:",top_k_idx)




                show_img = np.concatenate([data_img,reconstruct_img],axis=1)[:2]
                # print("show_img.shape:", show_img.shape)

                show_img = show_img.reshape([56,-1])
                # print("show_img.shape:", show_img.shape)
                # plot1 = plt.figure(1)
                # plt.clf()
                # plot1.suptitle('EMA_test', fontsize=20)
                # plot1.suptitle("EMA_test ,epoch:{} , step:{} , loss:{}".format(e, step, train_loss), fontsize=10)
                
                
                plt.imsave("./img/" + "epoch_" + str(e) + "_train_step_" + str(step).zfill(5) + ".jpg",show_img)
                # plt.imshow(show_img)
                plt.pause(0.000001)

                # print("cross_entropy:", cross_entropy, "R_cross_entropy:", R_cross_entropy, "G_cross_entropy:",
                #       G_cross_entropy, "B_cross_entropy:", B_cross_entropy, "logpx_z:", logpx_z, "segloss:", segloss,
                #       "top_VQ_loss:", top_VQ_loss, "bottom_VQ_loss:", bottom_VQ_loss)

                runtime = time.time() - start
                print("epoch:{} , step:{} , loss:{}".format(e, step, train_loss))
                summary_writer.add_summary(summary, global_step=update_count)
                update_count += 1
