import tensorflow as tf

from reg_net_config import cfg
# from reg_net_config200MF import cfg


from models.anynet import AnyNet
# from models.effnet import EffNet
from models.regnet import RegNet
# from models.resnet import ResNet


# Supported models
# _models = {"anynet": AnyNet, "effnet": EffNet, "resnet": ResNet, "regnet": RegNet}
_models = {"anynet": AnyNet, "regnet": RegNet}

def build_model():

    base_model =  _models[cfg.MODEL.TYPE]()
    
    # model_input = tf.keras.Input(shape=[256,256], name="model_input",dtype=tf.float32)
    # model_input_gray = tf.expand_dims(model_input,-1)
    # RGB_input = tf.image.grayscale_to_rgb(model_input_gray)
    # model_input = tf.keras.Input(shape=[144,256,3], name="model_input",dtype=tf.float32)
    model_input = tf.keras.Input(shape=[288,512,3], name="model_input",dtype=tf.float32)

    skips = base_model(model_input)
    print("skips[-1]:",skips[-1])
    out = skips[-1]
    model = tf.keras.Model(inputs= [model_input],outputs=[out])
    
    return model


if __name__ == "__main__":
    # define hyperparameters


    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
  

    data_enlarge_scale = 1 # 1920 *n

    epoch_num = 500
    batch_size = 32
    
    class_num = 10
    drop_rt = 0.2
    lr = 1e-4

     
    filters = 8

    
    # tf.compat.v1.disable_eager_execution()

    # define callback and set networ parameters
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    print(gpus)
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    if gpus:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)



    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


    '''
    Do augmentation and build model by functional API 
    '''
    def to_one_hot(int_label):
        one_hot_label = tf.one_hot(int_label, depth = 10)
        return (one_hot_label)

    x_tmp = tf.data.Dataset.from_tensor_slices(train_images)
    label_tmp = tf.data.Dataset.from_tensor_slices(train_labels)
    label_tmp = label_tmp.map(to_one_hot)
    train_dataset = tf.data.Dataset.zip((x_tmp,label_tmp))
    train_dataset = train_dataset.shuffle(buffer_size=128,reshuffle_each_iteration=True).batch(batch_size, drop_remainder=True)
    train_dataset = train_dataset.prefetch(batch_size)



    # Construct the model, loss_fun, and optimizer
    model = build_model()
    model.summary()






    # for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):