# ML_units

This is some unit test for ML algorithms.

## Contrastive Learning

This is my origin implementation of SimCLR.

The experiment is made on mini-imagenet dataset.(If the task is too easy, there will be no different between using SimCLR and not)

My experiment shows there's difference on 3 layers convolution network, but no difference for restnet20.

This is result of my experiment:

![alt text](https://github.com/FinnWeng/ML_units/blob/master/Contrastive_Learning/common/tf_log_image/3layer_conv.PNG "3 layer conv result")
![alt text](https://github.com/FinnWeng/ML_units/blob/master/Contrastive_Learning/common/tf_log_image/resnet20.PNG "resnet20 result")




## Coord Conv 

make the nerual network to know coordinate by channel of up-down and left-right.


## VQVAE
The easiest way to implement discrete latent variable. I know it by the paper of model-based learning paper. 

For updating the codebook, there're two different approach: one is normal gradient update; another one is Exponential Moving Average. The implementation of EMA is pretty lousy. But the result is astonishing: it convergs much faster since it wouldn't affect by bad gradient of decoder and encoder. 

I adapt the decomposed trick for ease the index collapse problem. Decomposed trick also accelerate converge. 

This is result of my experiment:

![alt text](https://github.com/FinnWeng/ML_units/blob/master/VQVAE/result_images/VQVAE_mnist.gif "Gradient V.S. EMA")





