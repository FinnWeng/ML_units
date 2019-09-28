# ML_units

This is some unit test for ML algorithms.

## Coord Conv 

make the nerual network to know coordinate by channel of up-down and left-right.


## VQVAE
The easiest way to implement discrete latent variable. I know it by the paper of model-based learning paper. 

For updating the codebook, there're two different approach: one is normal gradient update; another one is Exponential Moving Average. The implementation of EMA is pretty lousy. But the result sometimes is astonishing: it convergs much faster since it wouldn't affect by bad gradient of decoder and encoder. 

I adapt the decomposed trick for ease the index collapse problem. 

This is result of my experiment:

![alt text](https://github.com/FinnWeng/ML_units/blob/master/VQVAE/result_images/VQVAE_mnist.gif "Gradient V.S. EMA")


