import numpy as np
import matplotlib.pyplot as plt
import os

grad_img_list = os.listdir("../Gradient_experiment/img/")
grad_img_list.sort()

count = 0

for i in grad_img_list[:900]:
    grad_img = plt.imread("../Gradient_experiment/img/"+i)
    ema_img = plt.imread("./img/"+i)
    grad_img_without = plt.imread("../Gradient_experiment_without_decompose/img/"+i)
    ema_img_without = plt.imread("../EMA_slice_vector_without_decoposed/img/"+i)
    
    print(i)
    
    
    
    fig =plt.figure()
    plt.clf()
    fig.suptitle("step:{}".format(count), fontsize=10)
    fig.add_subplot(2,2,1).set_title("decomposed_Gradient")
    plt.imshow(grad_img)
    fig.add_subplot(2,2,2).set_title("decomposed_EMA")
    plt.imshow(ema_img)
    
    fig.add_subplot(2,2,3).set_title("Gradient")
    plt.imshow(grad_img_without)
    
    fig.add_subplot(2,2,4).set_title("EMA")
    plt.imshow(ema_img_without)
    
    # plt.show()
    # plt.pause(0.000001)
    
    plt.savefig("./merge_img/merge_img{}".format(count))
    count+=1