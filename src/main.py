# ############################################
# Contributed by Meenakshi Sarkar, email meenakshisar@iisc.ac.in for any queries.
# ############################################
# The main.py generates the vgg16 cosine similarity between #n_future of ground truth and predicted frames given by the data_path. Image frames can be of two types
# color and grayscale. 
# The directory ../data/color  has 60 color images (30 ground truth frames given by suffix: gt_ and 30 predicted frames given by suffix: pred_)
# with 3 color channels (RGB in that order, 64*64*3). The directory ../data/grayscale has 60 grayscale images (30 ground truth frames given by suffix: gt_ and 30 predicted frames given by suffix: pred_)
# with 3 color channels (All the 3 channels have the same value since its a grayscale image, 64*64*3)
# ############################################
# input arguments:
#   data_root=  root directory for data, default='../data'.
#   log_dir= path to the folder where you save your plot and .npz file, default='../log'.
#   seed = namually fixing the seed value for reproducibility of the code and values.
#   datatype= color/grayscale, default='color'.
#   n_past= number of past frames after which vgg16 cimilarity index to be calculated for the #n_future of image frames, type=int, default=5.
#   n_future= number of future frames on which we calculate vgg ccosine similarity, type=int, default=10.
#   image_h= Frame height,type=int, dest="image_h",default=64
#   image_w= Frame width,type=int, dest="image_w",default=64
#   gpu = gpu id on server, type=str, default="1". n.d: If you do not have gpu access you can modify this argument and write cpu compatible code.

# You can add any additional input arguments in the main program, but I am going to run your code only using these PREDEFINED SET OF ARGUMENTS during testing.
# I will also test the code with values other than the default values during testing except for seed, image_h, image_w, gpu and data_root.
# ############################################
# output: generates a vgg16_plot.png, and a vgg16_similarity.npz file in the log_dir. vgg16_plot.png shows the plot of vgg16 cosine similarity between the 
#        ground truth frames and predicted frames for the n_future no of frames. for example if n_past= 5 and n_future= 10, then for datatype= 'color',
#        it would evaluate the vgg16 cosine similarity between ground truth and predicted frames, starting with gt_0005 and pred_0005 till gt_0014 and pred_0014
#        (total 10 timesteps in future). These 10 values then need to be plotted in vgg16_plot.png also saved in vgg16_similarity.npz
# ############################################
import argparse
from argparse import ArgumentParser
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from scipy.spatial import distance
import keras
import matplotlib.pyplot as plt
import cv2

    
class VGG16CosineSim():
    def __init__(self):
        self.topTrainable = True
        self.weights = 'imagenet'
        self.pooling = 'max'
        self.input_shape = (224,224,3)
        self.model = keras.applications.VGG16(weights=self.weights, 
                                    include_top=self.topTrainable, 
                                    pooling=self.pooling, 
                                    input_shape=self.input_shape)
        self.baseModel = keras.Model(inputs=self.model.input, 
                                    outputs=self.model.get_layer('fc2').output)

    def getFeature(self,img):
        img1 = cv2.resize(img, (224, 224))
        return self.baseModel.predict(img1.reshape(1, 224, 224, 3))

    def calSim(self,im1, im2):
        return 1 - distance.cosine(im1, im2)
    
def main(args):
    n_past = args.n_past
    n_future = args.n_future
    vgg16_similarity = np.zeros(n_future)
    
    try:
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        print("Running on GPU")
    except Exception:
        config = ConfigProto(device_count = {'GPU':0})
        session = InteractiveSession(config=config)
        print("Running on CPU")

    model = VGG16CosineSim()
    root = args.data_root
    mode = args.datatype
    logs = os.path.abspath(args.log_dir + "/") 
    Labels = []
    
    try:
        for i in range(n_past,n_past+n_future):
            filePath = os.path.abspath(root + "/" + mode)
            gt = cv2.imread(filePath + "/gt_" + ("{:04d}".format(i)) + ".png")
            pred = cv2.imread(filePath + "/pred_" + ("{:04d}".format(i)) + ".png")
            vgg16_similarity[i-n_past] = model.calSim(model.getFeature(pred),model.getFeature(gt))
            Labels.append(("{:04d}".format(i)))
    except Exception:
        print("File Read unsuccessful, error with parameters.")
    
    session.close()
    print(Labels)
    print(vgg16_similarity)
    from tempfile import TemporaryFile
    outfile = TemporaryFile()
    np.savez(outfile, vgg16_similarity)
    
    plt.figure(figsize=(15,10))
    plt.plot(Labels,vgg16_similarity,label="Similarity Values")
    plt.title("VGG16 Cosine Similarity")
    plt.xlabel("Frame Number")
    plt.ylabel("Similarity Values")
    plt.legend()
    
    try:    
        plt.savefig(logs + "/vgg16_plot.png")
        np.save(logs + "/vgg16_similarity.npy",vgg16_similarity)
    except Exception:
        print("Error in identifying path. Saving the plot and file in src folder")
        plt.savefig("vgg16_plot.png") 
        np.save("vgg16_similarity.npy",vgg16_similarity)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='../data', help='root directory for data')
    parser.add_argument('--log_dir', default='../log', help='path to the folder where you save your plot and .npz file')
    parser.add_argument('--seed', default=77, type=int, help='manual seed')
    parser.add_argument('--datatype', default='color', help='color/grayscale')
    parser.add_argument('--n_past', type=int, default=5, help='number of past frames')
    parser.add_argument('--n_future', type=int, default=10, help='number of future frames on which we calculate vgg ccosine similarity')
    parser.add_argument("--image_h", type=int, dest="image_h",default=64, help="Frame height")
    parser.add_argument("--image_w", type=int, dest="image_w",default=64, help="Frame width")
    parser.add_argument('--gpu', type=str, default="1", help='gpu id on server')

    args = parser.parse_args()
    #main
    main(args)

