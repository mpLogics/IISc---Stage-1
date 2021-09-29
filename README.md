# Coding Assigment for the 2021 summer internship @GCDSL
#### Contributed by Meenakshi Sarkar, email meenakshisar@iisc.ac.in for any queries.

## Objective
  The main.py generates the vgg16 cosine similarity between #n_future of ground truth and predicted frames given by the data_path. Image frames can be  of two types **color** and **grayscale**. \
 The directory *../data/color*  has 60 color images (30 ground truth frames given by suffix: gt_ and 30 predicted frames given by suffix: pred_)
with 3 color channels (RGB in that order, [64,64,3]). The directory *../data/grayscale* has 60 grayscale images (30 ground truth frames given by suffix: gt_ and 30 predicted frames given by suffix: pred_) with 3 color channels (All the 3 channels have the same value since its a grayscale image,[64,64,3])

## input arguments:
  **data_root**=  root directory for data, default='../data'.\
  **log_dir**= path to the folder where you save your plot and .npz file, default='../log'.\
  **seed**= namually fixing the seed value for reproducibility of the code and values.\
  **datatype**= color/grayscale, default='color'.\
  **n_past**= number of past frames after which vgg16 cimilarity index to be calculated for the #n_future of image frames, type=int, default=5.\
  **n_future**= number of future frames on which we calculate vgg ccosine similarity, type=int, default=10.\
  **image_h**= Frame height,type=int, dest="image_h",default=64\
  **image_w**= Frame width,type=int, dest="image_w",default=64\
  **gpu**= gpu id on server, type=str, default="1". n.d: If you do not have gpu access you can modify this argument and write cpu compatible code.

You can add any additional input arguments in the main program, but I am going to run your code only using these **PREDEFINED SET OF ARGUMENTS** during testing.\
I will also test the code with values other than the default values during testing excpet for seed, image_h, image_w, gpu and data_root.

## output:
  generates a ***vgg16_plot.png***, and a ***vgg16_similarity.npz*** file in the log_dir. vgg16_plot.png shows the plot of vgg16 cosine similarity between the    ground truth frames and predicted frames for the n_future no of frames. for example if n_past= 5 and n_future= 10, then for datatype= 'color', it would evaluate the vgg16 cosine similarity between ground truth and predicted frames, starting with gt_0005 and pred_0005 till gt_0014 and pred_0014 (total 10 timesteps in future). These 10 values then need to be plotted in *vgg16_plot.png* also saved in *vgg16_similarity.npz*
  
  ## Test time code execution command from src folder
  
 ```
 python main.py --log_dir path/to/log_dir --datatype grayscale --n_past 10 --n_future 20 
 ```


