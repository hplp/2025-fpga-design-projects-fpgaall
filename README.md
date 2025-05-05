
## Team Members:
- Student 1: Jazzmin Poitier
- Student 2: Ndjiakou Kaptue Maiva

## Project Title: Comparative Performance of Neural Network-Based Image Classification on FPGA and Software Applications


## Project Description:

The application used for this project is Jupyter, a popular open-source interactive computing environment that enables the development and execution of Python-based machine learning models. In this project, we will use Jupyter to train a neural network model to classify lung cancer images. These images will be preprocessed and labeled before being fed into the model for training. The trained model will then be evaluated by running the classification task both on the PYNQ Z1 FPGA board and within the Jupyter environment on a traditional software platform. The PYNQ Z1 FPGA board, equipped with specialized hardware for parallel processing, will be leveraged to accelerate the inference process, while Jupyter will serve as the software-based approach for classification, running on a general-purpose CPU. The methodology involves training the neural network model in Python using well-established deep learning libraries, such as PyTorch, and then evaluating the performance of the classification on both platforms. Key metrics, such as the classification time and accuracy, will be recorded and compared. Additionally, the limitations of the PYNQ Z1 FPGA board will be examined, particularly focusing on the maximum image size it can handle before performance starts to degrade. This methodology will provide a comprehensive comparison of the strengths and weaknesses of both FPGA and software
solutions in real-time medical image analysis, ultimately guiding the best approach for scalable and efficient tumor detection systems.

## Objectives and Motivation:
The motivation behind this project is driven by the need for fast and efficient image classification in critical applications like medical diagnostics, where timely and accurate analysis of lung cancer images can be life-saving. 
-  Within 06 weeks, compare the performance (classification time) of FPGA and software platforms in classifying lung cancer images using the same dataset and algorithm, with a focus on measuring processing speed and efficiency.
- Investigate and document the largest image size the PYNQ Z1 board can handle without significant performance degradation, by testing and measuring processing time at different image sizes, to be completed during 06 weeks.
- Within 06 weeks, analyze the trade-offs between FPGA and software solutions in terms of speed, power, and scalability for real-time medical image analysis.
- Built a neural network for lung cancer image classification with an accurancy of more than   70%
  

##  Technology Stack:
(List the hardware platform, software tools, language(s), etc. you plan to use)
a. Hardware Technologies:
   . PYNQ Z1 Board
b. Programming Languages:
   . Python
   . VHDL/Verilog
c. Software Libraries:
   . OpenCV( for image processing)
   . TensorFlow/Pytorch(for machine learning)
   . PYNQ Library( for interfacing with the FPGA)
   . Jupyter Notebook
d. Data Analysis Tools:
   . NumPy/Pandas ( for data handling)
   . Matplotlib/Seaborn (for visualization)
e.    Resources:

# Description of the differents steps of the project 
the following software must be install vivado 2018.2, python 2.7( in the machine or install it in the virtual environment), ubuntu 2018+ 
# I. Board setup

1. connect the Board
We connected the PYNQ-Z1 board to our PC and retrieved its IP address to access it via a web browser.
![image](https://github.com/user-attachments/assets/da592bf3-43d6-4cb0-ae59-3fb2a39c9f9c) ![image](https://github.com/user-attachments/assets/7340f3e2-fbe1-479f-9dc6-aa460b602b57)

Set the JP4 / Boot jumper to the SD position by placing the jumper over the top two pins of JP4 as shown in the image. (This sets the board to boot from the Micro-SD card)
To power the PYNQ-Z1 from the micro USB cable, set the JP5 / Power jumper to the USB position.

a.Insert the Micro SD card loaded with the PYNQ-Z1 image into the Micro SD card slot underneath the board.

b.Connect the USB cable to your PC/Laptop, and to the PROG - UART / J14 MicroUSB port on the board

c.Connect the board to Ethernet by following the instructions below

d.Turn on the PYNQ-Z1 and check the boot sequence:
 
 -The Red LD13 LED will come on immediately to confirm that the board has power. After a few seconds, the Yellow/Green LD12 / Done LED will light up to show that the Zynq® device is operational.

 - After a minute you should see two Blue LD4 & LD5 LEDs and four Yellow/Green LD0-LD3 LEDs flash simultaneously.
   
 - The Blue LD4-LD5 LEDs will then turn on and off while the Yellow/Green LD0-LD3 LEDs remain on. The system is now booted and ready for use. Network connection
 
- Once your board is setup, you need to connect to it to start using Jupyter notebook.
  
- Find the COM port for your virtual serial interface. Open Device Manager and find the COM port number (in my case it is COM20).

2. Get the Ip address of the board and access the board online
   
Verify that the board has been registered, then open the application mobaxterm to obtain the IP address of the board.

a. Open device manager in microsft and search for USB connection, to find the COM where the board is connected in my case it is COM20
![image (3)](https://github.com/user-attachments/assets/d36e45ea-3e72-41cf-a651-625379582342)

b. In microsoft start icon, search the applicatiob MobaXterm if installed in your computer, open Mobaxterm . In the application follow the steps below to get the IP address of the board 
-click new session & serial connection
-define the parameters :
![image (4)](https://github.com/user-attachments/assets/f5a2116b-1929-4160-a28c-0dcb89c0ba07)
- in the terminal type : ifconfig -a
-The IP address of the board will appear
![image (5)](https://github.com/user-attachments/assets/a2d4de3c-1d96-4691-81a7-f9b7403d80f1)
-open a terminal and type : https://IP address

If ask a password type : xilinx
![image (6)](https://github.com/user-attachments/assets/0bd77414-fa89-4882-9823-8e223e3e00a1)

3. verify the pynq image present inside the SD card of the pynq board
   
The version of the pynq image used for this project is 2.5. In case you need to change the pynq image follow the instructions below:

a.verify the pynq image in the SD

open a jupyter notebook and type the command line :

import pinq

print (pynq.__version__)

![aaa](https://github.com/user-attachments/assets/2b8ef6cc-bfc1-47be-a021-c77c630574cf)

b. To install a new pinq image

- go to the link and follow the step to dowload a bootable pink image v 2.5, : https://pynq.readthedocs.io/en/latest/appendix/sdcard.html
  
- Follow the steps and install the pinq image inside the SD card
  
  ![image](https://github.com/user-attachments/assets/7175c02c-fa32-4195-b8ee-5ac86aadd472)
  ![image](https://github.com/user-attachments/assets/baf03734-94d8-4caf-bbac-ec5998b6b549)
  
# II Architecture selection and data preparation

1.  Install the BNN inside the jupyter notebook
   
In a terminal inside our jupyter notebook  type the folowing command line 

sudo pip3 install git+https://github.com/Xilinx/BNN-PYNQ.git (on PYNQ v2.3 and later versions, tested up to v2.5)

sudo pip3.6 install git+https://github.com/Xilinx/BNN-PYNQ.git (on PYNQ v2.2 and earlier)

In this directory we have several notebook . These notebooks use one of the two overlay ( an overlay is a  virtual, re-configurable architectures that sit on top of physical FPGA fabrics). It is specialized for an application, or a class of applications, offers both fast reconfiguration and minimized performance penalty.
the example notebook, the bnn package is first imported. The bnn.py file contains a detailed description of bnn, which includes the FPGA driver programming. In bnn.py, it mainly contains the definition of three classes: PynqBNN, CnvClassifier, LfcClassifier. Among them, PynqBNN is mainly used as a shared library to load the specified network and download the bitstream to the FPGA (PL); CnvClassifier is the classifier class of the CNV network, used to infer the image of the cifar10 format and the image that requires preprocessing; LfcClassifier is LFC The classifier class of the network is used to infer images in mnist format. When the two classes are constructed, they will load the shared library and download the bitstream to the FPGA (PL).

![image](https://github.com/user-attachments/assets/19c89749-a668-4151-b460-32ac1afe68fe) ![image](https://github.com/user-attachments/assets/d0f08500-e063-4d96-86ea-831287f14977)

![Capture d'écran 2025-04-24 204158](https://github.com/user-attachments/assets/c76a162a-88c7-45b2-9ce9-b81b575882f5)

2. Architecture selection
   
   Once our development environment was set up and the board was running PYNQ v2.5, the next major step was to evaluate which neural network architecture would be best suited for our lung cancer classification task. The BNN-PYNQ framework comes with two primary overlay types: CNV, which is a convolutional network designed for RGB images sized 32 by 32, and LFC, which is a fully connected network typically used for grayscale images sized 28 by 28.
   
We started by running several prebuilt tutorials provided in the BNN-PYNQ repository to test both overlays—CNV and LFC—on the provided CIFAR-10 dataset. Our objective was to compare both hardware and software executions based on two key metrics: classification accuracy and inference time.

For the CNV overlay, we evaluated different quantization modes including W1A1, W1A2, and W2A2. The best result on hardware was about 84.29% accuracy with W2A2, while the software equivalent achieved similar performance but took significantly longer—in our test, the inference time was around 700 seconds for the full batch.
For the LFC overlay, we saw even higher accuracy on the CIFAR dataset, with the hardware W2A2 configuration reaching 98.4%, and the software classifier scoring around 90%. However, since LFC is optimized for grayscale and our lung cancer images are in RGB, the CNV structure was more aligned with our data format.
After comparing both options, we decided to proceed with the CNV architecture, as it offered a good trade-off between accuracy, speed, and compatibility with our dataset format. This selection set the foundation for the next step—adapting our Kaggle lung cancer dataset for FPGA-based classification.
![image](https://github.com/user-attachments/assets/d3de7c9e-b8bf-4d40-800a-e0c13aaf0c37)
![image](https://github.com/user-attachments/assets/450ba9c1-1364-47f5-ae33-f38ad6ad09db) ![image](https://github.com/user-attachments/assets/b62b5937-a591-45f5-b607-f61ade26484c)

3. Data preparation
Our dataset was downloaded from Kaggle and required preprocessing before it could be used with the BNN overlay.
The preprocessing included:

a.  Writing Python scripts to reorganize the dataset into two labeled classes: benign and malignant.

![image](https://github.com/user-attachments/assets/82d60f9f-d9f1-4aca-b5ca-56ef0cd4b936)

b. Resizing all image data to compatible dimensions (32x32), to facilitate smoother integration with the FPGA overlay and ensure uniformity in training and inference.

![image](https://github.com/user-attachments/assets/51a4a096-32a6-4053-a9f4-8e2d70b3aa37)
![image](https://github.com/user-attachments/assets/9b1d43ee-82fe-459a-ae4c-71fd939b5390)


# III.  Train the software for image classification

1. Install the BNN locally in our computer
   
In a terminal ( Ubuntu shell)  type the folowing command line to clone the repository inside your computer and upload the dataset that have been preprocessed and put it in the directory 
sudo pip3 install git+https://github.com/Xilinx/BNN-PYNQ.git (on PYNQ v2.3 and later versions, tested up to v2.5)
sudo pip3.6 install git+https://github.com/Xilinx/BNN-PYNQ.git (on PYNQ v2.2 and earlier)

To begin the hardware acceleration phase of our lung cancer classification project, I worked within the BNN-PYNQ repository, which provides pre-built binaries and scripts for deploying quantized neural networks on the PYNQ Z1 board. The first step involved adapting the existing cifar10.py file. Originally, this script is tailored for classifying images from the CIFAR-10 dataset. However, since my project focuses on binary classification—distinguishing between benign and malignant lung cancer—I modified this script to handle two classes instead of ten. I updated the preprocessing pipeline to resize our medical images to 32x32 pixels, which is required for compatibility with the underlying BNN architecture.
![image](https://github.com/user-attachments/assets/8f991d28-3e69-48d1-81ed-1b05d2d2a6f4)

2.  create a python virtual environment
   
a. open the ubuntu terminal and type the command line : python -m venv pynq_bnn_env

b.Activate the virtual environment  with the command line:  source ~/pynq_bnn_env/bin/activate

3. Train  the model locally in your machine
   
a. Inside the virtual environment, enter into your training file with the command line : cd /path/to/your/python/file

b. In the training file open the cifar.py file and modify it with the code you have written for your dataset use the following command:

 - to open the cifar. py file : nano cifar.py
   
 - to save the modification you made :  press Ctrl + O, then press Enter.
   
 - To exit: press Ctrl + X.
   
c. type the command line : python cifar10.py to start the training of your model
![image](https://github.com/user-attachments/assets/8b35ffef-1e24-4fae-92c9-fa4dce24b765)

# IV. Hardware for image classification 

1. generate the weights files
   
The training command executed the quantized training or inference process, depending on how I configured the script. It leverages Theano and FINN-style BNN layers to generate intermediate feature outputs in a format optimized for deployment on the FPGA.
While running the code, I used:

bash

CopyEdit

ls *rz

to inspect the directory and confirm that the .npz file was generated. This .npz file is a compressed NumPy archive containing the trained network parameters, such as weights and biases. These are crucial because they are later used by the bitstream generation and inference engine on the PYNQ board.
Why is this important? Because this compressed file bridges the training environment and the FPGA runtime. Without it, the model could not be ported or run on the actual board.
![image](https://github.com/user-attachments/assets/647a490d-4db7-481e-b139-6271c50d035b)

2. Convert the training model into binary format

However, these .npz files are not directly usable by the FPGA overlay. To bridge this gap, we used a tool called FINNthesizer, provided within the same framework. FINNthesizer is responsible for converting the high-level model into a low-level representation that the hardware overlay can understand.
We ran the FINNthesizer script with the following command:

css

CopyEdit

python finnthesizer.py --network cnv --dataset lung_cancer
![image](https://github.com/user-attachments/assets/c254763d-775e-4442-b539-11ed6ef15d1d)

3. Generating the FPGA bitsream
   
After successfully converting our trained model into binary format using FINNthesizer, we moved to the final and most crucial step of the hardware implementation: generating the FPGA bitstream. This step is required to configure the programmable logic on the PYNQ-Z1 board so that it can execute our custom-trained BNN model.
To perform this, the make-hw script provided by the BNN-PYNQ framework must call Vivado High-Level Synthesis, also known as Vivado HLS. Vivado is the software tool from Xilinx that translates C++ or HLS-compatible models into synthesizable Verilog or VHDL and eventually into a bitstream (.bit file) that configures the FPGA hardware.
However, when we initially ran the hardware generation command:

css

CopyEdit

make-hw cnv-pynq pynqZ1-Z2 all
![image](https://github.com/user-attachments/assets/4e55d61c-ab86-4e7f-a395-a84a8866ec81)

# V. Challenges Faced 

BNN-PYNQ supports only CIFAR-10 model 

Incompatibility with custom lung dataset 

Python 2.7 limitation on PYNQ prevented usage of updated frameworks 

FINN installation failed (Docker versioning, dependencies) 

DNNWeaver raised Python errors in Jupyter environment 

Constrained memory on PYNQ Z1 limited full CNN deployment 

# VI.  Results & Evaluation 

Software  accuracy: ~ 71% 

Hardware feature extraction: Successful but limited to CIFAR-10 model 

Classification time: Faster with hardware preprocessing 

FPGA usage: Within resource limits using binary weights, no full CNN due to complexity 

# VII.   Conclusion 

This project demonstrates a hybrid classification architecture that balances software accuracy and hardware acceleration. Although a full CNN deployment was not feasible on the PYNQ Z1, the BNN-PYNQ model effectively served as a feature extractor. Software classification maintained high accuracy. Exploration of DNNWeaver and FINN highlighted current toolchain limitations. 

# Future Work 

Custom training of BNN with lung dataset 

Successful deployment of FINN with Docker 

Upgrade to more capable FPGA boards 

Use of quantization-aware training for better hardware mapping 

Deployment of lightweight CNN variants (e.g., MobileNet, TinyML models)

# Citation 

@inproceedings{finn,
author = {Umuroglu, Yaman and Fraser, Nicholas J. and Gambardella, Giulio and Blott, Michaela and Leong, Philip and Jahre, Magnus and Vissers, Kees},
title = {FINN: A Framework for Fast, Scalable Binarized Neural Network Inference},
booktitle = {Proceedings of the 2017 ACM/SIGDA International Symposium on Field-Programmable Gate Arrays},
series = {FPGA '17},
year = {2017},
pages = {65--74},
publisher = {ACM}
}

# Repo organization
The repo is organized as follows:
- Report phase 2 : outline the differents steps during the phase 2 of the project
  
- Report phase 2.2 error and Report phase 2.1 error : describe some of the errors we have faced during the project and the actions we have taken to solve them
  
-Lung_data_set  contains the Lung cancer image resized( 32*32) and reorganized ( 2 class: benign and malign ) fromm Kaggle website

-BNN_PYNQ: 

  * bnn: contains the LfcClassifier and CnvClassifier python class description

  * src: contains the sources of the different precision networks, the libraries to rebuild them, and scripts to train and pack the weights:

  * library: FINN library for HLS QNN descriptions, host code, script to rebuilt and drivers for the PYNQ and Ultra96 (please refer to README for more details)

  * network: HLS top functions for QNN topologies (CNV and LFC) with different implementations for weight and activation precision, host code and make script for HW and SW build (please refer to README for more details)

  * training: scripts to train on the Cifar10 that has been modified to classify our lung cancer dataset 

  * pynqZ1-Z2: bitstreams for Pynq devices

  * ultra96: bitstreams for Ultra96 devices

  * libraries: pre-compiled shared objects for low-level driver of the 5 overlays each for hardware and software runtime

  * pynqZ1-Z2: shared objects used by Pynq devices

  * ultra96: shared objects used by ultra96

  *  params: set of trained parameters for the 5 overlays:

MNIST and NIST dataset for LFC network. Note that NIST dataset is only applicable to LFC-W1A1 by default.

Cifar10 , SVHN and German Road Signs dataset for CNV network. Note that SVHN and German Road Signs databases are only applicable to CNV-W1A1 by default.

- notebooks: lists a set of python notebooks examples, that during installation will be moved in /home/xilinx/jupyter_notebooks/bnn/ folder
- tests: contains test script and test images
