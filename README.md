# Project_Template

## Team Name: FPGAALL
(Enter your team name from Canvas)

## Team Members:
- Student 1: Jazzmin Poitier
- Student 2: Ndjiakou Kaptue Maiva

## Project Title: Comparative Performance of Neural Network-Based Image Classification on FPGA and Software Applications
(Enter your project title - be creative)

## Project Description:
(Provide a short description of the problem you're addressing)
The application used for this project is Jupyter, a popular open-source interactive computing environment that enables the development and execution of Python-based machine learning models. In this project, we will use Jupyter to train a neural network model to classify lung cancer images. These images will be preprocessed and labeled before being fed into the model for training. The trained model will then be evaluated by running the classification task both on the PYNQ Z1 FPGA board and within the Jupyter environment on a traditional software platform. The PYNQ Z1 FPGA board, equipped with specialized hardware for parallel processing, will be leveraged to accelerate the inference process, while Jupyter will serve as the software-based approach for classification, running on a general-purpose CPU. The methodology involves training the neural network model in Python using well-established deep learning libraries, such as PyTorch, and then evaluating the performance of the classification on both platforms. Key metrics, such as the classification time and accuracy, will be recorded and compared. Additionally, the limitations of the PYNQ Z1 FPGA board will be examined, particularly focusing on the maximum image size it can handle before performance starts to degrade. This methodology will provide a comprehensive comparison of the strengths and weaknesses of both FPGA and software
solutions in real-time medical image analysis, ultimately guiding the best approach for scalable and efficient tumor detection systems.

## Objectives and Motivation:
The motivation behind this project is driven by the need for fast and efficient image classification in critical applications like medical diagnostics, where timely and accurate analysis of lung cancer images can be life-saving. 
-  Within 06 weeks, compare the performance (classification time) of FPGA and software platforms in classifying lung cancer images using the same dataset and algorithm, with a focus on measuring processing speed and efficiency.
- Investigate and document the largest image size the PYNQ Z1 board can handle without significant performance degradation, by testing and measuring processing time at different image sizes, to be completed during 06 weeks.
- Within 06 weeks, analyze the trade-offs between FPGA and software solutions in terms of speed, power, and scalability for real-time medical image analysis.
- Built a neural network for lung cancer image classification with an accurancy of more than 90%
  

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
  We will used a dataset from Kaggle with lung cancer image

## Key Tasks/Steps for the Project:

1. Research & Literature Review (Week 1)

 Objective: Understand existing CNN-based image classification methods and FPGA accelerators.

Tasks:

. Study the most commonly used CNN architectures for image classification (e.g., LeNet, AlexNet, VGGNet, ResNet).
. Explore FPGA-based CNN accelerators.
. Review medical image processing techniques, especially for lung cancer classification.
. Understand FPGA hardware and tools (e.g., Xilinx Vivado, HLS, OpenCL, etc.).

Duration: 1 week

2. Dataset Preparation & Preprocessing (Week 1 - Week 2)

Objective: Gather and preprocess the lung cancer image dataset.

 Tasks:

. Collect a lung cancer dataset (Kaggle).
. Preprocess the images (resize, normalize, augment) to feed into the CNN.
. Split the dataset into training and validation sets.
. Prepare labels for classification.

Duration: 3 days 

 3. CNN Model Design and Training (Week 1 - Week 2)

Objective: Design and train a CNN model for image classification.

 Tasks:

. Choose an appropriate CNN architecture (e.g., a simple CNN or lightweight model like MobileNet if hardware constraints are a concern).
. Train the model using the prepared dataset on a standard computing platform (e.g., using Python with TensorFlow or PyTorch).
. Evaluate model performance (accuracy, precision, recall, etc.).
. Save the trained model for deployment on FPGA.

Duration: 1 week

4. Hardware Design for FPGA (Week 2 - Week 3)
    Objective: Design and implement the CNN hardware accelerator for FPGA.
      Tasks:

      . Break down the CNN layers into hardware components.
      . Design hardware modules for each layer .
      . Implement memory management to efficiently store and load image data and intermediate results.
      . Optimize the design for parallel processing on FPGA.

Duration: 1 week

5. Integration and Testing on FPGA (Week 3 - Week 4)
 Objective: Integrate the CNN hardware accelerator on FPGA and test functionality.
   Tasks:
  
  . Integrate the CNN hardware accelerator with the FPGA platform
  . Load the pre-trained model onto the FPGA.
  . Run test images through the accelerator and validate that the FPGA correctly performs the classification task.
  . Debug and resolve any issues in data flow or performance bottlenecks.

Duration: 1 week

6. Optimization and Performance Evaluation (Week 4 - Week 6)
Objective: Optimize the FPGA implementation for better performance and evaluate the results.
 Tasks:

  .Implement hardware optimizations
  . Measure the performance of the accelerator 
  . Compare the FPGA results 
  . Analyze accuracy and inference time on FPGA.
  . Tune hyperparameters or hardware design for further improvements if needed.

Duration: 1 week

7. Documentation and Final Report (Week 6)
Objective: Complete the project documentation and prepare the final report.
Tasks:

  . Document the design process, including hardware design, model training, and implementation.
  . Include diagrams, flowcharts, and explanations for each step.
  . Write the final project report and presentation.
  . Prepare a demonstration for the final presentation.

Duration: 1 week

# Description of the different systems 

We propose a system combining image analysis, an edge detector, and a 3-layer Convolution Neural Network as shown in the figure1. In the image analysis part, the system reads 
the gray image, preprocesses it, detects edges, finally  normalizes it to a fixed size, and stores it as a binary image 
array. For training and classification, it uses a 3-layer CNN  with a flattened layer and fully connected layer followed by an output layer of four classes. We used Python programming language to implement this system and used OpenCV for processing and analyzing images. Python modules pandas, NumPy, Matplotlib, Sklearn, TensorFlow modules, and Keras framework are used to build CNN model, training, and classification of lung cancer

![figue 1 github](https://github.com/user-attachments/assets/8d59f756-5343-404e-a328-c7736db7452c)

we propose to use the image below as a guideline for our FPGA accelaerator 
![image](https://github.com/user-attachments/assets/b283fc0b-172e-4864-90cb-39e47a8f15a1)

## Teamwork Breakdown:

# Team member :Ndjiakou Kaptue Maiva
Focus on the CNN model design and training (Week 1 - Week 2).
Lead the hardware design for FPGA (Week 2 - Week 4).
Handle integration and testing on FPGA (Week 4 - Week 5).

# Team member : Jazzmin Poitier 
Handle dataset preparation and preprocessing (Week 1 ).
Support CNN model design and training (Week 2 - Week 3).
Focus on optimization and performance evaluation (Week 5 - Week 6).
Lead the documentation and final report (Week 6).





