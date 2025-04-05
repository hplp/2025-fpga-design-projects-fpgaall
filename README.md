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

## Key Objectives:
-  Within 06 weeks, compare the performance (classification time) of FPGA and software platforms in classifying brain tumor images using the same dataset and algorithm, with a focus on measuring processing speed and efficiency.
- Investigate and document the largest image size the PYNQ Z1 board can handle without significant performance degradation, by testing and measuring processing time at different image sizes, to be completed during 06 weeks.
- Within 06 weeks, analyze the trade-offs between FPGA and software solutions in terms of speed, power, and scalability for real-time medical image analysis.

## Technology Stack:
(List the hardware platform, software tools, language(s), etc. you plan to use)
1. Hardware Technologies:
   . PYNQ Z1 Board
2. Programming Languages:
   . Python
   . VHDL/Verilog
3. Software Libraries:
   . OpenCV( for image processing)
   . TensorFlow/Pytorch(for machine learning)
   . PYNQ Library( for interfacing with the FPGA)
4. Data Analysis Tools:
   . NumPy/Pandas ( for data handling)
   . Matplotlib/Seaborn (for visualization)
## Expected Outcomes:
(Describe what you expect to deliver at the end of the project)

## Tasks:
(Describe the tasks that need to be completed. Assign students to tasks)

## Timeline:
(Provide a timeline or milestones for the project)
