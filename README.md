# Lane Detection using U-Net
## Introduction
This project implements a U-Net deep learning model to perform lane detection on road images. Lane Detection is a task in Computer Vision and is an important component in the realm of autonomous driving and advanced driver assistance systems, ensuring vehicle safety.
The U-Net architecture, known for its efficiency in image segmentation tasks, is adapted here to specifically identify lane markings from regular traffic camera footage. This model can be particularly useful for autonomous driving systems and driver assistance technologies.
## Project Objective
The main goal of this project is to develop a robust lane detection tool that can accurately segment lanes from diverse road scenarios and varying lighting conditions. By leveraging the U-Net model, this tool aims to provide a reliable input for further processing in autonomous vehicle navigation systems.

## Dataset
The dataset used is the <a href ="https://www.kaggle.com/datasets/tusimple">TUSimple</a> dataset from Kaggle,consisting of 6,408 road images on US highways. The resolution of the image is 1280×720. The dataset is composed of 3,626 for training, 358 for validation, and 2,782 for testing called the TuSimple test set of which the images are under different weather conditions.

## Technologies Used
Python 3.8+
TensorFlow 2.x
OpenCV
NumPy



