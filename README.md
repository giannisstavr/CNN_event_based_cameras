# Weld Classification Using Event-Based Cameras and Machine Learning

## Description:
This GitHub project centers around the analysis of data gathered from a series of welding experiments, utilizing event-based cameras to capture footage during the welding process. The primary focus lies in processing the collected data and integrating it within a machine learning context. Ultimately, the goal is to develop an optimized machine learning model capable of effectively classifying welds based on the processed data.
In order to achieve that a binary CNN model was created that classifies weld spots taken from event based cameras into good or bad.
The footage captured from the event based cameras was proccesed into avi type format. Then by using Matlab it was possible to brake down the footage into individual frames. These frames were then labeled into "good" and "bad".

### Why Event-Based Cameras?
Event-based cameras offer distinct advantages over traditional frame-based cameras. By detecting changes in pixel intensity (events) rather than capturing images at fixed intervals, they provide high temporal resolution, low latency, and reduced data storage requirements. These qualities make event-based cameras an ideal choice for capturing dynamic welding processes with fine-grained temporal details.

## Packages needed:
Use the package manager pip to install the following:
  - tensorflow
  - opencv
  - imghdr
  - matplotlib
  - numpy
  - os
  - shutil

### Example of how to install:
```bash
pip install opencv-python
```
Similarly, for other missing packages, replace "opencv-python" with the name of the package you want to install.

## Additional Notes:
Possibly the model could have been better if a different base model was used. Due to time limitations and Colab restricting gpu usage I wasn't able to try anything else except from VGG16. ResNet and EfficientNet are a good starting point if you want to try it yourself.

