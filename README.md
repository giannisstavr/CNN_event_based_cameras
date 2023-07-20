# CNN_event_based_cameras
Binary CNN model that classifies weld spots taken from event based cameras into good or bad.
The images used as data where taken from videos recorded by event-based cameras during welding. These were then processed with Metavision Studio in order to export videos which preview "good" and "bad" welds. Then by using matlab it was possible to brake down the videos into individual frames. 

The input for the model was (256,256,3).

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
Possibly the model could have been better if a different based model was used. Due to time limitations and Colab restricting gpu usage I wasn't able to try anything else except from VGG16. ResNet and EfficientNet are a good starting point if you want to try it yourself.
