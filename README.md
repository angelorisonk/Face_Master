# The FaOb Master

The goal of this project is to capture a set of face images from an input device, such as a webcam, and save them to a file in a specified format, such as a hdf5 file. The project uses the MTCNN detector to detect and crop the faces in the images, and then resizes the face images to a specified size before saving them to the output file.

The project also includes a graphical user interface (GUI) built with Tkinter that allows the user to specify the number of images to capture, select the output file, and start the capture process. The GUI displays the number of images remaining, the time remaining, and the number of images captured during the capture process.


# Dependencies
To use the Intrusion Shield Image Trainer, you will need to install the following dependencies:

tensorflow
mtcnn
tkinter
opencv-python
h5py
You can install these dependencies by running the following command:

```shell script pip install -r requirements.txt```

# Usage:
To use this tool, simply run the script from the command line and follow the prompts in the graphical user interface. You can specify the number of images to capture, the size of the face images, the output file name and format, and the input device using the variables at the top of the script. Once the capture is complete, the images will be saved to the specified output file in either HDF5 or NumPy format.
```python image_trainer.py```
This will open the GUI window, where you can specify the number of images to capture,the output file name, and the input device.

To specify the number of images to capture, enter a value in the "Number of images remaining" field and click the "Start capture" button. The program will capture the specified number of images and save them to the output file.

To specify the output file name, click the "Select output file" button and select a file from the file dialog. The program will save the images to the specified file.

To specify the input device, use the "--input_device" command line argument. For example, to use the default webcam, run the following command:

python image_trainer.py --input_device 0
To specify the size of the face images to save, use the "--face_size" command line argument. For example, to save 256x256 images, run the following command:

python image_trainer.py --face_size 256
To specify the output format, use the "--output_format" command line argument. The program supports "h5" and "npz" formats. For example, to save the images in the "npz" format, run the following command:

```python image_trainer.py --output_format npz```

# Fetures

1. Face detection and alignment using the MTCNN library: The code uses the MTCNN library to detect and align faces in the video frames. This helps to ensure that the captured face images are correctly oriented and centered.

2. GPU acceleration: The code sets up GPU acceleration for TensorFlow to speed up the process of detecting and aligning faces.

3. `GUI window: The code uses tkinter to create a GUI window that displays the number of images remaining to capture, the time remaining, and the number of images captured so far. This allows the user to track the progress of the face capture process.

3. Output file options: The code allows the user to choose the output file format, either as individual image files or as a single h5 file. The h5 file format is a standardized binary file format that is often used for storing large datasets.

4. Face resizing: The code resizes the captured face images to a fixed size, which can be useful for training machine learning models or for other purposes.

5. Progress bar: The code uses the tqdm library to display a progress bar in the terminal window, which shows the progress of the face capture process. This can be helpful for tracking the progress of the code when it is running.


# Algorithm
1. Set the number of images to capture, the size of the face images, the output file name, the input device, and the output format.
2. Create an empty list to store the images.
3. Create an MTCNN detector.
4. Open the input device using OpenCV.
5. Create the GUI window and widgets.
6. Define the callback function for the Start capture button.
7. In the callback function, get the number of images to capture from the entry widget.
8. Update the label widgets with the number of images remaining and the time remaining.
9. Capture a set of images from the input device and add them to the list.
10.For each image, detect the faces in the image using the MTCNN detector.
11. If a face was detected, crop the image to just the face, resize the face image, and add it to the list.
12. Update the label widgets with the number of images remaining and the time remaining.
13. When all the images have been captured, save the images to the output file in the specified format.

License
This project is released under the MIT License https://opensource.org/licenses/MIT

Credits
This project uses the MTCNN face detection library, which was developed by David Sandberg. The MTCNN library is released under the Apache License 2.0.
The project also uses the Tkinter library for the graphical user interface. Tkinter is included in the Python standard library.

# Contribution guidelines
If you would like to contribute to this project, please follow these guidelines:

1. Create a new branch for your changes
2. Make sure your code follows the style guide
3. Test your code thoroughly
4. Submit a pull request
# Issues
If you encounter any issues while using this project, please report them here: (https://github.com/angelorisonk/Face_Master/pulls)


