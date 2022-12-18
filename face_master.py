# MTCNN library developed by David Sandberg
# https://github.com/anglorisonk/Face_Master

#imports nessasery libraries
import cv2
import numpy as np
import tensorflow as tf
import h5py
import time
import argparse
from tqdm import tqdm
from mtcnn import MTCNN
from tkinter import Tk, Label, Entry, Button

# Set up GPU acceleration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Set the number of images to capture(default=100)
num_images = 100

# Set the size of the face images to save
face_size = 256

# Set the output file name
output_file = 'D:\Home_Defensive_System\IntrusionShield_image_trainer\Dataset\ my_images.h5'

# Set the input device
input_device = 0

# Set the output format
output_format = 'h5'

# Create an empty list to store the images
images = []

# Create an MTCNN detector
detector = MTCNN()

# Open the input device using OpenCV
cap = cv2.VideoCapture(input_device)

# Create the GUI window
window = Tk()
window.title('Face Capture')

# Create the GUI widgets
label1 = Label(window, text='Number of images remaining:')
label2 = Label(window, text=str(num_images))
label3 = Label(window, text='Time remaining:')
label4 = Label(window, text='')
label5 = Label(window, text='Images captured:')
label6 = Label(window, text='')
entry1 = Entry(window)
entry1.insert(0, str(num_images))
button2 = Button(window, text='Start capture', command=lambda: start_capture())

# Add the GUI widgets to the window
label1.grid(row=0, column=0)
label2.grid(row=0, column=1)
label3.grid(row=1, column=0)
label4.grid(row=1, column=1)
label5.grid(row=2, column=0)
label6.grid(row=2, column=1)
entry1.grid(row=3, column=0)
button2.grid(row=4, column=0, columnspan=2)

# Define the callback function for the Start capture button
def start_capture():
    global num_images
    global images
    global label4
    global label6

    # Get the number of images to capture from the entry widget
    num_images = int(entry1.get())

    # Update the label widgets with the number of images remaining and the time remaining
    label2.config(text=str(num_images))
    label4.config(text='Calculating...')

    # Capture a set of images from the input device and add them to the list
    start_time = time.time()
    for i in tqdm(range(num_images)):
        ret, frame = cap.read()
        if ret:
            # Detect the faces in the frame
            faces = detector.detect_faces(frame)

            # If a face was detected
            if len(faces) > 0:
                # Get the first face
                face = faces[0]['box']

                # Crop the frame to just the face
                face_image = frame[face[1]:face[1]+face[3], face[0]:face[0]+face[2]]

                # Resize the face
                face_image = cv2.resize(face_image, (face_size, face_size))

                # Add the face to the list
                images.append(face_image)

            # Update the label widgets with the number of images remaining and the time remaining
            time_elapsed = time.time() - start_time
            time_remaining = (num_images - i - 1) * time_elapsed / (i + 1)
            label2.config(text=str(num_images - i - 1))
            label4.config(text='{:.2f} seconds'.format(time_remaining))
            label6.config(text=str(len(images)))

            # Display the video frame
            cv2.imshow('Video', frame)

            # Check if the user pressed a key to stop the loop
            key = cv2.waitKey(1)


            if key != -1:
                break

            # Release the input device
        cap.release()
                    

    # Convert the images to a NumPy array  
    images = np.array(images)

    # Save the images to the specified output format
    if output_format == 'h5':
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('D:\Home_Defensive_System\IntrusionShield_image_trainer\img_models\ Model_0.jpg', data=images)
    elif output_format == 'csv':
        np.savetxt(output_file, images.flatten(), delimiter=',')
    elif output_format == 'json':
        import json
        with open(output_file, 'w') as f:
            json.dump(images.tolist(), f)
    else:
        print('Error: Unrecognized output format')

    # Update the label widgets with the final number of images captured and the time elapsed
    label2.config(text='0')
    label4.config(text='{:.2f} seconds'.format(time_elapsed))
    label6.config(text=str(len(images)))

# Run the Tkinter event loop
window.mainloop()


