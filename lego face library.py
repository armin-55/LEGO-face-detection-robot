import cv2
import numpy as np
import os
import pickle
# this code is for add your image first and read them,"Armin"
# Function to remember multiple images
def remember_images(image_paths, output_file):
    orb = cv2.ORB_create()
    data = {}

    for image_path in image_paths:
        # Load each image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Error loading image {image_path}. Skipping.")
            continue

        # Compute keypoints and descriptors
        keypoints, descriptors = orb.detectAndCompute(image, None)
        if descriptors is not None:
            # Store the descriptors with the image name as key
            image_name = os.path.basename(image_path)  # Just the filename
            data[image_name] = descriptors

    # Save the data to a file
    with open(output_file, 'wb') as file:
        pickle.dump(data, file)
    print(f"Data saved to {output_file}")

# List of image paths
image_paths = [
    'D:\work\legoo facr\lego 1.png',
    'D:\work\legoo facr\lego 2.png',
    'D:\work\legoo facr\lego 3.png',
    'D:\work\legoo facr\lego 4.png',
    'D:\work\legoo facr\lego 5.png',
    'D:\work\legoo facr\lego 6.png',
    'D:\work\legoo facr\lego 7.png',
    'D:\work\legoo facr\lego 8.png',
    'D:\work\legoo facr\lego 9.png',
    'D:\work\legoo facr\lego 10.png',
    'D:\work\legoo facr\lego 11.png',
    'D:\work\legoo facr\lego 12.png',
    'D:\work\legoo facr\lego 13.png',
    'D:\work\legoo facr\lego 14.png',
    'D:\work\legoo facr\lego 15.png',
    'D:\work\legoo facr\lego 16.png',
    'D:\work\legoo facr\lego 17.png',
    'D:\work\legoo facr\lego 18.png',
    'D:\work\legoo facr\lego 19.png',
    'D:\work\legoo facr\lego 20.png',
    'D:\work\legoo facr\lego 21.png',
    'D:\work\legoo facr\lego 22.png',
    #'D:\work\legoo facr\lego 16.png',



    
    # Add paths to more images here
]

# Call the function to remember images and save to a file
remember_images(image_paths, 'image_features.pkl')
