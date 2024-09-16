import cv2
import numpy as np
import pickle
import time
from serial import Serial

def recognize_with_webcam(features_file, recognition_threshold=0.8, interval=5, process_frame_rate=5):
    # Load the saved features from the file
    try:
        with open(features_file, 'rb') as file:
            saved_features = pickle.load(file)
    except FileNotFoundError:
        print(f"Error: Features file '{features_file}' not found.")
        return

    # Initialize ORB detector and FLANN-based matcher
    orb = cv2.ORB_create()
    index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                        table_number=6,  # 12
                        key_size=12,  # 20
                        multi_probe_level=1)  # 2
    search_params = dict(checks=50)  # Adjust based on performance needs
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Start webcam capture
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    start_time = time.time()
    highest_match_name = None
    highest_match_ratio = 0
    frame_count = 0

    # List of specific image names to check against
    lego_employee = ["lego 2.png", "lego 3.png", "lego 4.png", "lego 5.png", "lego 6.png", "lego 11.png"]
    lego_boss = ["lego 7.png", "lego 19.png", "lego 20.png", "lego 215.png", "lego 22.png"]

    # Ask user for the current time once and send it to Arduino
    #current_hour = int(input("Enter the current hour (0-24): "))
    #send_command(f"{current_hour}")  # Send the time first

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Skip frames to reduce processing load
        frame_count += 1
        if frame_count % process_frame_rate != 0:
            continue

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute keypoints and descriptors of the current frame
        keypoints, frame_descriptors = orb.detectAndCompute(gray_frame, None)

        # Skip to next frame if no descriptors are found
        if frame_descriptors is None or len(frame_descriptors) < 2:
            continue

        # Compare the current frame descriptors with each saved image's descriptors
        for image_name, saved_descriptors in saved_features.items():
            # Skip if saved_descriptors is invalid
            if saved_descriptors is None or len(saved_descriptors) < 2:
                continue

            # Perform knnMatch to find the best matches
            matches = flann.knnMatch(saved_descriptors, frame_descriptors, k=2)

            # Ensure matches are valid and have at least 2 entries
            good_matches = []
            for m_n in matches:
                if len(m_n) == 2:  # Check if there are two matches
                    m, n = m_n
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)

            # Calculate the match ratio
            match_ratio = len(good_matches) / len(saved_descriptors) if len(saved_descriptors) > 0 else 0

            # Update best match if current match is better
            if match_ratio > highest_match_ratio:
                highest_match_ratio = match_ratio
                highest_match_name = image_name

        # Check if it's time to print the result (every interval seconds)
        current_time = time.time()
        #if current_time - start_time >= interval:
            # Print the name with the highest accuracy found in the interval
        if highest_match_name and highest_match_ratio > recognition_threshold:
                if highest_match_name in lego_employee:
                    print("Sending command 1")  # Command 1 based on recognition
                    send_command(f"1")  # Send command 1 to Arduino
                elif highest_match_name in lego_boss:
                    print("Sending command 2")  # Command 2 based on recognition
                    send_command(f"2")  # Send command 2 to Arduino
                else:
                    print("Sending command 3")  # Command 3 based on recognition
                    send_command(f"3")  # Send command 3 to Arduino
        else:
                print("No sufficient match found in this interval.")

            # Reset for the next interval
        start_time = current_time
        highest_match_name = None
        highest_match_ratio = 0

        # Display the resulting frame
        cv2.imshow('Webcam Recognition', frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

# Set up serial communication with Arduino
arduino_port = 'COM5'  # Adjust as needed
baud_rate = 9600       # Must match the Arduino's Serial.begin() rate
ser = Serial(arduino_port, baud_rate)

time.sleep(2)  # Give some time for the connection to be established

def send_command(command):
    ser.write(f"{command}\n".encode())  # Send the command as bytes
    time.sleep(1)  # Wait for the Arduino to process the command

try:
    # Call the function with the features file
    recognize_with_webcam('image_features.pkl', recognition_threshold=0.10, interval=1, process_frame_rate=5)
except KeyboardInterrupt:
    print("Program stopped by user.")
finally:
    ser.close()  # Close the serial connection
