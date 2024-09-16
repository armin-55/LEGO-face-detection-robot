import cv2
import numpy as np
import pickle
import time

def recognize_with_webcam(features_file, recognition_threshold=0.8, interval=5, process_frame_rate=5):
    # Load the saved features from the file (کتاب خانه داده تصاویر اولیه )
    try:
        with open(features_file, 'rb') as file:
            saved_features = pickle.load(file)
    except FileNotFoundError:
        print(f"Error: Features file '{features_file}' not found.")
        return

    # Initialize ORB detector and FLANN-based matcher همسایه نزدیک (با بردار تصویر)
    orb = cv2.ORB_create()
    index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                        table_number=6,  # 12
                        key_size=12,  # 20
                        multi_probe_level=1)  # 2
    search_params = dict(checks=50)  # Adjust based on performance needs
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Start webcam capture
    cap = cv2.VideoCapture(0)# وب کم 

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    start_time = time.time()# فرم و تایم رو با صفر یکسان آغاز میکند
    highest_match_name = None
    highest_match_ratio = 0
    frame_count = 0

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

        # Convert frame to grayscale تغییر رنگ از ار جی بی به خاکستری برای تشخیص راحت تر خطوط
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

        # Check if it's time to print the result (every 5 seconds)
        current_time = time.time()
        if current_time - start_time >= interval:
            # Print the name with the highest accuracy found in the interval
            if highest_match_name and highest_match_ratio > recognition_threshold:
                print(f"Most recognized: {highest_match_name}")
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

        lego_employee = ['lego 2.png', 'lego 3.png', 'lego 4.png', 'lego 5.png', 'lego 6.png', 'lego 7.png']
        lego_boss = ['lego 7.png', 'lego 19.png', 'lego 20.png', 'lego 215.png', 'lego 22.png', ]

    
     # Check if the highest_match_name is in the list of LEGO names
    if highest_match_name in lego_employee:
        print("1")
    elif highest_match_name in lego_boss:
        print("2")
    else:
        print("3")



    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

# Call the function with the features file
recognize_with_webcam('image_features.pkl', recognition_threshold=0.10, interval=1, process_frame_rate=5)

