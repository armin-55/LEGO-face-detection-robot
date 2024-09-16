import cv2
import numpy as np
import pickle

def recognize_with_webcam(features_file, recognition_threshold=0.8):
    # Load the saved features from the file
    with open(features_file, 'rb') as file:
        saved_features = pickle.load(file)

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

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute keypoints and descriptors of the current frame
        keypoints, frame_descriptors = orb.detectAndCompute(gray_frame, None)

        # Skip to next frame if no descriptors are found
        if frame_descriptors is None or len(frame_descriptors) < 2:
            continue

        best_match_name = None
        best_match_ratio = 0

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

            # Print the matching ratio for debugging purposes
            print(f"{image_name}: Match ratio = {match_ratio:.2f}")

            # Update best match if current match is better
            if match_ratio > best_match_ratio:
                best_match_ratio = match_ratio
                best_match_name = image_name

        # Print the recognized name in the terminal if the match ratio exceeds the threshold
        if best_match_ratio > recognition_threshold:
            print(f"Recognized as: {best_match_name} with match ratio: {best_match_ratio:.2f}")

            # Draw bounding box and label on the frame
            for match in good_matches[:10]:  # Only showing the top 10 matches for clarity
                pt1 = tuple(map(int, keypoints[match.trainIdx].pt))
                pt2 = (pt1[0] + 15, pt1[1] + 15)  # Scaled bounding box
                cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
                cv2.putText(frame, best_match_name, (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            print("No sufficient match found.")

        # Display the resulting frame
        cv2.imshow('Webcam Recognition', frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

# Call the function with the features file
recognize_with_webcam('image_features.pkl', recognition_threshold=0.8)
