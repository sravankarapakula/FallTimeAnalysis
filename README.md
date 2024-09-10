# FallTimeAnalysis
import cv2
import time
import matplotlib.pyplot as plt
import numpy as np

def detect_fall_time(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    start_time, end_time = None, None
    fall_times = []
    bounce_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to HSV (Hue, Saturation, Value)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define range of ball color in HSV (Red)
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])

        # Threshold the HSV image to get only ball colors
        ball_mask = cv2.inRange(hsv_frame, lower_red, upper_red)

        # Perform morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN, kernel)
        ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            if start_time is None:
                start_time = time.time()
            end_time = time.time()
            cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
            # Increment bounce count when a contour is detected
            bounce_count += 1

            # Calculate fall time and store it
            if start_time is not None:
                fall_time = end_time - start_time
                fall_times.append(fall_time)

            # Reset start time for the next bounce
            start_time = end_time

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    total_time = sum(fall_times)
    print("Total Fall Time:", total_time, "seconds")  # Display total fall time
    print("Number of Bounces:", bounce_count//100)

    # Adjust the number of bounces for plotting
    plot_bounce_count = bounce_count // 100 if bounce_count > 100 else 1

    # Plotting fall times for each bounce
    plt.plot(range(1, plot_bounce_count + 1), fall_times[:plot_bounce_count], marker='o', linestyle='-')
    plt.xlabel('Bounce Number')
    plt.ylabel('Fall Time (seconds)')
    plt.title('Fall Time for Each Bounce')

    # Annotate the graph with total fall time and adjusted number of bounces
    plt.annotate(f'Total Fall Time: {total_time:.2f} seconds\nNumber of Bounces: {plot_bounce_count}',
                 xy=(0.5, 0.95), xycoords='axes fraction',
                 xytext=(0.5, 0.95), textcoords='axes fraction',
                 horizontalalignment='center', verticalalignment='center')

    plt.grid(True)
    plt.show()

    cap.release()
    cv2.destroyAllWindows()

# Path to the input video file
video_path = r"C:\Users\mvnsa\OneDrive\Desktop\Python project\blender - Made with Clipchamp_1712598652168.mp4"

# Call the function to detect fall time and count bounces
detect_fall_time(video_path)
