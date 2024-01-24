import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Function to display the image
def display_image(title, image):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

# Function to update the image with instructions and points
def update_image_with_instructions():
    global image, points
    instructions = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]

    temp_image = image.copy()
    for i, point in enumerate(points):
        cv2.circle(temp_image, point, 5, (0, 255, 0), -1)
        cv2.putText(temp_image, instructions[i], (point[0] + 10, point[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if len(points) < 4:
        cv2.putText(temp_image, f"Mark the {instructions[len(points)]} corner", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    cv2.imshow("Image", temp_image)

# Function to select points
def select_points(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        update_image_with_instructions()

image=None
points=None

def trans_im(path):
    global image, points

    # Load the image
    image_path = path
    image = cv2.imread(image_path)

    # Make window resizable and adjust size to fit the screen
    screen_res = 1.0  # Adjust this factor to resize the window (1.0 for full size)
    screen_width, screen_height = 1366, 768  # Use your screen resolution here
    scale_width = screen_width / image.shape[1]
    scale_height = screen_height / image.shape[0]
    scale = min(scale_width, scale_height)

    window_width = int(image.shape[1] * scale * screen_res)
    window_height = int(image.shape[0] * scale * screen_res)

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", window_width, window_height)

    # Initialize points and display initial instructions
    points = []
    update_image_with_instructions()
    cv2.setMouseCallback("Image", select_points)

    # Wait until 4 points have been selected
    while len(points) < 4:
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    # Calculate the perspective transform matrix
    src_pts = np.array(points, dtype="float32")
    dst_pts = np.array([[0, 0], [image.shape[1], 0], [image.shape[1], image.shape[0]], [0, image.shape[0]]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Apply the perspective transformation
    transformed = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))

    print("Transformation Matrix:")
    print(matrix)
    # Display and print the results
    display_image("Transformed Image", transformed)

# for image in current dir
for img in os.listdir():
    if img.endswith('.jpg'):
        trans_im(img)
test="Transformed Image"
SSID='AAMCAR'
