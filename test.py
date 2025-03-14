import cv2
import numpy as np
from collections import Counter

# # Capture image from webcam
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()

# print("Press SPACE to capture the image.")
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Failed to capture image.")
#         break

#     cv2.imshow("Live Camera - Press SPACE to capture", frame)

#     key = cv2.waitKey(1)
#     if key == 32:  # SPACE key
#         image = frame.copy()
#         break
#     elif key == 27:  # ESC to quit
#         cap.release()
#         cv2.destroyAllWindows()
#         exit()

# cap.release()
# cv2.destroyAllWindows()


# Load image from file
image = cv2.imread("Resources/shapes5.jpg")
image = cv2.resize(image, (400, 400))

# Get dimensions
h, w, _ = image.shape

# Split into 4x4 regions
regions = {}
region_height = h // 4
region_width = w // 4

for i in range(4):
    for j in range(4):
        region_name = f"Region {i+1},{j+1}"
        regions[region_name] = image[i*region_height:(i+1)*region_height, j*region_width:(j+1)*region_width]

def get_dominant_color(region):
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape((-1, 3))
    pixel_list = [tuple(p) for p in pixels]
    most_common = Counter(pixel_list).most_common(1)[0][0]
    dominant_hsv = np.uint8([[most_common]])
    dominant_bgr = cv2.cvtColor(dominant_hsv, cv2.COLOR_HSV2BGR)[0][0]
    return most_common, dominant_bgr

# Loop through each region and detect color
for name, region in regions.items():
    hsv_color, bgr_color = get_dominant_color(region)

    # Create a patch for the detected color
    patch = np.zeros((100, 100, 3), dtype=np.uint8)
    patch[:] = bgr_color

    # # Show original region and detected color patch
    # cv2.imshow(f"{name} - Region", region)
    # cv2.imshow(f"{name} - Detected Color", patch)
    # Write the RGB value on the region
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"RGB: {bgr_color[2]}, {bgr_color[1]}, {bgr_color[0]}"
    cv2.putText(region, text, (10, region_height // 2), font, 0.25, (255, 255, 255), 1, cv2.LINE_AA)

    # Place the region back into the image
    image[i*region_height:(i+1)*region_height, j*region_width:(j+1)*region_width] = region


    # Print values
    print(f"{name}:")
    print("  Dominant HSV:", hsv_color)
    print("  Dominant BGR:", bgr_color)
    print()

# Show the final image with RGB values
cv2.imshow("Image with RGB values", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
