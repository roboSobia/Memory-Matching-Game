import cv2
import numpy as np

# DroidCam settings
HTTP = 'http://'
IP_ADDRESS = '192.168.2.19'  # Change to your IP
URL = HTTP + IP_ADDRESS + ':4747/mjpegfeed?640x480'

# Define colors with BGR values and HSV ranges
colors = [
    {'name': 'red', 'bgr': (0, 0, 255), 'lower': [(0, 100, 100), (170, 100, 100)], 'upper': [(10, 255, 255), (179, 255, 255)]},
    {'name': 'orange', 'bgr': (0, 165, 255), 'lower': [(11, 100, 100)], 'upper': [(20, 255, 255)]},
    {'name': 'yellow', 'bgr': (0, 255, 255), 'lower': [(25, 100, 100)], 'upper': [(35, 255, 255)]},
    {'name': 'green', 'bgr': (0, 255, 0), 'lower': [(36, 100, 100)], 'upper': [(70, 255, 255)]},
    {'name': 'blue', 'bgr': (255, 0, 0), 'lower': [(100, 100, 100)], 'upper': [(140, 255, 255)]},
    {'name': 'violet', 'bgr': (238, 130, 238), 'lower': [(141, 100, 100)], 'upper': [(160, 255, 255)]}
]

CELL_THRESHOLD = 500  # Minimum colored pixels to consider a cell filled

# Fixed grid size
GRID_ROWS = 4
GRID_COLS = 4

def find_board_corners(frame):
    """Find the four corners of the game board in the image"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blur, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour by area
    if not contours:
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Check if contour is large enough to be the board
    if cv2.contourArea(largest_contour) < 10000:  # Adjust this threshold as needed
        return None
    
    # Approximate the contour to a polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # If the polygon has 4 vertices, we assume it's the board
    if len(approx) == 4:
        # Sort corners for perspective transform
        corners = np.array([point[0] for point in approx], dtype=np.float32)
        return sort_corners(corners)
    
    return None

def sort_corners(corners):
    """Sort corners in order: top-left, top-right, bottom-right, bottom-left"""
    # Calculate sum and difference of x,y coordinates
    # top-left has smallest sum, bottom-right has largest sum
    # top-right has smallest difference, bottom-left has largest difference
    sum_corners = corners[:, 0] + corners[:, 1]
    diff_corners = corners[:, 0] - corners[:, 1]
    
    sorted_corners = np.zeros_like(corners)
    sorted_corners[0] = corners[np.argmin(sum_corners)]    # top-left
    sorted_corners[2] = corners[np.argmax(sum_corners)]    # bottom-right
    sorted_corners[1] = corners[np.argmin(diff_corners)]   # top-right
    sorted_corners[3] = corners[np.argmax(diff_corners)]   # bottom-left
    
    return sorted_corners

def transform_board(frame, corners):
    """Apply perspective transform to get top-down view of board"""
    # Define the destination points for the transform
    # Create a square output image
    board_size = 400  # Output size of the board
    dst_points = np.array([
        [0, 0],
        [board_size, 0],
        [board_size, board_size],
        [0, board_size]
    ], dtype=np.float32)
    
    # Calculate perspective transform matrix
    M = cv2.getPerspectiveTransform(corners, dst_points)
    
    # Apply transform
    warped = cv2.warpPerspective(frame, M, (board_size, board_size))
    
    return warped

cap = cv2.VideoCapture(URL)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Find board corners
    corners = find_board_corners(frame)
    
    if corners is not None:
        # Draw corners on original frame
        for corner in corners:
            cv2.circle(frame, (int(corner[0]), int(corner[1])), 5, (0, 0, 0), -1)
        
        # Draw board outline
        cv2.polylines(frame, [corners.astype(np.int32)], True, (0, 0, 0), 2)
        
        # Transform board to get top-down view
        warped_board = transform_board(frame, corners)
        
        # Get board dimensions
        board_height, board_width = warped_board.shape[:2]
        
        # Calculate cell dimensions
        cell_width = board_width // GRID_COLS
        cell_height = board_height // GRID_ROWS
        
        # Create empty 4x4 grid
        grid = [[None for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
        
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(warped_board, cv2.COLOR_BGR2HSV)
        
        # Draw grid lines on warped board
        for i in range(GRID_ROWS + 1):
            y = i * cell_height
            cv2.line(warped_board, (0, y), (board_width, y), (255, 255, 255), 1)
        
        for j in range(GRID_COLS + 1):
            x = j * cell_width
            cv2.line(warped_board, (x, 0), (x, board_height), (255, 255, 255), 1)
        
        # Initialize color counts for each cell
        color_counts = [[{} for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
        
        # Check each cell for colors
        for color in colors:
            mask = np.zeros(warped_board.shape[:2], dtype=np.uint8)
            for l, u in zip(color['lower'], color['upper']):
                lower = np.array(l, dtype=np.uint8)
                upper = np.array(u, dtype=np.uint8)
                mask_part = cv2.inRange(hsv, lower, upper)
                mask = cv2.bitwise_or(mask, mask_part)
            
            for i in range(GRID_ROWS):
                for j in range(GRID_COLS):
                    # Calculate cell coordinates
                    x1 = j * cell_width
                    y1 = i * cell_height
                    x2 = (j + 1) * cell_width
                    y2 = (i + 1) * cell_height
                    
                    # Count pixels of this color in the cell
                    cell_mask = mask[y1:y2, x1:x2]
                    pixel_count = cv2.countNonZero(cell_mask)
                    color_counts[i][j][color['name']] = pixel_count
        
        # Find dominant color for each cell
        for i in range(GRID_ROWS):
            for j in range(GRID_COLS):
                if color_counts[i][j]:  # If any colors detected
                    # Find color with maximum pixel count
                    dominant_color = max(color_counts[i][j], key=color_counts[i][j].get)
                    dominant_count = color_counts[i][j][dominant_color]
                    
                    if dominant_count > CELL_THRESHOLD:
                        grid[i][j] = dominant_color
                        # Get the BGR color for the dominant color
                        color_bgr = next(c['bgr'] for c in colors if c['name'] == dominant_color)
                        # Draw rectangle with the dominant color
                        x1 = j * cell_width
                        y1 = i * cell_height
                        x2 = (j + 1) * cell_width
                        y2 = (i + 1) * cell_height
                        cv2.rectangle(warped_board, (x1, y1), (x2, y2), color_bgr, 3)
                        cv2.putText(warped_board, f"({i},{j})", (x1 + 10, y1 + 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)
        
        print("\nCurrent Grid Detection:")
        for row in grid:
            print([cell if cell else "---" for cell in row])
        
        # Display the warped board
        cv2.imshow('Board Detection (4x4 Grid)', warped_board)
    else:
        cv2.putText(frame, "Looking for board...", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Display the original frame
    cv2.imshow('Original Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()