import pygame
import random
import serial
import time
import cv2
import numpy as np

# Arm Values
arm_values =[[110, 40, 125], [87, 65, 120], [87,110,120], [110,140,125], [150,55,155], [130,80,140], [130,105,140], [150,125,155]]

arm_home = [180, 90, 0]
arm_temp1 = [90, 10, 120] # change later
arm_temp2 = [90, 170, 120] # change later
arm_trash = [140, 0, 140] # change later
arm_trash2 = [140, 180, 140] # change later``````````

# Constants
SCREEN_WIDTH = 650
SCREEN_HEIGHT = 650
CARD_SIZE = 150
GRID_ROWS = 2
GRID_COLS = 4
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BACKGROUND_COLOR = (220, 220, 220)
SPACE = 10  # Space between cards
FLIP_DELAY = 1000  # Delay before hiding cards

# Define the card colors
colors = {
    "red": (255, 0, 0),
    "yellow": (255, 255, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255)
}

# Grid drawn - replacing all "orange" with "violet"
selected_colors = ["green", "red", "yellow", "blue", 
                   "blue", "yellow", "red", "green", 
                   "green", "red", "blue", "yellow",
                   "yellow", "blue", "green", "red"]

 
# Initialize card states: False for unvisited (not flipped before), and None for unknown color
card_states = {i: {"isFlippedBefore": False, "color": None} for i in range(GRID_ROWS * GRID_COLS)}
colors_found = {color: [] for color in colors.keys()}  # Initialize an empty list for each color

# Initialize the Pygame window
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Memory Puzzle Game")

# DroidCam settings
HTTP = 'http://'
IP_ADDRESS = '192.168.144.44'  # Change to your IP
URL = HTTP + IP_ADDRESS + ':4747/mjpegfeed?640x480'

colorRanges = [
    {'name': 'red', 'bgr': (0, 0, 255), 'lower': [(0, 100, 100), (170, 100, 100)], 'upper': [(10, 255, 255), (179, 255, 255)]},
    {'name': 'yellow', 'bgr': (0, 255, 255), 'lower': [(20, 100, 100)], 'upper': [(35, 255, 255)]},
    {'name': 'green', 'bgr': (0, 255, 0), 'lower': [(40, 40, 40)], 'upper': [(80, 255, 255)]},
    {'name': 'blue', 'bgr': (255, 0, 0), 'lower': [(100, 100, 100)], 'upper': [(130, 255, 255)]},
    {'name': 'black', 'bgr': (0, 0, 0), 'lower': [(0, 0, 0)], 'upper': [(180, 255, 50)]}  # Black detection
]
CELL_THRESHOLD = 600  # Minimum colored pixels to consider a cell filled

# Fixed grid size
GRID_ROWS = 2
GRID_COLS = 4

# def setup_serial():
#     ser = serial.Serial()
#     ser.baudrate = 9600
#     ser.port = 'COM5' 
#     ser.timeout = 2
    
#     try:
#         ser.open()
#         print("Serial port opened successfully!")
#         return ser
#     except Exception as e:
#         print(f"Error opening serial port: {e}")
#         return None


# ser = setup_serial()

# def send_arm_command(degree1, degree2, degree3, magnet, movement):
#     """
#     Send arm control parameters to the ESP32/ESP8266 through serial connection.
    
#     Parameters:
#     - ser: Serial object (already opened)
#     - degree1: first servo angle (integer)
#     - degree2: second servo angle (integer)
#     - degree3: third servo angle (integer)
#     - magnet: magnet state (integer)
#     - arm: arm selection (integer)
    
#     Returns:
#     - Response from the ESP (if any)
#     """
#     # Convert all parameters to strings and join with commas
#     command = f"{degree1},{degree2},{degree3},{magnet},{movement}\n"
    
#     # Send the command as bytes
#     ser.write(command.encode())
    
#     # Wait a moment for the ESP to process and respond
#     time.sleep(0.1)
    
#     # Read the response (if any)
#     response = b''
#     while ser.in_waiting:
#         response += ser.read(1)
    
#     return response.decode() if response else None



# def from_to(src, dest, id):
#     if src == "card" and dest == "temp1":
#         send_arm_command(arm_values[id][0], arm_values[id][1], arm_values[id][2], 1, 0) # pick card
#         time.sleep(2)
#         send_arm_command(arm_home[0], arm_home[1], arm_home[2], 1, 1) # home
#         time.sleep(2)
#         send_arm_command(arm_temp1[0], arm_temp1[1], arm_temp1[2], 1, 0) # put in temp1
#         time.sleep(2) 
#         send_arm_command(arm_home[0], arm_home[1], arm_home[2], 0, 1) # home
#         # Wait for "done" message from serial port
#         response = ""
#         while "done" not in response.lower():
#             if ser.in_waiting:
#                 byte = ser.read(1)
#                 response += byte.decode(errors='replace')
#         print("serial responded", response)
#     elif src == "card" and dest == "temp2":
#         send_arm_command(arm_values[id][0], arm_values[id][1], arm_values[id][2], 1, 0) # pick card
#         time.sleep(2)
#         send_arm_command(arm_home[0],arm_home[1], arm_home[2], 1, 1) # home
#         time.sleep(2)
#         send_arm_command(arm_temp2[0], arm_temp2[1], arm_temp2[2], 1, 0) # put in temp2
#         time.sleep(2)
#         send_arm_command(arm_home[0], arm_home[1], arm_home[2], 0, 1) # home
#         # Wait for "done" message from serial port
#         response = ""
#         while "done" not in response.lower():
#             if ser.in_waiting:
#                 byte = ser.read(1)
#                 response += byte.decode(errors='replace')
#         print("serial responded", response)
#     elif src == "card" and dest == "trash":
#         send_arm_command(arm_values[id][0], arm_values[id][1], arm_values[id][2], 1, 0) # pick card
#         time.sleep(2)
#         send_arm_command(arm_home[0], arm_home[1], arm_home[2], 1, 1) # home
#         time.sleep(2)
#         send_arm_command(arm_trash[0], arm_trash[1], arm_trash[2], 1, 0) # put in trash
#         time.sleep(2)
#         send_arm_command(arm_home[0], arm_home[1], arm_home[2], 0, 1) # home
#         # Wait for "done" message from serial port
#         response = ""
#         while "done" not in response.lower():
#             if ser.in_waiting:
#                 byte = ser.read(1)
#                 response += byte.decode(errors='replace')
#         print("serial responded", response)
#     elif src == "temp1" and dest == "trash":
#         send_arm_command(arm_temp1[0], arm_temp1[1], arm_temp1[2], 1, 0) # pick from temp1
#         time.sleep(2)
#         send_arm_command(arm_home[0], arm_home[1], arm_home[2], 1, 1) # home
#         time.sleep(2)
#         send_arm_command(arm_trash[0], arm_trash[1], arm_trash[2], 1, 0) # put in trash
#         time.sleep(2)
#         send_arm_command(arm_home[0], arm_home[1], arm_home[2], 0, 1) # home
#         # Wait for "done" message from serial port
#         response = ""
#         while "done" not in response.lower():
#             if ser.in_waiting:
#                 byte = ser.read(1)
#                 response += byte.decode(errors='replace')
#         print("serial responded", response)
#     elif src == "temp2" and dest == "trash":
#         send_arm_command(arm_temp2[0], arm_temp2[1], arm_temp2[2], 1, 0) # pick from temp2
#         time.sleep(2)
#         send_arm_command(arm_home[0], arm_home[1], arm_home[2], 1, 1) # home
#         time.sleep(2)
#         send_arm_command(arm_trash[0], arm_trash[1], arm_trash[2], 1, 0) # put in trash
#         time.sleep(2)
#         send_arm_command(arm_home[0], arm_home[1], arm_home[2], 0, 1) # home
#         # Wait for "done" message from serial port
#         response = ""
#         while "done" not in response.lower():
#             if ser.in_waiting:
#                 byte = ser.read(1)
#                 response += byte.decode(errors='replace')
#         print("serial responded", response)
#     elif src == "temp1" and dest == "card":
#         send_arm_command(arm_temp1[0], arm_temp1[1], arm_temp1[2], 1, 0) # pick from temp1
#         time.sleep(2)
#         send_arm_command(arm_home[0], arm_home[1], arm_home[2], 1, 1) # home
#         time.sleep(2)
#         send_arm_command(arm_values[id][0], arm_values[id][1], arm_values[id][2], 1, 0) # put in place
#         time.sleep(2)
#         send_arm_command(arm_home[0], arm_home[1], arm_home[2], 0, 1) # home
#         # Wait for "done" message from serial port
#         response = ""
#         while "done" not in response.lower():
#             if ser.in_waiting:
#                 byte = ser.read(1)
#                 response += byte.decode(errors='replace')
#         print("serial responded", response)
#     elif src == "temp2" and dest == "card":
#         send_arm_command(arm_temp2[0], arm_temp2[1], arm_temp2[2], 1, 0) # pick from temp2
#         time.sleep(2)
#         send_arm_command(arm_home[0], arm_home[1], arm_home[2], 1, 1) # home
#         time.sleep(2)
#         send_arm_command(arm_values[id][0], arm_values[id][1], arm_values[id][2], 1, 0) # put in place
#         time.sleep(2)
#         send_arm_command(arm_home[0], arm_home[1], arm_home[2], 0, 1) # home
#         # Wait for "done" message from serial port
#         response = ""
#         while "done" not in response.lower():
#             if ser.in_waiting:
#                 byte = ser.read(1)
#                 response += byte.decode(errors='replace')
#         print("serial responded", response)

def find_board_corners(frame):
    """Find the four corners of the game board with a white frame in the image"""
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define white color range in HSV
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])
    
    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Optional: Morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Check if contour is large enough to be the board
    if cv2.contourArea(largest_contour) < 10000:  # Adjust this threshold as needed
        return None
    
    # Approximate the contour to a polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # If the polygon has 4 vertices, we assume it's the board
    if len(approx) == 4:
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
    sorted_corners[3] = corners[np.argmin(diff_corners)]   # top-right
    sorted_corners[1] = corners[np.argmax(diff_corners)]   # bottom-left
    
    return sorted_corners

def transform_board(frame, corners):
    """Apply perspective transform to get top-down view of board"""
    # Define the destination points for the transform
    # Create a square output image
    width = 400  # Output size of the board
    height = 200
    
    dst_points = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ], dtype=np.float32)
    
    # Calculate perspective transform matrix
    M = cv2.getPerspectiveTransform(corners, dst_points)
    
    # Apply transform
    warped = cv2.warpPerspective(frame, M, (width, height))
    
    return warped



# Function to detect card color
def detectColor(card_id):
    
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
            row = card_id // GRID_COLS
            col = card_id % GRID_COLS
            
            # Check each cell for colors
            for color in colorRanges:
                mask = np.zeros(warped_board.shape[:2], dtype=np.uint8)
                for l, u in zip(color['lower'], color['upper']):
                    lower = np.array(l, dtype=np.uint8)
                    upper = np.array(u, dtype=np.uint8)
                    mask_part = cv2.inRange(hsv, lower, upper)
                    mask = cv2.bitwise_or(mask, mask_part)
                
                    x1 = col * cell_width
                    y1 = row * cell_height
                    x2 = (col + 1) * cell_width
                    y2 = (row + 1) * cell_height
                    
                    # Count pixels of this color in the cell
                    cell_mask = mask[y1:y2, x1:x2]
                    pixel_count = cv2.countNonZero(cell_mask)
                    color_counts[row][col][color['name']] = pixel_count
        
            # Find dominant color for cell at grid[row][col]
            if color_counts[row][col]:  # If any colors detected
                # Find color with maximum pixel count
                dominant_color = max(color_counts[row][col], key=color_counts[row][col].get)
                dominant_count = color_counts[row][col][dominant_color]
                
                if dominant_count > CELL_THRESHOLD:
                    if dominant_color == "black":
                        print(f"Cell ({row},{col}) is mostly black, treating as not flipped.")
                        grid[row][col] = None  # Or skip updating card state
                    else:
                        grid[row][col] = dominant_color
                    # Get the BGR color for the dominant color
                    color_bgr = next(c['bgr'] for c in colorRanges if c['name'] == dominant_color)
                    # Draw rectangle with the dominant color
                    x1 = col * cell_width
                    y1 = row * cell_height
                    x2 = (col + 1) * cell_width
                    y2 = (row + 1) * cell_height
                    cv2.rectangle(warped_board, (x1, y1), (x2, y2), color_bgr, 3)
                    cv2.putText(warped_board, f"({row},{col})", (x1 + 10, y1 + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)
            
            print("\nCurrent Grid Detection:")
            for grid_row in grid:  # Changed variable name from 'row' to 'grid_row' 
                print([cell if cell else "---" for cell in grid_row])
            
            # Convert card_id to grid coordinates
            
            # Get the color at those coordinates
            detected_color = grid[row][col]
            
            # For debugging
            print(f"Card {card_id} at position ({row},{col}) is {detected_color if detected_color else 'not detected'}")
            
            # Return the detected color
            if(detected_color):
              return detected_color
            
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

# Function to choose a random card that has not been flipped or visited before
def chooseRandomCard(): 
    unvisited_cards = [card_id for card_id, state in card_states.items() if not state["isFlippedBefore"]]
    print(f"Unvisited cards: {unvisited_cards}")  # Debug
    return random.choice(unvisited_cards) if unvisited_cards else None

def showCard(card_id):
    # Check if the card's color is already identified
    if card_states[card_id]["color"] is None:
        # If not, detect the color and update the card's state
        print(f"Showing card {card_id} with newly detected color {selected_colors[card_id]}")  # Debug
        # card_states[card_id] = {"isFlippedBefore": True, "color": selected_colors[card_id]}
        # Wait for a mouse click to continue
        # print(card_id)
        # print("Press Enter to continue...")
        # waiting_for_key = True
        # while waiting_for_key:
        #     for event in pygame.event.get():
        #         if event.type == pygame.QUIT:
        #             pygame.quit()
        #             exit()
        #         if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
        #             waiting_for_key = False
        #             break

        # Add the card to the colors_found dictionary
        color = detectColor(card_id)
        card_states[card_id] = {"isFlippedBefore": True, "color": color}
        colors_found[color].append(card_id)
        drawGrid()
    else:
        # If the color is already identified, just mark the card as flipped
        print(f"Showing card {card_id} with already identified color {card_states[card_id]['color']}")  # Debug  
        card_states[card_id]["isFlippedBefore"] = True
        # print("Press Enter to continue...")
        # print(card_id)
        # waiting_for_key = True
        # while waiting_for_key:
        #     for event in pygame.event.get():
        #         if event.type == pygame.QUIT:
        #             pygame.quit()
        #             exit()
        #         if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
        #             waiting_for_key = False
        #             break
        drawGrid()


# Function to hide card (flip it back)
def hideCard(card_id):
    print(f"Hiding card {card_id}")  # Debug
    # to avoid choosing same card again 
    # (to be more clear please don't set it back to false, it removes the whole purpose of it)
    # card_states[card_id]["isFlippedBefore"] = False 
    # print(card_id)
    # print("Press Enter to continue...")
    # waiting_for_key = True
    # while waiting_for_key:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             pygame.quit()
    #             exit()
    #         if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
    #             waiting_for_key = False
    #             break
    drawGrid()

# Function to find a pair of cards with the same color
def findPair():
    for color, card_ids in colors_found.items():
        if len(card_ids) >= 2:
            print(f"Found pair: {card_ids[0]} and {card_ids[1]} with color {color}")  # Debug
            return card_ids[0], card_ids[1]
    print("No pairs found")  # Debug
    return None

# Function to find a matching card for a flipped card
def findMatch(card_id): 
    color = card_states[card_id]["color"]
    print(f"Looking for match for card {card_id} with color {color}")  # Debug
    return next((other_card_id for other_card_id in colors_found[color] if other_card_id != card_id), None)

# Function to draw the grid of cards
def drawGrid():
    for i in range(GRID_ROWS):
        for j in range(GRID_COLS):
            index = i * GRID_COLS + j
            state = card_states[index]
            color = colors[state["color"]] if state["isFlippedBefore"] else BLACK
            pygame.draw.rect(screen, color, 
                             (j * (CARD_SIZE + SPACE) + SPACE, 
                              i * (CARD_SIZE + SPACE) + SPACE, 
                              CARD_SIZE, CARD_SIZE))
            pygame.draw.rect(screen, BLACK, 
                             (j * (CARD_SIZE + SPACE) + SPACE, 
                              i * (CARD_SIZE + SPACE) + SPACE, 
                              CARD_SIZE, CARD_SIZE), 3)
    pygame.display.flip()

# Draw the grid and update the display
screen.fill(BACKGROUND_COLOR)
drawGrid()
cap = cv2.VideoCapture(URL)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Game loop
running = True
pairs_found = 0
current_flipped_cards = []
clock = pygame.time.Clock()
# send_arm_command(arm_home[0], arm_home[1], arm_home[2], 0, 1) # home


while pairs_found < (GRID_ROWS * GRID_COLS // 2) and running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    print(f"\nCurrent flipped cards: {current_flipped_cards}")  # Debug
    print(f"Pairs found: {pairs_found}")  # Debug

    pygame.time.wait(FLIP_DELAY)

    # If no cards are flipped
    if len(current_flipped_cards) == 0:
        print("No cards flipped. Checking for pairs...")  # Debug
        # Check if any known pair exists in colors_found
        pair_ids = findPair()
        if pair_ids:
            print(f"Pair found: {pair_ids}")  # Debug
            # Flip both matching cards
            #from_to("card", "trash", pair_ids[0])
            showCard(pair_ids[0])
            #from_to("card", "trash", pair_ids[1])
            showCard(pair_ids[1])
            pygame.time.wait(FLIP_DELAY)

            # Update states
            colors_found[card_states[pair_ids[0]]["color"]].remove(pair_ids[0])
            colors_found[card_states[pair_ids[1]]["color"]].remove(pair_ids[1])
            pairs_found += 1
            print(f"Updated pairs_found: {pairs_found}")  # Debug
            pygame.time.wait(FLIP_DELAY)
            # flip 
            continue  # Continue to next loop iteration

        # If no known pair, choose a random card and flip it
        card_id = chooseRandomCard()
        if card_id is not None:
            print(f"Flipping random card: {card_id}")  # Debug
            #from_to("card", "temp1", card_id)
            current_flipped_cards.append(card_id)
            showCard(card_id)
            pygame.time.wait(FLIP_DELAY)

    # If one card is flipped
    elif len(current_flipped_cards) == 1:
        print("One card flipped. Looking for match...")  # Debug
        # Check if its match has been flipped before
        matched_card_id = findMatch(current_flipped_cards[0])
        if matched_card_id:
            print(f"Match found: {matched_card_id}")  # Debug
            # Flip the matching card
            #from_to("temp1", "trash", current_flipped_cards[0])
            #from_to("card", "trash", matched_card_id)
            showCard(matched_card_id)  # Fix: changed from showCard(0) to show the actual matching card
            pygame.time.wait(FLIP_DELAY)

            # Update states
            colors_found[card_states[current_flipped_cards[0]]["color"]].remove(current_flipped_cards[0])
            colors_found[card_states[matched_card_id]["color"]].remove(matched_card_id)
            pairs_found += 1
            print(f"Updated pairs_found: {pairs_found}")  # Debug
            
            # Add delay to see both matched cards before proceeding
            pygame.time.wait(FLIP_DELAY)
            
            current_flipped_cards.clear()
            continue  # Continue to next loop iteration

        # If no match found, choose another random card and flip it
        card_id = chooseRandomCard()
        if card_id is not None:
            print(f"Flipping another random card: {card_id}")  # Debug
            #from_to("card", "temp2", card_id)
            showCard(card_id)
            current_flipped_cards.append(card_id)
            pygame.time.wait(FLIP_DELAY)

    # If two cards are flipped
    elif len(current_flipped_cards) == 2:
        print("Two cards flipped. Checking for match...")  # Debug
        # Check if they match
        color1 = card_states[current_flipped_cards[0]]["color"]
        color2 = card_states[current_flipped_cards[1]]["color"]

        if color1 == color2:
            print("Cards match! Keeping them flipped.")  # Debug
            # If match, keep them flipped and update states
            colors_found[color1].remove(current_flipped_cards[0])
            colors_found[color2].remove(current_flipped_cards[1])
            pairs_found += 1
            print(f"Updated pairs_found: {pairs_found}")  # Debug#from_to
            #from_to("temp1", "trash", current_flipped_cards[0])
            #from_to("temp2", "trash", current_flipped_cards[1])
            pygame.time.wait(FLIP_DELAY)
        
        else:
            print("Cards do not match. Hiding them.")  # Debug
            # If no match, hide both cards
            #from_to("temp1", "card", current_flipped_cards[0])
            hideCard(current_flipped_cards[0])
            #from_to("temp2", "card", current_flipped_cards[1])
            hideCard(current_flipped_cards[1])
            pygame.time.wait(FLIP_DELAY)

        # Reset current_flipped_cards for the next round
        current_flipped_cards.clear()

    clock.tick(30)  # Limit the frame rate to 30 FPS

pygame.quit()
