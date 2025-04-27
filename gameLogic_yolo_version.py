import pygame
import random
import serial
import time
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import os

# Arm Values
arm_values =[[110, 40, 125], [90, 65, 120], [90,110,120], [110,140,125], [150,55,155], [130,80,140], [130,105,140], [150,125,155]]
arm_home = [180, 90, 0]
arm_temp1 = [90, 0, 120] # change later
arm_temp2 = [90, 180, 120] # change later
arm_trash = [140, 0, 140] # change later

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

# Update the object colors for display (these will be used as fallbacks)
object_colors = {
    "orange": (255, 165, 0),
    "apple": (255, 0, 0),
    "cat": (0, 0, 255),
    "car": (0, 255, 255),
    "umbrella": (0, 255, 0),
    "banana": (255, 255, 255),
    "fire hydrant": (255, 128, 0),
    "person": (0, 0, 0)
}

# Define object detection labels and their colors
def assign_color(label):
    color_map = {
        'orange': (0, 165, 255),    # Orange color (BGR)
        'apple': (0, 0, 255),       # Red color for apple
        'cat': (255, 0, 0),         # Blue for cat
        'car': (255, 255, 0),       # Cyan for car
        'umbrella': (0, 255, 0),    # Green for umbrella
        'banana': (255, 255, 255),  # White for banana
        'fire hydrant': (0, 128, 255),  # Orange for fire hydrant
        'person': (0, 0, 0),        # Black for person
    }
    return color_map.get(label, (255, 255, 255))  # Default to white if label not found

# Initialize YOLO model
model = YOLO("yolov5s.pt")  # Using standard YOLOv5s model
target_labels = ['orange', 'apple', 'cat', 'car', 'umbrella', 'banana', 'fire hydrant', 'person']
print("YOLO model loaded. Detecting:", target_labels)

# Camera frame dimensions
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Fixed grid dimensions in the camera frame
GRID_TOP = 90    # Y coordinate of top of grid in camera frame
GRID_LEFT = 30   # X coordinate of left of grid in camera frame
GRID_WIDTH = 550  # Width of grid in camera frame
GRID_HEIGHT = 280 # Height of grid in camera frame

# Add a constant for the window name (add this near other constants)
CAMERA_WINDOW = "Memory Game - Card Detection"

# Add a new constant for the board window
BOARD_WINDOW = "Memory Game - Board View"

def predict(model, img, target_labels, conf=0.5, rectangle=2, text=2):
    results = model.predict(img, conf=conf, verbose=False)
    detected_objects = []

    # Visualize detections
    for result in results:
        for box in result.boxes:
            label = result.names[int(box.cls[0])].lower()
            if label in target_labels:  # Only detect specific objects
                score = box.conf.item()
                color = assign_color(label)
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]),
                              color=color, thickness=rectangle)
                cv2.putText(img, f"{label} ({score:.2f})", (xyxy[0], xyxy[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=color, thickness=text)
                detected_objects.append(label)
    
    return img, results, detected_objects

# Initialize card states: False for unvisited (not flipped before), and None for unknown object
card_states = {i: {"isFlippedBefore": False, "object": None} for i in range(GRID_ROWS * GRID_COLS)}
objects_found = {obj: [] for obj in target_labels}  # Initialize an empty list for each target object

# Initialize the Pygame window
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Memory Puzzle Game - YOLO Edition")

# DroidCam settings
HTTP = 'http://'
IP_ADDRESS = '192.168.2.19'  # Change to your IP
URL = HTTP + IP_ADDRESS + ':4747/mjpegfeed?640x480'

# Fixed grid size
GRID_ROWS = 2
GRID_COLS = 4

def setup_serial():
    ser = serial.Serial()
    ser.baudrate = 9600
    ser.port = 'COM5' 
    ser.timeout = 2
    
    try:
        ser.open()
        print("Serial port opened successfully!")
        return ser
    except Exception as e:
        print(f"Error opening serial port: {e}")
        return None


ser = setup_serial()

def send_arm_command(degree1, degree2, degree3, magnet, movement):
    """
    Send arm control parameters to the ESP32/ESP8266 through serial connection.
    
    Parameters:
    - ser: Serial object (already opened)
    - degree1: first servo angle (integer)
    - degree2: second servo angle (integer)
    - degree3: third servo angle (integer)
    - magnet: magnet state (integer)
    - arm: arm selection (integer)
    
    Returns:
    - Response from the ESP (if any)
    """
    # Convert all parameters to strings and join with commas
    command = f"{degree1},{degree2},{degree3},{magnet},{movement}\n"
    
    # Send the command as bytes
    ser.write(command.encode())
    
    # Wait a moment for the ESP to process and respond
    time.sleep(0.1)
    
    # Read the response (if any)
    response = b''
    while ser.in_waiting:
        response += ser.read(1)
    
    return response.decode() if response else None



def from_to(src, dest, id):
    if src == "card" and dest == "temp1":
        send_arm_command(arm_values[id][0], arm_values[id][1], arm_values[id][2], 1, 0) # pick card
        time.sleep(2)
        send_arm_command(arm_home[0], arm_home[1], arm_home[2], 1, 1) # home
        time.sleep(2)
        send_arm_command(arm_temp1[0], arm_temp1[1], arm_temp1[2], 1, 0) # put in temp1
        time.sleep(2) 
        send_arm_command(arm_home[0], arm_home[1], arm_home[2], 0, 1) # home
        # Wait for "done" message from serial port
        response = ""
        while "done" not in response.lower():
            if ser.in_waiting:
                byte = ser.read(1)
                response += byte.decode(errors='replace')
        print("serial responded", response)
    elif src == "card" and dest == "temp2":
        send_arm_command(arm_values[id][0], arm_values[id][1], arm_values[id][2], 1, 0) # pick card
        time.sleep(2)
        send_arm_command(arm_home[0],arm_home[1], arm_home[2], 1, 1) # home
        time.sleep(2)
        send_arm_command(arm_temp2[0], arm_temp2[1], arm_temp2[2], 1, 0) # put in temp2
        time.sleep(2)
        send_arm_command(arm_home[0], arm_home[1], arm_home[2], 0, 1) # home
        # Wait for "done" message from serial port
        response = ""
        while "done" not in response.lower():
            if ser.in_waiting:
                byte = ser.read(1)
                response += byte.decode(errors='replace')
        print("serial responded", response)
    elif src == "card" and dest == "trash":
        send_arm_command(arm_values[id][0], arm_values[id][1], arm_values[id][2], 1, 0) # pick card
        time.sleep(2)
        send_arm_command(arm_home[0], arm_home[1], arm_home[2], 1, 1) # home
        time.sleep(2)
        send_arm_command(arm_trash[0], arm_trash[1], arm_trash[2], 1, 0) # put in trash
        time.sleep(2)
        send_arm_command(arm_home[0], arm_home[1], arm_home[2], 0, 1) # home
        # Wait for "done" message from serial port
        response = ""
        while "done" not in response.lower():
            if ser.in_waiting:
                byte = ser.read(1)
                response += byte.decode(errors='replace')
        print("serial responded", response)
    elif src == "temp1" and dest == "trash":
        send_arm_command(arm_temp1[0], arm_temp1[1], arm_temp1[2], 1, 0) # pick from temp1
        time.sleep(2)
        send_arm_command(arm_home[0], arm_home[1], arm_home[2], 1, 1) # home
        time.sleep(2)
        send_arm_command(arm_trash[0], arm_trash[1], arm_trash[2], 1, 0) # put in trash
        time.sleep(2)
        send_arm_command(arm_home[0], arm_home[1], arm_home[2], 0, 1) # home
        # Wait for "done" message from serial port
        response = ""
        while "done" not in response.lower():
            if ser.in_waiting:
                byte = ser.read(1)
                response += byte.decode(errors='replace')
        print("serial responded", response)
    elif src == "temp2" and dest == "trash":
        send_arm_command(arm_temp2[0], arm_temp2[1], arm_temp2[2], 1, 0) # pick from temp2
        time.sleep(2)
        send_arm_command(arm_home[0], arm_home[1], arm_home[2], 1, 1) # home
        time.sleep(2)
        send_arm_command(arm_trash[0], arm_trash[1], arm_trash[2], 1, 0) # put in trash
        time.sleep(2)
        send_arm_command(arm_home[0], arm_home[1], arm_home[2], 0, 1) # home
        # Wait for "done" message from serial port
        response = ""
        while "done" not in response.lower():
            if ser.in_waiting:
                byte = ser.read(1)
                response += byte.decode(errors='replace')
        print("serial responded", response)
    elif src == "temp1" and dest == "card":
        send_arm_command(arm_temp1[0], arm_temp1[1], arm_temp1[2], 1, 0) # pick from temp1
        time.sleep(2)
        send_arm_command(arm_home[0], arm_home[1], arm_home[2], 1, 1) # home
        time.sleep(2)
        send_arm_command(arm_values[id][0], arm_values[id][1], arm_values[id][2], 1, 0) # put in place
        time.sleep(2)
        send_arm_command(arm_home[0], arm_home[1], arm_home[2], 0, 1) # home
        # Wait for "done" message from serial port
        response = ""
        while "done" not in response.lower():
            if ser.in_waiting:
                byte = ser.read(1)
                response += byte.decode(errors='replace')
        print("serial responded", response)
    elif src == "temp2" and dest == "card":
        send_arm_command(arm_temp2[0], arm_temp2[1], arm_temp2[2], 1, 0) # pick from temp2
        time.sleep(2)
        send_arm_command(arm_home[0], arm_home[1], arm_home[2], 1, 1) # home
        time.sleep(2)
        send_arm_command(arm_values[id][0], arm_values[id][1], arm_values[id][2], 1, 0) # put in place
        time.sleep(2)
        send_arm_command(arm_home[0], arm_home[1], arm_home[2], 0, 1) # home
        # Wait for "done" message from serial port
        response = ""
        while "done" not in response.lower():
            if ser.in_waiting:
                byte = ser.read(1)
                response += byte.decode(errors='replace')
        print("serial responded", response)

# New function to extract just the game board region from the frame
def extract_board_region(frame):
    """Extract just the game board region from the full camera frame"""
    if frame is None:
        return None
    # Extract the board region based on our grid coordinates
    board_region = frame[GRID_TOP:GRID_TOP+GRID_HEIGHT, GRID_LEFT:GRID_LEFT+GRID_WIDTH].copy()
    return board_region

# Function to detect all objects on the board at once
def detect_all_objects(frame):
    """Run YOLO on the entire frame and return all detected objects with their bounding boxes"""
    # Extract just the board region for detection
    board_region = extract_board_region(frame)
    if board_region is None:
        return frame, []
    
    # Show the board region in a separate window
    cv2.imshow(BOARD_WINDOW, board_region)
    
    # Run YOLO only on the board region
    results = model.predict(board_region, conf=0.4, verbose=False)
    detected_objects = []
    
    for result in results:
        for box in result.boxes:
            label = result.names[int(box.cls[0])].lower()
            if label in target_labels:
                score = box.conf.item()
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                # Adjust coordinates to be relative to the board region
                detected_objects.append({
                    'label': label,
                    'confidence': score,
                    'bbox': (xyxy[0], xyxy[1], xyxy[2], xyxy[3])  # x1, y1, x2, y2 (within board region)
                })
                
                # Draw bounding box on board region
                cv2.rectangle(board_region, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]),
                              color=assign_color(label), thickness=2)
                cv2.putText(board_region, f"{label} ({score:.2f})", (xyxy[0], xyxy[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=assign_color(label), thickness=2)
                
                # Also draw on main frame for reference
                cv2.rectangle(frame, 
                             (xyxy[0] + GRID_LEFT, xyxy[1] + GRID_TOP), 
                             (xyxy[2] + GRID_LEFT, xyxy[3] + GRID_TOP),
                             color=assign_color(label), thickness=2)
    
    # Update the board window with detections
    cv2.imshow(BOARD_WINDOW, board_region)
    
    return frame, detected_objects

# Function to check if a bounding box overlaps with a card position
def is_object_in_card(card_id, detected_objects):
    """Check if any detected object overlaps with the given card position"""
    # Calculate card position in grid
    row = card_id // GRID_COLS
    col = card_id % GRID_COLS
    
    # Calculate cell dimensions
    cell_width = GRID_WIDTH // GRID_COLS
    cell_height = GRID_HEIGHT // GRID_ROWS
    
    # Calculate card position within the board region (not the full frame)
    card_x = col * cell_width
    card_y = row * cell_height
    card_right = card_x + cell_width
    card_bottom = card_y + cell_height
    
    # Calculate center point of card
    card_center_x = card_x + cell_width // 2
    card_center_y = card_y + cell_height // 2
    
    # For each detected object, check if its bounding box overlaps with this card
    for obj in detected_objects:
        x1, y1, x2, y2 = obj['bbox']
        
        # Simple center point check - if the center of the card is within the bounding box
        if (x1 <= card_center_x <= x2 and y1 <= card_center_y <= y2):
            return obj['label']
            
    return None

# Function to detect card object using YOLO with fixed camera position
def detectObject(card_id):
    print(f"Detecting card {card_id} using YOLO...")
    
    while True:
        frame = get_frame()  # Use our threaded frame getter
        if frame is None:
            print("Failed to read frame")
            time.sleep(0.5)
            continue
        
        # Calculate cell dimensions based on fixed grid
        cell_width = GRID_WIDTH // GRID_COLS
        cell_height = GRID_HEIGHT // GRID_ROWS
        
        # Calculate card position in grid
        row = card_id // GRID_COLS
        col = card_id % GRID_COLS
        
        # Calculate card position in camera frame
        card_x = GRID_LEFT + (col * cell_width)
        card_y = GRID_TOP + (row * cell_height)
        
        # Draw the grid overlay on the frame for debugging
        cv2.rectangle(frame, (GRID_LEFT, GRID_TOP), 
                     (GRID_LEFT + GRID_WIDTH, GRID_TOP + GRID_HEIGHT), 
                     (255, 255, 255), 2)
        
        for i in range(1, GRID_ROWS):
            y = GRID_TOP + (i * cell_height)
            cv2.line(frame, (GRID_LEFT, y), (GRID_LEFT + GRID_WIDTH, y), (255, 255, 255), 1)
            
        for j in range(1, GRID_COLS):
            x = GRID_LEFT + (j * cell_width)
            cv2.line(frame, (x, GRID_TOP), (x, GRID_TOP + GRID_HEIGHT), (255, 255, 255), 1)
        
        # Highlight the current card in both windows
        cv2.rectangle(frame, 
                     (card_x, card_y), 
                     (card_x + cell_width, card_y + cell_height), 
                     (0, 255, 0), 3)  # Make it thicker and more visible
        
        # Run YOLO detection on the board region
        processed_frame, all_objects = detect_all_objects(frame)
        
        # Check if any object is detected at this card's position
        detected_object_label = is_object_in_card(card_id, all_objects)
        
        # Display debugging info on frame
        cv2.putText(processed_frame, f"Detecting Card {card_id}", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if detected_object_label:
            cv2.putText(processed_frame, f"Detected: {detected_object_label}", (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(processed_frame, "No object detected - Searching...", (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the frame with detections (using the consistent window name)
        cv2.imshow(CAMERA_WINDOW, processed_frame)
        
        # Wait a short time for a keypress
        key = cv2.waitKey(100) & 0xFF
        if key == 27:  # ESC key - emergency exit
            print("Detection aborted by user")
            return random.choice(target_labels)
        
        # If we detected an object, return it
        if detected_object_label:
            print(f"Detected object: {detected_object_label}")
            return detected_object_label
        
        print(f"No object detected at card {card_id}, still searching...")

# Wait for user to position the camera - update to use the board window
def wait_for_camera_positioning():
    print("Positioning the camera...")
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    while True:
        frame = get_frame()
        if frame is None:
            time.sleep(0.1)
            continue
        
        # Draw grid overlay to help with positioning
        cv2.rectangle(frame, (GRID_LEFT, GRID_TOP), 
                     (GRID_LEFT + GRID_WIDTH, GRID_TOP + GRID_HEIGHT), 
                     (255, 255, 255), 2)
        
        cell_width = GRID_WIDTH // GRID_COLS
        cell_height = GRID_HEIGHT // GRID_ROWS
        
        for i in range(1, GRID_ROWS):
            y = GRID_TOP + (i * cell_height)
            cv2.line(frame, (GRID_LEFT, y), (GRID_LEFT + GRID_WIDTH, y), (255, 255, 255), 1)
            
        for j in range(1, GRID_COLS):
            x = GRID_LEFT + (j * cell_width)
            cv2.line(frame, (x, GRID_TOP), (x, GRID_TOP + GRID_HEIGHT), (255, 255, 255), 1)
        
        # Draw instructions
        cv2.putText(frame, "Position camera to see all cards within the grid", 
                   (20, FRAME_HEIGHT - 60), font, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press SPACE when ready to start the game", 
                   (20, FRAME_HEIGHT - 30), font, 0.7, (0, 255, 0), 2)
        
        # Use the consistent window name
        cv2.imshow(CAMERA_WINDOW, frame)
        
        # Also show just the board region
        board_region = extract_board_region(frame)
        if board_region is not None:
            cv2.imshow(BOARD_WINDOW, board_region)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # Space key
            break

# Function to choose a random card that has not been flipped or visited before
def chooseRandomCard(): 
    unvisited_cards = [card_id for card_id, state in card_states.items() if not state["isFlippedBefore"]]
    print(f"Unvisited cards: {unvisited_cards}")  # Debug
    return random.choice(unvisited_cards) if unvisited_cards else None

# Add a function to wait for key press with a prompt
def wait_for_key_press(message, key=32, window_name=CAMERA_WINDOW):
    """
    Wait for a specific key press while displaying a message
    
    Parameters:
    - message: Message to display
    - key: ASCII code of the key to wait for (default is 32 = SPACE)
    - window_name: OpenCV window to display the message in
    
    Returns:
    - True if the specific key was pressed, False if ESC was pressed
    """
    print(f"Waiting for key press: {message}")
    
    # Get the current frame
    frame = get_frame()
    if frame is None:
        # If no frame available, create a blank one
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    
    # Add the message to the frame
    cv2.putText(frame, message, (20, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Press SPACE to continue or ESC to abort", 
               (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow(window_name, frame)
    
    # Wait for key press
    while True:
        pressed_key = cv2.waitKey(100) & 0xFF
        if pressed_key == 27:  # ESC
            return False
        elif pressed_key == key:  # Default is SPACE
            return True

def showCard(card_id):
    # Check if the card's object is already identified
    if card_states[card_id]["object"] is None:
        # If not, detect the object and update the card's state
        print(f"Showing card {card_id}")
        
        # # Pause and wait for user to continue
        # if not wait_for_key_press(f"Ready to detect card {card_id}?"):
        #     print("Detection aborted by user")
        #     return
        
        # Detect the object on this card
        detected_object = detectObject(card_id)
        
        # Update card state
        card_states[card_id] = {"isFlippedBefore": True, "object": detected_object}
        objects_found[detected_object].append(card_id)
        drawGrid()
        
        # Pause to let user see the card
        # wait_for_key_press(f"Detected {detected_object} on card {card_id}. Continue?")
    else:
        # If the object is already identified, just mark the card as flipped
        print(f"Showing card {card_id} with already identified object {card_states[card_id]['object']}")
        card_states[card_id]["isFlippedBefore"] = True
        drawGrid()
        
        # Pause to let user see the card
        # wait_for_key_press(f"Showing card {card_id} with {card_states[card_id]['object']}. Continue?")

# Function to hide card (flip it back)
def hideCard(card_id):
    print(f"Hiding card {card_id}")
    
    # Pause and wait for user to continue
    # if not wait_for_key_press(f"Ready to hide card {card_id}?"):
    #     print("Hide operation aborted by user")
    #     return
    
    drawGrid()

# Function to find a pair of cards with the same object
def findPair():
    for obj, card_ids in objects_found.items():
        if len(card_ids) >= 2:
            print(f"Found pair: {card_ids[0]} and {card_ids[1]} with object {obj}")
            return card_ids[0], card_ids[1]
    print("No pairs found")
    return None

# Function to find a matching card for a flipped card
def findMatch(card_id): 
    obj = card_states[card_id]["object"]
    print(f"Looking for match for card {card_id} with object {obj}")
    return next((other_card_id for other_card_id in objects_found[obj] if other_card_id != card_id), None)

# Create a dictionary to store loaded images
object_images = {}

# Function to load all object images
def load_object_images():
    global object_images
    resource_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Resources")
    
    if not os.path.exists(resource_path):
        print(f"Warning: Resources directory not found at {resource_path}")
        print("Creating Resources directory...")
        os.makedirs(resource_path, exist_ok=True)
        print(f"Please add object images to: {resource_path}")
    
    print(f"Loading images from {resource_path}...")
    
    # Try to load an image for each object
    for obj in target_labels:
        # Try different common file extensions
        for ext in ['.png', '.jpg', '.jpeg']:
            img_path = os.path.join(resource_path, f"{obj.replace(' ', '_')}{ext}")
            if os.path.exists(img_path):
                try:
                    # Load and scale the image to fit the card size
                    img = pygame.image.load(img_path)
                    img = pygame.transform.scale(img, (CARD_SIZE - 10, CARD_SIZE - 10))
                    object_images[obj] = img
                    print(f"Loaded image for {obj}: {img_path}")
                    break
                except Exception as e:
                    print(f"Error loading image for {obj}: {e}"
                          )
        
        # If no image was loaded for this object, inform the user
        if obj not in object_images:
            print(f"No image found for {obj}. Please add {obj}.png, {obj}.jpg or {obj}.jpeg to the Resources directory.")

# Function to draw the grid of cards
def drawGrid():
    for i in range(GRID_ROWS):
        for j in range(GRID_COLS):
            index = i * GRID_COLS + j
            state = card_states[index]
            
            # Default to black background for unflipped cards
            bg_color = BLACK
            
            # Calculate card position
            card_x = j * (CARD_SIZE + SPACE) + SPACE
            card_y = i * (CARD_SIZE + SPACE) + SPACE
            
            # Draw the card background
            pygame.draw.rect(screen, bg_color, (card_x, card_y, CARD_SIZE, CARD_SIZE))
            
            # Draw border
            pygame.draw.rect(screen, (50, 50, 50), (card_x, card_y, CARD_SIZE, CARD_SIZE), 3)
            
            # If the card is flipped, display the object image or name
            if state["isFlippedBefore"] and state["object"] is not None:
                obj = state["object"]
                
                if obj in object_images:
                    # Display the image
                    screen.blit(object_images[obj], (card_x + 5, card_y + 5))
                else:
                    # Fallback to colored background with text
                    bg_color = object_colors.get(obj, (200, 200, 200))
                    pygame.draw.rect(screen, bg_color, (card_x + 3, card_y + 3, CARD_SIZE - 6, CARD_SIZE - 6))
                    
                    # Also display the object name
                    font = pygame.font.SysFont('Arial', 20)
                    text_surface = font.render(obj, True, (255, 255, 255))
                    text_rect = text_surface.get_rect(center=(card_x + CARD_SIZE // 2, card_y + CARD_SIZE // 2))
                    screen.blit(text_surface, text_rect)
                
    pygame.display.flip()

# Draw the grid and update the display
screen.fill(BACKGROUND_COLOR)
load_object_images()  # Load all object images before drawing the grid
drawGrid()

# Create a class for threading camera capture
class CameraThread:
    def __init__(self, source):
        self.source = source
        self.cap = None
        self.frame = None
        self.frame_lock = threading.Lock()
        self.running = False
        self.thread = None
    
    def start(self):
        self.cap = cv2.VideoCapture(self.source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not self.cap.isOpened():
            print("Error: Could not open video stream.")
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self.update)
        self.thread.daemon = True
        self.thread.start()
        return True
    
    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.frame = frame
            time.sleep(0.01)  # Small sleep to prevent CPU hogging
    
    def get_frame(self):
        with self.frame_lock:
            if self.frame is not None:
                return self.frame.copy()
            return None
    
    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()
        if self.cap is not None:
            self.cap.release()

# Function to get the latest frame
def get_frame():
    return camera.get_frame()

# Initialize camera with threading
camera = CameraThread(URL)
if not camera.start():
    print("Failed to start camera thread")
    pygame.quit()
    exit()

# Wait for user to position the camera
wait_for_camera_positioning()

# Wait for user to start the game
# wait_for_key_press("Camera positioned. Press SPACE to start the game!")

# Game loop
running = True
pairs_found = 0
current_flipped_cards = []
clock = pygame.time.Clock()
send_arm_command(arm_home[0], arm_home[1], arm_home[2], 0, 1) # home

while pairs_found < (GRID_ROWS * GRID_COLS // 2) and running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    print(f"\nCurrent flipped cards: {current_flipped_cards}")
    print(f"Pairs found: {pairs_found}")

    pygame.time.wait(FLIP_DELAY)

    # If no cards are flipped
    if len(current_flipped_cards) == 0:
        print("No cards flipped. Checking for pairs...")
        # Check if any known pair exists in objects_found
        pair_ids = findPair()
        if pair_ids:
            print(f"Pair found: {pair_ids}")
            # Flip both matching cards
            from_to("card", "trash", pair_ids[0])
            showCard(pair_ids[0])
            from_to("card", "trash", pair_ids[1])
            showCard(pair_ids[1])
            pygame.time.wait(FLIP_DELAY)

            # Update states
            objects_found[card_states[pair_ids[0]]["object"]].remove(pair_ids[0])
            objects_found[card_states[pair_ids[1]]["object"]].remove(pair_ids[1])
            pairs_found += 1
            print(f"Updated pairs_found: {pairs_found}")
            pygame.time.wait(FLIP_DELAY)
            continue

        # If no known pair, choose a random card and flip it
        card_id = chooseRandomCard()
        if card_id is not None:
            print(f"Flipping random card: {card_id}")
            from_to("card", "temp1", card_id)
            current_flipped_cards.append(card_id)
            showCard(card_id)
            pygame.time.wait(FLIP_DELAY)

    # If one card is flipped
    elif len(current_flipped_cards) == 1:
        print("One card flipped. Looking for match...")
        # Check if its match has been flipped before
        matched_card_id = findMatch(current_flipped_cards[0])
        if matched_card_id:
            print(f"Match found: {matched_card_id}")
            # Flip the matching card
            from_to("temp1", "trash", current_flipped_cards[0])
            from_to("card", "trash", matched_card_id)
            showCard(matched_card_id)
            pygame.time.wait(FLIP_DELAY)

            # Update states
            objects_found[card_states[current_flipped_cards[0]]["object"]].remove(current_flipped_cards[0])
            objects_found[card_states[matched_card_id]["object"]].remove(matched_card_id)
            pairs_found += 1
            print(f"Updated pairs_found: {pairs_found}")
            
            pygame.time.wait(FLIP_DELAY)
            current_flipped_cards.clear()
            continue

        # If no match found, choose another random card and flip it
        card_id = chooseRandomCard()
        if card_id is not None:
            print(f"Flipping another random card: {card_id}")
            from_to("card", "temp2", card_id)
            showCard(card_id)
            current_flipped_cards.append(card_id)
            pygame.time.wait(FLIP_DELAY)

    # If two cards are flipped
    elif len(current_flipped_cards) == 2:
        print("Two cards flipped. Checking for match...")
        # Check if they match
        object1 = card_states[current_flipped_cards[0]]["object"]
        object2 = card_states[current_flipped_cards[1]]["object"]

        if object1 == object2:
            print("Cards match! Keeping them flipped.")
            # If match, keep them flipped and update states
            objects_found[object1].remove(current_flipped_cards[0])
            objects_found[object2].remove(current_flipped_cards[1])
            pairs_found += 1
            print(f"Updated pairs_found: {pairs_found}")
            from_to("temp1", "trash", current_flipped_cards[0])
            from_to("temp2", "trash", current_flipped_cards[1])
            pygame.time.wait(FLIP_DELAY)
        
        else:
            print("Cards do not match. Hiding them.")
            # If no match, hide both cards
            from_to("temp1", "card", current_flipped_cards[0])
            hideCard(current_flipped_cards[0])
            from_to("temp2", "card", current_flipped_cards[1])
            hideCard(current_flipped_cards[1])
            pygame.time.wait(FLIP_DELAY)

        # Reset current_flipped_cards for the next round
        current_flipped_cards.clear()

    clock.tick(30)  # Limit the frame rate to 30 FPS

# Clean up when game ends
camera.stop()  # Stop the camera thread
cv2.destroyWindow(CAMERA_WINDOW)  # Destroy our specific window
cv2.destroyWindow(BOARD_WINDOW)  # Destroy the board window
cv2.destroyAllWindows()  # Just to be sure all windows are closed
pygame.quit()