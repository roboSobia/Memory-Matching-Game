<div align='center'>
  <h1>🤖 Memory Matching Game with Robotic Arm 🎮</h1>
</div>

## 🌟 Game Description

This project features a robotic arm playing a memory-matching game. The primary goal is to have the robot autonomously:
1.  **Reveal** cards on a game board.
2.  **Identify** the content of the cards (either by color or by recognizing objects).
3.  **Memorize** the locations and contents of revealed cards.
4.  **Match** pairs of cards with the same value/pattern.
5.  **Remove** matched pairs from the board.

The project explores two main approaches for card content identification:
*   **Solid Colored Cards:** The initial and more tested approach, where cards are identified by their distinct solid colors.
*   **Object-Based Cards (YOLO):** An advanced approach using YOLO (You Only Look Once) object detection to identify specific items depicted on the cards.

>[!NOTE]
> **Current Progress:**
> - Color detection using OpenCV (HSV color spaces) is implemented and tested, including dynamic board detection and perspective warping.
> - Object detection using YOLOv5s is also implemented, capable of recognizing a predefined set of objects from images, with a fixed grid assumption for card locations.
> - Game logic for both modes (tracking card states, finding pairs) is in place.
> - Robotic arm control logic (`from_to` movement sequences) is defined in the Python scripts, ready for full integration with the physical arm via serial communication.

---

## ✨ Key Features

*   **Dual Detection Modes:**
    *   🎨 **Color-Based Detection:** Identifies cards by their solid color using OpenCV.
    *   👁️ **Object-Based Detection (YOLO):** Recognizes objects on cards using a pre-trained YOLOv5s model.
*   **Robotic Arm Integration:**
    *   Predefined arm movement sequences (`arm_values`, `from_to` functions) for picking, placing, and discarding cards.
    *   Serial communication setup for interfacing with an ESP32/Arduino-controlled arm.
*   **Dynamic & Static Board Recognition:**
    *   Dynamic board corner detection and perspective transformation for color mode.
    *   Fixed grid layout assumption for YOLO mode.
*   **Interactive Pygame UI:**
    *   Visual representation of the game board and card states.
    *   Displays detected objects/colors on flipped cards.
*   **Camera Input:**
    *   Utilizes DroidCam for live video feed from a smartphone.
    *   Threaded camera capture for smoother performance (in YOLO version).
*   **Modular Game Logic:** Manages card states, identifies pairs, and controls the game flow.

---

## 🛠️ Tech Stack

*   **Programming Language:** Python
*   **Computer Vision:** OpenCV, Ultralytics (for YOLOv5)
*   **User Interface:** Pygame
*   **Robotics Control:** ESP32/Arduino (via Serial Communication)
*   **Camera:** DroidCam (or any standard webcam)

---

## 🔩 Hardware Mechanism</color>

 Pick, Show, and Place (Magnet/Gripper)
*   **Concept:** The robot arm uses a gripper (possibly with a small electromagnet if cards have a metallic component, or a suction cup/mechanical gripper) to pick up a card, rotate it to show its face to the camera, and then place it back or into a temporary holding spot.
*   **Steps:**
    1.  Arm moves to a card position.
    2.  Grips/lifts the card.
    3.  Orients the card towards the camera for detection.
    4.  Python script processes the image and identifies the card.
    5.  Arm places the card back or to a temporary "revealed" spot (e.g., `arm_temp1`, `arm_temp2` in the code).
    6.  If a pair is made, cards are moved to a "trash" or "matched" pile (`arm_trash`).
*   **Advantages:** More robust against slight misalignments, allows for temporary holding spots.
*   **Considerations:**
    *   As noted: <mark style="background: #FFF3A3A6;">This method, if cards are immediately placed back, doesn't easily keep two cards revealed simultaneously for human visual confirmation. The current code logic with `arm_temp1` and `arm_temp2` addresses this by providing temporary holding locations for up to two cards.</mark>

---

## 💻 Software Mechanism</color>

### <mark style="background: #ABF7F7A6;"> Responsibilities </mark>

1.  **Python Script (Laptop/PC):**
    *   Captures video frames (e.g., via DroidCam).
    *   **Card Content Detection:**
        *   **Color Mode:** Processes frames using OpenCV, detects board corners, warps perspective, segments colors in HSV space to identify the dominant color of a card.
        *   **YOLO Mode:** Uses a YOLOv5 model to detect predefined objects within specific regions of the camera feed (assuming a fixed grid).
    *   Manages game state: tracks flipped cards, identified contents, and found pairs (`card_states`, `objects_found`/`colors_found`).
    *   Implements game strategy (e.g., choosing which card to flip next).
    *   Sends commands (servo angles, magnet state) to the ESP32/Arduino via serial communication to control the robotic arm.
    *   Displays the game status through a Pygame interface.

2.  **ESP32/Arduino Code (Microcontroller):**
    *   Receives commands from the Python script over the serial port.
    *   Parses commands to determine target servo angles and effector (magnet/gripper) actions.
    *   Controls the servo motors of the robotic arm to execute the required movements (e.g., moving to card, picking, placing, flipping).
    *   Sends a "done" signal back to Python upon completing a movement sequence.

### <mark style="background: #ABF7F7A6;"> Robotic Arm Control Logic </mark>
*   **Predefined Positions:** `arm_values` list stores servo angles for each card slot on the board. `arm_home`, `arm_temp1`, `arm_temp2`, `arm_trash` define key locations.
*   **Movement Sequences:** The `from_to(src, dest, id)` function orchestrates complex movements like "pick card `id` and move to temporary spot 1" or "move card from temporary spot 2 to trash pile."
*   **Serial Commands:** `send_arm_command(degree1, degree2, degree3, magnet, movement_type)` formats and sends the control parameters to the ESP32.

### <mark style="background: #ABF7F7A6;"> Communication </mark>

*   **Primary Method (Implemented): Serial Communication**
    *   The Python script uses the `pyserial` library to establish a connection with the ESP32/Arduino over a USB COM port.
    *   Commands are sent as formatted strings (e.g., `"angle1,angle2,angle3,magnet_state,movement_flag\n"`).
    *   The ESP32 reads these serial strings, parses them, and controls the arm. It can send simple acknowledgments like "done" back.

*   **Alternative/Future: Wi-Fi Communication**
    *   The ESP32 could create a Wi-Fi access point or connect to an existing network.
    *   It would run a simple HTTP server or a WebSocket server.
    *   Python script would send commands as HTTP requests or WebSocket messages.
    *   *Benefit:* Frees up a USB port, allows for more remote operation.
    *   *Current Status:* Not implemented in the provided code but remains a viable option.

---
## 🚀 How to Run

1.  **Prerequisites:**
    *   Python 3.x
    *   Install required libraries:
        ```bash
        pip install pygame pyserial opencv-python numpy ultralytics
        ```
    *   DroidCam (or similar) installed on your PC and smartphone (or a standard webcam).
    *   Arduino IDE set up for your ESP32/Arduino, with the arm control sketch uploaded.

2.  **Setup:**
    *   Connect the ESP32/Arduino controlling the robotic arm to your PC via USB. Note the COM port (e.g., `COM5`).
    *   **For YOLO Mode:**
        *   Ensure you have the `yolov5s.pt` model file (or your chosen model) in the project directory.
        *   Create a `Resources` folder and place images for each detectable object (e.g., `apple.png`, `cat.jpg`). The script will try to load these for the Pygame UI.
    *   Start DroidCam on your phone and PC, and note the IP address if using the URL method.

3.  **Configuration (in Python script):**
    *   Update `URL` with your DroidCam IP address and port (e.g., `URL = 'http://192.168.2.19:4747/mjpegfeed?640x480'`).
    *   Update `ser.port = 'COM5'` to your ESP32's actual COM port.
    *   Adjust `GRID_TOP`, `GRID_LEFT`, etc. (for YOLO mode) or camera positioning (for color mode's dynamic detection) to align with your physical card setup.
    *   Modify `arm_values` and other arm position arrays to match your robot arm's calibration for each card slot and special position.

4.  **Running the Game:**
    *   Execute the desired Python script (e.g., `python memory_game_yolo.py` or `python memory_game_color.py`).
    *   The script will guide you through camera positioning (if applicable).
    *   The Pygame window will display the game, and the OpenCV window will show the camera feed with detections.
    *   The robotic arm should start interacting with the cards based on the game logic.

---
## 📈 Current Status & Future Work

*   ✅ **Core Game Logic:** Implemented for both color and object detection.
*   ✅ **Computer Vision:**
    *   Color detection with dynamic board warping is functional.
    *   YOLO object detection with a fixed grid is functional.
*   ✅ **Pygame UI:** Provides a visual representation of the game.
*   📝 **Robotic Arm Control:** Detailed movement sequences and serial communication protocol are defined. Full physical integration and calibration are the next critical steps.
*   💡 **Potential Future Enhancements:**
    *   More sophisticated game-playing strategies for the AI.
    *   Improved robustness of card flipping/handling mechanisms.
    *   Support for a wider variety of card types (shapes, numbers, complex patterns) using more advanced CV/AI models.
    *   Enhanced UI/UX.
    *   Calibration routines for the robotic arm and camera setup.
    *   Transition to Wi-Fi communication for more flexibility.
