<div align='center'>
  <h1>Memory Matching Game</h1>
</div>

## $\color{rgba(255, 192, 203, 1)}{\textsf{Game Description}}$

The robot plays a memory-matching game by flipping over cards, identifying pairs with the same value/pattern, and eliminating them. This tests its ability to recognize patterns, remember card locations, and make strategic decisions.

For now, we build our assumption on matching <mark style="background: #ABF7F7A6;"> solid colored cards </mark> only. 
Matching shapes or numbers requires more complex AI models, we might consider a level up once we get this working properly.

>[!NOTE]
> So Far,
> We have a color detection code working and successfully tested it!

---
## $\color{rgba(255, 192, 203, 1)}{\textsf{Hardware Mechanism}}$

>[!IMPORTANT]
>These are the current scenarios that we're working on choosing one of which, arranged by Priority:

### 1. Flipping Cards

<mark style="background: pink;"> check this video </mark>: https://youtube.com/shorts/c3VXFufrVV4?si=mrFHUmhxs6JlsQAE

![image](https://github.com/user-attachments/assets/c42e5eb7-d07b-4cc2-911d-3f3873396959)

We'll use the arm only to flip the fixed cards. 
Front faced = some color and the other side will be black.

##### The whole idea will be in building the cards station, but on the other hand this will make the code very simple(only pushing on cards)
- Any arm type will work with this implementation

> We're still deciding on whether making it vertical/horizontal will make it more stable.


### 2. Magnet Clipper 
Instead of physically flipping the card—to avoid any complex mechanisms—the robot arm will grip each card individually and lift it for the camera to detect. 

**Steps:**
1. Grip the card using the robot arm.
2. Lift the card and rotate it to make its face visible to the camera.
3. Detect the card's color.
4. Place the card back in its original position.
5. Pick another Card
6. Check if it matches with the first one

>[!CAUTION]
The problem with this is that it doesn't keep two cards open at the same time that's why we're keeping it as a last resort

---
## $\color{rgba(255, 192, 203, 1)}{\textsf{Software Mechanism}}$

## <mark style="background: pink;"> Responsibilities </mark> 

1. The Python script (running on the laptop) handles:
    - Capturing frames from the camera.
    - Detecting the card's color using OpenCV.
    - Keeping track of the game state (memorizing positions and pairs).
    - Sending commands to the ESP to flip or remove cards.

2. The ESP code (written in C++/Arduino) handles:
    - Receiving commands from the Python script.
    - Controlling the robot arm to flip/remove cards.

</br>

## <mark style="background: pink;"> How Communication works </mark>

We can either use serial communication or wifi communication.
we'll be using wifi probably because it'll save us a USB

##### *Flow:*
- ESP creates a wifi access point or connects to a router.
- The ESP runs a basic HTTP server or socket server.
- The Python script sends HTTP requests or socket messages to the ESP when a color match is detected.
- The ESP reads the request and triggers the robot arm action.

We’ll use HTTP communication for simplicity:
- Python → Sends HTTP POST request to ESP when an action is needed.
- ESP → Handles the request, decodes the command, and controls the servo.
