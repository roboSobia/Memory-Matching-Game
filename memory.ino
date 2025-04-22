#include <ESP32Servo.h>
// Define the pin for the magnet control
#define MAGNET_PIN 25 // Use GPIO 25 for the magnet
/*
180
118, 44, 118, 0
48, 67, 108
48, 121, 108
120, 138, 124
149, 62, 135
127, 90, 122
127, 109, 127
175, 130, 165
*/
// Declare three servo objects
Servo servo1;
Servo servo2;
Servo servo3;

// Variable to store the desired magnet state (0 = OFF, 1 = ON)
int magnetState;
int pickSig;
int flg;

void setup() {
  // Initialize serial communication at 9600 baud
  Serial.begin(9600);

  // Configure the magnet pin as an output
  pinMode(MAGNET_PIN, OUTPUT);
  // Ensure the magnet is initially OFF
  digitalWrite(MAGNET_PIN, HIGH);
  magnetState = 0; // Initialize magnet state variable
  flg = 4;
  // Attach servos to GPIO pins
  // Using default pulse widths (500us to 2400us) is usually fine for most servos
  servo1.attach(26);  // Servo 1 on pin 26
  servo2.attach(27);  // Servo 2 on pin 27
  servo3.attach(14);  // Servo 3 on pin 14

  // Optional: Seet servos to initial position (e.g., 90 degrees)
  // servo1.write(90);
  // servo2.write(90);
  // servo3.write(90);
  // delay(500); // Allow time for servos to reach initial position

  Serial.println("ESP32 Servo and Magnet Control Ready.");
  Serial.println("Send commands in format: angle1,angle2,angle3,magnetState (e.g., 90,90,90,1)");
}

void smoothMoveSync(Servo& s1, int& curr1, int target1,
                    Servo& s2, int& curr2, int target2,
                    Servo& s3, int& curr3, int target3,
                    int delayTime, int pickSig) {

  // Calculate the number of steps based on the largest angle difference
  int diff1 = abs(target1 - curr1);
  int diff2 = abs(target2 - curr2);
  int diff3 = abs(target3 - curr3);
  int maxDiff = max(diff1, max(diff2, diff3));

  if (maxDiff == 0) return; // No movement needed
  if(pickSig % 2 == 0) {
    for(int step = 1; step <= maxDiff; ++step) {
    int angle2 = curr2 + ((target2 - curr2) * step) / maxDiff;
    s2.write(angle2);
    delay(delayTime); // Small delay for smoother movement and servo settling
    }
    // Move all servos in sync step by step
    for (int step = 1; step <= maxDiff; ++step) {
    // For each servo, interpolate the angle if there's a change
    // Integer division provides approximation for intermediate steps
      int angle1 = curr1 + ((target1 - curr1) * step) / maxDiff;
      int angle3 = curr3 + ((target3 - curr3) * step) / maxDiff;

      s1.write(angle1);
      s3.write(angle3);
      delay(delayTime); // Small delay for smoother movement and servo settling
    }
  } else {
    for (int step = 1; step <= maxDiff; ++step) {
    // For each servo, interpolate the angle if there's a change
    // Integer division provides approximation for intermediate steps
      int angle1 = curr1 + ((target1 - curr1) * step) / maxDiff;
      int angle3 = curr3 + ((target3 - curr3) * step) / maxDiff;

      s1.write(angle1);
      s3.write(angle3);
      delay(delayTime); // Small delay for smoother movement and servo settling
    }
    for(int step = 1; step <= maxDiff; ++step) {
    int angle2 = curr2 + ((target2 - curr2) * step) / maxDiff;
    s2.write(angle2);
    delay(delayTime); // Small delay for smoother movement and servo settling
    }
    // Move all servos in sync step by step
  }
  

  // Ensure servos reach the exact target angles at the end
  s1.write(target1);
  s2.write(target2);
  s3.write(target3);

  // Update current angles
  curr1 = target1;
  curr2 = target2;
  curr3 = target3;
}

void loop() {
  // Static variables to remember the current angles between loop cycles
  static int currentAngle1 = 90; // Initialize to a known state
  static int currentAngle2 = 90;
  static int currentAngle3 = 90;

  if (Serial.available() > 0) {
    String line = Serial.readStringUntil('\n');
    line.trim(); // Remove any leading/trailing whitespace

    if (line.length() > 0) {
      // Find the positions of the three commas needed for 3 angles + magnet state
      int comma1 = line.indexOf(',');
      int comma2 = line.indexOf(',', comma1 + 1);
      int comma3 = line.indexOf(',', comma2 + 1);
      int comma4 = line.indexOf(',', comma3 + 1);

      // Verify that three commas are present (separating 4 values)
      if (comma1 != -1 && comma2 != -1 && comma3 != -1) {
        // Extract substrings for each angle and the magnet state
        String angle1Str = line.substring(0, comma1);
        String angle2Str = line.substring(comma1 + 1, comma2);
        String angle3Str = line.substring(comma2 + 1, comma3);
        String magnetStr = line.substring(comma3 + 1, comma4);
        String pickSigStr = line.substring(comma4 + 1);

        // Convert strings to integers
        int targetAngle1 = angle1Str.toInt();
        int targetAngle2 = angle2Str.toInt();
        int targetAngle3 = angle3Str.toInt();
        magnetState = magnetStr.toInt(); // Update the desired magnet state
        pickSig = pickSigStr.toInt(); // Update the desired magnet state
        if(pickSig) {
          flg = 4;
        }
        // Optional: Add angle constraints if needed
        targetAngle1 = constrain(targetAngle1, 0, 180);
        targetAngle2 = constrain(targetAngle2, 0, 180);
        targetAngle3 = constrain(targetAngle3, 0, 180);

        Serial.print("Received Command: ");
        Serial.print("A1="); Serial.print(targetAngle1);
        Serial.print(", A2="); Serial.print(targetAngle2);
        Serial.print(", A3="); Serial.print(targetAngle3);
        Serial.print(", Magnet="); Serial.println(magnetState);

        // Move all servos together in sync
        if (magnetState == 1) {
          digitalWrite(MAGNET_PIN, LOW); // Turn magnet ON
          Serial.println("Magnet Activated (Pin HIGH)");
        } else {
          digitalWrite(MAGNET_PIN, HIGH);  // Turn magnet OFF (for 0 or any other value)
          Serial.println("Magnet Deactivated (Pin LOW)");
        }
        smoothMoveSync(servo1, currentAngle1, targetAngle1,
                       servo2, currentAngle2, targetAngle2,
                       servo3, currentAngle3, targetAngle3,
                       10, pickSig); // Delay between steps (lower = faster, but might strain servos/power)
        flg--;
        Serial.write("done");
        // --- Control the Magnet ---
        // --------------------------

      } else {
        Serial.print("Error: Invalid command format received: ");
        Serial.println(line);
        Serial.println("Expected format: angle1,angle2,angle3,magnetState (e.g., 90,90,90,1)");
      }
    }
  }
  // Add a small delay to prevent the loop from running too fast when idle
  // Adjust as needed, but be aware smoothMoveSync already has delays
  // delay(10);
}