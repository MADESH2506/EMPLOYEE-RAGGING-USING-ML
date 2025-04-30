import cv2
import mediapipe as mp

def detect_hand_landmarks(image):
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image
    results = hands.process(image_rgb)

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Print hand landmarks
            for landmark, point in enumerate(hand_landmarks.landmark):
                height, width, _ = image.shape
                cx, cy = int(point.x * width), int(point.y * height)
                print(f"Landmark {landmark}: ({cx}, {cy})")

                # Draw a circle at each landmark point
                cv2.circle(image, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

    return image

# Open the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Detect hand landmarks
    frame_with_landmarks = detect_hand_landmarks(frame)

    # Display the result
    cv2.imshow("Hand Landmarks Detection", frame_with_landmarks)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()