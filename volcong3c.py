import cv2
import mediapipe as mp
from math import hypot
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

def set_volume(vol, vol_min, vol_max):
    volume = vol_min + (vol_max - vol_min) * vol
    return max(vol_min, min(volume, vol_max))  # Clamp volume between vol_min and vol_max


cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

vol_min, vol_max = 0, 1

# Get the default audio endpoint for rendering (speakers)
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, 0, None)
volume_control = interface.QueryInterface(IAudioEndpointVolume)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture an image from the camera.")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

    lmList = []
    if results.multi_hand_landmarks:
        for handlandmark in results.multi_hand_landmarks:
            for id, lm in enumerate(handlandmark.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            mpDraw.draw_landmarks(img, handlandmark, mpHands.HAND_CONNECTIONS)

    if lmList != []:
        x1, y1 = lmList[4][1], lmList[4][2]  # Thumb
        x2, y2 = lmList[8][1], lmList[8][2]  # Index finger

        cv2.circle(img, (x1, y1), 13, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 13, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        length = hypot(x2 - x1, y2 - y1)
        vol = length / 350

        volume_level = set_volume(vol, vol_min, vol_max)
        print(volume_level, int(length))

        volume_control.SetMasterVolumeLevelScalar(volume_level, None)

    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()
