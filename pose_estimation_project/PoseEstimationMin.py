import cv2
import mediapipe as mp
import time


mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture('..\PoseVideos\k5.mp4')

pTime = 0
cTime = 0

def resizeFrame(frame, scale = 0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width,height)

    return cv2.resize(frame, dimensions, interpolation = cv2.INTER_AREA)



while True:
    success, img = cap.read()
    img = resizeFrame(img,0.25)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = pose.process(imgRGB)

    # print(results.pose_landmarks)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        h, w, c = img.shape
        for id, lm in enumerate(results.pose_landmarks.landmark):
            # print(id,lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            # print(id,cx,cy)
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)