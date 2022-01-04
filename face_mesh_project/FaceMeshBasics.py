import cv2
import mediapipe as mp
import time




def resizeFrame(frame, scale = 0.25):
    width, height = int(frame.shape[1] * scale), int(frame.shape[0] * scale)
    dimensions = (width,height)
    return cv2.resize(frame, dimensions, interpolation = cv2.INTER_AREA)


cap = cv2.VideoCapture("../videos/k7.mp4")

pTime = 0
cTime = 0

mpDraw = mp.solutions.drawing_utils
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=4)




while True:
    success, img = cap.read()
    img = resizeFrame(img)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    print(results.multi_face_landmarks)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
            for id, lm in enumerate(faceLms.landmark):
                # print(lm)
                ih, iw, ic = img.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                print(id, x, y)

    cTime = time.time()
    fps = 1 / (cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cv2.imshow("Image",img)
    cv2.waitKey(1)