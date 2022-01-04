import cv2
import mediapipe as mp
import time


def resizeFrame(frame, scale = 0.25):
    width, height = int(frame.shape[1] * scale), int(frame.shape[0] * scale)
    dimensions = (width,height)
    return cv2.resize(frame, dimensions, interpolation = cv2.INTER_AREA)


cap = cv2.VideoCapture("./videos/k1.mp4")

mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection(0.75, 1)
mpDraw = mp.solutions.drawing_utils


pTime = 0
cTime = 0
while True:
    success, img = cap.read()
    img = resizeFrame(img)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    # print(results.detections)
    if results.detections:
        for id, detection in enumerate(results.detections):
            # print(id,detection)
            # mpDraw.draw_detection(img, detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            ih, iw, ic = img.shape
            bboxC = detection.location_data.relative_bounding_box
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            cv2.putText(img, f'Face: {int(detection.score[0] * 100)}%',(bbox[0], bbox[1]-20),cv2.FONT_HERSHEY_PLAIN,
                        1, (255, 0, 255), 2)

    cTime = time.time()
    fps = 1 / (cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cv2.imshow("Image",img)
    cv2.waitKey(1)