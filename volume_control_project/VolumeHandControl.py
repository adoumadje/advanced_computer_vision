import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume



################################
wCam, hCam = 640, 480
################################



################################



#volume.GetMute()
#volume.GetMasterVolumeLevel()
#
#volume.SetMasterVolumeLevel(-20.0, None)

################################


detector = htm.handDetector(detectionCon=0.75)
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0


cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    detector.findHands(img)
    lmlist = detector.findPosition(img,draw=False)
    if len(lmlist) != 0:
        index_f = (lmlist[8][1], lmlist[8][2])
        thumb_f = (lmlist[4][1], lmlist[4][2])
        #print(index_f,thumb_f)
        index_fx, index_fy = index_f[0], index_f[1]
        thumb_fx, thumb_fy = thumb_f[0], thumb_f[1]
        mdl_pt = ((index_fx+thumb_fx)//2,(index_fy+thumb_fy)//2)
        f_distance = math.hypot(index_fx - thumb_fx, index_fy - thumb_fy)
        #print(f_distance,volume,volRange)
        cv2.circle(img, index_f, 5, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, thumb_f, 5, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (index_fx, index_fy), (thumb_fx, thumb_fy), (255, 0, 255), 2)
        cv2.circle(img, mdl_pt, 5, (255, 0, 255), cv2.FILLED)
        vol = np.interp(f_distance,[15,160],[minVol,maxVol])
        volBar = np.interp(f_distance,[15,160],[400,150])
        volPer = np.interp(f_distance,[15,160],[0,100])
        print(int(f_distance), vol)
        volume.SetMasterVolumeLevel(vol, None)

    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    cTime = time.time()
    fps = 1 / (cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Image",img)
    cv2.waitKey(1)