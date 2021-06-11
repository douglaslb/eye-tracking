import cv2
import dlib
from imutils import face_utils
import numpy as np
 
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 

# pegar os índices do previsor, para olhos esquerdo e direito
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

def detect_face(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray_frame, 1)

    for react in rects:
        shape = predictor(gray_frame, react)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        
        #Cria o contorno com base nos landmarks dos olhos
        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

        # Coordenadas do olho esquerdo
        xLeftStart, yLeftStart = leftEye[0][0], leftEye[2][1]
        xLeftEnd, yLeftEnd = leftEye[3][0], leftEye[4][1]

        centerLeftX = (xLeftEnd-xLeftStart)/2
        centerLeftY = (yLeftEnd-yLeftStart)/2

        #centerLeft = (int(centerLeftX), int(centerLeftY))

        # Coordenadas do olho direito
        xRightStart, yRightStart = rightEye[0][0], rightEye[2][1]
        xRightEnd, yRightEnd = rightEye[3][0], rightEye[4][1]

        centerRightX = (xRightEnd-xRightStart)/2
        centerRightY = (yRightEnd-yRightStart)/2

       #centerRight =  (int(centerRightX), int(centerRightY))
    
        eyesCenter = ((centerLeftX + centerRightX)/2, (centerLeftY + centerRightY)/2)

        #Crop do olho esquerdo (preto/branco, colorido)
        leftEye = gray_frame[yLeftStart:yLeftEnd, xLeftStart:xLeftEnd]
        leftEyeColor = frame[yLeftStart:yLeftEnd, xLeftStart:xLeftEnd]

        #Crop do olho direito (preto/branco, colorido)
        rightEye = gray_frame[yRightStart:yRightEnd , xRightStart:xRightEnd]
        rightEyeColor = frame[yRightStart:yRightEnd , xRightStart:xRightEnd]

        cv2.rectangle(frame, (xRightStart, yRightStart), (xRightEnd, yRightEnd), (0, 255, 255), 2)
        cv2.rectangle(frame, (xLeftStart, yLeftStart), (xLeftEnd, yLeftEnd), (0, 255, 255), 2)

        #Busca a posição do valor minimo (mais escuro) dentro do crop do olho 
        _, _, min_loc, _ = cv2.minMaxLoc(leftEye)
        cv2.circle(leftEyeColor, min_loc, 5, (0, 0, 255), 2)
        cv2.circle(leftEyeColor, min_loc, 2, (255, 0, 0), 2)

        pupil_left = min_loc

        _, _, min_loc, _ = cv2.minMaxLoc(rightEye)
        cv2.circle(rightEyeColor, min_loc, 5, (0, 0, 255), 2)
        cv2.circle(rightEyeColor, min_loc, 2, (255, 0, 0), 2)

        pupil_right = min_loc


        pupil_median_x = (pupil_left[0] + pupil_right[0])/2
        pupil_median_y = (pupil_left[1] + pupil_right[1])/2

        pupil_median = (pupil_median_x, pupil_median_y)
        
        pixelX = (eyesCenter[0] - pupil_median[0])*10/100
        pixelY = -((eyesCenter[1] - pupil_median[1])*10/100)

        if pixelX > 1:
            pixelX = 1
        elif pixelX < -1:
            pixelX = -1

        if pixelY > 1:
            pixelY = 1
        elif pixelY < -1:
            pixelY = -1
        

        pov = np.zeros([600,800,3],dtype=np.uint8)
        pov.fill(255)
        cv2.circle(pov, (int(400+(400*pixelX)),int(300+(300*pixelY))), radius=10, color=(0, 0, 255), thickness=5)

        cv2.imshow('POV', pov)

    return frame

vc = cv2.VideoCapture(0)

if vc.isOpened(): 
    rval, frame = vc.read()

else:
    rval = False

while rval:
    img = detect_face(frame)
    cv2.imshow("Result", img)


    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: 
        break
vc.release()
cv2.destroyAllWindows()

