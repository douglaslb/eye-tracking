import cv2
import dlib
from imutils import face_utils
 
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
        xLeft, yLeft = leftEye[0][0], leftEye[2][1]
        widthL, heightL = leftEye[3][0], leftEye[4][1]

        # Coordenadas do olho direito
        xRight, yRight = rightEye[0][0], rightEye[2][1]
        widthR, heightR = rightEye[3][0], rightEye[4][1]

        #Crop do olho esquerdo (preto/branco, colorido)
        leftEye = gray_frame[yLeft:heightL, xLeft:widthL]
        leftEyeColor = frame[yLeft:heightL, xLeft:widthL]

        #Crop do olho direito (preto/branco, colorido)
        rightEye = gray_frame[yRight:heightR , xRight:widthR]
        rightEyeColor = frame[yRight:heightR , xRight:widthR]


        cv2.imshow('ROI  - Olho esquerdo', leftEyeColor)
        cv2.imshow('ROI  - Olho direito', rightEyeColor)

        #Busca a posição do valor minimo (mais escuro) dentro do crop do olho 
        _, _, min_loc, _ = cv2.minMaxLoc(leftEye)
        cv2.circle(leftEyeColor, min_loc, 5, (0, 0, 255), 2)
        cv2.circle(leftEyeColor, min_loc, 2, (255, 0, 0), 2)

        _, _, min_loc, _ = cv2.minMaxLoc(rightEye)
        cv2.circle(rightEyeColor, min_loc, 5, (0, 0, 255), 2)
        cv2.circle(rightEyeColor, min_loc, 2, (255, 0, 0), 2)

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

