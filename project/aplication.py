import cv2 as cv
import mediapipe as mp
import math
import time
import numpy as np

tired_flag =0
ESTATUS = "ok"
color = (0, 0, 0)
px = 0
size =0.0
bool_blink =0 #para achar o tempo da piscada
end_time =0 # calcular o tempo

# constantes
FONTS =cv.FONT_HERSHEY_COMPLEX # Font of text on the window

# posiçaõ do rosto
FACE_OVAL=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

# a boca
LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
LOWER_LIPS =[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS =[ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 

# olho esquerdo
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]

# olho direito
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]

# para achar os pontos do rosto
map_face_mesh = mp.solutions.face_mesh
# escolher a camera
camera = cv.VideoCapture(2)
# (blue, green, red) BGR
YELLOW =(0, 255, 255)
GREEN = (0, 255, 0)
PINK = (147, 20, 255)
VERMELHO = (0, 0, 255)

#Bota o texto na janela
def colorBackgroundText(img, text, font, fontScale, textPos, textThickness=1,textColor=(0,255,0), bgColor=(0,0,0), pad_x=3, pad_y=3):

    (t_w, t_h), _= cv.getTextSize(text, font, fontScale, textThickness) # getting the text size
    x, y = textPos
    cv.rectangle(img, (x-pad_x, y+ pad_y), (x+t_w+pad_x, y-t_h-pad_y), bgColor,-1) # draw rectangle 
    cv.putText(img,text, textPos,font, fontScale, textColor,textThickness ) # draw in text

    return img

#Bota o texto na janela 
def textWithBackground(img, text, font, fontScale, textPos, textThickness=1,textColor=(0,255,0), bgColor=(0,0,0), pad_x=3, pad_y=3, bgOpacity=0.5):

    (t_w, t_h), _= cv.getTextSize(text, font, fontScale, textThickness)
    x, y = textPos
    overlay = img.copy()
    cv.rectangle(overlay, (x-pad_x, y+ pad_y), (x+t_w+pad_x, y-t_h-pad_y), bgColor,-1) 
    new_img = cv.addWeighted(overlay, bgOpacity, img, 1 - bgOpacity, 0)
    cv.putText(new_img,text, textPos,font, fontScale, textColor,textThickness )
    img = new_img
    return img

# Detecta os ponto do rosto
def landmarksDetection(img, results, draw=False):
    img_height, img_width= img.shape[:2]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw :
        [cv.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]
    return mesh_coord # a lista de todas as marcas

# Euclaidean distance 
def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

# Formula da piscada
def blinkRatio(img, landmarks, right_indices, left_indices):
    # Olho direito
    # linha horizontal
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # linha vertical
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]

    # olho esquerdo 
    # linha horizontal
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]
    # linha verical
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]
    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)
    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)
    reRatio = rhDistance/rvDistance
    leRatio = lhDistance/lvDistance
    ratio = (reRatio + leRatio) / 2
    return ratio 


with map_face_mesh.FaceMesh(min_detection_confidence =0.5, min_tracking_confidence=0.5) as face_mesh:

    while True:
        ret, frame = camera.read() # pegando a imagem da camera
        if not ret: 
            break # se não estiver mais imagem
        frame = cv.flip(frame,180) #espelha a imagem
        frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
        frame_height, frame_width= frame.shape[:2]
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results  = face_mesh.process(rgb_frame)
        colorBackgroundText(frame, f'{ESTATUS}', FONTS, size, (px ,100),4, 0, color)
        #calculo de fadiga
        if tired_flag == 1:
            ESTATUS = " FADIGADO !"
            size = 2.0
            px = 250
            color = (0, 0, 255)
        if results.multi_face_landmarks:
            mesh_coords = landmarksDetection(frame, results, True)
            ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
            if ratio >4.0:
                if bool_blink == 0:
                    start_time = time.time()
                    bool_blink =1
                colorBackgroundText(frame,  f'Blink', FONTS, 1.7, (int(frame_height/2), 600), 2, YELLOW, pad_x=6, pad_y=6, )
            else:
                if bool_blink == 1:
                    end_time = time.time()-start_time
                else:
                    if end_time >0.5:
                        tired_flag += 1
                bool_blink =0
            cv.polylines(frame,  [np.array([mesh_coords[p] for p in LEFT_EYE ], dtype=np.int32)], True, GREEN, 1, cv.LINE_AA)
            cv.polylines(frame,  [np.array([mesh_coords[p] for p in RIGHT_EYE ], dtype=np.int32)], True, GREEN, 1, cv.LINE_AA)
        cv.imshow('Eye Blink', frame)
        key = cv.waitKey(2)
        if key == 27:
            break
        else:
            if key == 13:
                size = 0.0
                px =0
                tired_flag =0
    cv.destroyAllWindows()
    camera.release()
