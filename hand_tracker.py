import cv2
import mediapipe as mp
import numpy as np
import time

### Tipagem ------>
confidence = float
webcam_image = np.ndarray
rgb_tuple = tuple[int, int, int]

#coords_vector - 



class Detector:
    def __init__(self,
                 mode: bool = False,
                 number_hands: int = 2,
                 model_complexity: int = 1,
                 min_detec_confidence: confidence = 0.5,
                 min_tranking_confidence: confidence = 0.5):
        # Parametros necessários para inicializar o Hands
        self.mode = mode 
        self.max_num_hands = number_hands
        self.complexity = model_complexity
        self.detection_con = min_detec_confidence
        self.tracking_con = min_tranking_confidence

        # Inicializar o Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode,
                                        self.max_num_hands,
                                        self.complexity,
                                        self.detection_con,
                                        self.tracking_con)    
        self.mp_draw = mp.solutions.drawing_utils
        self.tip_ids = [4, 8, 12, 16, 20]

    def find_hands(self,
                   img: webcam_image,
                   draw_hands: bool = True):
        # Correção de cor 
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        
        # Coletando resultados do processo das hands e analisando-os
        self.results = self.hands.process(img_RGB)
        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                if draw_hands:
                    self.mp_draw.draw_landmarks(img, hand, self.mp_hands.HAND_CONNECTIONS)  

        return img
    
    def find_posicition(self, 
                        img: webcam_image,
                        hand_number: int = 0):
        self.required_landmark_list = []

        my_hand = None
        if self.results.multi_hand_landmarks:
            height, width, _ = img.shape
            my_hand = self.results.multi_hand_landmarks[hand_number]
            for id, lm in enumerate(my_hand.landmark):
                center_x, center_y = int(lm.x*width), int(lm.y*height)
            
                self.required_landmark_list.append([id, center_x, center_y]) 

            return self.required_landmark_list



# Teste de classse ===============
if __name__ == '__main__':
    Detec = Detector()

    # Coletando o Framerate
    previous_time = 0
    current_time = 0

     # Captura de imagem
    capture = cv2.VideoCapture(0)

    while True:
        # Captura de imagem
        capture = cv2.VideoCapture(0)
        # Coletando o Framerate
        previous_time = 0

        # Captura do frame
        _, img = capture.read()

        # Manipulação de frame 
        img = Detec.find_hands(img) 
        landmark_list = Detec.find_posicition(img)
        if landmark_list:
            print(landmark_list[8])

        # Calcucando fps
        # current_time = time.time()
        # fps = 1/(current_time - previous_time)
        # previous_time = current_time



        # Mostrando o frame
        # cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_DUPLEX, 2, (255,0,255), 3)
        cv2.imshow('Camera Beatriz', img)
        cv2.waitKey(1)

        # Testes 
        teste = Detec.find_posicition(img)
        print(teste)

        # Quitando
        if cv2.waitKey(20) & 0xFF==ord('q'):
            break

