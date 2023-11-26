import cv2
import mediapipe as mp
import pickle
import numpy as np

model_dict = pickle.load(open('./model.p','rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'HI', 1: 'THUMBS UP', 2:'THUMBS DOWN', 3:'YO', 4:'EXCUSE ME', 5:'Y', 6:'1', 7:'4', 8:'2'}
while True:
    data_aux = []
    x_ =[]
    y_ =[]
    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,  # image to draw
                        hand_landmarks,  # model output
                        mp_hands.HAND_CONNECTIONS,  # hand connections
                        mp_drawing_styles.get_default_hand_landmarks_style(),  # drawing style
                        mp_drawing_styles.get_default_hand_connections_style()  # use get_default_hand_connections_style
                    )

        for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                   x = hand_landmarks.landmark[i].x
                   y = hand_landmarks.landmark[i].y
                   data_aux.append(x)
                   data_aux.append(y)
                   x_.append(x)
                   y_.append(y)

        # getting the corners of hand
        x1 = int(min(x_) * W)
        x2 = int(max(x_) * H)
        y1 = int(min(y_) * W)
        y2 = int(max(y_) * H)

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame,(x1,y2), (x2,y2),(0,0,0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)
    cv2.imshow('frame', frame)
    cv2.waitKey(25)

cap.release()
cv2.destroyAllWindows()