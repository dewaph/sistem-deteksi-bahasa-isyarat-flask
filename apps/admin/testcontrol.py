import mediapipe as mp
import pickle
import numpy as np
import mysql.connector
import cv2
import os

def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="sistem_deteksi_isyarat"
    )

def load_model_from_database(model_name):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT model_data FROM models WHERE model_name = %s", (model_name,))
        model_data = cursor.fetchone()
        if model_data is not None:
            return pickle.loads(model_data[0])
        else:
            return None
    finally:
        conn.close()

# def load_model(model_name):
#     model_dir = os.path.join("apps", "static", "model")  # Adjust the path as needed
#     model_path = os.path.join(model_dir, f"{model_name}.p")
    
#     if os.path.isfile(model_path):
#         with open(model_path, 'rb') as model_file:
#             model_data = model_file.read()
#             return pickle.loads(model_data)
#     else:
#         return None

# # Load model1 from the database
# model1_dict = load_model('model1')
# if model1_dict is not None and 'model1' in model1_dict:
#     model1 = model1_dict['model1']
# else:
#     # Handle the case when model1 is not found in the database or loading fails
#     model1 = None  # You can provide a default model or raise an exception if needed

# # Load model2 from the database
# model2_dict = load_model('model2')
# if model2_dict is not None and 'model2' in model2_dict:
#     model2 = model2_dict['model2']
# else:
#     # Handle the case when model2 is not found in the database or loading fails
#     model2 = None  # You can provide a default model or raise an exception if needed

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def get_label_mapping(kategori_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(f"SELECT class_label, huruf FROM dataset WHERE kategori_id = {kategori_id}")
    label_mapping = {row[0]: row[1] for row in cursor.fetchall()}

    cursor.close()
    conn.close()

    return label_mapping

labels_dict1 = get_label_mapping(1)
labels_dict2 = get_label_mapping(2)

def update_label_dictionaries_and_models():
    # Update labels_dict1 and labels_dict2
    global labels_dict1, labels_dict2
    labels_dict1 = get_label_mapping(1)
    labels_dict2 = get_label_mapping(2)

    # # Load model1 from the database
    # model1_dict = load_model('model1')
    # if model1_dict is not None and 'model1' in model1_dict:
    #     global model1  # Use global to modify the variable outside the function
    #     model1 = model1_dict['model1']
    # else:
    #     # Handle the case when model1 is not found in the database or loading fails
    #     model1 = None  # You can provide a default model or raise an exception if needed

    # # Load model2 from the database
    # model2_dict = load_model('model2')
    # if model2_dict is not None and 'model2' in model2_dict:
    #     global model2  # Use global to modify the variable outside the function
    #     model2 = model2_dict['model2']
    # else:
    #     # Handle the case when model2 is not found in the database or loading fails
    #     model2 = None  # You can provide a default model or raise an exception if needed

def generate_frames():
    model1 = pickle.load(open('apps/static/model/model1.p', 'rb'))
    model2 = pickle.load(open('apps/static/model/model2.p', 'rb'))
    # model1 = model_dict1['model1']
    # model2 = model_dict2['model2']
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=2,  
    min_detection_confidence=0.3  
)
    while True:
        data_aux = []
        x_ = []
        y_ = []
        ret, frame = cap.read()
        cv2.putText(frame, 'Arahkan Tangan ke Kamera', (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        box_width = 250
        box_height = 300  
        box_y1 = int(H * 0.3) 
        box_y2 = box_y1 + box_height
        box_x_center = int(W / 2)
        box_x1 = box_x_center - int(box_width / 2) - 50  
        box_x2 = box_x1 + box_width

        # Draw the bounding box on the frame
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), 2)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            n = len(results.multi_hand_landmarks)
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))
            if n == 1:
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10
                prediction1 = model1.predict([np.asarray(data_aux)])
                confidence1 = model1.predict_proba([np.asarray(data_aux)])
                predicted_character1 = labels_dict1[int(prediction1[0])]
                confidence1 = np.max(confidence1)  # Adjust depending on the method used
                bar_width1 = int(confidence1 * 200)
                if confidence1 < 0.80:
                    cv2.putText(frame, f'{predicted_character1} Kurang tepat', (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, f'{predicted_character1} ({confidence1:.2f})', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.rectangle(frame, (0, H - 40), (250, H - 0), (245, 117, 16), -1)
                cv2.putText(frame, f'{confidence1:.2f}', (100, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (200, H - 20), (200 + bar_width1, H - 10), (0, 255, 0), -1)
            else:
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10
                prediction2 = model2.predict([np.asarray(data_aux)])
                confidence2 = model2.predict_proba([np.asarray(data_aux)])
                predicted_character2 = labels_dict2[int(prediction2[0])]
                confidence2 = np.max(confidence2)  # Adjust depending on the method used
                bar_width1 = int(confidence1 * 200)
                if confidence2 < 0.70:
                    cv2.putText(frame, f'{predicted_character2} Kurang tepat', (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, f'{predicted_character2} ({confidence2:.2f})', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.rectangle(frame, (0, H - 40), (250, H - 0), (245, 117, 16), -1)
                cv2.putText(frame, f'{confidence2:.2f}', (100, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (200, H - 20), (200 + bar_width1, H - 10), (0, 255, 0), -1)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        



