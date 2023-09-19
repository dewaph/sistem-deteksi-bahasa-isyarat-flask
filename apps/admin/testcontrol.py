import mediapipe as mp
import pickle
import numpy as np
import mysql.connector
import cv2

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

# Load model1 from the database
model1_dict = load_model_from_database('model1')
if model1_dict is not None and 'model1' in model1_dict:
    model1 = model1_dict['model1']
else:
    # Handle the case when model1 is not found in the database or loading fails
    model1 = None  # You can provide a default model or raise an exception if needed

# Load model2 from the database
model2_dict = load_model_from_database('model2')
if model2_dict is not None and 'model2' in model2_dict:
    model2 = model2_dict['model2']
else:
    # Handle the case when model2 is not found in the database or loading fails
    model2 = None  # You can provide a default model or raise an exception if needed

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def get_label_mapping(table_name):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(f"SELECT class_label, huruf FROM {table_name}")
    label_mapping = {row[0]: row[1] for row in cursor.fetchall()}

    cursor.close()
    conn.close()

    return label_mapping

labels_dict1 = get_label_mapping('data_1_tangan')
labels_dict2 = get_label_mapping('data_2_tangan')

def update_label_dictionaries_and_models():
    # Update labels_dict1 and labels_dict2
    global labels_dict1, labels_dict2
    labels_dict1 = get_label_mapping('data_1_tangan')
    labels_dict2 = get_label_mapping('data_2_tangan')

    # Load model1 from the database
    model1_dict = load_model_from_database('model1')
    if model1_dict is not None and 'model1' in model1_dict:
        global model1  # Use global to modify the variable outside the function
        model1 = model1_dict['model1']
    else:
        # Handle the case when model1 is not found in the database or loading fails
        model1 = None  # You can provide a default model or raise an exception if needed

    # Load model2 from the database
    model2_dict = load_model_from_database('model2')
    if model2_dict is not None and 'model2' in model2_dict:
        global model2  # Use global to modify the variable outside the function
        model2 = model2_dict['model2']
    else:
        # Handle the case when model2 is not found in the database or loading fails
        model2 = None  # You can provide a default model or raise an exception if needed

def generate_frames():
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)
    while True:
        data_aux = []
        x_ = []
        y_ = []
        ret, frame = cap.read()
        cv2.putText(frame, 'Arahkan Tangan ke Kamera', (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
                predicted_character1 = labels_dict1[int(prediction1[0])]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character1, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
            else:
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10
                prediction2 = model2.predict([np.asarray(data_aux)])
                predicted_character2 = labels_dict2[int(prediction2[0])]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character2, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



