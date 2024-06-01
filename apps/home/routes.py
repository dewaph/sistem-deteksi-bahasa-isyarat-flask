from apps.home import blueprint
from flask import render_template, request, Response, jsonify, send_file
from flask_login import login_required, current_user
from jinja2 import TemplateNotFound
import os
import io
import mysql.connector
from PIL import Image
from moviepy.editor import VideoFileClip, concatenate_videoclips
from flask import current_app
import mediapipe as mp
import numpy as np
import pickle
import cv2
import time
from gtts import gTTS


# Fungsi untuk menghubungkan ke database
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="sistem_deteksi_isyarat"
    )

# Fungsi untuk meresize gambar
def resize_image(image, size):
    resized_image = image.resize(size)
    return resized_image

def get_username():
    return current_user.username if current_user.is_authenticated else None

def get_context(segment):
    context = {'segment': segment}
    context['username'] = current_user.username if current_user.is_authenticated else None
    return context

@blueprint.route('/beranda')
@login_required
def beranda():
    username = get_username()
    return render_template('home/index.html', segment='index', username=username)

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

huruf_index = 0

def update_label_dictionaries_and_models():
    # Update labels_dict1 and labels_dict2
    global labels_dict1, labels_dict2
    labels_dict1 = get_label_mapping(1)
    labels_dict2 = get_label_mapping(2)


def generate_frames():
    update_label_dictionaries_and_models()
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=2,  
    min_detection_confidence=0.3 
)

    cap = cv2.VideoCapture(0)

    model1 = pickle.load(open('apps/static/model/model1.p', 'rb'))
    model2 = pickle.load(open('apps/static/model/model2.p', 'rb'))

    # Merge the two dictionaries
    merged_labels_dict = {}

    for key in set(labels_dict1) | set(labels_dict2):
        values = []
        if key in labels_dict1:
            values.append(labels_dict1[key])
        if key in labels_dict2:
            values.append(labels_dict2[key])
        merged_labels_dict[key] = values

    # Extract the quiz letters from the merged dictionary
    huruf = [value for values in merged_labels_dict.values() for value in values]
    huruf.sort()

    # Initialize an index for quiz letters
    rekam = False
    huruf = huruf[huruf_index]

    while True:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()

        if not ret:
            break

        # cv2.putText(frame, f'Peragakan huruf {huruf}', (130, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        # image_path = 'apps/static/kamusku/{}.png'.format(huruf)
        # image = cv2.resize(cv2.imread(image_path), (0, 0), fx=0.6, fy=0.6)

        # # Display the loaded and resized image
        # y_offset, x_offset = 0, frame.shape[1] - image.shape[1]
        # frame[y_offset:y_offset + image.shape[0], x_offset:x_offset + image.shape[1]] = image

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
        # cv2.rectangle(frame, (W - 200, H - 40), (W, H), (245, 117, 16), -1)
        # cv2.putText(frame, '0', (W - 150, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            hand_detected_outside_box = any(any(box_x1 > landmark.x * W or landmark.x * W > box_x2 or box_y1 > landmark.y * H or landmark.y * H > box_y2 for landmark in hand_landmarks.landmark) for hand_landmarks in results.multi_hand_landmarks)

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

            if hand_detected_outside_box:
                cv2.putText(frame, "Posisikan Tangan dalam Box", (box_x1, box_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,  
                        hand_landmarks,  
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10

                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                # Check if the hand is inside the bounding box
                hand_inside_box = box_x1 < x1 < box_x2 and box_y1 < y1 < box_y2

                rect_y1 = box_y1 - 30
                rect_y2 = box_y1 - 1
                rect_x1 = box_x1  # Adjust this value to move the rectangle to the right
                rect_x2 = box_x2 - 120  # Adjust this value to move the rectangle to the right
                text_y = box_y1 - 11
                text_x = box_x1 + 5  # Adjust this value to move the text to the right

                if hand_inside_box and not rekam:
                    if huruf == 'C':
                            confidence_threshold1 = 0.90
                            confidence_threshold2 = 0.90       
                    elif huruf == 'E':
                            confidence_threshold1 = 0.80
                            confidence_threshold2 = 0.80
                    elif huruf == 'F':
                            confidence_threshold1 = 0.80
                            confidence_threshold2 = 0.80
                    elif huruf == 'G':
                            confidence_threshold1 = 0.50
                            confidence_threshold2 = 0.50
                    elif huruf == 'H':
                            confidence_threshold1 = 0.50
                            confidence_threshold2 = 0.50
                    elif huruf == 'I':
                            confidence_threshold1 = 0.90
                            confidence_threshold2 = 0.90
                    elif huruf == 'J':
                            confidence_threshold1 = 0.90
                            confidence_threshold2 = 0.90
                    elif huruf == 'K':
                            confidence_threshold1 = 0.80
                            confidence_threshold2 = 0.80
                    elif huruf == 'L':
                            confidence_threshold1 = 0.80
                            confidence_threshold2 = 0.80
                    elif huruf == 'N':
                            confidence_threshold1 = 0.80
                            confidence_threshold2 = 0.80
                    elif huruf == 'O':
                            confidence_threshold1 = 0.90
                            confidence_threshold2 = 0.90
                    elif huruf == 'P':
                            confidence_threshold1 = 0.60
                            confidence_threshold2 = 0.60
                    elif huruf == 'T':
                            confidence_threshold1 = 0.60
                            confidence_threshold2 = 0.60
                    elif huruf == 'U':
                            confidence_threshold1 = 0.90
                            confidence_threshold2 = 0.90
                    elif huruf == 'V':
                            confidence_threshold1 = 0.90
                            confidence_threshold2 = 0.90
                    elif huruf == 'W':
                            confidence_threshold1 = 0.80
                            confidence_threshold2 = 0.80
                    elif huruf == 'Z':
                            confidence_threshold1 = 0.80
                            confidence_threshold2 = 0.80
                    else:
                            confidence_threshold1 = 0.50
                            confidence_threshold2 = 0.50

                    if len(results.multi_hand_landmarks) == 1:
                        prediction1 = model1.predict([np.asarray(data_aux)])
                        predicted_character = labels_dict1[int(prediction1[0])]
                        confidence1 = model1.predict_proba([np.asarray(data_aux)])
                        confidence1 = np.max(confidence1)

                        if predicted_character == huruf:
                            if confidence1 > confidence_threshold1:
                                rekam = True
                                cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 2)
                                cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 255, 0), -1)
                                cv2.putText(frame, f'Huruf : {predicted_character}', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                                # cv2.rectangle(frame, (W - 200, H - 40), (W, H), (245, 117, 16), -1)
                                # cv2.putText(frame, f'{confidence1:.2f}', (W - 150, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                            else:
                                cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 255), 2)
                                cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 255), -1)
                                cv2.putText(frame, f'Kurang Tepat', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                                # cv2.rectangle(frame, (W - 200, H - 40), (W, H), (245, 117, 16), -1)
                                # cv2.putText(frame, f'{confidence1:.2f}', (W - 150, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        else:
                            cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 255), 2)
                            cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 255), -1)
                            cv2.putText(frame, f'Kurang Tepat', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        
                    elif len(results.multi_hand_landmarks) == 2:
                        prediction2 = model2.predict([np.asarray(data_aux)])
                        predicted_character = labels_dict2[int(prediction2[0])]
                        confidence2 = model2.predict_proba([np.asarray(data_aux)])
                        confidence2 = np.max(confidence2)

                        if predicted_character == huruf:
                            if confidence2 > confidence_threshold2:
                                rekam = True
                                cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 2)
                                cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 255, 0), -1)
                                cv2.putText(frame, f'Huruf : {predicted_character}', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                                # cv2.rectangle(frame, (W - 200, H - 40), (W, H), (245, 117, 16), -1)
                                # cv2.putText(frame, f'{confidence2:.2f}', (W - 150, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                                
                            else:
                                cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 255), 2)
                                cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 255), -1)
                                cv2.putText(frame, f'Kurang Tepat', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                                # cv2.rectangle(frame, (W - 200, H - 40), (W, H), (245, 117, 16), -1)
                                # cv2.putText(frame, f'{confidence2:.2f}', (W - 150, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        else:
                            cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 255), 2)
                            cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 255), -1)
                            cv2.putText(frame, f'Kurang Tepat', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                if rekam:
                    rekam = False

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Setelah loop selesai, lepaskan kamera
    cap.release()

    # Tutup semua jendela OpenCV
    cv2.destroyAllWindows()

@blueprint.route('/belajar', methods=['GET'])
@login_required
def belajar():
    merged_labels_dict = {}

    for key in set(labels_dict1) | set(labels_dict2):
        values = []
        if key in labels_dict1 and not labels_dict1[key].startswith('SALAH'):
            values.append(labels_dict1[key])
        if key in labels_dict2 and not labels_dict2[key].startswith('SALAH'):
            values.append(labels_dict2[key])
        merged_labels_dict[key] = values

    # Extract the quiz letters from the merged dictionary
    huruf = [value for values in merged_labels_dict.values() for value in values]
    huruf.sort()

    kunci_dict = {}
    for letter in huruf:
        file_path = f'apps/static/Kunci/{letter}.txt'
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                kunci_dict[letter] = file.read()
        except FileNotFoundError:
            kunci_dict[letter] = f'File for {letter} not found'

    return render_template('home/belajar.html', huruf=huruf, kunci=kunci_dict)

@blueprint.route('/update_quiz/<letter>')
def update_quiz(letter):
    global huruf_index
    merged_labels_dict = {}

    for key in set(labels_dict1) | set(labels_dict2):
        values = []
        if key in labels_dict1:
            values.append(labels_dict1[key])
        if key in labels_dict2:
            values.append(labels_dict2[key])
        merged_labels_dict[key] = values

    # Extract the quiz letters from the merged dictionary
    huruf = [value for values in merged_labels_dict.values() for value in values]
    huruf.sort()
    huruf_index = huruf.index(letter)
    return "Quiz index updated"

@blueprint.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

recognized_characters = ""

@blueprint.route('/transkrip', methods=['GET'])
@login_required
def transkrip():
    return render_template('home/transkrip.html')

@blueprint.route('/recognized_characters', methods=['GET'])
def get_recognized_characters():
    global recognized_characters
    return jsonify({'recognized_characters': recognized_characters})

@blueprint.route('/add_space', methods=['POST'])
@login_required
def add_space():
    global recognized_characters
    recognized_characters += ' '
    return jsonify({'recognized_characters': recognized_characters})

@blueprint.route('/remove_last_character', methods=['POST'])
def remove_last_character():
    global recognized_characters
    if recognized_characters.endswith(' '):
        recognized_characters = recognized_characters[:-2]
    else:
        recognized_characters = recognized_characters[:-1]
    return jsonify({'recognized_characters': recognized_characters})

@blueprint.route('/reset_recognized_characters', methods=['POST'])
def reset_recognized_characters():
    global recognized_characters
    recognized_characters = ""
    return jsonify({'recognized_characters': recognized_characters})

def generate_trans():
    update_label_dictionaries_and_models()
    global cap, start_time, recognized_characters
    # Inisialisasi webcam dan variabel lainnya
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=2,  
    min_detection_confidence=0.3 
)
    
    model1 = pickle.load(open('apps/static/model/model1.p', 'rb'))
    model2 = pickle.load(open('apps/static/model/model2.p', 'rb'))

    start_time = None
    rekam = False

    while True:
        data_aux = []
        x_ = []  # Initialize the x_ list for hand landmarks
        y_ = []  # Initialize the y_ list for hand landmarks

        ret, frame = cap.read()

        if not ret:
            break

        # cv2.putText(frame, 'Mulai Menerjemahkan', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
        #             cv2.LINE_AA)

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
            hand_detected_outside_box = any(any(box_x1 > landmark.x * W or landmark.x * W > box_x2 or box_y1 > landmark.y * H or landmark.y * H > box_y2 for landmark in hand_landmarks.landmark) for hand_landmarks in results.multi_hand_landmarks)

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

            if hand_detected_outside_box:
                cv2.putText(frame, "Posisikan Tangan dalam Box", (box_x1, box_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,  
                        hand_landmarks,  
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10

                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                # Check if the hand is inside the bounding box
                hand_inside_box = box_x1 < x1 < box_x2 and box_y1 < y1 < box_y2

                rect_y1 = box_y1 - 30
                rect_y2 = box_y1 - 1
                rect_x1 = box_x1  # Adjust this value to move the rectangle to the right
                rect_x2 = box_x2 - 120  # Adjust this value to move the rectangle to the right
                text_y = box_y1 - 11
                text_x = box_x1 + 5  # Adjust this value to move the text to the right

                if hand_inside_box and not rekam:
                    if len(results.multi_hand_landmarks) == 1:
                        prediction1 = model1.predict([np.asarray(data_aux)])
                        predicted_character = labels_dict1[int(prediction1[0])]
                        confidence1 = model1.predict_proba([np.asarray(data_aux)])
                        confidence1 = np.max(confidence1)

                        if start_time is None:
                            start_time = time.time()

                        if predicted_character:
                            if confidence1 > 0.50:
                                rekam = True
                                cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 2)
                                cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 255, 0), -1)
                                cv2.putText(frame, f'Huruf : {predicted_character}', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                                # cv2.rectangle(frame, (W - 200, H - 40), (W, H), (245, 117, 16), -1)
                                # cv2.putText(frame, f'{confidence1:.2f}', (W - 150, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                                if start_time is not None and time.time() - start_time >= 1.5:
                                    prediction1 = model1.predict([np.asarray(data_aux)])
                                    predicted_character = labels_dict1[int(prediction1[0])]
                                    recognized_characters += predicted_character

                                    # Tambahkan hasil prediksi huruf selanjutnya
                                    if predicted_character != ' ':
                                        with open("transcript.txt", "a") as f:
                                            f.write(predicted_character)

                                    start_time = None  # Reset waktu awal
                            else:
                                cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 255), 2)
                                cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 255), -1)
                                cv2.putText(frame, f'Kurang Tepat', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                                # cv2.rectangle(frame, (W - 200, H - 40), (W, H), (245, 117, 16), -1)
                                # cv2.putText(frame, f'{confidence1:.2f}', (W - 150, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        else:
                            cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 255), 2)
                            cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 255), -1)
                            cv2.putText(frame, f'Kurang Tepat', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        
                    elif len(results.multi_hand_landmarks) == 2:
                        prediction2 = model2.predict([np.asarray(data_aux)])
                        predicted_character = labels_dict2[int(prediction2[0])]
                        confidence2 = model2.predict_proba([np.asarray(data_aux)])
                        confidence2 = np.max(confidence2)
                        if start_time is None:
                            start_time = time.time()

                        if predicted_character:
                            if confidence2 > 0.5:
                                rekam = True
                                cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 2)
                                cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 255, 0), -1)
                                cv2.putText(frame, f'Huruf : {predicted_character}', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                                # cv2.rectangle(frame, (W - 200, H - 40), (W, H), (245, 117, 16), -1)
                                # cv2.putText(frame, f'{confidence2:.2f}', (W - 150, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)  
                                if start_time is not None and time.time() - start_time >= 1.5:
                                    prediction2 = model2.predict([np.asarray(data_aux)])
                                    predicted_character = labels_dict2[int(prediction2[0])]
                                    recognized_characters += predicted_character

                                    # Tambahkan hasil prediksi huruf selanjutnya
                                    if predicted_character != ' ':
                                        with open("transcript.txt", "a") as f:
                                            f.write(predicted_character)

                                    start_time = None  # Reset waktu awa
                            else:
                                cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 255), 2)
                                cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 255), -1)
                                cv2.putText(frame, f'Kurang Tepat', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                                # cv2.rectangle(frame, (W - 200, H - 40), (W, H), (245, 117, 16), -1)
                                # cv2.putText(frame, f'{confidence2:.2f}', (W - 150, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        else:
                            cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 255), 2)
                            cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 255), -1)
                            cv2.putText(frame, f'Kurang Tepat', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                if rekam:
                    rekam = False

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@blueprint.route('/video_trans')
def video_trans():
    return Response(generate_trans(), mimetype='multipart/x-mixed-replace; boundary=frame')

@blueprint.route('/<template>')
@login_required
def route_template(template):
    try:
        if not template.endswith('.html'):
            template += '.html'
            
        segment = get_segment(request)
        context = get_context(segment)
        return render_template("home/" + template, **context)

    except TemplateNotFound:
        return render_template('home/page-404.html'), 404

    except:
        return render_template('home/page-500.html'), 500


# Helper - Extract current page name from request
def get_segment(request):

    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment

    except:
        return None

# @blueprint.route('/video_chunk')
# def video_chunk():
#     combined_video_path = os.path.join(current_app.root_path, 'static', 'kamusku', 'combined_video.mp4')

#     start_byte = int(request.args.get('start', 0))
#     chunk_size = 1024 * 1024  # Ubah sesuai kebutuhan
    
#     with open(combined_video_path, 'rb') as video_file:
#         video_file.seek(start_byte)
#         chunk = video_file.read(chunk_size)
        
#     return Response(chunk, content_type='video/mp4')

# @blueprint.route('/kamus', methods=['GET', 'POST'])
# def kamus():
#     output_folder = os.path.join(current_app.root_path, 'static', 'kamusku')

#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     if request.method == "POST":
#         user_input = request.form.get("user_input").upper()

#         num_columns = 4  # Jumlah kolom yang diinginkan
        
#         video_clips = []

#         if user_input:
#             alphabet_chunks = [user_input[i:i + num_columns] for i in range(0, len(user_input), num_columns)]
#             image_number = 0
#             for chunk in alphabet_chunks:
#                 for i, letter in enumerate(chunk):
#                     image_number += 1
#                     image_path = os.path.join(output_folder, f"{letter}.png")

#                     if os.path.exists(image_path):
#                         image = Image.open(image_path)
#                         resized_image = resize_image(image, (200, 200))
#                         resized_image.save(image_path)

#                     video_path = os.path.join(output_folder, f"{letter}.mp4")

#                     if os.path.exists(video_path):
#                         video_clip = VideoFileClip(video_path)
#                         video_clips.append(video_clip)

#             if video_clips:
#                 combined_clip = concatenate_videoclips(video_clips)
#                 combined_video_file = os.path.join(output_folder, "combined_video.mp4")
#                 combined_clip.write_videofile(combined_video_file, codec="libx264", audio_codec="aac")

#                 combined_video_path = os.path.join(output_folder, "combined_video.mp4")
#                 return render_template("home/kamus.html", user_input=user_input, combined_video_path=combined_video_path)
#             else:
#                 return render_template("home/kamus.html", user_input=user_input, error_message="Data tidak ditemukan untuk kata ini.")

#     else:
#         alphabet_set = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

#         letters_data = []

#         for letter in sorted(alphabet_set):
#             letter_data = {"letter": letter, "images": [], "videos": []}

#             image_path = os.path.join(output_folder, f"{letter}.png")

#             if os.path.exists(image_path):
#                 letter_data["images"].append(f"{letter}.png")

#             video_path = os.path.join(output_folder, f"{letter}.mp4")

#             if os.path.exists(video_path):
#                 letter_data["videos"].append(f"{letter}.mp4")

#             letters_data.append(letter_data)

#         return render_template("home/kamus.html", letters_data=letters_data)
    
# @blueprint.route('/terjemahan', methods=['GET'])
# @login_required
# def terjemahan():
#     try:
#         with open("transcript.txt", "r") as f:
#             transcript_content = f.read()
#     except FileNotFoundError:
#         transcript_content = "Belum ada Terjemahan"
#     with open("transcript.txt", "w") as f:
#         f.truncate(0)
#     return render_template('home/terjemahan.html', transcript_content=transcript_content)

# def load_model_from_database(model_name):
#     conn = get_db_connection()
#     try:
#         cursor = conn.cursor()
#         cursor.execute("SELECT model_data FROM models WHERE model_name = %s", (model_name,))
#         model_data = cursor.fetchone()
#         if model_data is not None:
#             return pickle.loads(model_data[0])
#         else:
#             return None
#     finally:
#         conn.close()

# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# @blueprint.route('/next')
# def next_quiz():
#     global huruf_index
#     huruf_index = (huruf_index + 1) % 26
#     return "Quiz index updated"

# @blueprint.route('/prev')
# def prev_quiz():
#     global huruf_index
#     huruf_index = (huruf_index - 1) % 26
#     return "Quiz index updated"


# @blueprint.route('/close')
# def close():
#     # Setelah loop selesai, lepaskan kamera
#     cap.release()
#     with open("transcript.txt", "w") as f:
#         f.truncate(0)

#     # Tutup semua jendela OpenCV
#     cv2.destroyAllWindows()
#     return"tutup windows"
