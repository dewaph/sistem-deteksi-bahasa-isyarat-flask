import cv2
import csv
import mediapipe as mp
import numpy as np
import mysql.connector
import os


# Function to connect to the database
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="sistem_deteksi_isyarat"
    )


# Function to get images from the database based on class_label
def get_images_from_db(class_label, num_images=5, dataset=1):
    if dataset == 1:
        table_name = "data_1_tangan"
    elif dataset == 2:
        table_name = "data_2_tangan"
    else:
        return None

    connection = get_db_connection()
    cursor = connection.cursor()

    query = f"SELECT image_data FROM {table_name} WHERE class_label = %s LIMIT %s"
    cursor.execute(query, (class_label, num_images))
    images_data = cursor.fetchall()

    connection.close()

    return images_data

def get_labels_from_db(dataset):
    connection = get_db_connection()
    cursor = connection.cursor()

    if dataset == 1:
        table_name = "data_1_tangan"
    elif dataset == 2:
        table_name = "data_2_tangan"
    else:
        connection.close()
        return []

    query = f"SELECT class_label, huruf FROM {table_name}"
    cursor.execute(query)
    labels = cursor.fetchall()

    connection.close()

    return labels

def create_label_dictionary(labels):
    label_dict = {}
    for class_label, huruf in labels:
        label_dict[class_label] = huruf
    return label_dict

# Fetch labels for dataset 1 and create label dictionary
dataset_1_labels = get_labels_from_db(dataset=1)
label_dataset_1 = create_label_dictionary(dataset_1_labels)

# Fetch labels for dataset 2 and create label dictionary
dataset_2_labels = get_labels_from_db(dataset=2)
label_dataset_2 = create_label_dictionary(dataset_2_labels)

def update_label_dictionaries():
    # Fetch labels for dataset 1 and update label dictionary
    dataset_1_labels = get_labels_from_db(dataset=1)
    global label_dataset_1  # Use global to modify the variable outside the function
    label_dataset_1 = create_label_dictionary(dataset_1_labels)

    # Fetch labels for dataset 2 and update label dictionary
    dataset_2_labels = get_labels_from_db(dataset=2)
    global label_dataset_2  # Use global to modify the variable outside the function
    label_dataset_2 = create_label_dictionary(dataset_2_labels)

# Call this function whenever you want to update the label dictionaries

def process_images(dataset):
    mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)
    # Initialize MediaPipe Hands
    if dataset == 1:
        label_dict = label_dataset_1
    elif dataset == 2:
        label_dict = label_dataset_2
    else:
        return "Invalid dataset"

    processed_images = []

    for class_label, huruf in label_dict.items():
        # Get images from the database for the current class
        images_data = get_images_from_db(class_label, num_images=5, dataset=dataset)

        class_images = []

        for i, image_data in enumerate(images_data):
            nparr = np.frombuffer(image_data[0], np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = mp_hands.process(img_rgb)

            img_with_landmarks = img_rgb.copy()

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        img_with_landmarks,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                        mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )

            class_images.append(img_with_landmarks)

        processed_images.append((class_label, class_images))

    # Close MediaPipe Hands
    mp_hands.close()

    return processed_images

def fetch_data_from_db1():
    connection = get_db_connection()
    cursor = connection.cursor()

    query = "SELECT class_label, image_data FROM data_1_tangan"
    cursor.execute(query)
    result = cursor.fetchall()

    connection.close()

    return result

def fetch_data_from_db2():
    connection = get_db_connection()
    cursor = connection.cursor()

    query = "SELECT class_label, image_data FROM data_2_tangan"
    cursor.execute(query)
    result = cursor.fetchall()

    connection.close()

    return result


def process_hand_landmarks_data1():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)
    data_from_db = fetch_data_from_db1()
    data = []
    labels = []
    for class_label, image_data in data_from_db:
        data_aux = []
        x_ = []
        y_ = []
        image_np = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        if not (results.multi_hand_landmarks is None):
            n = len(results.multi_hand_landmarks)
            if n == 1:
                try:
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
                    data.append(data_aux)
                    labels.append(class_label)
                except:
                    data_aux = np.zeros([1, 63], dtype=np.float32)[0]
    if len(data) > 0 and len(labels) > 0:
        folder_path = "apps/static/csv"
        csv_filename = "onehand.csv"
        file_path = os.path.join(folder_path, csv_filename)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_path = os.path.abspath(file_path)
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['label'] + [f'data_{i}' for i in range(len(data[0]))])  # Writing column headers
            for i in range(len(data)):
                label = labels[i]
                data_row = np.asarray(data[i], dtype=np.float32)
                writer.writerow([label] + data_row.tolist())
        return file_path
    else:
        return None

def process_hand_landmarks_data2():
    mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)

    # Fetch image data and class labels from the database
    data_from_db = fetch_data_from_db2()

    data = []
    labels = []
    for class_label, image_data in data_from_db:
        data_aux = []

        x_ = []
        y_ = []

        # Convert image data from bytes to image
        image_np = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = mp_hands.process(img_rgb)

        if not (results.multi_hand_landmarks is None):
            n = len(results.multi_hand_landmarks)
            if n == 2:
                try:
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
                    data.append(data_aux)
                    labels.append(class_label)
                except:
                    data_aux = np.zeros([1, 189], dtype=np.float32)[0]

    if len(data) > 0 and len(labels) > 0:
        folder_path = "apps/static/csv"
        csv_filename = "twohand.csv"
        file_path = os.path.join(folder_path, csv_filename)
        # Membuat direktori jika belum ada
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_path = os.path.abspath(file_path)
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['label'] + [f'data_{i}' for i in range(len(data[0]))])  # Writing column headers

            for i in range(len(data)):
                label = labels[i]
                data_row = np.asarray(data[i], dtype=np.float32)
                writer.writerow([label] + data_row.tolist())

        
        return file_path
    else:
        return None
