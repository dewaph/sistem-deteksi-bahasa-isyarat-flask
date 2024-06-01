import cv2
import csv
import mediapipe as mp
import numpy as np
import mysql.connector
import os
import base64
from math import ceil
import pandas as pd



# Function to connect to the database
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="sistem_deteksi_isyarat"
    )


# Function to get images from the database based on class_label
def get_images_from_db(class_label, num_images=20, dataset=1):
    if dataset == 1:
        table_name = "dataset"
        kategori_id = 1
    elif dataset == 2:
        table_name = "dataset"
        kategori_id = 2
    else:
        return None

    connection = get_db_connection()
    cursor = connection.cursor()

    query = f"SELECT image_data FROM {table_name} WHERE class_label = %s AND kategori_id = %s LIMIT %s"
    cursor.execute(query, (class_label, kategori_id, num_images))
    images_data = cursor.fetchall()

    connection.close()

    return images_data

def get_labels_process(dataset):
    connection = get_db_connection()
    cursor = connection.cursor()

    if dataset == 1:
        dataset = 1
    elif dataset == 2:
        dataset = 2
    else:
        connection.close()
        return []

    query = f"SELECT DISTINCT class_label, huruf FROM processed_images WHERE dataset = {dataset}"
    cursor.execute(query)
    labels = cursor.fetchall()

    connection.close()

    return labels

def create_label_process(labels):
    label_dict = {}
    for class_label, huruf in labels:
        label_dict[class_label] = huruf
    return label_dict

# Fetch labels for dataset 1 and create label dictionary
dataset_1_labels = get_labels_process(dataset=1)
label_dataset_1 = create_label_process(dataset_1_labels)

# Fetch labels for dataset 2 and create label dictionary
dataset_2_labels = get_labels_process(dataset=2)
label_dataset_2 = create_label_process(dataset_2_labels)

def update_label_process():
    # Fetch labels for dataset 1 and update label dictionary
    dataset_1_labels = get_labels_process(dataset=1)
    global label_dataset_1  # Use global to modify the variable outside the function
    label_dataset_1 = create_label_process(dataset_1_labels)

    # Fetch labels for dataset 2 and update label dictionary
    dataset_2_labels = get_labels_process(dataset=2)
    global label_dataset_2  # Use global to modify the variable outside the function
    label_dataset_2 = create_label_process(dataset_2_labels)

def get_labels_from_db(dataset):
    if dataset == 1:
        table_name = "dataset"
        kategori_id = 1
    elif dataset == 2:
        table_name = "dataset"
        kategori_id = 2
    else:
        return []

    connection = get_db_connection()
    cursor = connection.cursor()

    query = f"SELECT class_label, huruf FROM {table_name} WHERE kategori_id = %s"
    cursor.execute(query, (kategori_id,))
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

def process_and_save_images(dataset):
    mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1 if dataset == 1 else 2, min_detection_confidence=0.3, min_tracking_confidence=0.5)

    label_dict = label_dataset_1 if dataset == 1 else label_dataset_2

    for class_label, huruf in label_dict.items():
        # Get images from the database for the current class
        images_data = get_images_from_db(class_label, num_images=20, dataset=dataset)

        for i, image_data in enumerate(images_data):
            nparr = np.frombuffer(image_data[0], np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = mp_hands.process(img_rgb)

            img_with_landmarks = img.copy()

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        img_with_landmarks,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),  
                        mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)  
                    )

            # Save processed image to the database
            save_image_to_db(class_label, huruf, img_with_landmarks, dataset)

    mp_hands.close()

# New function to save images to the database
def save_image_to_db(class_label, huruf, img_with_landmarks, dataset):
    connection = get_db_connection()
    cursor = connection.cursor()

    # Convert image to bytes
    _, buffer = cv2.imencode('.jpg', img_with_landmarks)
    image_data = buffer.tobytes()

    # Insert the image into the database with timestamp
    query = "INSERT INTO processed_images (class_label, huruf, dataset, image_data) VALUES (%s, %s, %s, %s)"
    cursor.execute(query, (class_label, huruf, dataset, image_data))

    connection.commit()
    connection.close()

# New function to fetch processed images from the database
def fetch_processed_images(dataset, page, per_page=3):
    connection = get_db_connection()
    cursor = connection.cursor()

    label_dict = label_dataset_1 if dataset == 1 else label_dataset_2

    processed_images = []

    for class_label, huruf in label_dict.items():
        query = "SELECT image_data FROM processed_images WHERE class_label = %s AND dataset = %s"
        cursor.execute(query, (class_label, dataset))
        images_data = cursor.fetchall()

        class_images = []

        for image_data in images_data:
            class_images.append({'class_label': class_label, 'image_data': base64.b64encode(image_data[0]).decode('utf-8')})

        processed_images.append(class_images)

    connection.close()

    # Paginate the processed images
    start = (page - 1) * per_page
    end = start + per_page
    paginated_images = processed_images[start:end]

    # Calculate the total number of pages
    total_pages = ceil(len(processed_images) / per_page)

    return paginated_images, total_pages

def fetch_data_from_db1():
    connection = get_db_connection()
    cursor = connection.cursor()

    query = "SELECT class_label, image_data FROM dataset WHERE kategori_id = 1"
    cursor.execute(query)
    result = cursor.fetchall()

    connection.close()

    return result

def fetch_data_from_db2():
    connection = get_db_connection()
    cursor = connection.cursor()

    query = "SELECT class_label, image_data FROM dataset WHERE kategori_id = 2"
    cursor.execute(query)
    result = cursor.fetchall()

    connection.close()

    return result

def process_hand_landmarks_data1():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3, min_tracking_confidence=0.5)
    data_from_db = fetch_data_from_db1()
    data = []
    labels = []
    for class_label, image_data in data_from_db:
        data_aux = []
        x_ = []
        # y_ = []

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
                            # y = hand_landmarks.landmark[i].y
                            x_.append(x)
                            # y_.append(y)

                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            # y = hand_landmarks.landmark[i].y
                            data_aux.append(x - min(x_))
                            # data_aux.append(y - min(y_))

                    data.append(data_aux)
                    labels.append(class_label)
                except:
                    data_aux = np.zeros([1, 63], dtype=np.float32)[0]
    if len(data) > 0 and len(labels) > 0:
        folder_path = "apps/static/csv"
        csv_filename = "onehand.csv"
        file_path = os.path.join(folder_path, csv_filename)
        if os.path.exists(file_path):
            print(f"sukses")
        else:
            # os.makedirs(folder_path)
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
    mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3, min_tracking_confidence=0.5)

    # Fetch image data and class labels from the database
    data_from_db = fetch_data_from_db2()

    data = []
    labels = []
    for class_label, image_data in data_from_db:
        data_aux = []
        x_ = []
        # y_ = []

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
                            # y = hand_landmarks.landmark[i].y
                            x_.append(x)
                            # y_.append(y)

                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            # y = hand_landmarks.landmark[i].y
                            data_aux.append(x - min(x_))
                            # data_aux.append(y - min(y_))

                    data.append(data_aux)
                    labels.append(class_label)
                except:
                    data_aux = np.zeros([1, 189], dtype=np.float32)[0]

    if len(data) > 0 and len(labels) > 0:
        folder_path = "apps/static/csv"
        csv_filename = "twohand.csv"
        file_path = os.path.join(folder_path, csv_filename)
        # Membuat direktori jika belum ada
        if os.path.exists(file_path):
            print(f"sukses")
        else:
            # os.makedirs(folder_path)
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
    




































































































































































def process_images(dataset):
    mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1 if dataset == 1 else 2, min_detection_confidence=0.3, min_tracking_confidence=0.5)
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
        images_data = get_images_from_db(class_label, num_images=20, dataset=dataset)

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


# New function to convert processed images to base64 format
def convert_to_base64(processed_images):
    base64_images = []

    for class_images in processed_images:
        base64_class_images = []
        for image_record in class_images:
            base64_class_images.append({'class_label': image_record['class_label'], 'image_data': image_record['image_data']})
        base64_images.append(base64_class_images)

    return base64_images





























































# def process_hand_landmarks_data1():
#     mp_hands = mp.solutions.hands
#     mp_drawing = mp.solutions.drawing_utils
#     mp_drawing_styles = mp.solutions.drawing_styles
#     hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3, min_tracking_confidence=0.5)
#     data_from_db = fetch_data_from_db1()
#     data = []
#     labels = []
    
#     for class_label, image_data in data_from_db:
#         data_aux = []
#         x_ = []
        
#         image_np = np.frombuffer(image_data, np.uint8)
#         img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         results = hands.process(img_rgb)
        
#         if not (results.multi_hand_landmarks is None):
#             n = len(results.multi_hand_landmarks)
            
#             if n == 1:
#                 try:
#                     for hand_landmarks in results.multi_hand_landmarks:
#                         for i in range(len(hand_landmarks.landmark)):
#                             x = hand_landmarks.landmark[i].x
                           
#                             x_.append(x)
                            
                            
#                         for i in range(len(hand_landmarks.landmark)):
#                             x = hand_landmarks.landmark[i].x
                            
#                             data_aux.append(x - min(x_))
                            
                            
#                     data.append(data_aux)
#                     labels.append(class_label)
                
#                 except:
#                     data_aux = np.zeros([1, 63], dtype=np.float32)[0]

#     if len(data) > 0 and len(labels) > 0:
#         # Create a DataFrame using pandas
#         df = pd.DataFrame(data, columns=[f'data_{i}' for i in range(len(data[0]))])
#         df.insert(0, 'label', labels)
        
#         # Specify the Excel file path
#         folder_path = "apps/static/excel"
#         excel_filename = "onehand.xlsx"
#         excel_path = os.path.join(folder_path, excel_filename)
        
#         # Create the folder if it doesn't exist
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)
        
#         # Save the DataFrame to an Excel file
#         df.to_excel(excel_path, index=False)
        
#         return excel_path
    
#     else:
#         return None
    
# def process_hand_landmarks_data2():
#     mp_hands = mp.solutions.hands
#     mp_drawing = mp.solutions.drawing_utils
#     mp_drawing_styles = mp.solutions.drawing_styles
#     hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3, min_tracking_confidence=0.5)
#     data_from_db = fetch_data_from_db2()
#     data = []
#     labels = []
    
#     for class_label, image_data in data_from_db:
#         data_aux = []
#         x_ = []
        
#         image_np = np.frombuffer(image_data, np.uint8)
#         img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         results = hands.process(img_rgb)
        
#         if not (results.multi_hand_landmarks is None):
#             n = len(results.multi_hand_landmarks)
            
#             if n == 2:
#                 try:
#                     for hand_landmarks in results.multi_hand_landmarks:
#                         for i in range(len(hand_landmarks.landmark)):
#                             x = hand_landmarks.landmark[i].x
                            
#                             x_.append(x)
                            
                            
                            
#                         for i in range(len(hand_landmarks.landmark)):
#                             x = hand_landmarks.landmark[i].x
                            
#                             data_aux.append(x - min(x_))
                            
                            
                    
#                     data.append(data_aux)
#                     labels.append(class_label)
                
#                 except:
#                     data_aux = np.zeros([1, 63], dtype=np.float32)[0]

#     if len(data) > 0 and len(labels) > 0:
#         # Create a DataFrame using pandas
#         df = pd.DataFrame(data, columns=[f'data_{i}' for i in range(len(data[0]))])
#         df.insert(0, 'label', labels)
        
#         # Specify the Excel file path
#         folder_path = "apps/static/excel"
#         excel_filename = "twohand.xlsx"
#         excel_path = os.path.join(folder_path, excel_filename)
        
#         # Create the folder if it doesn't exist
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)
        
#         # Save the DataFrame to an Excel file
#         df.to_excel(excel_path, index=False)
        
#         return excel_path
    
#     else:
#         return None