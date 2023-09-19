import mysql.connector
import cv2
import numpy as np

def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="sistem_deteksi_isyarat"
    )


# Function to resize the image using OpenCV
def resize_and_save_image(image_data, max_size=(256, 256)):
    try:
        # Convert the byte to a NumPy array
        np_array = np.frombuffer(image_data, np.uint8)
        image_np = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        # Resize the image while maintaining the aspect ratio
        height, width = image_np.shape[:2]
        ratio = min(max_size[0] / width, max_size[1] / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        resized_image = cv2.resize(image_np, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Encode the resized image to bytes
        _, image_data_resized = cv2.imencode(".jpg", resized_image)
        image_data_resized_bytes = image_data_resized.tobytes()

        return image_data_resized_bytes

    except Exception as e:
        print(f"Error during image resizing: {e}")
        return None

def save_photo_to_db(image_data, huruf, table_name):
    connection = get_db_connection()
    cursor = connection.cursor()
    query = f"SELECT class_label FROM {table_name} WHERE huruf = %s"
    cursor.execute(query, (huruf,))
    result = cursor.fetchall()
    if result:
        class_label = result[0][0]
    else:
        query = f"SELECT MAX(class_label) FROM {table_name}"
        cursor.execute(query)
        max_class_label = cursor.fetchone()[0]
        class_label = max_class_label + 1 if max_class_label is not None else 0
    query = f"INSERT INTO {table_name} (class_label, image_data, huruf) VALUES (%s, %s, %s)"
    cursor.execute(query, (class_label, image_data, huruf))
    connection.commit()
    connection.close()


def huruf_exists_in_data_2_tangan(huruf):
    connection = get_db_connection()
    cursor = connection.cursor()
    
    query = "SELECT COUNT(*) FROM data_2_tangan WHERE huruf = %s"
    cursor.execute(query, (huruf,))
    count = cursor.fetchone()[0]
    
    connection.close()
    
    return count > 0

def huruf_exists_in_data_1_tangan(huruf):
    connection = get_db_connection()
    cursor = connection.cursor()
    
    query = "SELECT COUNT(*) FROM data_1_tangan WHERE huruf = %s"
    cursor.execute(query, (huruf,))
    count = cursor.fetchone()[0]
    
    connection.close()
    
    return count > 0