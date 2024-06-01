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
def resize_and_save_image(image_data, target_size=(480, 480)):
    try:
        # Convert the byte to a NumPy array
        np_array = np.frombuffer(image_data, np.uint8)
        image_np = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        # Resize the image to the target size
        resized_image = cv2.resize(image_np, target_size, interpolation=cv2.INTER_AREA)

        # Encode the resized image to bytes
        _, image_data_resized = cv2.imencode(".jpg", resized_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        image_data_resized_bytes = image_data_resized.tobytes()

        return image_data_resized_bytes

    except cv2.error as e:
        print(f"OpenCV error during image processing: {e}")
        return None
    except Exception as e:
        print(f"Error during image resizing: {e}")
        return None


def save_photo_to_db(image_data, huruf, kategori_id):
    connection = get_db_connection()  
    cursor = connection.cursor()
    query = f"SELECT class_label FROM dataset WHERE huruf = %s AND kategori_id = %s"
    cursor.execute(query, (huruf, kategori_id))
    result = cursor.fetchall()

    if result:
        class_label = result[0][0]
    else:
        query = f"SELECT MAX(class_label) FROM dataset WHERE kategori_id = %s"
        cursor.execute(query, (kategori_id,))
        max_class_label = cursor.fetchone()[0]
        class_label = max_class_label + 1 if max_class_label is not None else 0
    query = f"INSERT INTO dataset (kategori_id, class_label, image_data, huruf) VALUES (%s, %s, %s, %s)"
    cursor.execute(query, (kategori_id, class_label, image_data, huruf))
    connection.commit()
    connection.close()


def huruf_exists_in_data_tangan(huruf):
    connection = get_db_connection()
    cursor = connection.cursor()
    
    query = "SELECT COUNT(*) FROM dataset WHERE huruf = %s"
    cursor.execute(query, (huruf,))
    count = cursor.fetchone()[0]
    
    connection.close()
    
    return count > 0


















# def save_photo_to_db(image_data, huruf, table_name):
#     connection = get_db_connection()
#     cursor = connection.cursor()
#     query = f"SELECT class_label FROM {table_name} WHERE huruf = %s"
#     cursor.execute(query, (huruf,))
#     result = cursor.fetchall()
#     if result:
#         class_label = result[0][0]
#     else:
#         query = f"SELECT MAX(class_label) FROM {table_name}"
#         cursor.execute(query)
#         max_class_label = cursor.fetchone()[0]
#         class_label = max_class_label + 1 if max_class_label is not None else 0
#     query = f"INSERT INTO {table_name} (class_label, image_data, huruf) VALUES (%s, %s, %s)"
#     cursor.execute(query, (class_label, image_data, huruf))
#     connection.commit()
#     connection.close()



# def huruf_exists_in_data_tangan(huruf):
#     connection = get_db_connection()
#     cursor = connection.cursor()
    
#     query = "SELECT COUNT(*) FROM dataset WHERE huruf = %s UNION ALL SELECT COUNT(*) FROM data_2_tangan WHERE huruf = %s"
#     cursor.execute(query, (huruf, huruf))
#     counts = cursor.fetchall()
    
#     connection.close()
    
#     total_count = sum(count[0] for count in counts)
    
#     return total_count > 0

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
