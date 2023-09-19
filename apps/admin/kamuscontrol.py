import mysql.connector
import base64

def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="sistem_deteksi_isyarat"
    )

def is_image_exists(letter):
    connection = get_db_connection()
    cursor = connection.cursor()

    query = "SELECT image_data_id FROM kamus WHERE letter=%s"
    cursor.execute(query, (letter,))
    result = cursor.fetchone()

    connection.close()

    return result[0] is not None if result else False

# Function to check if video_data exists for the given letter in the video_table
def is_video_exists(letter):
    connection = get_db_connection()
    cursor = connection.cursor()

    query = "SELECT video_data_id FROM kamus WHERE letter=%s"
    cursor.execute(query, (letter,))
    result = cursor.fetchone()

    connection.close()

    return result[0] is not None if result else False

# Function to save image and video data to the database and associate them with the kamus entry
def save_data_to_db(letter, image_data, video_data):
    connection = get_db_connection()
    cursor = connection.cursor()

    # Save the image data to the image_table
    query = "INSERT INTO image_table (image_data) VALUES (%s)"
    cursor.execute(query, (image_data,))
    image_data_id = cursor.lastrowid

    # Save the video data to the video_table
    query = "INSERT INTO video_table (video_data) VALUES (%s)"
    cursor.execute(query, (video_data,))
    video_data_id = cursor.lastrowid

    # Save the image_data_id and video_data_id to the kamus table
    query = "INSERT INTO kamus (letter, image_data_id, video_data_id) VALUES (%s, %s, %s)"
    cursor.execute(query, (letter, image_data_id, video_data_id))

    connection.commit()
    connection.close()

# Function to update image data in the image_table and associate it with the kamus entry
def update_data_in_db(letter, image_data, video_data):
    connection = get_db_connection()
    cursor = connection.cursor()

    # Check if the letter exists in the kamus table
    query = "SELECT id, image_data_id, video_data_id FROM kamus WHERE letter=%s"
    cursor.execute(query, (letter,))
    result = cursor.fetchone()

    if result:
        kamus_id, existing_image_id, existing_video_id = result

        # Update the image data in the image_table
        query = "UPDATE image_table SET image_data=%s WHERE id=%s"
        cursor.execute(query, (image_data, existing_image_id))

        # Update the video data in the video_table
        query = "UPDATE video_table SET video_data=%s WHERE id=%s"
        cursor.execute(query, (video_data, existing_video_id))

    else:
        # If the letter does not exist, insert new data
        query = "INSERT INTO image_table (image_data) VALUES (%s)"
        cursor.execute(query, (image_data,))
        image_data_id = cursor.lastrowid

        query = "INSERT INTO video_table (video_data) VALUES (%s)"
        cursor.execute(query, (video_data,))
        video_data_id = cursor.lastrowid

        query = "INSERT INTO kamus (letter, image_data_id, video_data_id) VALUES (%s, %s, %s)"
        cursor.execute(query, (letter, image_data_id, video_data_id))

    connection.commit()
    connection.close()

def delete_data_from_db(letter):
    connection = get_db_connection()
    cursor = connection.cursor()

    # Get the data types (image and video) associated with the kamus entry
    query = "SELECT image_data_id, video_data_id FROM kamus WHERE letter=%s"
    cursor.execute(query, (letter,))
    result = cursor.fetchone()
    image_data_id, video_data_id = result if result else (None, None)

    if image_data_id is not None:
        # Delete the image data from the image_table
        query = "DELETE FROM image_table WHERE id=%s"
        cursor.execute(query, (image_data_id,))

    if video_data_id is not None:
        # Delete the video data from the video_table
        query = "DELETE FROM video_table WHERE id=%s"
        cursor.execute(query, (video_data_id,))

    # Remove the association in the kamus entry
    query = "UPDATE kamus SET image_data_id=NULL, video_data_id=NULL WHERE letter=%s"
    cursor.execute(query, (letter,))

    connection.commit()
    connection.close()

def get_data_from_db(letter, data_type):
    connection = get_db_connection()
    cursor = connection.cursor()

    if data_type == "image":
        query = "SELECT image_data FROM image_table WHERE id = (SELECT image_data_id FROM kamus WHERE letter = %s)"
    elif data_type == "video":
        query = "SELECT video_data FROM video_table WHERE id = (SELECT video_data_id FROM kamus WHERE letter = %s)"
    else:
        raise ValueError("Invalid data_type. Use 'image' or 'video'.")

    cursor.execute(query, (letter,))
    result = cursor.fetchone()

    connection.close()

    if result:
        data = result[0]
        data_base64 = base64.b64encode(data).decode('utf-8')
        return data_base64
    else:
        return None