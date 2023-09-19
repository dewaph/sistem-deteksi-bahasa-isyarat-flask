import mysql.connector

# Fungsi untuk menghubungkan ke database
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="sistem_deteksi_isyarat"
    )

# Fungsi untuk menyimpan gambar ke tabel gambar
def save_image_to_db(table_name, class_label, image_data, letter):
    connection = get_db_connection()
    cursor = connection.cursor()

    query = f"INSERT INTO {table_name} (class_label, image_data, huruf) VALUES (%s, %s, %s)"
    cursor.execute(query, (class_label, image_data, letter))

    connection.commit()
    connection.close()

# Function to get the next available index from the specified table
def get_next_available_index(table_name):
    connection = get_db_connection()
    cursor = connection.cursor()

    query = f"SELECT MAX(class_label) FROM {table_name}"
    cursor.execute(query)
    result = cursor.fetchone()[0]

    connection.close()

    return result + 1 if result is not None else 0

# Function to check if hand gesture exists in the specified table and return its class label
def get_existing_label(table_name, hand_gesture):
    connection = get_db_connection()
    cursor = connection.cursor()

    query = f"SELECT class_label FROM {table_name} WHERE huruf = %s"
    cursor.execute(query, (hand_gesture,))
    result = cursor.fetchone()

    connection.close()

    return result[0] if result else None

# Get user input for hand gesture letters and class labels
def get_label_dict(table_choice, hand_gesture):
    if table_choice == 1:
        table_name = "data_1_tangan"
    elif table_choice == 2:
        table_name = "data_2_tangan"
    else:
        print("Invalid table choice.")
        exit()

    label_dict = {}
    # Assuming you have functions like get_next_available_index() and get_existing_label() defined
    next_available_index = get_next_available_index(table_name)
    
    existing_label = get_existing_label(table_name, hand_gesture)

    if existing_label is not None:
        label_dict[existing_label] = hand_gesture
    else:
        label_dict[next_available_index] = hand_gesture

    return label_dict

def hand_gesture_exists_in_data_2_tangan(hand_gesture):
    connection = get_db_connection()
    cursor = connection.cursor()

    query = "SELECT COUNT(*) FROM data_2_tangan WHERE huruf = %s"
    cursor.execute(query, (hand_gesture,))
    count = cursor.fetchone()[0]

    connection.close()

    return count > 0

def hand_gesture_exists_in_data_1_tangan(hand_gesture):
    connection = get_db_connection()
    cursor = connection.cursor()

    query = "SELECT COUNT(*) FROM data_1_tangan WHERE huruf = %s"
    cursor.execute(query, (hand_gesture,))
    count = cursor.fetchone()[0]

    connection.close()

    return count > 0