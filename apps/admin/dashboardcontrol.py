import mysql.connector

def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="sistem_deteksi_isyarat"
    )

def count_data_in_table(table_name):
    # Fungsi ini akan menghitung jumlah data di dalam suatu tabel di database
    connection = get_db_connection()
    cursor = connection.cursor()

    query = f"SELECT COUNT(*) FROM {table_name}"
    cursor.execute(query)
    result = cursor.fetchone()[0]

    connection.close()

    return result

def count_unique_labels(table_name):
    connection = get_db_connection()
    cursor = connection.cursor()

    query = f"SELECT class_label, huruf, COUNT(*) AS count FROM {table_name} GROUP BY class_label"
    cursor.execute(query)
    result = cursor.fetchall()

    connection.close()

    return result

def is_huruf_exists_in_tables(huruf):
    connection = get_db_connection()
    cursor = connection.cursor()

    query = """
        SELECT COUNT(*) FROM (
            SELECT huruf FROM data_1_tangan
            UNION
            SELECT huruf FROM data_2_tangan
        ) AS combined_table
        WHERE huruf = %s
    """
    cursor.execute(query, (huruf,))
    result = cursor.fetchone()

    connection.close()

    return result[0] > 0


def is_label_exists(table_name, label):
    connection = get_db_connection()
    cursor = connection.cursor()

    query = f"SELECT COUNT(*) FROM {table_name} WHERE class_label = %s"
    cursor.execute(query, (label,))
    result = cursor.fetchone()

    connection.close()

    return result[0] > 0

def delete_data_by_huruf(table_name, huruf):
    connection = get_db_connection()
    cursor = connection.cursor()

    query = f"DELETE FROM {table_name} WHERE huruf = %s"
    cursor.execute(query, (huruf,))
    connection.commit()

    connection.close()