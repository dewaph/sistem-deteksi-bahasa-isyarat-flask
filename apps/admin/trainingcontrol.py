import pandas as pd
import csv
import pickle
import numpy as np
import mysql.connector

def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="sistem_deteksi_isyarat"
    )

def save_model_to_database(model_name, model):
    model_data = pickle.dumps({model_name: model})
    
    # Explicitly open a connection and cursor
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT COUNT(*) FROM models WHERE model_name = %s", (model_name,))
        count = cursor.fetchone()[0]
        if count == 0:
            cursor.execute("INSERT INTO models (model_name, model_data) VALUES (%s, %s)", (model_name, model_data))
        else:
            cursor.execute("UPDATE models SET model_data = %s WHERE model_name = %s", (model_data, model_name))
        
        conn.commit()
    finally:
        cursor.close()
        conn.close()

def calculate_metrics(conf_matrix):
    TP = np.sum(np.diag(conf_matrix))
    FP = np.sum(conf_matrix.sum(axis=0) - np.diag(conf_matrix))
    FN = np.sum(conf_matrix.sum(axis=1) - np.diag(conf_matrix))
    TPk = np.diag(conf_matrix)
    FPk = conf_matrix.sum(axis=0) - np.diag(conf_matrix)
    FNk = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
    TN = np.sum(np.sum(conf_matrix) - (TPk + FPk + FNk))
    return TP, TN, FP, FN

def calculate_precision_recall_f1(TP, FP, FN, TN):
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return accuracy, precision, recall, f1_score

# Function to load data from CSV
def load_data_from_csv(file_path):
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)
        data = list(reader)
        df = pd.DataFrame(data, columns=headers)
        x = df.drop('label', axis=1)
        y = df['label']
    return x, y