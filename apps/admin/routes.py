from apps.admin import blueprint
from flask import render_template, request, jsonify, Response, send_file, send_from_directory
from flask_login import login_required
from jinja2 import TemplateNotFound
from .dashboardcontrol import count_data_in_table, count_unique_labels, is_label_exists, is_huruf_exists_in_tables, get_db_connection
import pandas as pd
from .kamuscontrol import is_image_exists, is_video_exists, get_data_from_db, update_data_in_db, save_data_to_db, delete_data_from_db
import os
from base64 import b64encode
from .datacontrol import get_db_connection, resize_and_save_image, save_photo_to_db, huruf_exists_in_data_tangan
from .preprocesscontrol import get_db_connection, process_images, process_hand_landmarks_data1, process_hand_landmarks_data2, update_label_dictionaries, process_and_save_images, fetch_processed_images, convert_to_base64, update_label_process
import cv2
import base64
from .trainingcontrol import get_db_connection, save_model, calculate_metrics, calculate_precision_recall_f1, load_data_from_csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from .testcontrol import get_db_connection, generate_frames, update_label_dictionaries_and_models
from .auto import get_db_connection, get_label_dict, save_image_to_db, hand_gesture_exists_in_data_1_tangan, hand_gesture_exists_in_data_2_tangan



@blueprint.route('/admin', methods=['GET', 'POST'])
@login_required
def admin():
    data1tangan_count = count_data_in_table(1)
    data2tangan_count = count_data_in_table(2)
    total_data_count = data1tangan_count + data2tangan_count

    unique_labels_data1 = count_unique_labels(1)
    unique_labels_data2 = count_unique_labels(2)
    tabel = unique_labels_data1 + unique_labels_data2
    total_unique_labels = len(set(unique_labels_data1 + unique_labels_data2))

    data1_df = pd.DataFrame(tabel, columns=['huruf', 'jumlah data gambar', 'kategori'])
    data1_df = data1_df.sort_values(by='huruf')

    return render_template('admin/home/index.html', segment='index', total_data_count=total_data_count,
                           data1tangan_count=data1tangan_count,
                           data2tangan_count=data2tangan_count,
                           total_unique_labels=total_unique_labels,
                           data1_df=data1_df,
                           )

@blueprint.route('/get_huruf_options', methods=['GET'])
def get_huruf_options():
    table_name = request.args.get('table_name')
    
    connection = get_db_connection()
    cursor = connection.cursor()

    query = f"SELECT DISTINCT huruf FROM dataset WHERE kategori_id = {table_name}"

    cursor.execute(query)
    result = cursor.fetchall()

    huruf_options = [row[0] for row in result]

    connection.close()

    return jsonify(huruf_options)
    
@blueprint.route('/delete_data', methods=['POST'])
def delete_data():
    data = request.get_json()
    huruf = data.get('huruf')

    connection = get_db_connection()
    cursor = connection.cursor()

    query = f"DELETE FROM dataset WHERE huruf = %s"
    cursor.execute(query, (huruf,))

    processquery = f"DELETE FROM processed_images WHERE huruf = %s"
    cursor.execute(processquery, (huruf,))
    connection.commit()

    connection.close()

    response = {"message": f"Data dengan huruf '{huruf}' telah dihapus."}
    return jsonify(response), 200

@blueprint.route('/tambah_data')
@login_required
def tambah_data():
    return render_template('admin/home/data.html')

@blueprint.route('/upload', methods=['POST'])
def upload_photo():
    data = request.form.to_dict()
    uploaded_files = request.files.getlist('uploaded_files')
    selected_table = data['table_name']
    new_value = data['new_value']
    connection = get_db_connection()
    cursor = connection.cursor()

    # Tentukan "kategori_id" berdasarkan "selected_table"
    if selected_table == 'data_1_tangan':
        kategori_id = 1
    elif selected_table == 'data_2_tangan':
        kategori_id = 2
    else:
        kategori_id = None  

    query = f"SELECT DISTINCT huruf FROM dataset WHERE kategori_id = {kategori_id}"
    cursor.execute(query)
    existing_huruf_choices = [row[0] for row in cursor.fetchall()]
    connection.close()

    if kategori_id is None:
        return jsonify({'error': 'Tabel tidak dikenali'}), 400

    if huruf_exists_in_data_tangan(new_value):
        return jsonify({'error': 'huruf sudah ada dalam database'}), 400

    if new_value:
        for uploaded_file in uploaded_files:
            try:
                image_data = uploaded_file.read()
                image_data_resized = resize_and_save_image(image_data)
                if image_data is not None:
                    save_photo_to_db(image_data_resized, new_value, kategori_id)
                else:
                    return jsonify({'error': 'Terjadi kesalahan saat merubah ukuran gambar. Silakan unggah gambar yang valid.'}), 400
            except Exception as e:
                return jsonify({'error': f'Error dalam pemrosesan gambar yang diunggah: {e}'}), 500

    return jsonify({'message': 'Gambar berhasil ditambahkan'}), 200

@blueprint.route('/preprocessing')
@login_required
def preprocessing():
    update_label_process()
    def get_labels_from_db(dataset):
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

    def create_label_dictionary(labels):
        label_dict = {}
        for class_label, huruf in labels:
            label_dict[class_label] = huruf
        return label_dict

    dataset_1_labels = get_labels_from_db(dataset=1)
    label_dataset_1 = create_label_dictionary(dataset_1_labels)

    dataset_2_labels = get_labels_from_db(dataset=2)
    label_dataset_2 = create_label_dictionary(dataset_2_labels)

    # Fetch processed images from the database for dataset 1 (page 1)
    page_dataset_1 = int(request.args.get('page_dataset_1', 1))
    dataset_1_images, total_pages_dataset_1 = fetch_processed_images(dataset=1, page=page_dataset_1)

    # Fetch processed images from the database for dataset 2 (page 1)
    page_dataset_2 = int(request.args.get('page_dataset_2', 1))
    dataset_2_images, total_pages_dataset_2 = fetch_processed_images(dataset=2, page=page_dataset_2)

    # Convert processed images to base64 format
    base64_dataset_1_images = convert_to_base64(dataset_1_images)
    base64_dataset_2_images = convert_to_base64(dataset_2_images)

    show_card = bool(dataset_1_images)
    show_card2 = bool(dataset_2_images)
    
    return render_template('admin/home/preprocessing.html', dataset_1_images=base64_dataset_1_images, dataset_2_images=base64_dataset_2_images, label_dataset_1=label_dataset_1, label_dataset_2=label_dataset_2, show_card=show_card, show_card2=show_card2, total_pages_dataset_1=total_pages_dataset_1, total_pages_dataset_2=total_pages_dataset_2,
                           current_page_dataset_1=page_dataset_1, current_page_dataset_2=page_dataset_2)

def clean():
    connection = get_db_connection()
    cursor = connection.cursor()

    # Add statements to delete or truncate tables as needed
    cursor.execute("DELETE FROM processed_images")
    # Add more delete or truncate statements for other tables if necessary

    connection.commit()
    connection.close()

@blueprint.route('/preprocess', methods=['POST'])
def preprocess():
    clean()
    update_label_dictionaries()
    def get_labels_from_db(dataset):
        connection = get_db_connection()
        cursor = connection.cursor()

        if dataset == 1:
            kategori_id = 1
        elif dataset == 2:
            kategori_id = 2
        else:
            connection.close()
            return []

        query = f"SELECT class_label, huruf FROM dataset WHERE kategori_id = {kategori_id}"
        cursor.execute(query)
        labels = cursor.fetchall()

        connection.close()

        return labels

    def create_label_dictionary(labels):
        label_dict = {}
        for class_label, huruf in labels:
            label_dict[class_label] = huruf
        return label_dict

    dataset_1_labels = get_labels_from_db(dataset=1)
    label_dataset_1 = create_label_dictionary(dataset_1_labels)

    dataset_2_labels = get_labels_from_db(dataset=2)
    label_dataset_2 = create_label_dictionary(dataset_2_labels)

    # Process and save images for dataset 1
    process_and_save_images(dataset=1)

    # Process and save images for dataset 2
    process_and_save_images(dataset=2)

    # Fetch processed images from the database for dataset 1 (page 1)
    page_dataset_1 = int(request.args.get('page_dataset_1', 1))
    dataset_1_images, total_pages_dataset_1 = fetch_processed_images(dataset=1, page=page_dataset_1)

    # Fetch processed images from the database for dataset 2 (page 1)
    page_dataset_2 = int(request.args.get('page_dataset_2', 1))
    dataset_2_images, total_pages_dataset_2 = fetch_processed_images(dataset=2, page=page_dataset_2)


    # Convert processed images to base64 format
    base64_dataset_1_images = convert_to_base64(dataset_1_images)
    base64_dataset_2_images = convert_to_base64(dataset_2_images)

    show_card = bool(dataset_1_images)
    show_card2 = bool(dataset_2_images)

    return render_template('admin/home/preprocessing.html', dataset_1_images=base64_dataset_1_images, dataset_2_images=base64_dataset_2_images, label_dataset_1=label_dataset_1, label_dataset_2=label_dataset_2, show_card=show_card, show_card2=show_card2, total_pages_dataset_1=total_pages_dataset_1, total_pages_dataset_2=total_pages_dataset_2,
                           current_page_dataset_1=page_dataset_1, current_page_dataset_2=page_dataset_2)

@blueprint.route('/export-csv', methods=['POST'])
def export_csv():
    csv_type = request.form.get('csv')

    if csv_type == 'onehand':
        file_path = process_hand_landmarks_data1()
    elif csv_type == 'twohand':
        file_path = process_hand_landmarks_data2()
    else:
        file_path = None

    if file_path:
        return jsonify({"success": True, "message": "Landmark CSV berhasil dibuat", "file_path": file_path})
    else:
        return jsonify({"success": False, "message": "gagal membuat Landmark CSV"})
    
@blueprint.route('/training')
@login_required
def training():
    UPLOAD_FOLDER = os.path.join("apps", "static", "csv")
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    csv_file1 = os.path.join(UPLOAD_FOLDER, "onehand.csv")
    csv_file2 = os.path.join(UPLOAD_FOLDER, "twohand.csv")
    table_data1 = "Data Kosong, buat CSV 1 tangan terlebih dahulu pada fitur buat landmark"
    table_data2 = "Data Kosong, buat CSV 2 tangan terlebih dahulu pada fitur buat landmark"
    if os.path.isfile(csv_file1):
        df1 = pd.read_csv(csv_file1)
        limited_df1 = pd.concat([df1.head(20), df1.tail(20)])
        table_data1 = limited_df1.to_html(classes="table", index=False)

    if os.path.isfile(csv_file2):
        df2 = pd.read_csv(csv_file2)
        limited_df2 = pd.concat([df2.head(20), df2.tail(20)])
        table_data2 = limited_df2.to_html(classes="table", index=False)
    
    return render_template('admin/home/training.html', table_data1=table_data1, table_data2=table_data2)

@blueprint.route('/start_training', methods=['POST'])
def start_training():
    def get_unique_labels(kategori_id):
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(f"SELECT DISTINCT huruf FROM dataset WHERE kategori_id = {kategori_id}")
        unique_labels = [row[0] for row in cursor.fetchall()]

        cursor.close()
        conn.close()

        return unique_labels

    def create_label_dict(label_list):
        return {label: index for index, label in enumerate(label_list)}

    unique_labels1 = get_unique_labels(1)
    unique_labels2 = get_unique_labels(2)

    label1 = create_label_dict(unique_labels1)
    label2 = create_label_dict(unique_labels2)
    # Load data from CSV files
    x, y = load_data_from_csv('apps/static/csv/onehand.csv')
    x2, y2 = load_data_from_csv('apps/static/csv/twohand.csv')

    # Split data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, stratify=y)
    x_train2, x_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size=0.2, shuffle=True, stratify=y2)

    # Train models
    model1 = RandomForestClassifier()
    model2 = RandomForestClassifier()

    model1.fit(x_train, y_train)
    model2.fit(x_train2, y_train2)

    # Make predictions on test data
    y_predict = model1.predict(x_test)
    y_predict2 = model2.predict(x_test2)

    # Calculate True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN) for Model 1
    cm1 = confusion_matrix(y_test, y_predict)
    TP1, TN1, FP1, FN1 = calculate_metrics(cm1)

    # Calculate Precision, Recall, and F1-score for Model 1
    accuracy1, precision1, recall1, f1_score1 = calculate_precision_recall_f1(TP1, FP1, FN1, TN1)

    # Tampilkan hasil untuk Model 1
    TP_1 = f"{TP1}"
    TN_1 = f"{TN1}"
    FP_1 = f"{FP1}"
    FN_1 = f"{FN1}"
    akurasi_1 = f"{accuracy1 * 100:.2f}%"
    precision_1 = f"{precision1 * 100:.2f}%"
    recall_1 = f"{recall1 * 100:.2f}%"
    f_1 = f"{f1_score1 * 100:.2f}%" 

    # Calculate True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN) for Model 2
    cm2 = confusion_matrix(y_test2, y_predict2)
    TP2, TN2, FP2, FN2 = calculate_metrics(cm2)

    # Calculate Precision, Recall, and F1-score for Model 2
    accuracy2, precision2, recall2, f1_score2 = calculate_precision_recall_f1(TP2, FP2, FN2, TN2)

    # Tampilkan hasil untuk Model 2
    TP_2 = f"{TP2}"
    TN_2 = f"{TN2}"
    FP_2 = f"{FP2}"
    FN_2 = f"{FN2}"
    akurasi_2 = f"{accuracy2 * 100:.2f}%"
    precision_2 = f"{precision2 * 100:.2f}%"
    recall_2 = f"{recall2 * 100:.2f}%"
    f_2 = f"{f1_score2 * 100:.2f}%"

    # Simpan hasil visualisasi ke dalam file gambar (heatmap)
    heatmap1_path = 'heatmap1.png'
    heatmap2_path = 'heatmap2.png'

    plt.figure(figsize=(10, 6))
    sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', xticklabels=label1.keys(), yticklabels=label1.keys())
    plt.title("Confusion Matrix - Model 1")
    plt.xlabel('Predicted Label')
    plt.ylabel('Test Label')
    plt.savefig(heatmap1_path)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues', xticklabels=label2.keys(), yticklabels=label2.keys())
    plt.title("Confusion Matrix - Model 2")
    plt.xlabel('Predicted Label')
    plt.ylabel('Test Label')
    plt.savefig(heatmap2_path)
    plt.close()

    heatmap1_data = get_base64_encoded_image(heatmap1_path)
    heatmap2_data = get_base64_encoded_image(heatmap2_path)

    models1, models2 = load_model()

    # Assuming you have trained models model1 and model2
    save_model('model1', models1)
    save_model('model2', models2)

    return jsonify({
        'success': True,
        'message': 'Training selesai dan Model berhasil disimpan',
        'TP_1': TP_1,
        'TN_1': TN_1,
        'FP_1': FP_1,
        'FN_1': FN_1,
        'akurasi_1': akurasi_1,
        'precision_1': precision_1,
        'recall_1': recall_1,
        'f_1': f_1,
        'TP_2': TP_2,
        'TN_2': TN_2,
        'FP_2': FP_2,
        'FN_2': FN_2,
        'akurasi_2': akurasi_2,
        'precision_2': precision_2,
        'recall_2': recall_2,
        'f_2': f_2,
        'heatmap1_data': heatmap1_data,
        'heatmap2_data': heatmap2_data
    })


def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

    
@blueprint.route('/test')
@login_required
def test():
    return render_template('admin/home/test.html')

@blueprint.route('/video')
def video():
    update_label_dictionaries_and_models()
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@blueprint.route('/tutup')
def tutup():

    # Tutup semua jendela OpenCV
    cv2.destroyAllWindows()
    return"tutup windows"

@blueprint.route('/<template>')
@login_required
def route_template(template):

    try:

        if not template.endswith('.html'):
            template += '.html'

        # Detect the current page
        segment = get_segment(request)

        # Serve the file (if exists) from app/templates/home/FILE.html
        return render_template('/home/' + template, segment=segment)

    except TemplateNotFound:
        return render_template('/home/page-404.html'), 404

    except:
        return render_template('/home/page-500.html'), 500
    

# Helper - Extract current page name from request
def get_segment(request):

    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment

    except:
        return None
    



































































































































#optional
    

# @blueprint.route('/edit_data_or_label', methods=['POST'])
# def edit_data_or_label():
#     data = request.get_json()
#     table_name = data.get('table_name')
#     huruf = data.get('huruf')
#     new_value = data.get('new_value')
#     edit_option = data.get('edit_option')
    
#     # Proses pengecekan keberadaan huruf atau label sesuai opsi edit yang dipilih
#     if edit_option == 'edit_data' and is_huruf_exists_in_tables(new_value):
#         response = {"message": f"Data dengan huruf '{new_value}' sudah ada di database. Edit gagal."}
#         return jsonify(response), 400
#     elif edit_option == 'edit_label' and is_label_exists(table_name, new_value):
#         response = {"message": f"Data dengan label '{new_value}' sudah ada di database. Edit gagal."}
#         return jsonify(response), 400
    
#     # Proses permintaan sesuai pilihan edit
#     if edit_option == 'edit_data':
#         edit_data_huruf(table_name, huruf, new_value)
#         response = {"message": f"Data dengan huruf '{huruf}' telah diubah menjadi '{new_value}'"}
#     elif edit_option == 'edit_label':
#         edit_label_huruf(table_name, huruf, new_value)
#         response = {"message": f"Label data dengan huruf '{huruf}' telah diubah menjadi '{new_value}'"}
#     else:
#         response = {"message": "Pilihan edit tidak valid."}
#         return jsonify(response), 400
    
#     return jsonify(response), 200


# def edit_data_huruf(table_name, huruf, new_value):

#     connection = get_db_connection()
#     cursor = connection.cursor()

#     query = f"UPDATE {table_name} SET huruf = %s WHERE huruf = %s"
#     cursor.execute(query, (new_value, huruf))
#     connection.commit()

#     connection.close()

#     response = {"message": f"Data dengan huruf '{huruf}' telah diubah menjadi '{new_value}'"}
#     return jsonify(response), 200

# def edit_label_huruf(table_name, huruf, new_value):

#     connection = get_db_connection()
#     cursor = connection.cursor()

#     query = f"UPDATE {table_name} SET class_label = %s WHERE huruf = %s"
#     cursor.execute(query, (new_value, huruf))
#     connection.commit()

#     connection.close()

#     response = {"message": f"Label data dengan huruf '{huruf}' telah diubah menjadi '{new_value}'"}
#     return jsonify(response), 200


@blueprint.route('/edit_kamus', methods=['GET', 'POST'])
@login_required
def edit_kamus():
    selected_letter = None
    image_data = None
    video_data = None
    image_data_base64 = None  
    video_data_base64 = None
    swal_success_message = None

    if request.method == 'POST':
        selected_letter = request.form.get('selected_letter')

        alphabet_set = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

        if selected_letter in alphabet_set:
            uploaded_image = request.files.get('uploaded_image')
            uploaded_video = request.files.get('uploaded_video')

            if uploaded_image is not None:
                image_data = uploaded_image.read()
                image_data_base64 = b64encode(image_data).decode('utf-8')

            if uploaded_video is not None:
                video_data = uploaded_video.read()
                video_data_base64 = b64encode(video_data).decode('utf-8')

            save_button = request.form.get('save_button')

            if save_button:
                if is_image_exists(selected_letter) and is_video_exists(selected_letter):
                    # Update data in the database

                    if image_data is not None:
                        update_data_in_db(selected_letter, image_data, video_data)

                    if video_data is not None:
                        update_data_in_db(selected_letter, image_data, video_data)

                    # Show SweetAlert success message
                    swal_success_message = f"Data for letter {selected_letter} has been updated."

                else:
                    # Save data to the database
                    save_data_to_db(selected_letter, image_data, video_data)

                    # Show SweetAlert success message
                    swal_success_message = f"Data for letter {selected_letter} has been saved."

    return render_template('admin/home/edit.html', selected_letter=selected_letter, image_data_base64=image_data_base64, video_data_base64=video_data_base64, swal_success_message=swal_success_message)

# @blueprint.route("/delete_letter", methods=['DELETE'])
# def delete_letter():
#     letter = request.args.get('letter')

#     try:
#         delete_data_from_db(letter)
#         return jsonify({"message": f"Data for letter {letter} deleted successfully"}), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
    
# @blueprint.route('/get_data', methods=['GET'])
# def get_data():
#     letter = request.args.get('letter')
#     data_type = request.args.get('data_type')

#     try:
#         result = get_data_from_db(letter, data_type)
#         return jsonify(result)
#     except ValueError as e:
#         return jsonify({'error': str(e)}), 400


# @blueprint.route('/upload', methods=['POST'])
# def upload_photo():
#     data = request.form.to_dict()
#     uploaded_files = request.files.getlist('uploaded_files')
#     selected_table = data['table_name']
#     # huruf = data['huruf']
#     new_value = data['new_value']
#     connection = get_db_connection()
#     cursor = connection.cursor()
#     query = f"SELECT DISTINCT huruf FROM {selected_table}"
#     cursor.execute(query)
#     existing_huruf_choices = [row[0] for row in cursor.fetchall()]
#     connection.close()
#     if selected_table == 'data_1_tangan' and huruf_exists_in_data_tangan(new_value):
#         return jsonify({'error': f'Data sudah ada pada database'}), 400
#     elif selected_table == 'data_2_tangan' and huruf_exists_in_data_tangan(new_value):
#         return jsonify({'error': f'Data sudah ada pada database'}), 400
    
#     if new_value:
#     # if huruf == "Tambah huruf baru":
#     #     huruf = new_value
#         # if huruf:
#         for uploaded_file in uploaded_files:
#             try:
#                 image_data = uploaded_file.read()
#                 image_data_resized = resize_and_save_image(image_data)
#                 if image_data_resized is not None:
#                     save_photo_to_db(image_data_resized, new_value, selected_table)
#                 else:
#                     return jsonify({'error': 'Error occurred during image resizing. Please upload a valid image.'}), 400
#             except Exception as e:
#                 return jsonify({'error': f'Error processing the uploaded image: {e}'}), 500
#     return jsonify({'message': 'Gambar berhasil di tambahkan'}), 200
    # elif huruf in existing_huruf_choices:
    #     for uploaded_file in uploaded_files:
    #         try:
    #             image_data = uploaded_file.read()
    #             image_data_resized = resize_and_save_image(image_data)
    #             if image_data_resized is not None:
    #                 save_photo_to_db(image_data_resized, huruf, selected_table)
    #             else:
    #                 return jsonify({'error': 'Error occurred during image resizing. Please upload a valid image.'}), 400
    #         except Exception as e:
    #             return jsonify({'error': f'Error processing the uploaded image: {e}'}), 500
    #     return jsonify({'message': 'Photos uploaded successfully'}), 200
    # else:
    #     return jsonify({'error': 'Invalid input or no files uploaded'}), 400

# @blueprint.route('/download_csv1')
# def download_csv1():
#     file_path = process_hand_landmarks_data1()
#     if file_path:
#         return send_file(file_path, as_attachment=True)
#     else:
#         return "No data to download."

# @blueprint.route('/download_csv2')
# def download_csv2():
#     file_path = process_hand_landmarks_data2()
#     if file_path:
#         return send_file(file_path, as_attachment=True)
#     else:
#         return "No data to download."
    

# @blueprint.route('/csv_upload', methods=['POST'])
# def csv_upload():
#     UPLOAD_FOLDER = os.path.join("apps", "static", "csv")
#     if not os.path.exists(UPLOAD_FOLDER):
#         os.makedirs(UPLOAD_FOLDER)
#     table_data1 = None
#     table_data2 = None
#     if 'csv_file1' in request.files:
#         file = request.files['csv_file1']
#         if file.filename != '':
#             filename = os.path.join(UPLOAD_FOLDER, "onehand.csv")
#             file.save(filename)
            
#             df = pd.read_csv(filename)
#             limited_df = pd.concat([df.head(20), df.tail(20)])
#             table_data1 = limited_df.to_html(classes="table", index=False)

#     if 'csv_file2' in request.files:
#         file = request.files['csv_file2']
#         if file.filename != '':
#             filename = os.path.join(UPLOAD_FOLDER, "twohand.csv")
#             file.save(filename)
            
#             df = pd.read_csv(filename)
#             limited_df = pd.concat([df.head(20), df.tail(20)])
#             table_data2 = limited_df.to_html(classes="table", index=False)
    
#     if table_data1 and table_data2:
#         response = {'table_data1': table_data1, 'table_data2': table_data2}
#     elif table_data1:
#         response = {'table_data1': table_data1}
#     elif table_data2:
#         response = {'table_data2': table_data2}
#     else:
#         response = {}
    
#     return jsonify(response)

# @blueprint.route('/x')
# def x():

#     # Tutup semua jendela OpenCV
#     cv2.destroyAllWindows()
#     return"tutup windows"

# @blueprint.route('/auto')
# @login_required
# def auto():
#     return render_template('admin/home/auto.html')


# @blueprint.route("/start_recording", methods=["GET", "POST"])
# def start_recording():
#     error_message = None
#     if request.method == "POST":
#         dataset_size = request.form.get("dataset_size")
#         table_choice = int(request.form.get("table_choice"))

#         hand_gesture = request.form.get("hand_gesture").upper()

#         if table_choice == 1 and hand_gesture_exists_in_data_2_tangan(hand_gesture):
#             error_message = f'Huruf {hand_gesture} sudah ada pada tabel data 2 tangan'
#             return render_template("admin/home/auto.html", error_message=error_message)
        
#         if table_choice == 2 and hand_gesture_exists_in_data_1_tangan(hand_gesture):
#             error_message = f'Huruf {hand_gesture} sudah ada pada tabel data 1 tangan'
#             return render_template("admin/home/auto.html", error_message=error_message)

#         # Call get_label_dict() with the required arguments
#         label_dict = get_label_dict(table_choice, hand_gesture)
        
#         return render_template("admin/home/auto.html", label_dict=label_dict, table_choice=table_choice, hand_gesture=hand_gesture, dataset_size=dataset_size)

#     return render_template("admin/home/auto.html")

# def frames(table_choice, hand_gesture):

#     cap = cv2.VideoCapture(0)
#     label = get_label_dict(table_choice, hand_gesture)
#     for class_label in label.keys():

#         print('Collecting data for class {}'.format(label[class_label]))

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             cv2.putText(frame, 'Siap merekam untuk huruf {}'.format(label[class_label]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)

#             _, buffer = cv2.imencode('.jpg', frame)
#             frame_data = buffer.tobytes()

#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')   

#         cap.release()      
            
# def gen(table_choice, hand_gesture, dataset_size):
#     image_size = (256, 256)

#     cap = cv2.VideoCapture(0)
#     label = get_label_dict(table_choice, hand_gesture)
#     if table_choice == 1:
#         table_name = "data_1_tangan"
#     elif table_choice == 2:
#         table_name = "data_2_tangan"
#     else:
#         raise ValueError("Invalid table_choice value")
#     for class_label in label.keys():
#         # class_label = table_choice
#         counter = 0
#         while counter < dataset_size:
#             ret, frame = cap.read()
#             resized_frame = cv2.resize(frame, image_size)
            
#             _, buffer = cv2.imencode('.jpg', resized_frame)
            
#             image_data = buffer.tobytes()
            
#             save_image_to_db(table_name, class_label, image_data, label[class_label])  # Implement this function
            
#             counter += 1
#             yield (b'--frame\r\n'
#                 b'Content-Type: image/jpeg\r\n\r\n' + image_data + b'\r\n')
#             if counter >= dataset_size:
#                 yield (b'--frame\r\n'
#                 b'Content-Type: text/javascript\r\n\r\n'
#                 b'window.location.reload(true);'  # Refresh the page
#                 b'\r\n\r\n')
        
# @blueprint.route('/feed/<int:table_choice>/<string:hand_gesture>/<int:dataset_size>')
# def feed(table_choice, hand_gesture, dataset_size):
#     return Response(gen(table_choice, hand_gesture, dataset_size),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')
# @blueprint.route('/kamera/<int:table_choice>/<string:hand_gesture>')
# def kamera(table_choice, hand_gesture):
#     return Response(frames(table_choice, hand_gesture),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')


 
def load_model():
    x, y = load_data_from_csv('apps/static/csvv/onehand.csv')
    x2, y2 = load_data_from_csv('apps/static/csvv/twohand.csv')

    # Split data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, stratify=y)
    x_train2, x_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size=0.2, shuffle=True, stratify=y2)

    # Train models
    models1 = RandomForestClassifier()
    models2 = RandomForestClassifier()

    models1.fit(x_train, y_train)
    models2.fit(x_train2, y_train2)

    y_predict = models1.predict(x_test)
    y_predict2 = models2.predict(x_test2)

    return models1, models2