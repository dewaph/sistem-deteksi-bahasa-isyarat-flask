from apps.admin import blueprint
from flask import render_template, request, jsonify, Response
from flask_login import login_required
from jinja2 import TemplateNotFound
from .dashboardcontrol import count_data_in_table, count_unique_labels, is_label_exists, is_huruf_exists_in_tables, get_db_connection
import pandas as pd
from .kamuscontrol import is_image_exists, is_video_exists, get_data_from_db, update_data_in_db, save_data_to_db, delete_data_from_db
import os
from base64 import b64encode
from .datacontrol import get_db_connection, resize_and_save_image, save_photo_to_db, huruf_exists_in_data_1_tangan, huruf_exists_in_data_2_tangan
from .preprocesscontrol import get_db_connection, process_images, process_hand_landmarks_data1, process_hand_landmarks_data2, update_label_dictionaries
import cv2
import base64
from .trainingcontrol import get_db_connection, save_model_to_database, calculate_metrics, calculate_precision_recall_f1, load_data_from_csv
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
    data1tangan_count = count_data_in_table('data_1_tangan')
    data2tangan_count = count_data_in_table('data_2_tangan')
    total_data_count = data1tangan_count + data2tangan_count

    unique_labels_data1 = count_unique_labels('data_1_tangan')
    unique_labels_data2 = count_unique_labels('data_2_tangan')
    total_unique_labels = len(set(unique_labels_data1 + unique_labels_data2))

    data1_df = pd.DataFrame(unique_labels_data1, columns=['label','huruf', 'jumlah'])
    data2_df = pd.DataFrame(unique_labels_data2, columns=['label','huruf', 'jumlah'])

    return render_template('admin/home/index.html', segment='index', total_data_count=total_data_count,
                           data1tangan_count=data1tangan_count,
                           data2tangan_count=data2tangan_count,
                           total_unique_labels=total_unique_labels,
                           data1_df=data1_df,
                           data2_df=data2_df)

@blueprint.route('/edit_data_or_label', methods=['POST'])
def edit_data_or_label():
    data = request.get_json()
    table_name = data.get('table_name')
    huruf = data.get('huruf')
    new_value = data.get('new_value')
    edit_option = data.get('edit_option')
    
    # Proses pengecekan keberadaan huruf atau label sesuai opsi edit yang dipilih
    if edit_option == 'edit_data' and is_huruf_exists_in_tables(new_value):
        response = {"message": f"Data dengan huruf '{new_value}' sudah ada di database. Edit gagal."}
        return jsonify(response), 400
    elif edit_option == 'edit_label' and is_label_exists(table_name, new_value):
        response = {"message": f"Data dengan label '{new_value}' sudah ada di database. Edit gagal."}
        return jsonify(response), 400
    
    # Proses permintaan sesuai pilihan edit
    if edit_option == 'edit_data':
        edit_data_huruf(table_name, huruf, new_value)
        response = {"message": f"Data dengan huruf '{huruf}' telah diubah menjadi '{new_value}'"}
    elif edit_option == 'edit_label':
        edit_label_huruf(table_name, huruf, new_value)
        response = {"message": f"Label data dengan huruf '{huruf}' telah diubah menjadi '{new_value}'"}
    else:
        response = {"message": "Pilihan edit tidak valid."}
        return jsonify(response), 400
    
    return jsonify(response), 200


def edit_data_huruf(table_name, huruf, new_value):

    connection = get_db_connection()
    cursor = connection.cursor()

    query = f"UPDATE {table_name} SET huruf = %s WHERE huruf = %s"
    cursor.execute(query, (new_value, huruf))
    connection.commit()

    connection.close()

    response = {"message": f"Data dengan huruf '{huruf}' telah diubah menjadi '{new_value}'"}
    return jsonify(response), 200

def edit_label_huruf(table_name, huruf, new_value):

    connection = get_db_connection()
    cursor = connection.cursor()

    query = f"UPDATE {table_name} SET class_label = %s WHERE huruf = %s"
    cursor.execute(query, (new_value, huruf))
    connection.commit()

    connection.close()

    response = {"message": f"Label data dengan huruf '{huruf}' telah diubah menjadi '{new_value}'"}
    return jsonify(response), 200

@blueprint.route('/delete_data', methods=['POST'])
def delete_data():
    data = request.get_json()
    table_name = data.get('table_name')
    huruf = data.get('huruf')

    connection = get_db_connection()
    cursor = connection.cursor()

    query = f"DELETE FROM {table_name} WHERE huruf = %s"
    cursor.execute(query, (huruf,))
    connection.commit()

    connection.close()

    response = {"message": f"Data dengan huruf '{huruf}' telah dihapus."}
    return jsonify(response), 200


@blueprint.route('/get_huruf_options', methods=['GET'])
def get_huruf_options():
    table_name = request.args.get('table_name')
    
    connection = get_db_connection()
    cursor = connection.cursor()

    query = f"SELECT DISTINCT huruf FROM {table_name}"

    cursor.execute(query)
    result = cursor.fetchall()

    huruf_options = [row[0] for row in result]

    connection.close()

    return jsonify(huruf_options)

@blueprint.route('/edit_kamus', methods=['GET', 'POST'])
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

@blueprint.route("/delete_letter", methods=['DELETE'])
def delete_letter():
    letter = request.args.get('letter')

    try:
        delete_data_from_db(letter)
        return jsonify({"message": f"Data for letter {letter} deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@blueprint.route('/get_data', methods=['GET'])
def get_data():
    letter = request.args.get('letter')
    data_type = request.args.get('data_type')

    try:
        result = get_data_from_db(letter, data_type)
        return jsonify(result)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    
@blueprint.route('/tambah_data')
@login_required
def tambah_data():
    return render_template('admin/home/data.html')

@blueprint.route('/upload', methods=['POST'])
def upload_photo():
    data = request.form.to_dict()
    uploaded_files = request.files.getlist('uploaded_files')
    selected_table = data['table_name']
    huruf = data['huruf']
    new_value = data['new_value']
    connection = get_db_connection()
    cursor = connection.cursor()
    query = f"SELECT DISTINCT huruf FROM {selected_table}"
    cursor.execute(query)
    existing_huruf_choices = [row[0] for row in cursor.fetchall()]
    connection.close()
    if selected_table == 'data_1_tangan' and huruf_exists_in_data_2_tangan(new_value):
        return jsonify({'error': f'Huruf {new_value} sudah ada pada tabel data 2 tangan'}), 400
    elif selected_table == 'data_2_tangan' and huruf_exists_in_data_1_tangan(new_value):
        return jsonify({'error': f'Huruf {new_value} sudah ada pada tabel data 1 tangan'}), 400
    if huruf == "Tambah huruf baru":
        huruf = new_value
        if huruf:
            for uploaded_file in uploaded_files:
                try:
                    image_data = uploaded_file.read()
                    image_data_resized = resize_and_save_image(image_data)
                    if image_data_resized is not None:
                        save_photo_to_db(image_data_resized, huruf, selected_table)
                    else:
                        return jsonify({'error': 'Error occurred during image resizing. Please upload a valid image.'}), 400
                except Exception as e:
                    return jsonify({'error': f'Error processing the uploaded image: {e}'}), 500
        return jsonify({'message': 'Gambar berhasil di tambahkan'}), 200
    elif huruf in existing_huruf_choices:
        for uploaded_file in uploaded_files:
            try:
                image_data = uploaded_file.read()
                image_data_resized = resize_and_save_image(image_data)
                if image_data_resized is not None:
                    save_photo_to_db(image_data_resized, huruf, selected_table)
                else:
                    return jsonify({'error': 'Error occurred during image resizing. Please upload a valid image.'}), 400
            except Exception as e:
                return jsonify({'error': f'Error processing the uploaded image: {e}'}), 500
        return jsonify({'message': 'Photos uploaded successfully'}), 200
    else:
        return jsonify({'error': 'Invalid input or no files uploaded'}), 400

@blueprint.route('/preprocessing')
@login_required
def preprocessing():
    return render_template('admin/home/preprocessing.html')

@blueprint.route('/preprocess', methods=['POST'])
def preprocess():
    update_label_dictionaries()
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

    dataset_1_labels = get_labels_from_db(dataset=1)
    label_dataset_1 = create_label_dictionary(dataset_1_labels)

    dataset_2_labels = get_labels_from_db(dataset=2)
    label_dataset_2 = create_label_dictionary(dataset_2_labels)
    
    dataset_1_images = process_images(dataset=1)
    dataset_2_images = process_images(dataset=2)
    
    base64_dataset_1_images = []
    for class_label, images in dataset_1_images:
        base64_images = []
        for img in images:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            _, buffer = cv2.imencode('.jpg', img_rgb)
            base64_images.append(base64.b64encode(buffer).decode('utf-8'))
        base64_dataset_1_images.append((class_label, base64_images))

    base64_dataset_2_images = []
    for class_label, images in dataset_2_images:
        base64_images = []
        for img in images:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            _, buffer = cv2.imencode('.jpg', img_rgb)
            base64_images.append(base64.b64encode(buffer).decode('utf-8'))
        base64_dataset_2_images.append((class_label, base64_images))
    show_csv_button = True  

    return render_template('admin/home/preprocessing.html', dataset_1_images=base64_dataset_1_images, dataset_2_images=base64_dataset_2_images, label_dataset_1=label_dataset_1, label_dataset_2=label_dataset_2, show_csv_button=show_csv_button)

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
        return jsonify({"success": True, "message": "CSV berhasil diexport", "file_path": file_path})
    else:
        return jsonify({"success": False, "message": "gagal membuat CSV"})

UPLOAD_FOLDER = 'uploads'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@blueprint.route('/training')
@login_required
def training():
    return render_template('admin/home/training.html')

@blueprint.route('/csv_upload', methods=['POST'])
def csv_upload():
    table_data1 = None
    table_data2 = None
    if 'csv_file1' in request.files:
        file = request.files['csv_file1']
        if file.filename != '':
            filename = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filename)
            
            df = pd.read_csv(filename)
            table_data1 = df.to_html(classes="table", index=False)

    if 'csv_file2' in request.files:
        file = request.files['csv_file2']
        if file.filename != '':
            filename = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filename)
            
            df = pd.read_csv(filename)
            table_data2 = df.to_html(classes="table", index=False)
    
    if table_data1 and table_data2:
        response = {'table_data1': table_data1, 'table_data2': table_data2}
    elif table_data1:
        response = {'table_data1': table_data1}
    elif table_data2:
        response = {'table_data2': table_data2}
    else:
        response = {}
    
    return jsonify(response)

@blueprint.route('/start_training', methods=['POST'])
def start_training():
    def get_unique_labels(table_name):
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(f"SELECT DISTINCT huruf FROM {table_name}")
        unique_labels = [row[0] for row in cursor.fetchall()]

        cursor.close()
        conn.close()

        return unique_labels

    def create_label_dict(label_list):
        return {label: index for index, label in enumerate(label_list)}

    unique_labels1 = get_unique_labels('data_1_tangan')
    unique_labels2 = get_unique_labels('data_2_tangan')

    label1 = create_label_dict(unique_labels1)
    label2 = create_label_dict(unique_labels2)
    # Load data from CSV files
    x, y = load_data_from_csv('uploads/onehand.csv')
    x2, y2 = load_data_from_csv('uploads/twohand.csv')

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

    # Save model1 to the database
    save_model_to_database('model1', model1)

    # Save model2 to the database
    save_model_to_database('model2', model2)

    return jsonify({
        'success': True,
        'message': 'Training selesai dan Model di save ke dalam Database',
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

@blueprint.route('/x')
def x():

    # Tutup semua jendela OpenCV
    cv2.destroyAllWindows()
    return"tutup windows"

@blueprint.route('/auto')
@login_required
def auto():
    return render_template('admin/home/auto.html')


@blueprint.route("/start_recording", methods=["GET", "POST"])
def start_recording():
    error_message = None
    if request.method == "POST":
        dataset_size = request.form.get("dataset_size")
        table_choice = int(request.form.get("table_choice"))

        hand_gesture = request.form.get("hand_gesture").upper()

        if table_choice == 1 and hand_gesture_exists_in_data_2_tangan(hand_gesture):
            error_message = f'Huruf {hand_gesture} sudah ada pada tabel data 2 tangan'
            return render_template("admin/home/auto.html", error_message=error_message)
        
        if table_choice == 2 and hand_gesture_exists_in_data_1_tangan(hand_gesture):
            error_message = f'Huruf {hand_gesture} sudah ada pada tabel data 1 tangan'
            return render_template("admin/home/auto.html", error_message=error_message)

        # Call get_label_dict() with the required arguments
        label_dict = get_label_dict(table_choice, hand_gesture)
        
        return render_template("admin/home/auto.html", label_dict=label_dict, table_choice=table_choice, hand_gesture=hand_gesture, dataset_size=dataset_size)

    return render_template("admin/home/auto.html")

def frames(table_choice, hand_gesture):

    cap = cv2.VideoCapture(0)
    label = get_label_dict(table_choice, hand_gesture)
    for class_label in label.keys():

        print('Collecting data for class {}'.format(label[class_label]))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.putText(frame, 'Siap merekam untuk huruf {}'.format(label[class_label]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)

            _, buffer = cv2.imencode('.jpg', frame)
            frame_data = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')   

        cap.release()      
            
def gen(table_choice, hand_gesture, dataset_size):
    image_size = (256, 256)

    cap = cv2.VideoCapture(0)
    label = get_label_dict(table_choice, hand_gesture)
    if table_choice == 1:
        table_name = "data_1_tangan"
    elif table_choice == 2:
        table_name = "data_2_tangan"
    else:
        raise ValueError("Invalid table_choice value")
    for class_label in label.keys():
        # class_label = table_choice
        counter = 0
        while counter < dataset_size:
            ret, frame = cap.read()
            resized_frame = cv2.resize(frame, image_size)
            
            _, buffer = cv2.imencode('.jpg', resized_frame)
            
            image_data = buffer.tobytes()
            
            save_image_to_db(table_name, class_label, image_data, label[class_label])  # Implement this function
            
            counter += 1
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + image_data + b'\r\n')
            if counter >= dataset_size:
                yield (b'--frame\r\n'
                b'Content-Type: text/javascript\r\n\r\n'
                b'window.location.reload(true);'  # Refresh the page
                b'\r\n\r\n')
        
@blueprint.route('/feed/<int:table_choice>/<string:hand_gesture>/<int:dataset_size>')
def feed(table_choice, hand_gesture, dataset_size):
    return Response(gen(table_choice, hand_gesture, dataset_size),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@blueprint.route('/kamera/<int:table_choice>/<string:hand_gesture>')
def kamera(table_choice, hand_gesture):
    return Response(frames(table_choice, hand_gesture),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

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
    
