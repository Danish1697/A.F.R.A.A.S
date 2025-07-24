from flask import Flask, render_template, request, redirect, send_file
import os
import pandas as pd
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'ImagesAttendance'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return 'No file part'
    file = request.files['image']
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect('/')

@app.route('/attendance')
def attendance():
    if os.path.exists('Attendance.csv'):
        df = pd.read_csv('Attendance.csv')
        return render_template('attendance.html', tables=[df.to_html(classes='data')], titles=df.columns.values)
    else:
        return "No attendance file found!"

@app.route('/download')
def download():
    return send_file('Attendance.csv', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
