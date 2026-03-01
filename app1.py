from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from werkzeug.utils import secure_filename
import os
import pickle
import nltk
import pdfkit
import pymysql
import chardet
import fitz
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from objective import ObjectiveTest
from subjective import SubjectiveTest

# Install PyMySQL as MySQLdb replacement
pymysql.install_as_MySQLdb()

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "supersecretkey")

# ================= DATABASE CONFIG =================
def get_db_connection():
    return pymysql.connect(
        host=os.environ.get("MYSQL_HOST"),
        user=os.environ.get("MYSQL_USER"),
        password=os.environ.get("MYSQL_PASSWORD"),
        database=os.environ.get("MYSQL_DB"),
        cursorclass=pymysql.cursors.DictCursor
    )

# ================= FOLDER CONFIG =================
UPLOAD_FOLDER = os.path.join("static", "uploads")
PROFILE_FOLDER = os.path.join("static", "profile_images")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROFILE_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

REFERENCE_FILE = "reference_answers.txt"

# ================= MAIL CONFIG =================
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get("MAIL_USERNAME")
app.config['MAIL_PASSWORD'] = os.environ.get("MAIL_PASSWORD")
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get("MAIL_USERNAME")

mail = Mail(app)
serializer = URLSafeTimedSerializer(app.secret_key)

# ================= ROUTES =================

@app.route('/')
def home():
    return render_template('home.html')

# ---------------- SIGNUP ----------------
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    message = ''
    if request.method == 'POST':
        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute('SELECT * FROM users WHERE username=%s', (request.form['username'],))
        existing_user = cursor.fetchone()

        if existing_user:
            message = "Username already exists!"
        else:
            cursor.execute(
                'INSERT INTO users (first_name,last_name,username,email,password) VALUES (%s,%s,%s,%s,%s)',
                (
                    request.form['first_name'],
                    request.form['last_name'],
                    request.form['username'],
                    request.form['email'],
                    request.form['password']
                )
            )
            connection.commit()
            return redirect(url_for('login'))

        cursor.close()
        connection.close()

    return render_template('signup.html', message=message)

# ---------------- LOGIN ----------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    message = ''
    if request.method == 'POST':
        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute(
            'SELECT * FROM users WHERE username=%s AND password=%s',
            (request.form['username'], request.form['password'])
        )
        user = cursor.fetchone()

        cursor.close()
        connection.close()

        if user:
            session['loggedin'] = True
            session['user_id'] = user['id']
            session['first_name'] = user['first_name']
            return redirect(url_for('user_dashboard'))
        else:
            message = "Invalid username or password!"

    return render_template('login.html', message=message)

# ---------------- DASHBOARD ----------------
@app.route('/user_dashboard')
def user_dashboard():
    if 'loggedin' in session:
        return render_template('user_dashboard.html',
                               first_name=session['first_name'])
    return redirect(url_for('login'))

# ---------------- FILE UPLOAD ----------------
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'loggedin' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            session['uploaded_file'] = filepath
            return redirect(url_for('train_model'))

    return render_template('upload.html')

# ---------------- TRAIN MODEL ----------------
@app.route('/train_model')
def train_model():
    if 'uploaded_file' not in session:
        return redirect(url_for('upload'))

    model = pickle.load(open('AI_model.pkl', 'rb'))
    return render_template('train_model.html')

# ---------------- QUESTION GENERATION ----------------
@app.route('/question_generation', methods=['GET', 'POST'])
def question_generation():
    if request.method == "POST":
        inputText = request.form.get("itext")
        testType = request.form.get("test_type")
        noOfQues = request.form.get("noq")

        if testType == "objective":
            generator = ObjectiveTest(inputText, noOfQues)
        else:
            generator = SubjectiveTest(inputText, noOfQues)

        question_list, answer_list = generator.generate_test()
        return render_template("generatedtestdata.html",
                               cresults=zip(question_list, answer_list))

    return render_template("index.html")

# ---------------- ANSWER EVALUATION ----------------
@app.route("/eval_upload", methods=["GET", "POST"])
def eval_upload():
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return render_template("eval_ans_upload.html", error="No file selected.")

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        student_answer = extract_text(file_path)
        reference_answer = load_reference_answer()
        score = calculate_similarity(student_answer, reference_answer)

        return render_template("result.html",
                               student_answer=student_answer,
                               reference_answer=reference_answer,
                               score=score)

    return render_template("eval_ans_upload.html")

# ================= UTILITIES =================

def load_reference_answer():
    if os.path.exists(REFERENCE_FILE):
        with open(REFERENCE_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    return "Reference file not found"

def calculate_similarity(a, b):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([a, b])
    score = cosine_similarity(vectors[0], vectors[1])[0][0]
    return round(score * 10, 2)

def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
        return text.strip()

    with open(file_path, 'rb') as f:
        raw = f.read()
        encoding = chardet.detect(raw)['encoding'] or 'utf-8'
        return raw.decode(encoding, errors='replace')

# ================= MAIN =================
if __name__ == '__main__':
    app.run()