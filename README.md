# 🧠 AI Enhanced Learning Assistant Application

This project is a Flask-based AI-powered learning assistant developed as a **final year project**. It allows users to register/login, upload content, generate test questions (objective/subjective), evaluate student answers against a reference, and manage user profiles with password reset capabilities.

---

## 🚀 Features

- 🔐 **User Authentication**: Signup, Login, Logout, and Forgot Password functionality.
- 🧾 **Profile Management**: Update personal details and profile image.
- 📄 **File Upload**: Upload learning material (PDF/TXT).
- 🧠 **Model Training**: Load pre-trained AI model (`AI_model.pkl`) for educational use cases.
- ❓ **Question Generation**: Generate Objective and Subjective questions using NLP.
- 📊 **Answer Evaluation**: Evaluate student answers by comparing with a reference answer using cosine similarity.
- 📤 **PDF Report Download**: Download evaluation results as a PDF.
- 📧 **Email Support**: Password reset via email (configured with Gmail SMTP).

---

## 🧰 Tech Stack

- **Backend**: Flask, MySQL
- **Frontend**: HTML, CSS (Jinja templates)
- **AI/ML**: scikit-learn, TF-IDF, Cosine Similarity
- **Others**: NLTK, Flask-Mail, Pickle, PDFKit, PyMuPDF (`fitz`), chardet

---

## 📂 Project Structure

```
.
├── app.py                     # Main Flask application
├── AI_model.pkl               # Pickled ML model
├── templates/                 # HTML templates (signup, login, dashboard, etc.)
├── static/                    # CSS, JS, images, uploads
├── reference_answers.txt      # Reference data for evaluation
├── objective.py               # Module for generating objective questions
├── subjective.py              # Module for generating subjective questions
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-repo/ai-learning-assistant.git
cd ai-learning-assistant
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

#### Key Libraries

- Flask
- flask-mysqldb
- scikit-learn
- nltk
- chardet
- flask-mail
- PyMuPDF
- pdfkit

> Make sure `wkhtmltopdf` is installed and its path is configured in `app.py`.

### 3. Configure MySQL

Create a database `mydatabase` and a `users` table:

```sql
CREATE DATABASE mydatabase;
USE mydatabase;

CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    username VARCHAR(100) UNIQUE,
    email VARCHAR(100),
    password VARCHAR(100),
    profile_image VARCHAR(255)
);
```

### 4. Run the App

```bash
python app.py
```

Visit: `http://localhost:5000`

---

## 🧪 Model Training

- The model used is loaded from `AI_model.pkl`.
- It is invoked during the `/train_model` route after uploading content.
- Ensure your model is compatible with the expected format.

---

## 📬 Email Configuration

The app uses **Gmail SMTP** to send reset links.

```python
app.config['MAIL_USERNAME'] = 'your_email@gmail.com'
app.config['MAIL_PASSWORD'] = 'your_app_password'  
```



