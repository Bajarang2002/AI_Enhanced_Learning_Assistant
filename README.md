# 🧠 AI-Enhanced Learning Assistant Application

An advanced AI-powered learning and assessment platform built with **Flask**, **PostgreSQL**, **PyTorch (RNN, LSTM & BiLSTM)**, and **JWT (JSON Web Token) Security**. The application empowers users to register/login securely, upload large educational datasets to train deep learning models, inspect live model convergence and loss curves, synthesize semantically coherent objective MCQs and comprehensive subjective questions with detailed answers, and evaluate student answer sheets against reference answer keys with automated scoring and concept diagnostic feedback.

---

## 🚀 Key Features

1. 🔐 **JWT-Based Authentication System (`jwt_auth.py`)**
   - **RFC 7519 Compliant JSON Web Token (JWT)** engine using cryptographic HMAC-SHA256 (HS256) signatures.
   - Automatically issues secure JWT tokens on both Sign Up and Login.
   - Dual-layer security: Stores JWT tokens in HTTP-only cookies and Authorization headers.
   - User profile management, password encryption, and reset mechanisms backed by PostgreSQL `users` table.

2. 🧠 **RNN, LSTM & BiLSTM Deep Learning Model Training**
   - Upload large training datasets (`.txt`, `.pdf`, `.csv`).
   - Select between **LSTM** (Long Short-Term Memory), **RNN** (Recurrent Neural Network), or **BiLSTM** (Bidirectional LSTM).
   - Completely separate training pipeline: [`train_model_pipeline.py`](file:///d:/Bajarang/AI_Enhanced_Learning_Assistant/AI_Enhanced_Learning_Assistant/train_model_pipeline.py).
   - Real-time training loop with epoch loss, training accuracy, validation loss, and validation accuracy.
   - Automatically logs model parameters and performance history to PostgreSQL (`model_metrics` table).

3. ❓ **Semantic Question & Detailed Answer Generation**
   - Understands true **semantic propositions** (definitions, capabilities, mechanisms, problem-solving, and contrasts) instead of naive random-word masking.
   - **Objective MCQs**: Generates well-formed question stems, 4 plausible domain-matching options (A, B, C, D) belonging to the same semantic category, the correct answer, and an explanatory rationale.
   - **Subjective Questions**: In-depth analytical questions with **detailed, multi-paragraph solutions** (Core Definition & Principle, Operational Mechanism, and Significance).

4. 📊 **Model Accuracy & Diagnostic Performance Page**
   - Live KPI cards: Current Model Type, Accuracy Percentage (e.g. 97.8%), Validation Accuracy (82.3%), and Final Loss.
   - Interactive charts (Chart.js) rendering Training & Validation Accuracy and Loss curves across epochs.
   - Hyperparameter and dataset specification table.
   - PostgreSQL historical training runs log.

5. ⚖️ **Dual Answer Sheet Evaluation System**
   - Upload both:
     1. **Correct / Reference Answer Sheet** (`.pdf`, `.txt`, or text input)
     2. **Student Answer Sheet** (`.pdf`, `.txt`, or text input)
   - Evaluates:
     - TF-IDF Cosine Semantic Similarity (%).
     - Domain Key Concept Coverage (%).
     - Answer Completeness & Length Ratio.
   - Generates:
     - Final score out of 10 and 100 with letter grade.
     - Matched core concepts vs Missing concepts from answer key.
     - Actionable recommendations for student improvement.
     - Side-by-side comparative inspection.
     - Evaluation history recorded in PostgreSQL (`evaluation_history` table).

6. ⚙️ **Settings & Customization Suite (`settings.html`)**
   - **Reset / Change Password**: Direct in-dashboard password update with instant verification.
   - **Theme & Appearance Switcher**: 4 sleek palettes:
     - 🍷 *Academic Crimson* (Classic University Maroon & Gold)
     - 🌟 *Modern Indigo SaaS* (Executive Slate-900 & Indigo)
     - 🌊 *Deep Oceanic Navy* (Marine Blue & Cyan)
     - ☀️ *Clean Minimalist Light* (Soft Gray & Emerald)
   - **General App Preferences**: Set default neural model, default question count, and evaluation strictness.

7. 🎨 **Professional, Modern UI Overhaul**
   - Sleek glassmorphism navigation, modern card elevations, responsive sidebar with active indicators, and high-impact SaaS landing page.

---

## 🧰 Tech Stack

- **Backend**: Flask 3.0, Python 3.11
- **Security / Auth**: JSON Web Tokens (JWT) HS256 HMAC-SHA256 (`jwt_auth.py`), itsdangerous
- **Database**: PostgreSQL (`psycopg2`) with auto-reconnection and SQLite fallback
- **Deep Learning / NLP**: PyTorch (RNN, LSTM, BiLSTM), scikit-learn (TF-IDF, Cosine Similarity), NLTK
- **Frontend**: HTML5, Bootstrap 5, FontAwesome 6, Chart.js, Tailwind CSS
- **Document Processing**: PyMuPDF (`fitz`), chardet

---

## 📂 Project Structure

```
AI_Enhanced_Learning_Assistant/
├── app.py                     # Main Flask application & routes (JWT secured)
├── jwt_auth.py                # Pure-Python RFC 7519 JWT authentication engine
├── db.py                      # PostgreSQL database manager & CRUD operations
├── schema.sql                 # PostgreSQL table definitions
├── setup_postgres.py          # Standalone PostgreSQL setup & test script
├── train_model_pipeline.py    # Separate RNN/LSTM data loading, training & semantic Q&A
├── evaluator.py               # Dual answer sheet comparison & scoring engine
├── models/                    # Saved PyTorch model checkpoints & metadata
│   ├── trained_model.pt
│   └── model_metadata.json
├── templates/                 # Clean, intuitive Jinja2 HTML templates
│   ├── home.html              # Simple, readable classic landing page
│   ├── login.html             # Clean, intuitive login with password toggle
│   ├── signup.html            # Simple user registration with instant JWT issuance
│   ├── user_dashboard.html    # Clean, focused dashboard with left sidebar navigation
│   ├── settings.html          # Password reset, model & app preferences
│   ├── upload_train.html      # Model training page
│   ├── model_accuracy.html    # Model accuracy & Chart.js loss curves
│   ├── eval_ans_upload.html   # Dual answer sheet evaluation upload
│   ├── result.html            # Evaluation scorecard & diagnostic report
│   ├── generatedtestdata.html # Generated Q&A test report
│   └── partials/              # Component partials
└── static/                    # CSS styles, images, and user uploads
```

---

## ⚙️ Setup & Execution Instructions

### 1. PostgreSQL Database Configuration (`.env`)

All database connection and security parameters are isolated in the **`.env`** file for maximum security (not hardcoded in the main files):

```env
# .env file
PG_HOST=localhost
PG_PORT=5432
PG_USER=postgres
PG_PASSWORD=root
PG_DATABASE=learning_assistant_db

SECRET_KEY=learning_assistant_flask_session_key_2026_secured
JWT_SECRET_KEY=ai_learning_assistant_jwt_super_secret_2026_key_#889!
```

To automatically create the database and verify table initialization, run:
```bash
python setup_postgres.py
```

### 2. Standalone Model Training (Optional CLI)

You can train the RNN/LSTM model directly from the command line:
```bash
python train_model_pipeline.py --file New_Ref.txt --model LSTM --epochs 5 --batch_size 64
```
Available models: `LSTM`, `RNN`, `BiLSTM`.

### 3. Run the Application

Start the Flask server:
```bash
python app.py
```

Open your browser at:
```
http://localhost:5000
```
