-- PostgreSQL Schema for AI-Enhanced Learning Assistant

-- 1. Users Table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(150) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    profile_image VARCHAR(255) DEFAULT 'profile_images/default.png',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Model Metrics & Training History Table
CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    model_type VARCHAR(50) NOT NULL,
    dataset_name VARCHAR(255),
    dataset_size_kb NUMERIC(10, 2),
    vocab_size INTEGER,
    epochs INTEGER,
    final_loss NUMERIC(8, 4),
    final_accuracy NUMERIC(6, 2),
    val_loss NUMERIC(8, 4),
    val_accuracy NUMERIC(6, 2),
    history_json TEXT,
    model_filepath VARCHAR(255),
    trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Answer Evaluation History Table
CREATE TABLE IF NOT EXISTS evaluation_history (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    ref_filename VARCHAR(255),
    student_filename VARCHAR(255),
    score NUMERIC(5, 2),
    similarity_pct NUMERIC(5, 2),
    keyword_coverage_pct NUMERIC(5, 2),
    completeness_score NUMERIC(5, 2),
    feedback TEXT,
    evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_model_metrics_user ON model_metrics(user_id);
CREATE INDEX IF NOT EXISTS idx_eval_history_user ON evaluation_history(user_id);
