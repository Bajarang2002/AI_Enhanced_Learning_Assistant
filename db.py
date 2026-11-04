import os
import json
import logging
import sqlite3
import tempfile
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

# Load database security details from .env file
_env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(_env_path):
    load_dotenv(dotenv_path=_env_path, override=True)
else:
    load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PostgreSQL Configuration (Loaded from .env environment variables)
PG_HOST = os.environ.get('PG_HOST', 'localhost')
PG_PORT = os.environ.get('PG_PORT', '5432')
PG_USER = os.environ.get('PG_USER', 'postgres')
PG_PASSWORD = os.environ.get('PG_PASSWORD', '')
PG_DATABASE = os.environ.get('PG_DATABASE', 'learning_assistant_db')

SQLITE_FALLBACK_DB = os.path.join(os.path.dirname(__file__), 'database.db')

USE_POSTGRES = True

def get_writable_sqlite_path():
    """Returns local database.db if writable; otherwise uses system temp directory."""
    try:
        test_conn = sqlite3.connect(SQLITE_FALLBACK_DB)
        test_conn.execute("CREATE TABLE IF NOT EXISTS _test_write (id INT)")
        test_conn.execute("DROP TABLE IF EXISTS _test_write")
        test_conn.close()
        return SQLITE_FALLBACK_DB
    except Exception:
        return os.path.join(tempfile.gettempdir(), 'learning_assistant_fallback.db')

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
except ImportError:
    psycopg2 = None
    RealDictCursor = None
    logger.warning("psycopg2 is not installed. Database will use SQLite fallback.")


def get_pg_admin_connection():
    """Connect to postgres administrative database to create application db if needed."""
    if not psycopg2:
        return None
    try:
        conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            user=PG_USER,
            password=PG_PASSWORD,
            database='postgres',
            connect_timeout=3
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        return conn
    except Exception as e:
        logger.warning(f"Could not connect to PostgreSQL admin DB: {e}")
        return None


def ensure_postgres_database_exists():
    """Ensure that the target PostgreSQL database exists; if not, create it."""
    if not psycopg2:
        return False
    try:
        admin_conn = get_pg_admin_connection()
        if not admin_conn:
            return False
        with admin_conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (PG_DATABASE,))
            exists = cur.fetchone()
            if not exists:
                logger.info(f"Database '{PG_DATABASE}' not found. Creating it...")
                cur.execute(f'CREATE DATABASE "{PG_DATABASE}"')
                logger.info(f"Database '{PG_DATABASE}' created successfully.")
        admin_conn.close()
        return True
    except Exception as e:
        logger.warning(f"Failed to ensure PostgreSQL database '{PG_DATABASE}': {e}")
        return False


def get_db_connection():
    """
    Get a database connection.
    Attempts PostgreSQL first. If PostgreSQL server is unreachable, falls back to SQLite
    to prevent application crash, while notifying in logs.
    """
    global USE_POSTGRES

    if USE_POSTGRES and psycopg2:
        try:
            ensure_postgres_database_exists()
            conn = psycopg2.connect(
                host=PG_HOST,
                port=PG_PORT,
                user=PG_USER,
                password=PG_PASSWORD,
                database=PG_DATABASE,
                connect_timeout=3
            )
            return conn, 'postgres'
        except Exception as e:
            logger.warning(f"PostgreSQL connection failed ({e}). Falling back to local SQLite: {SQLITE_FALLBACK_DB}")
            # Do not permanently disable USE_POSTGRES so next call can retry
    
    # SQLite Fallback Connection
    db_path = get_writable_sqlite_path()
    sqlite_conn = sqlite3.connect(db_path)
    sqlite_conn.row_factory = sqlite3.Row
    return sqlite_conn, 'sqlite'


def init_db():
    """Initialize all required tables in PostgreSQL (and fallback SQLite)."""
    conn, db_type = get_db_connection()
    try:
        cur = conn.cursor()
        if db_type == 'postgres':
            cur.execute("""
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
            """)
            conn.commit()
            logger.info("PostgreSQL tables verified/created successfully.")
        else:
            # SQLite compatible syntax
            cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                first_name TEXT,
                last_name TEXT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                profile_image TEXT DEFAULT 'profile_images/default.png',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            """)
            cur.execute("""
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                model_type TEXT NOT NULL,
                dataset_name TEXT,
                dataset_size_kb REAL,
                vocab_size INTEGER,
                epochs INTEGER,
                final_loss REAL,
                final_accuracy REAL,
                val_loss REAL,
                val_accuracy REAL,
                history_json TEXT,
                model_filepath TEXT,
                trained_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            """)
            cur.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                ref_filename TEXT,
                student_filename TEXT,
                score REAL,
                similarity_pct REAL,
                keyword_coverage_pct REAL,
                completeness_score REAL,
                feedback TEXT,
                evaluated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            """)
            conn.commit()
            logger.info("SQLite fallback tables verified/created successfully.")
        cur.close()
    except Exception as e:
        logger.error(f"Error during init_db: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()


def query_db(query, params=(), one=False, commit=False):
    """
    Execute a parameterized SQL query and return dict results.
    Works seamlessly across PostgreSQL and SQLite.
    """
    conn, db_type = get_db_connection()
    try:
        if db_type == 'postgres':
            cur = conn.cursor(cursor_factory=RealDictCursor)
        else:
            cur = conn.cursor()
            # Replace PostgreSQL %s with ? for SQLite if needed
            query = query.replace('%s', '?')

        cur.execute(query, params)

        if commit:
            conn.commit()
            last_id = None
            if db_type == 'postgres':
                try:
                    last_id = cur.fetchone()
                except Exception:
                    pass
            else:
                last_id = cur.lastrowid
            cur.close()
            return last_id

        if one:
            result = cur.fetchone()
            if result and db_type == 'sqlite':
                result = dict(result)
        else:
            result = cur.fetchall()
            if result and db_type == 'sqlite':
                result = [dict(r) for r in result]

        cur.close()
        return result
    except Exception as e:
        logger.error(f"Database query error: {e} | Query: {query}")
        if conn:
            conn.rollback()
        raise e
    finally:
        if conn:
            conn.close()


# User CRUD Operations
def get_user_by_username(username):
    return query_db("SELECT * FROM users WHERE username = %s", (username,), one=True)


def get_user_by_email(email):
    return query_db("SELECT * FROM users WHERE email = %s", (email,), one=True)


def get_user_by_id(user_id):
    return query_db("SELECT * FROM users WHERE id = %s", (user_id,), one=True)


def create_user(first_name, last_name, username, email, password):
    """Creates a new user with password securely hashed using PBKDF2:SHA256."""
    if not (password.startswith('pbkdf2:') or password.startswith('scrypt:') or password.startswith('argon2:')):
        password_to_store = generate_password_hash(password, method='pbkdf2:sha256')
    else:
        password_to_store = password

    query = """
        INSERT INTO users (first_name, last_name, username, email, password, profile_image)
        VALUES (%s, %s, %s, %s, %s, 'profile_images/default.png')
    """
    return query_db(query, (first_name, last_name, username, email, password_to_store), commit=True)


def verify_user(username, password):
    """
    Verifies user credentials securely against stored hash.
    Includes backward-compatibility auto-rehash for any legacy plaintext records.
    """
    user = get_user_by_username(username)
    if not user:
        return None

    stored_pwd = user.get('password', '')
    
    # 1. Verify standard hash
    try:
        if check_password_hash(stored_pwd, password):
            return user
    except Exception:
        pass

    # 2. Legacy plaintext verification with automatic upgrade to secure hash
    if stored_pwd == password:
        try:
            upgraded_hash = generate_password_hash(password, method='pbkdf2:sha256')
            update_user_password_by_id(user['id'], upgraded_hash)
            logger.info(f"Upgraded legacy plaintext password to secure hash for user '{username}'.")
        except Exception as ex:
            logger.warning(f"Could not upgrade password hash: {ex}")
        return user

    return None


def update_user_profile(user_id, first_name, last_name, email, profile_image):
    query = """
        UPDATE users 
        SET first_name = %s, last_name = %s, email = %s, profile_image = %s 
        WHERE id = %s
    """
    return query_db(query, (first_name, last_name, email, profile_image, user_id), commit=True)


def update_user_password(email, new_password):
    """Updates password securely in hash format by user email."""
    if not (new_password.startswith('pbkdf2:') or new_password.startswith('scrypt:')):
        pwd_hash = generate_password_hash(new_password, method='pbkdf2:sha256')
    else:
        pwd_hash = new_password
    query = "UPDATE users SET password = %s WHERE email = %s"
    return query_db(query, (pwd_hash, email), commit=True)


def update_user_password_by_id(user_id, new_password):
    """Updates password securely in hash format by user ID."""
    if not (new_password.startswith('pbkdf2:') or new_password.startswith('scrypt:')):
        pwd_hash = generate_password_hash(new_password, method='pbkdf2:sha256')
    else:
        pwd_hash = new_password
    query = "UPDATE users SET password = %s WHERE id = %s"
    return query_db(query, (pwd_hash, user_id), commit=True)


# Model Metrics Operations
def save_model_metrics(user_id, model_type, dataset_name, dataset_size_kb, vocab_size,
                       epochs, final_loss, final_accuracy, val_loss, val_accuracy,
                       history_json, model_filepath):
    query = """
        INSERT INTO model_metrics (
            user_id, model_type, dataset_name, dataset_size_kb, vocab_size,
            epochs, final_loss, final_accuracy, val_loss, val_accuracy,
            history_json, model_filepath
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    return query_db(
        query,
        (user_id, model_type, dataset_name, dataset_size_kb, vocab_size,
         epochs, final_loss, final_accuracy, val_loss, val_accuracy,
         history_json, model_filepath),
        commit=True
    )


def get_latest_model_metrics(user_id=None):
    if user_id:
        return query_db(
            "SELECT * FROM model_metrics WHERE user_id = %s ORDER BY trained_at DESC LIMIT 1",
            (user_id,),
            one=True
        )
    return query_db(
        "SELECT * FROM model_metrics ORDER BY trained_at DESC LIMIT 1",
        one=True
    )


def get_all_model_metrics(user_id=None, limit=20):
    if user_id:
        return query_db(
            "SELECT * FROM model_metrics WHERE user_id = %s ORDER BY trained_at DESC LIMIT %s",
            (user_id, limit)
        )
    return query_db(
        "SELECT * FROM model_metrics ORDER BY trained_at DESC LIMIT %s",
        (limit,)
    )


def delete_model_metric(metric_id, user_id=None):
    """Deletes a specific model training history record by ID."""
    if user_id:
        query = "DELETE FROM model_metrics WHERE id = %s AND (user_id = %s OR user_id IS NULL)"
        return query_db(query, (metric_id, user_id), commit=True)
    else:
        query = "DELETE FROM model_metrics WHERE id = %s"
        return query_db(query, (metric_id,), commit=True)


def delete_user_profile_image(user_id):
    """Resets user's profile image to default logo."""
    query = "UPDATE users SET profile_image = 'profile_images/default.png' WHERE id = %s"
    return query_db(query, (user_id,), commit=True)


# Evaluation History Operations
def save_evaluation_result(user_id, ref_filename, student_filename, score,
                           similarity_pct, keyword_coverage_pct, completeness_score, feedback):
    query = """
        INSERT INTO evaluation_history (
            user_id, ref_filename, student_filename, score,
            similarity_pct, keyword_coverage_pct, completeness_score, feedback
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    return query_db(
        query,
        (user_id, ref_filename, student_filename, score,
         similarity_pct, keyword_coverage_pct, completeness_score, feedback),
        commit=True
    )


def get_user_evaluations(user_id, limit=20):
    return query_db(
        "SELECT * FROM evaluation_history WHERE user_id = %s ORDER BY evaluated_at DESC LIMIT %s",
        (user_id, limit)
    )
