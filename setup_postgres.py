"""
PostgreSQL Setup & Table Initialization Script
Run this script to configure the PostgreSQL database and tables for the AI-Enhanced Learning Assistant:
    python setup_postgres.py
"""

import os
import sys
from dotenv import load_dotenv

# Load database security details from .env file
_env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(_env_path):
    load_dotenv(dotenv_path=_env_path, override=True)
else:
    load_dotenv(override=True)

def main():
    print("=" * 60)
    print("AI-Enhanced Learning Assistant - PostgreSQL Database Setup")
    print("=" * 60)

    from db import (
        PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DATABASE,
        ensure_postgres_database_exists, init_db, get_db_connection
    )

    print(f"PostgreSQL Configuration:")
    print(f"  Host:     {PG_HOST}")
    print(f"  Port:     {PG_PORT}")
    print(f"  User:     {PG_USER}")
    print(f"  Database: {PG_DATABASE}")
    print("-" * 60)

    print("Step 1: Ensuring database exists...")
    db_created = ensure_postgres_database_exists()
    if db_created:
        print(f"  -> Database '{PG_DATABASE}' is ready.")
    else:
        print(f"  -> Could not verify admin database creation. Will attempt direct connection...")

    print("Step 2: Connecting and creating tables...")
    try:
        init_db()
        conn, db_type = get_db_connection()
        print(f"  -> Successfully connected using: {db_type.upper()}")
        if db_type == 'postgres':
            print("  -> PostgreSQL tables (users, model_metrics, evaluation_history) are ACTIVE!")
        else:
            print("  -> PostgreSQL server was not reached. SQLite fallback tables are ACTIVE.")
            print("  -> TIP: Ensure PostgreSQL service is running on your machine and credentials match.")
        conn.close()
    except Exception as e:
        print(f"  -> Error initializing database: {e}")
        sys.exit(1)

    print("=" * 60)
    print("Setup completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
