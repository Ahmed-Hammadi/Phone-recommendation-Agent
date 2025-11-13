"""
Database Migration Script: CSV to PostgreSQL
Loads mobile.csv data into PostgreSQL database with proper schema and indexes.
"""

import os
import pandas as pd
from sqlalchemy import create_engine, text, Column, Integer, String, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# Database configuration
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "phone_recommender")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Paths
BASE_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE_DIR, "database", "mobile.csv")

Base = declarative_base()

def normalize_column_name(col):
    """Normalize column names for database compatibility"""
    return col.strip().lower().replace(" ", "_").replace("(", "").replace(")", "").replace(":", "").replace(".", "")

def create_database():
    """Create the database if it doesn't exist"""
    from sqlalchemy import create_engine
    from sqlalchemy.exc import ProgrammingError
    
    # Connect to default postgres database to create our database
    default_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/postgres"
    engine = create_engine(default_url, isolation_level="AUTOCOMMIT")
    
    try:
        with engine.connect() as conn:
            # Check if database exists
            result = conn.execute(text(f"SELECT 1 FROM pg_database WHERE datname = '{DB_NAME}'"))
            if not result.fetchone():
                conn.execute(text(f"CREATE DATABASE {DB_NAME}"))
                print(f"✅ Created database: {DB_NAME}")
            else:
                print(f"ℹ️  Database {DB_NAME} already exists")
    except Exception as e:
        print(f"⚠️  Error creating database: {e}")
    finally:
        engine.dispose()

def load_csv_to_postgres():
    """Load CSV data into PostgreSQL"""
    
    # Create database if needed
    create_database()
    
    # Connect to our database
    engine = create_engine(DATABASE_URL)
    
    print(f"\n📂 Loading CSV from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    
    # Normalize column names
    df.columns = [normalize_column_name(col) for col in df.columns]
    
    # Drop the first unnamed index column if it exists
    if 'unnamed_0' in df.columns or '' in df.columns:
        df = df.drop(columns=['unnamed_0', ''], errors='ignore')
    
    # Create search_name column for fuzzy matching
    df["brand"] = df.get("brand", "").astype(str).fillna("").str.strip()
    df["model"] = df.get("model", "").astype(str).fillna("").str.strip()
    df["search_name"] = (df["brand"] + " " + df["model"]).str.strip().str.lower()
    
    print(f"📊 Total rows: {len(df)}")
    print(f"📋 Columns: {len(df.columns)}")
    print(f"🔤 Sample columns: {list(df.columns[:10])}")
    
    # Load data into PostgreSQL
    print(f"\n🔄 Loading data into PostgreSQL table 'phones'...")
    df.to_sql(
        'phones',
        engine,
        if_exists='replace',  # Replace table if it exists
        index=True,
        index_label='id',
        chunksize=1000
    )
    
    print(f"✅ Data loaded successfully!")
    
    # Create indexes for better query performance
    print(f"\n🔧 Creating indexes...")
    with engine.connect() as conn:
        # Index on search_name for fuzzy matching
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_phones_search_name ON phones (search_name)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_phones_brand ON phones (brand)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_phones_model ON phones (model)"))
        
        # Text search index (PostgreSQL full-text search)
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_phones_search_name_trgm ON phones USING gin (search_name gin_trgm_ops)"))
        conn.commit()
        print(f"✅ Indexes created!")
    
    # Verify data
    print(f"\n🔍 Verifying data...")
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM phones"))
        count = result.scalar()
        print(f"✅ Total rows in database: {count}")
        
        # Show sample data
        result = conn.execute(text("SELECT id, brand, model, search_name FROM phones LIMIT 5"))
        print(f"\n📱 Sample phones:")
        for row in result:
            print(f"  - {row.brand} {row.model} (search: {row.search_name})")
    
    engine.dispose()
    print(f"\n✅ Migration complete!")

if __name__ == "__main__":
    print("=" * 80)
    print("📱 Phone Database Migration: CSV → PostgreSQL")
    print("=" * 80)
    
    # Check if CSV exists
    if not os.path.exists(CSV_PATH):
        print(f"❌ CSV file not found: {CSV_PATH}")
        exit(1)
    
    try:
        load_csv_to_postgres()
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    print("\n" + "=" * 80)
    print("🎉 Migration successful! You can now use PostgreSQL.")
    print("=" * 80)
