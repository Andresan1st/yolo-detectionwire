"""
Database configuration and connection for SQL Server
"""
import pyodbc
from contextlib import contextmanager
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# Database configuration
DB_CONFIG = {
    'server': os.getenv('DB_HOST', '192.168.10.15'),
    'database': os.getenv('DB_NAME', 'db_it'),
    'username': os.getenv('DB_USER', 'user_sql'),
    'password': os.getenv('DB_PASSWORD', '1234'),
    'table': os.getenv('DB_TABLE', 'machine_detection_copper'),
}

def get_connection_string():
    """Build SQL Server connection string"""
    return (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={DB_CONFIG['server']};"
        f"DATABASE={DB_CONFIG['database']};"
        f"UID={DB_CONFIG['username']};"
        f"PWD={DB_CONFIG['password']};"
    )

@contextmanager
def get_db_connection():
    """Context manager for database connection"""
    conn = None
    try:
        conn = pyodbc.connect(get_connection_string())
        yield conn
    except pyodbc.Error as e:
        print(f"Database connection error: {e}")
        raise
    finally:
        if conn:
            conn.close()

def init_database():
    """Initialize database and create table if not exists"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Create table if not exists
            cursor.execute(f"""
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='{DB_CONFIG['table']}' AND xtype='U')
                CREATE TABLE {DB_CONFIG['table']} (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    timestamp DATETIME DEFAULT GETDATE(),
                    detection_count INT DEFAULT 0,
                    confidence_avg FLOAT DEFAULT 0.0,
                    image_path NVARCHAR(500),
                    element_type NVARCHAR(50) DEFAULT 'copper',
                    status NVARCHAR(50) DEFAULT 'detected',
                    camera_source NVARCHAR(100),
                    created_at DATETIME DEFAULT GETDATE()
                )
            """)
            conn.commit()
            print(f"Database table '{DB_CONFIG['table']}' is ready")

    except Exception as e:
        print(f"Database initialization error: {e}")
        # Create SQLite fallback
        create_sqlite_fallback()

def create_sqlite_fallback():
    """Create SQLite fallback database"""
    import sqlite3
    db_path = os.path.join(os.path.dirname(__file__), 'detections.db')

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS machine_detection_copper (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            detection_count INTEGER DEFAULT 0,
            confidence_avg REAL DEFAULT 0.0,
            image_path TEXT,
            element_type TEXT DEFAULT 'copper',
            status TEXT DEFAULT 'detected',
            camera_source TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    print(f"SQLite fallback database created at {db_path}")

def save_detection(detection_data: dict) -> int:
    """Save detection record to database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(f"""
                INSERT INTO {DB_CONFIG['table']}
                (detection_count, confidence_avg, image_path, element_type, status, camera_source)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                detection_data.get('detection_count', 0),
                detection_data.get('confidence_avg', 0.0),
                detection_data.get('image_path'),
                detection_data.get('element_type', 'copper'),
                detection_data.get('status', 'detected'),
                detection_data.get('camera_source')
            ))
            conn.commit()
            return cursor.identity

    except Exception as e:
        print(f"Error saving to SQL Server: {e}")
        # SQLite fallback
        import sqlite3
        db_path = os.path.join(os.path.dirname(__file__), 'detections.db')

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO machine_detection_copper
            (detection_count, confidence_avg, image_path, element_type, status, camera_source)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            detection_data.get('detection_count', 0),
            detection_data.get('confidence_avg', 0.0),
            detection_data.get('image_path'),
            detection_data.get('element_type', 'copper'),
            detection_data.get('status', 'detected'),
            detection_data.get('camera_source')
        ))
        conn.commit()
        row_id = cursor.lastrowid
        conn.close()
        return row_id

def get_recent_detections(limit: int = 100):
    """Get recent detection records"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT TOP {limit} id, timestamp, detection_count, confidence_avg,
                       image_path, element_type, status, camera_source, created_at
                FROM {DB_CONFIG['table']}
                ORDER BY created_at DESC
            """)
            columns = [column[0] for column in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]
            return results

    except Exception as e:
        print(f"Error reading from SQL Server: {e}")
        # SQLite fallback
        import sqlite3
        db_path = os.path.join(os.path.dirname(__file__), 'detections.db')

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, timestamp, detection_count, confidence_avg,
                   image_path, element_type, status, camera_source, created_at
            FROM machine_detection_copper
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))

        columns = [column[0] for column in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        return results
