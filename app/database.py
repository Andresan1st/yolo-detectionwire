"""
Database configuration - SQL Server only
"""
import pymssql
import os
from datetime import datetime
from typing import List, Dict, Optional

# SQL Server configuration
DB_CONFIG = {
    'server': os.getenv('DB_HOST', '192.168.10.15'),
    'database': os.getenv('DB_NAME', 'db_it'),
    'user': os.getenv('DB_USER', 'user_sql'),
    'password': os.getenv('DB_PASSWORD', '1234'),
}

def get_connection():
    """Get SQL Server connection"""
    return pymssql.connect(
        server=DB_CONFIG['server'],
        database=DB_CONFIG['database'],
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password']
    )

def init_database():
    """Initialize database - create table if not exists"""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='machine_detection_copper' AND xtype='U')
            CREATE TABLE machine_detection_copper (
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
        conn.close()
        print("✅ SQL Server connection ready")
        print(f"   Server: {DB_CONFIG['server']}")
        print(f"   Database: {DB_CONFIG['database']}")
        print(f"   Table: machine_detection_copper")

    except Exception as e:
        print(f"❌ Database error: {e}")
        raise

def save_detection(detection_data: dict) -> int:
    """Save detection record to database"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO machine_detection_copper
        (detection_count, confidence_avg, image_path, element_type, status, camera_source)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (
        str(detection_data.get('detection_count', 0)),
        str(detection_data.get('confidence_avg', 0.0)),
        str(detection_data.get('image_path', '')),
        str(detection_data.get('element_type', 'copper')),
        str(detection_data.get('status', 'detected')),
        str(detection_data.get('camera_source', ''))
    ))

    conn.commit()
    cursor.execute("SELECT @@IDENTITY")
    row_id = int(cursor.fetchone()[0])
    conn.close()
    return row_id

def get_recent_detections(limit: int = 100) -> List[Dict]:
    """Get recent detection records"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(f"""
        SELECT TOP {limit} id, timestamp, detection_count, confidence_avg,
               image_path, element_type, status, camera_source, created_at
        FROM machine_detection_copper
        ORDER BY created_at DESC
    """)

    columns = [column[0] for column in cursor.description]
    results = [dict(zip(columns, row)) for row in cursor.fetchall()]
    conn.close()

    return results
