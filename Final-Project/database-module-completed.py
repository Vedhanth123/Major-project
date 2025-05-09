import sqlite3
import os

# Database configuration
DATABASE_PATH = "user_data.db"

def create_database():
    """Create the database and necessary tables if they don't exist"""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Create emotion logs table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS emotion_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            emotion TEXT,
            timestamp INTEGER,
            city TEXT,
            country TEXT
        )
        ''')
        
        # Create user preferences table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_preferences (
            name TEXT PRIMARY KEY,
            favorite_genre TEXT,
            preferred_duration TEXT,
            last_updated INTEGER
        )
        ''')
        
        conn.commit()
        print("Database created successfully")
        return True
    except Exception as e:
        print(f"Database creation error: {e}")
        return False
    finally:
        if conn:
            conn.close()

def log_emotion_data(name, emotion, timestamp, city, country):
    """Log emotion data to the database"""
    conn = None
    try:
        # Ensure database exists
        if not os.path.exists(DATABASE_PATH):
            create_database()
            
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO emotion_logs (name, emotion, timestamp, city, country)
        VALUES (?, ?, ?, ?, ?)
        ''', (name, emotion, timestamp, city, country))
        
        conn.commit()
        print(f"Emotion data logged for {name}")
        return True
    except Exception as e:
        print(f"Database error: {e}")
        return False
    finally:
        if conn:
            conn.close()

def get_user_preferences(name):
    """Retrieve user preferences from the database"""
    conn = None
    try:
        if not os.path.exists(DATABASE_PATH):
            return None
            
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT favorite_genre, preferred_duration
        FROM user_preferences
        WHERE name = ?
        ''', (name,))
        
        result = cursor.fetchone()
        if result:
            return {
                'favorite_genre': result[0],
                'preferred_duration': result[1]
            }
        return None
    except Exception as e:
        print(f"Database error: {e}")
        return None
    finally:
        if conn:
            conn.close()

def update_user_preferences(name, favorite_genre, preferred_duration, timestamp):
    """Update or insert user preferences in the database"""
    conn = None
    try:
        # Ensure database exists
        if not os.path.exists(DATABASE_PATH):
            create_database()
            
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Check if user preferences already exist
        cursor.execute('''
        SELECT name FROM user_preferences WHERE name = ?
        ''', (name,))
        
        result = cursor.fetchone()
        
        if result:
            # Update existing preferences
            cursor.execute('''
            UPDATE user_preferences
            SET favorite_genre = ?, preferred_duration = ?, last_updated = ?
            WHERE name = ?
            ''', (favorite_genre, preferred_duration, timestamp, name))
        else:
            # Insert new preferences
            cursor.execute('''
            INSERT INTO user_preferences (name, favorite_genre, preferred_duration, last_updated)
            VALUES (?, ?, ?, ?)
            ''', (name, favorite_genre, preferred_duration, timestamp))
        
        conn.commit()
        print(f"User preferences updated for {name}")
        return True
    except Exception as e:
        print(f"Database error: {e}")
        return False
    finally:
        if conn:
            conn.close()
