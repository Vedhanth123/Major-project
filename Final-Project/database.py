import sqlite3

# Initialize the database and create the table
def init_db():

    conn = sqlite3.connect('user_data.db')  # Creates or connects to a database
    c = conn.cursor()
    c.execute('''DROP TABLE IF EXISTS user_emotions''')  # Drop the table if it exists (for testing purposes)

    c.execute('''
        CREATE TABLE IF NOT EXISTS user_emotions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age_range TEXT NOT NULL,
            dominant_emotion TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            city TEXT,
            country TEXT
        )
    ''')
    conn.commit()
    conn.close()


# Function to insert data into the database
def log_emotion_data(name, age_range, dominant_emotion, timestamp, city, country):
    conn = sqlite3.connect('user_data.db')
    c = conn.cursor()
    c.execute('''INSERT INTO user_emotions (name, age_range, dominant_emotion, timestamp, city, country)
                 VALUES (?, ?, ?,?, ?, ?, ?)''', (name, age_range, dominant_emotion, timestamp, city, country))
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
