
import sqlite3

# Connect to your SQLite database
conn = sqlite3.connect('emotion_data.db')
cursor = conn.cursor()

# Show database tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables in the database:")
for table in tables:
    print(table[0])

# Show the first 5 rows of the 'emotion' table
cursor.execute("SELECT * FROM emotion_logs;")
rows = cursor.fetchall()
print("\nFirst 5 rows of the 'emotion' table:")
for row in rows:
    print(row)