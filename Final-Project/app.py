from flask import Flask, render_template
import sqlite3

app = Flask(__name__)

@app.route('/')
def dashboard():
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, age_range, dominant_emotion, city, country FROM user_emotions")
    logs = cursor.fetchall()
    conn.close()
    return render_template('dashboard.html', logs=logs)


if __name__ == '__main__':
    app.run(debug=True)
