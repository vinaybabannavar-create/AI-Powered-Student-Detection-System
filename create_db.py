# create_db.py
# Run once: creates database/users.db and users table
import os, sqlite3

DB_DIR = os.path.join(os.path.dirname(__file__), 'database')
DB_PATH = os.path.join(DB_DIR, 'users.db')
os.makedirs(DB_DIR, exist_ok=True)

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL
)
''')
conn.commit()
conn.close()
print("Database created at:", DB_PATH)
