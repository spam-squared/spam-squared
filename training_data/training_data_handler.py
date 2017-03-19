import sqlite3

training_data = None


def init_db():
    """Init database"""
    conn = sqlite3.connect('training_data.db')
    c = conn.cursor()
    c.execute('''
      CREATE TABLE IF NOT EXISTS train_data (
        id INTEGER PRIMARY KEY,
        target TEXT NOT NULL,
        data TEXT NOT NULL
      );
    ''')

    conn.commit()
    c.close()


def load_db():
    """Load training data from database and initialize them"""
    conn = sqlite3.connect('training_data.db')
    c = conn.cursor()
    c.execute("SELECT * FROM train_data")

    global training_data
    training_data = list(map(lambda row: [row[0], row[1]], c))

    conn.commit()
    conn.close()


def insert_sample(data, target):
    """Insert a training sample into database"""
    conn = sqlite3.connect('training_data.db')
    c = conn.cursor()
    c.execute("""
      INSERT INTO train_data (target, data)
      VALUES ('%s',  '%s');
    """ % data, target)

    conn.commit()
    conn.close()


def get_training_data():
    """Get training data"""
    global training_data
    return training_data # Get with responses["responses"][index]["text"]
