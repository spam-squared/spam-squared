import sqlite3


class TrainingDataHandler(object):
    def init_db(self):
        """Init database"""
        conn = sqlite3.connect('training_data.db')
        c = conn.cursor()
        c.execute('''
          CREATE TABLE IF NOT EXISTS train_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            target INTEGER NOT NULL,
            data TEXT NOT NULL
          );
        ''')

        conn.commit()
        c.close()

    def load_db(self):
        """Load training data from database and initialize them"""
        conn = sqlite3.connect('training_data.db')
        c = conn.cursor()
        c.execute("SELECT * FROM train_data")

        data = list(map(lambda row: [row[0], row[1]], c))

        conn.commit()
        conn.close()
        return data

    def insert_sample(self, data, target):
        """Insert a training sample into database"""
        conn = sqlite3.connect('training_data.db')
        c = conn.cursor()
        c.execute("""
          INSERT INTO train_data (target, data)
          VALUES ('%s',  '%s');
        """ % data, target)

        conn.commit()
        conn.close()

    def get_training_data(self):
        """Get training data"""
        return self.load_db()
