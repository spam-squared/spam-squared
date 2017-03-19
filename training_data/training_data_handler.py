import sqlite3


class TrainingDataHandler(object):
    training_data = None

    def init_db(self):
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

    def load_db(self):
        """Load training data from database and initialize them"""
        conn = sqlite3.connect('training_data.db')
        c = conn.cursor()
        c.execute("SELECT * FROM train_data")

        self.training_data = list(map(lambda row: [row[0], row[1]], c))

        conn.commit()
        conn.close()

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
        return self.training_data # Get with responses["responses"][index]["text"]
