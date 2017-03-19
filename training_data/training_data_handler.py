import sqlite3


class TrainingDataHandler(object):
    conn = sqlite3.connect('training_data.db')

    def init_db(self):
        """Init database"""
        c = self.conn.cursor()
        c.execute('''
          CREATE TABLE IF NOT EXISTS train_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            target INTEGER NOT NULL,
            answered BOOLEAN NOT NULL,
            data TEXT NOT NULL,
            spammer TEXT NOT NULL
          );
        ''')

        self.conn.commit()
        c.close()

    def load_db(self):
        """Load training data from database and initialize them"""
        conn = sqlite3.connect('training_data.db')
        c = conn.cursor()
        c.execute("SELECT * FROM train_data")

        data = list(c)

        conn.commit()
        return data

    def load_without_target(self):
        conn = sqlite3.connect('training_data.db')
        c = conn.cursor()
        c.execute("SELECT * FROM train_data WHERE target = -1")

        data = list(c)

        conn.commit()
        return data

    def set_target(self, spammer, target):
        conn = sqlite3.connect('training_data.db')

        c = conn.cursor()
        c.execute(f"""
          UPDATE train_data
          SET target = {target}
          WHERE id = (
            SELECT id FROM train_data
            WHERE spammer = {spammer}
            ORDER BY id DESCENDING
            LIMIT 1
          )
        """)

        conn.commit()

    def insert_sample(self, data, target, spammer):
        """Insert a training sample into database, target -1 if not yet present"""


        c = self.conn.cursor()
        c.execute(f"""
          UPDATE train_data
           SET answered = 1
          WHERE spammer = '{spammer}'
          """)

        self.conn.commit()
        c.close()

        c = self.conn.cursor()
        c.execute("""
          INSERT INTO train_data (target, answered, data, spammer)
          VALUES ('%s',  '%s', '%s', '%s');
        """ % (target, "FALSE", data, spammer))

        self.conn.commit()
        c.close()

    def get_training_data(self):
        """Get training data"""
        return self.load_db()
