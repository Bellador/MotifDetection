import psycopg2
import datetime
import os
import sys

class DbQuerier:
    path_db_psw = "C:/Users/mhartman/Documents/100mDataset/db_password.txt"

    def __init__(self, query, project_name):
        self.query = query
        self.project_name = project_name
        self.project_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), project_name)
        self.csv_output_path = self.project_path + '/metadata_{}_{:%Y_%m_%d}.csv'.format(self.project_name, datetime.datetime.now())
        self.conn = self.connect_db()
        print("Connected to database")
        print("--" * 30)
        print("Querying database - hold on...")
        self.export_query_to_csv()
        print("Query complete.")
        print("--" * 30)

    def connect_db(self):
        while True:
            #check if db password file exists, otherwise manual entry
            if os.path.isfile(DbQuerier.path_db_psw):
                with open(DbQuerier.path_db_psw, 'r') as f:
                    password = f.read()
            else:
                password = input("Input database password: ")
            try:
                conn = psycopg2.connect("host=127.0.0.1 dbname=100m_dataset user=postgres port=5432 password={}".format(password))
                return conn
            except Exception as e:
                print(f"Error {e}. Try again.")
                sys.exit(1)

    def export_query_to_csv(self):
        with self.conn.cursor() as cursor:
            outputquery = "COPY ({0}) TO STDOUT WITH DELIMITER ';' CSV HEADER".format(self.query)
            with open(self.csv_output_path, 'w', encoding='utf-8') as f:
                cursor.copy_expert(outputquery, f)
        print("Export file created at: {}".format(self.csv_output_path))
