
import os

cwd = os.getcwd()
artwork_data_path = "Daegu_new.json"
visitor_data_path = "VisitorData/preAURA_1025_1030.csv"
artwork_data_filename, _ = os.path.splitext(os.path.basename(artwork_data_path))
visitor_data_filename, _ = os.path.splitext(os.path.basename(visitor_data_path))

print(visitor_data_filename)