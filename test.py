
# import os

# cwd = os.getcwd()
# artwork_data_path = "Daegu_new.json"
# visitor_data_path = "VisitorData/preAURA_1025_1030.csv"
# artwork_data_filename, _ = os.path.splitext(os.path.basename(artwork_data_path))
# visitor_data_filename, _ = os.path.splitext(os.path.basename(visitor_data_path))

# print(visitor_data_filename)


dict = {'A1':['w4', 10], 'A2':['w1', 20], 'A6':['w1', 10], 'A3':['w6', 30], 'A4':['w2', 40], 'A7':['w4', 1], 'A5':['w5', 50], 'A8':['w2', 1]}

keys = list(dict.keys())
values = list(dict.values())

z = {k:v for v, k in sorted(zip(values, keys), key=(lambda x : (x[0][0], x[0][1])))}

print(z)
