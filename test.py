
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



init_scene_data = {'PA-0023': ['w5', 28], 'PA-0026': ['w5', 65], 'KO-0009': ['w6', 29], 'PA-0095': ['w8', 12], 'PA-0098': ['w8', 33], 'PA-0076': ['w8', 49], 'PA-0074': ['w9', 16], 'PA-0075': ['w9', 31], 'PA-0077': ['w9', 49], 'KO-0010': ['w9', 80], 'KO-0008': ('w9', 116), 'PA-0101': ['w9', 132], 'PA-0057': ['w10', 11], 'PA-0052': ['w10', 29], 'PA-0061': ['w10', 49], 'PA-0001': ['w14', 33], 'PA-0003': ['w18', 41], 'PA-0004': ['w18', 93], 'PA-0082': ['w24', 17], 'PA-0084': ['w24', 48], 'PA-0083': ['w26', 21], 'PA-0063': ['w26', 46], 'PA-0067': ['w27', 22], 'PA-0064': ['w27', 49], 'PA-0024': ['w31', 53], 'PA-0087': ['w41', 28], 'PA-0027': ['w42', 16], 'PA-0025': ['w42', 46], 'PA-0036': ['w43', 19], 'PA-0085': ['w43', 40], 'PA-0086': ['w43', 55], 'PA-0070': ['w45', 12], 'PA-0065': ['w45', 50], 'PA-0031': ['w46', 35], 'PA-0088': ['w50', 34], 'PA-0100': ['w52', 29], 'PA-0099': ['w52', 58], 'KO-0007': ['w56', 33], 'KO-0006': ['w57', 49], 'KO-0004': ['w57', 86], 'KO-0005': ['w57', 112], 'PA-0090': ['w57', 154], 'PA-0089': ['w58', 34]}
keys, values = list(init_scene_data.keys()), list(init_scene_data.values())
ordered_scene_data = {k:v for v, k in sorted(zip(values, keys), key=(lambda x : (x[0][0], x[0][1])))}

print(ordered_scene_data)