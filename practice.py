import numpy as np

print("============================ Practice Starts! ============================")

array1 = np.arange(4)
# print(array1)

array2 = np.zeros((4, 4), dtype=float)
# print(array2)

array3 = np.ones((4, 4), dtype=str)
# print(array3)

array4 = np.arange(16).reshape(4, 4)
array5 = array4 < 10
# masking 연산
array4[array5] = 100
np.max(array4)

# print(array5)
# print(array5[3][2])
# print(array4[3][2])

# for xy, i in np.ndenumerate(array4):
#     print(xy[0] * 20)

# array4It = np.nditer(array4, flags=['buffered', 'multi_index'], op_dtypes=['S'])
# for x in array4It:
#     print(array4It.multi_index, x)


dic = dict()
array1 = [dic] * 20 * 10
b = np.reshape(array1, (10, 20))
print(b)
