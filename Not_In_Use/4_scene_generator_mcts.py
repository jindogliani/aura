# Given lists
A = [5, 3, 2, 1, 5, 3]
B = ['A', 'B', 'C', 'D', 'E', 'F']

# Sorting both lists based on the values in A
A, B = zip(*sorted(zip(A, B)))
print(A)
print(B)