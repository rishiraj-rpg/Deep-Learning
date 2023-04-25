import numpy as np
n = 3
arr = []
for i in range(n):
    item = [int(x) for x in input("Enter").split()]
    arr.append(item)
arr=np.array(arr)
print(arr)
print(type(arr))

input_matrix = np.array([[1,2,3],[4,5,6],[7,8,9]])
filter_matrix = np.array([[1,1],[1,1]])
filter_size = 2
padding = 1
stride = 2
actual_size = n+2*padding
actual_matrix= np.zeros((actual_size,actual_size))

for i in range(actual_size):
    for j in range(actual_size):
        if i<padding or i>=n+padding or j<padding or j>=n+padding:
            actual_matrix[i][j] = 0
        else:
            actual_matrix[i][j] = input_matrix[i-padding][j-padding]

print(actual_matrix)

output_size = int((n+(2*padding)-filter_size)/stride) + 1
output_matrix = np.zeros((output_size,output_size))

for i in range(output_size):
    for j in range(output_size):
        window = actual_matrix[(i*stride):(i*stride+filter_size),(j*stride):(j*stride + filter_size)]
        maxi = np.multiply(window,filter_matrix)
        output_matrix[i][j] = np.sum(maxi)

print(output_matrix)