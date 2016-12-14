import matplotlib.pyplot as plt

with open("sample.txt", "r") as f:
	text = f.read()

matrix = text.split("\n")
##print(matrix)
for i in range(len(matrix)):
	if(matrix[i] == ' ' or matrix[i] == '\n'):
		continue
	matrix[i] = matrix[i].split(" ")
	##print(matrix[i])
	for j in range(len(matrix[i])):
		if(matrix[i][j] == ' ' or matrix[i][j] == '\n'):
			continue
		##print(matrix[i][j])
		matrix[i][j] = float(matrix[i][j])

plt.imshow(matrix, cmap='hot', interpolation='nearest')
plt.show()
