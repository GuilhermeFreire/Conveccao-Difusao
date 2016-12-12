import matplotlib.pyplot as plt
import matplotlib.animation as animation

def parseText(text_input):
	matrix = text_input.split("\n")
	for i in range(len(matrix)):
		if(matrix[i] == ' ' or matrix[i] == '\n'):
			continue
		matrix[i] = matrix[i].split(" ")
		for j in range(len(matrix[i])):
			matrix[i][j] = float(matrix[i][j])
	return matrix

def updatefig(*args):
	global file
	file.seek(0)
	text = file.read()
	m = parseText(text)
	img.set_array(m)
	return img,

fig = plt.figure()

file = open("sample.txt", "r")
text = file.read()

matrix = parseText(text)

img = plt.imshow(matrix, cmap='hot', interpolation='nearest', animated=True)

ani = animation.FuncAnimation(fig, updatefig, interval=100, blit=True)
plt.show()