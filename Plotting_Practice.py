# Q1
import matplotlib.pyplot as plt
group_A = [12, 15, 14, 13, 16, 18, 19, 15, 14, 20, 17, 14, 15, 40, 45, 50, 62]
group_B = [12, 17, 15, 13, 19, 20, 21, 18, 17, 16, 15, 14, 16, 15]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.boxplot(group_A)
ax1.set_title('Group A')
ax1.set_ylabel('Measurement Values')

ax2.boxplot(group_B)
ax2.set_title('Group B')
ax2.set_ylabel('Measurement Values')

fig.suptitle('Box Plots for Group A and Group B')

plt.show()

#Q2
import matplotlib.pyplot as plt
import numpy as np

with open('genome.txt', 'r') as file:
    genome_sequence = file.read().strip()

genome_list = list(genome_sequence)
genome_length = len(genome_list)

t = np.linspace(0, 4 * np.pi, genome_length)  # 4*pi gives about 2 turns
x = np.cos(t)
y = np.sin(t)
z = np.linspace(0, 5, genome_length)  # z increases linearly to spread out the helix vertically

coordinates = np.column_stack((x, y, z))

color_map = {'A': 'red', 'T': 'blue', 'C': 'green', 'G': 'yellow'}
colors = [color_map[nucleotide] for nucleotide in genome_list]

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], c=colors)

ax.set_title('3D Helix Structure')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

#Q3 
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from PIL import Image

img_url = 'https://images-na.ssl-images-amazon.com/images/S/pv-target-images/25e802f96e59fb4030334078973976ccce4ddb03fe806aded78cf0106736eae6._RI_V_TTW_.jpg'

with urllib.request.urlopen(img_url) as url:
    img = Image.open(url)
    img_array = np.array(img)  

plt.figure(figsize=(12, 4))
plt.subplot(1, 4, 1)
plt.imshow(img_array)
plt.axis('off')
print("Original Image Array:\n", img_array)


rotated_img = np.rot90(img_array)

plt.subplot(1, 4, 2)
plt.imshow(rotated_img)
plt.axis('off')
plt.title("Rotated 90°")

flipped_img = np.fliplr(img_array)

plt.subplot(1, 4, 3)
plt.imshow(flipped_img)
plt.axis('off')
plt.title("Flipped Left-Right")

# Convert image to grayscale using the formula: Y = 0.299*R + 0.587*G + 0.114*B
gray_img = np.dot(img_array[..., :3], [0.299, 0.587, 0.114])

# Plot grayscale image
plt.subplot(1, 4, 4)
plt.imshow(gray_img, cmap='gray')
plt.axis('off')
plt.title("Grayscale Image")

plt.tight_layout()
plt.show()


#Q4
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()

X = np.array(iris.data) 
Y = np.array(iris.target)  

mean_values = np.mean(X, axis=0)
median_values = np.median(X, axis=0)
std_values = np.std(X, axis=0)

print("Mean values:", mean_values)
print("Median values:", median_values)
print("Standard deviation values:", std_values)

min_values = np.min(X, axis=0)
max_values = np.max(X, axis=0)

print("Minimum values:", min_values)
print("Maximum values:", max_values)

# Extract only the sepal length and sepal width as a NumPy array
sepal_features = X[:, :2]  # Extracting first two columns (sepal length and sepal width)
# print("Sepal length and width:", sepal_features)

plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Sepal Length vs Sepal Width')
plt.show()

plt.hist(X[:, 0], bins=20, color='yellow', alpha=0.7)
plt.xlabel('Sepal Length')
plt.ylabel('Frequency')
plt.title('Distribution of Sepal Length')
plt.show()

plt.plot(X[:, 2], X[:, 3], 'g-')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Petal Length vs Petal Width')
plt.show()