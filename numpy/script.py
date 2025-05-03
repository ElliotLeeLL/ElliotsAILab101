import numpy as np
import matplotlib.pyplot as plt


# Creating Arrays
zero_array = np.zeros(5)
print(zero_array)

two_dimension_zero_array = np.zeros((3,4))
print(two_dimension_zero_array)
print(two_dimension_zero_array.shape)
print(two_dimension_zero_array.ndim)
print(two_dimension_zero_array.size)

one_array = np.ones((3,4))
full_array = np.full((3,4), np.pi)
empty_array = np.empty((3,4))
array_array = np.array([[1,2,3,4],[6,7,8,9]])
range_array = np.arange(1,5)
print(range_array)

lin_space_array = np.linspace(0, 5/3, 6)
print(lin_space_array)

uniform_dis_random = np.random.rand(3, 4)
print(uniform_dis_random)
gaussian_dis_random = np.random.randn(3, 4)
print(gaussian_dis_random)

plt.hist(np.random.rand(100000), density=True, bins=100, histtype='step', color='blue', label='rand')
plt.hist(np.random.randn(100000), density=True, bins=100, histtype='step', color='red', label='randn')
plt.axis([-2.5, 2.5, 0, 1.1])
plt.legend()
plt.title("Random Distribution")
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()

# Array data