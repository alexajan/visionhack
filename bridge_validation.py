import os
import light
import pickle
import numpy as np

files = os.listdir('/Users/alexajan/Downloads/validationset/validationset/')
files = list(filter(lambda x: x.split('.')[-1] == 'avi', files))
# print(files)
training_pixels = []

for index in range(int(len(files))):
    bridge = files[index]
    training_pixels.append(light.bw(bridge))

    print("finished " + str(index))

print(training_pixels)


with open("training_pixels.txt", 'wb') as f:
    pickle.dump(training_pixels, f)

# with open("bridge_list.txt", 'rb') as f:
#     bridge_list = pickle.load(f)

print("finished training data")


print(np.shape(training_pixels))

