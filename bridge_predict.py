import light
import pickle
import numpy as np
from sklearn.svm import NuSVC

with open('/Users/alexajan/Desktop/VisionHack/visionhack_mit/trainset/trainset/train.txt', 'r') as f:
    info = f.readlines()
files_and_tags = list(map(lambda x: (x.split(' ')[0], x.split(' ')[1].split('\r')[0]), info))

predict_pixels = []
training_pixels = []
bridge_list = []
bridge_test = []

for index in range(int(len(files_and_tags)/2)):
    bridge = files_and_tags[index]
    if bridge[1][0] == "1":
        training_pixels.append(light.bw(bridge[0]))
        bridge_list.append(1)

    elif bridge[1][0] == "0":
        training_pixels.append(light.bw(bridge[0]))
        bridge_list.append(0)

    print("finished " + str(index))

print(bridge_list)
print(training_pixels)

with open("bridge_list.txt", 'wb') as f:
    pickle.dump(bridge_list, f)

with open("training_pixels.txt", 'wb') as f:
    pickle.dump(training_pixels, f)

# with open("bridge_list.txt", 'rb') as f:
#     bridge_list = pickle.load(f)

print("finished training data")

# with open("training_pixels.txt", 'rb') as f:
#     training_pixels = pickle.load(f)
#     training_pixels = np.transpose(list(zip(*training_pixels)))

print(np.shape(training_pixels))

# bridge_list = np.reshape(bridge_list, (-1,1))
print(np.shape(bridge_list))

# bridge_list = np.transpose(list(zip(*bridge_list)))



# training_pixels = np.reshape(training_pixels, (-1, 1), dtype=float)
# print(np.shape(training_pixels))
for index in range(int(len(files_and_tags)/2), len(files_and_tags)):
    predict_pixels.append(light.bw(files_and_tags[index][0]))
    bridge_test.append(int(files_and_tags[index][1][0]))

#
# with open("bridge_test.txt", 'rb') as f:
#     bridge_test = pickle.load(f)
# print(bridge_test)
#
# with open("test_pixels.txt", 'rb') as f:
#     predict_pixels = pickle.load(f)
#     predict_pixels = np.transpose(list(zip(*predict_pixels)))

# print(predict_pixels.shape, training_pixels.shape)


# for arr in predict_pixels:
#   print(arr)
# print("starting training")
#
# model = NuSVC(nu=0.4)
# model.fit(training_pixels, bridge_list)
#
# print("finished training, starting predicting")
#
# print(model.predict(predict_pixels))