import numpy
import cv2
import os
import matplotlib.pyplot as plt

# code from guys to parse train.txt
files = os.listdir('/Users/alexajan/Downloads/validationset/validationset/')
files = list(filter(lambda x: x.split('.')[-1] == 'avi', files))

# get names of all files with bridges


def bridge_names():
    with open('/Users/alexajan/Desktop/VisionHack/visionhack_mit/trainset/trainset/train.txt', 'r') as f:
        info = f.readlines()
    files_and_tags = list(map(lambda x: (x.split(' ')[0], x.split(' ')[1].split('\r')[0]), info))
    bridges = []
    for x in files_and_tags:
        if x[1][0] == "1":
            bridges.append(x[0])

    not_bridges = []
    for x in files_and_tags:
        if x[1][0] == "0":
            not_bridges.append(x[0])


# print(bridges)


# function that turns video into greyscale and then black and white to count black pixels
def bw(vid):
    cap = cv2.VideoCapture('/Users/alexajan/Downloads/validationset/validationset/' + vid)
    # list of num black pixels in each each frame
    black_pixels = []

    while (cap.isOpened()):
        ret, frame = cap.read()
        # break at end of vid
        if frame is None:
            break
        # height,width,layers=frame.shape
        # frame=frame[0:int(.1*height),int(.2*width):int(.8*width)]
        # # resize info
        height, width, layers = frame.shape
        height2 = int(height / 2)
        width2 = int(width / 2)

        # make greyscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # resize
        resize = cv2.resize(frame, (width2, height2))

        # make black and white
        retval, threshold = cv2.threshold(resize, 255/3.5, 255, cv2.THRESH_BINARY)

        # count black pixels for each frame and append to black pixels array
        # print(numpy.sum(threshold) / 255)
        black_pix = numpy.sum(threshold) / 255
        black_pixels.append(black_pix)

        # show vid
        # cv2.imshow('frame', threshold)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # clean up after vid
    cap.release()
    cv2.destroyAllWindows()

    # print("black_pixels", black_pixels)

    # plot num pixels vs time
    # plt.plot(black_pixels)
    # plt.axis([0,300,0,600000])
    # plt.ylabel('black pixels')
    # plt.xlabel('time by frame')
    # plt.show()
    # plt.close()

    # plt.savefig("/Users/alexajan/Desktop/Visionhack/visionhack_mit/light_plots/" + vid + '-try1.pdf')
    # plt.close()

    return black_pixels

if __name__ == '__main__':
    for bridge in files:
        print(bw(bridge))



