import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import ImageGrab
import datetime
import os
# with open('train_mod.txt', 'r') as f:
#     info = f.readlines()
# files_and_tags = list(map(lambda x: (x.split(' ')[0], x.split(' ')[1].split('\r')[0]), info))
files = os.listdir('.')
files = list(filter(lambda x: x.split('.')[-1] == 'avi', files))

def produce_bump_charts(vidfile,treshold,interval=10):
    #counter=0
    is_ped = 0
    count_detect = 0
    cap = cv2.VideoCapture(vidfile)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orb = cv2.xfeatures2d.SIFT_create()
    matchlist=[]
    print(length)
    for i in range(0,int(length),interval):
        cap.set(1,i)


        ret, frame = cap.read()
        if ret:
            try:
    #             print(i)
                height , width , layers =  frame.shape
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame=frame[0:int(.7*height),int(width*.5):width]
                #frame = cv2.resize(frame, (width2, height2))
                #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #frame = apply_smoothing(frame)
                #frame=cv2.Canny(frame,10,50)
                #frame=select_region(frame)
                img1 = cv2.imread('pedestrain.png',0)

                kp1, des1 = orb.detectAndCompute(img1,None)
                kp2, des2 = orb.detectAndCompute(frame,None)
                # create BFMatcher object
                bf = cv2.BFMatcher()

                # Match descriptors.
                matches = bf.knnMatch(des1,des2, k=2)
                # Apply ratio test
                good = []
                for m,n in matches:
    #                 if  .01*n.distance< m.distance < 1*n.distance:
                    if   m.distance < .75*n.distance:
                        good.append([m])
                print(len(good))
                if len(good)>treshold:
                    print('true')
                    img3 = cv2.drawMatchesKnn(img1,kp1,frame,kp2,good,None,flags=2)
                    # plt.imshow(img3),plt.show()
                    count_detect+=1
                    is_ped=1
                    if count_detect==2:
                        return is_ped, matchlist
                #cv2.drawMatchesKnn expects list of lists as matches.
                #img3 = cv2.drawMatchesKnn(img1,kp1,frame,kp2,good,None,flags=2)
                #print(i,len(good))
                matchlist.append(len(good))
                #plt.imshow(img3),plt.show()
                #cv2.imshow('frame', frame)
            except:
                pass
    cap.release()
    #cv2.destroyAllWindows()
    #matchlist=calcSma(matchlist,7)
    print('done')
    # if int(truebit)==1:
    #     plt.plot(matchlist,'b')
    #     plt.show()
    # elif int(truebit)==0:
    #     plt.plot(matchlist,'r')
    #     plt.show()
    return( is_ped, matchlist)

allbumpcharts=[]
print(datetime.datetime.now())
total_ped = 0
total_non_ped = 0
total_ped_detect = 0
total_false_detect = 0
total_cases = 0
ped_detect_array = []
ped_false_detect_array = []
#
# for k in range(10,20):
k=6

for i in files:
#     if i=='akn.098.056.left.avi':
#     if j == 1:
#         total_ped += 1
#     elif j == 0:
#         total_non_ped += 1
#     print(i,j)

    total_cases += 1
    is_ped, matchlist = produce_bump_charts(i,k)
    result = open('/Users/codeWorm/Documents/GitHub/openCV_proj/Challenge/validationset/result_ped_r_2.txt', 'a+')
    result_line = str(i) + ' 00000' + str(is_ped) + '\n'
    result.write(result_line)
    result.close()
    allbumpcharts.append(matchlist)
    if is_ped == 1:
        total_ped_detect += 1
    print('Found : %d pedestrains'%total_ped_detect)
    print('Total cases already passed: %d'%total_cases)




    # if is_ped == j == 1:
    #     total_ped_detect += 1
    # elif is_ped == 1 and j == 0:
    #     total_false_detect += 1
    # print(datetime.datetime.now())
    # print('Total TP = %d' %(total_ped_detect))
    # print('Total FP = %d' %total_false_detect)
    # print('Total total_ped = %d' %total_ped)
    # print('Total total_non_ped = %d' %total_non_ped)

# ped_detect_array.append(total_ped_detect)
# ped_false_detect_array.append(total_false_detect)

# plt.plot(range(20,40),ped_detect_array)
# plt.plot(range(20,40),ped_false_detect_array)
# plt.show()