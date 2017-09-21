import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

def filter_region(image, vertices):
    """
    Create the mask using the vertices and apply it to the input image
    """
    mask = np.zeros_like(image)
    if len(mask.shape)==2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,)*mask.shape[2]) # in case, the input image has a channel dimension        
    return cv2.bitwise_and(image, mask)

    
def select_region(image):
    # first, define the polygon by vertices
    rows, cols = image.shape[:2]
    bottom_left  = [cols*0.1, rows*0.8]
    top_left     = [cols*0.2, rows*0.4]
    bottom_right = [cols*0.8, rows*0.8]
    top_right    = [cols*0.9, rows*0.4] 
    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return filter_region(image, vertices)

if __name__ == "__main__":
    path = '/Users/emmanuelazuh/Downloads/testset/'
    vids = list(filter(lambda x: x.split('.')[-1] == 'avi', os.listdir(path)))
    
    ctr = 0
    prev_four_frame = np.empty(0)
    lower_black = np.array([0, 0, 0], dtype = "uint8")
    upper_black = np.array([30, 30, 30], dtype = "uint8")
    contour_area = []

    vid_idx = 0
    cap = cv2.VideoCapture(path + vids[vid_idx])
    correct = 0
    prev_centroid = 0
    centroids = []
    positives = 0
    prev_area = 0
    while(cap.isOpened()):
        ret, frame = cap.read()

        if (ret) and (ctr % 4 == 0):
            h,w,l = frame.shape
            h,w = h/2, w/2

            # Use hsv, blur and compute edges
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            blurred_image = cv2.GaussianBlur(hsv, (11, 11), 0)
            mask = cv2.bitwise_not(cv2.inRange(blurred_image, lower_black, upper_black))
            edge_image = cv2.Canny(blurred_image, 10, 40)
            mask = cv2.adaptiveThreshold(mask,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,-3)

            region_of_interest = select_region(mask)

            _, contours, _ = cv2.findContours(region_of_interest, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            Centroid_x = sum(list(map(lambda x: int(cv2.moments(x)['m10']/cv2.moments(x)['m00']), 
                                          filter(lambda y: cv2.moments(y)['m00'] != 0, contours))))
            # find intensity centroid of all contours. thresholding gets mostly dark objects in a 
            # selected region of the frame through which the wiper oscillates
            contour_area_sum = sum(list(map(lambda x: cv2.contourArea(x), contours)))

            contour_area.append(contour_area_sum - prev_area)
            centroids.append(Centroid_x)
            prev_area = contour_area_sum

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if ctr == 303:
            # plot blue for reported positive and red for reported negative
            plt_cent = np.array(centroids)

            centroid_c = np.sum(plt_cent > 100000) > 0
            contour_c = np.sum(np.array(contour_area) > 7000) > 0
            event_occurs = 1 if (centroid_c and contour_c) else 0
            positives += event_occurs
            
            print('ran', vid_idx + 1, 'tests')
            print('found', positives, 'positives')
            
            with open('wiper_out.txt', 'a+') as out:
                out.write(vids[vid_idx] + ' 0000' + str(event_occurs) + '0\n')
                

            # reset values to run on next video
            vid_idx += 1
            if vid_idx == len(vids):
                break
            cap = cv2.VideoCapture(path + vids[vid_idx])
            ctr = 0
            contour_area = []
            prev_cnt_count = 0
            prev_centroid = 0
            event_occurs = 0
        ctr += 1
    # print('accuracy =', correct/float(len(vids)))         
    cap.release()
    cv2.destroyAllWindows()