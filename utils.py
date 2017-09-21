import numpy as np
import cv2


def select_region(image):
    # first, define the polygon by vertices
    rows, cols = image.shape[:2]
    bottom_left  = [cols*0.1, rows*0.95]
    top_left     = [cols*0.4, rows*0.6]
    bottom_right = [cols*0.9, rows*0.95]
    top_right    = [cols*0.6, rows*0.6] 
    
    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return filter_region(image, vertices)

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

def get_crossing(array, threshold=15, frame_range=6, crossing_per_sig=1):
    """
    array is an array representing a sinusoidal signal and a 
    threshold for occurence ofsignal of interest. 
    frame_range is the number of points to look at to get the number of crossings.
    crossing_per_sig is the number of crossings within frame_range that represents occurence of
                     signal of interest
    Returns the number of occurences of the signal of interest
    """
    oscillations = 0
    for i in xrange(0, len(array) - frame_range, 4):
        # select threshold confirming wiper movement (black intensity)
        if np.sum(np.abs(array[i: i + frame_range]) >= threshold) >= crossing_per_sig:
            # check for at least crossing_per_sig zero crossings to confirm oscillation
            change = 0
            prev_point = array[i]
            for point in array[i + 1 : i + frame_range]:
                if point*prev_point < 0:
                    change += 1
                    prev_point = point
            if change >= crossing_per_sig:
                oscillations += 1
            return oscillations
    return count

def read_files_and_tags(path_to_txt):
    """
    Returns list of tuples of the form [(filename.avi, 001110), ...]
    """
    with open(path_to_txt, 'r') as f:
        info = f.readlines()
    files_and_tags = list(map(lambda x: (x.split(' ')[0], x.split(' ')[1].split('\r')[0]), info))
    return files_and_tags

