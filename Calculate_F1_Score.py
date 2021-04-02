#!/usr/bin/env python
# coding: utf-8

# In[48]:


import numpy as np
from scipy.io import loadmat
import caffe
from sklearn.metrics import f1_score, recall_score
import cv2

#======== Define util functions ====================================
def create_circular_mask(h, w, center=None, radius=None):
    # Create a circular mask from center on an (h,w) map with euclidean distance radius
    if center is None: # use the middle of the image
        center = (int(h/2), int(w/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], h-center[0], w-center[1])

    X, Y = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def vpg_lane_f1(map1, map2, mask_R = 12):
    # For single class metric, calculate two map1,2 whose label are binary (0 for no, and !0 for label)
    # Input:
    #     map1: (h,w,1) size label predicted map
    #     map2: (h,w,1) size label groundtruth map, every 8*8 pixels an grid with same label
    #     mask_R: euclidean distance of radius R, default 4
    # Return:
    #     single_class_f1: f1 score for map1 and map2 described in VPGNet Sec5.3

    # First extend the grid-labeled map2 to circle-labeled map extend_mask with boundary R
    
    map2_mask = map2 > 0 # Assume map1,2 only have one class
    if True not in map2: # If all black map in groundtruth, set as 1 rather than 0 (default zero)
        return 1.0
    map1_mask = map1 > 0
    extend_mask = np.zeros((480, 640), dtype=bool) # extended groundtruth (from 8*8 square grid to radius R circle)
    for i in range(0, 480):
        for j in range(0, 640):
            if map2_mask[i,j] == True: # if this pixel have label, this 8*8 grid should have same label
                area_mask = create_circular_mask(480, 640, center = (i,j), radius = mask_R)
                extend_mask = extend_mask + area_mask # add the area_mask to blank mask
                
    # Compare map1 and the extended mask for f1 score
    single_class_f1 = f1_score(extend_mask.flatten(), map1_mask.flatten(), zero_division = 1)
    return single_class_f1

def calcSingleImage(mat_path, sensivity = 10, mask_R = 20, CAFFEMODEL = None):
    #==== Load Ground Truth ====
    mat = loadmat(mat_path) # load mat file
    rgb_seg_vp_label = mat['rgb_seg_vp']
    gt = rgb_seg_vp_label[:, :, 3].reshape(480,640)
    test_img = rgb_seg_vp_label[:, :, :3]/255
    #==== Load the Prediciton ====
    # Set the right path to your model definition file, pretrained model weights,
    # and the image you would like to classify.
    # load the model
    net = CAFFEMODEL
#     test_img = caffe.io.load_image(image_path) # Being replaced by load mat directely
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2, 1, 0))
    transformed_img = transformer.preprocess('data', test_img) # swap R, B channel, the final input to the network should be RGB
    net.blobs['data'].data[...] = transformed_img
    net.forward()
    mlabel = net.blobs['multi-label'].data # mlabel: saves 18 feature maps for different classes
    mlabel = np.delete(mlabel, 1, 1) # delete the second channel in prediction, which I don't know what is it.
    x = 10 # vertical shifting
    y = 12 # horizontal shifting
    score = [None]*17 # Save the scores
    for i in range(1, 18):
        # Loop
        small_mask = mlabel[0, i, ...] * 255 # normalize from (0,1) to (0,255) 
        resized_mask = cv2.resize(small_mask, (640, 480))
        translationM = np.float32([[1, 0, x], [0, 1, y]])
        resized_mask = cv2.warpAffine(resized_mask, translationM, (640, 480)) 
        imggray = resized_mask.astype(np.uint8)
        ret,thresh = cv2.threshold(imggray,sensivity, 1, cv2.THRESH_BINARY) # do binarize: if higher than 'sensivity', set as 1; otherwise set as 0
        thresh = thresh.reshape(480, 640).astype(np.uint8)
        predn = thresh.astype(np.bool)
        gtn = (gt == i)
        score[i - 1] = vpg_lane_f1(predn, gtn, mask_R = mask_R)
    return score


# In[49]:


#======== Load the necessary parameters ====================================
### Load the model
MODEL_FILE = './deploy.prototxt'
PRETRAINED = 'snapshots/split_iter_100000.caffemodel'
caffe.set_mode_gpu()
caffe.set_device(0)
net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
print ("successfully loaded classifier")


# In[50]:


#========  Predict on the single image and calculate the f1 score ====================================
mat_path = '/mnt/diska/VPGNet/VPGNet-DB-5ch/scene_3/20160503_0945_43/000091.mat'
score = calcSingleImage(mat_path, sensivity = 50, mask_R = 13, CAFFEMODEL = net)


# In[51]:


#======== Print the f1 score for each class and the average score ====================================
for ind, i in enumerate(score):
    print("{}:{}".format(ind, i))
print("avg: {}".format(np.average(score)))