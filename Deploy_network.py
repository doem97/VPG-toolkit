#!/usr/bin/env python
# coding: utf-8

# ## Deploy Network

# In[1]:


# Import Libs
import numpy as np
import matplotlib.pyplot as plt
import sys
import caffe
import cv2

# Show image
import IPython.display
import PIL.Image


# In[2]:


# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = './deploy.prototxt'
PRETRAINED = 'snapshots/split_iter_100000.caffemodel'


# In[3]:


# # load the model
caffe.set_mode_gpu()
caffe.set_device(0)
net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
print ("successfully loaded classifier")


# ## V2

# #### Load .mat and see label

# In[4]:


from scipy.io import loadmat

i = '/mnt/diska/VPGNet/VPGNet-DB-5ch/scene_1/20160512_1330_59/000001.mat'
mat_file = loadmat(i) # load .mat file
rgb_seg_vp_label = mat_file['rgb_seg_vp']
rgb_image = rgb_seg_vp_label[:, :, :3] # rgb_image saves raw image
seg_label = rgb_seg_vp_label[:, :, 3].reshape(480,640) # seg_label saves pixel_level classification map
# seg_label= np.where(seg_label > 1, 1, 0)


# In[5]:


IPython.display.display(PIL.Image.fromarray(seg_label.astype(np.bool))) # Take a look at classification map


# #### Load image and train

# In[6]:


image_path = '/mnt/diska/VPGNet/VPGNet-DB-5ch/scene_1/20160512_1330_59/000001.png' # the image you want to see
mask_path = image_path.replace(".png","_mask.png") # the masked image (generated by gen_label_v4.py)
image = cv2.imread(image_path) # read image
# net.blobs['data'].reshape(1, image.shape[2], image.shape[0], image.shape[1])
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # for cv2.imread, it reads in BGR so change to RGB
IPython.display.display(PIL.Image.fromarray(image)) # show image
mask_im = cv2.imread(mask_path) # read masked image
# net.blobs['data'].reshape(1, image.shape[2], image.shape[0], image.shape[1])
mask_im = cv2.cvtColor(mask_im, cv2.COLOR_RGB2BGR) # also, exchange the position for R and B
IPython.display.display(PIL.Image.fromarray(mask_im)) # show masked image


# In[7]:


# Test on Image
test_img = caffe.io.load_image(image_path)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2, 1, 0))
transformed_img = transformer.preprocess('data', test_img) # swap R, B channel, the final input to the network should be RGB
net.blobs['data'].data[...] = transformed_img

net.forward()


# In[8]:


#Binary Mask blob masking
obj_mask = net.blobs['binary-mask'].data # object mask: detect whether have obj
mlabel = net.blobs['multi-label'].data # mlabel: saves 18 feature maps for different classes
bbox = net.blobs['bb-output-tiled'].data # bbox: not sure


# ## Show Output Featuremap

# ### Show Object Mask

# In[ ]:


for i in range(0,2): # for object mask, have 2 channels of feature maps
    small_mask = obj_mask[0, i, ...] * 255 # normalize to 0-255
    resized_mask = cv2.resize(small_mask, (640, 480)) # resize to 640*480
    image = resized_mask.astype('uint8') # for next line showing, the element of array should be in format np.uint8
    IPython.display.display(PIL.Image.fromarray(image)) # show them


# ### Show Multi-label

# In[9]:


for i in range(0,18): # 18 classes in total, corresponding to github/VPGNet/vpgnet-labels.txt
    small_mask = mlabel[0, i, ...] * 255
    resized_mask = cv2.resize(small_mask, (640, 480))
    image = resized_mask.astype('uint8')
    print(i) # show which class it is
    IPython.display.display(PIL.Image.fromarray(image))


# ## Mlabel output and contour

# In[ ]:


print("ground truth")
IPython.display.display(PIL.Image.fromarray(seg_label.astype(np.bool))) # groundtruth image, from seg_label

ch = 15 # select which channel you want to see. E.g. to see safe zoon should be 15
small_mask = mlabel[0, 0, ...] * 255 # normalize from (0,1) to (0,255)
resized_mask = cv2.resize(small_mask, (640, 480))
o_image = resized_mask.astype('uint8')
print("predicted feature map")
IPython.display.display(PIL.Image.fromarray(o_image)) # show predicted feature map

imggray = o_image
sensitivity = 0
ret,thresh = cv2.threshold(imggray,sensitivity, 255, cv2.THRESH_BINARY) # do binarize: if higher than 'sensitivity', set as 1; otherwise set as 0
print("after binarize over threshold {}".format(sensitivity))
IPython.display.display(PIL.Image.fromarray(thresh)) # Show image after binarization

# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # draw contour on the binarized output map
cv2.drawContours(imggray, contours, -1, (255,255,255), 3)
print("draw contour on feature map")
IPython.display.display(PIL.Image.fromarray(imggray))

image = cv2.imread(image_path) # read image
# net.blobs['data'].reshape(1, image.shape[2], image.shape[0], image.shape[1])
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # for cv2.imread, it reads in BGR so change to RGB
cv2.drawContours(image, contours, -1, (255,0,0), 3)# drow contour on original image
print("draw contour on original image")
IPython.display.display(PIL.Image.fromarray(image))


# # Can ignore below

# ### Show Bounding Box

# In[ ]:


for i in range(0,4):
    small_mask = bbox[0, i, ...] * 255
    resized_mask = cv2.resize(small_mask, (640, 480))
    image = resized_mask.astype('uint8')
    IPython.display.display(PIL.Image.fromarray(image))


# ## V1

# In[ ]:


# Transform Image for pre-processing
# Input in Caffe data layer is (C, H, W)
test_img = caffe.io.Transformer({'data': (1, image.shape[2], image.shape[0], image.shape[1])})
transformer.set_transpose('data', (2, 0, 1)) # To reshape from (H, W, C) to (C, H, W) ...
transformer.set_raw_scale('data', 1/255.) # To scale to [0, 1] ...
net.blobs['data'].data[...] = transformer.preprocess('data', image)


# In[ ]:


# Forward Pass for prediction
net.forward()
score_bb = net.blobs['bb-output-tiled'].data  #blobs['blob name']
score_multi = net.blobs['multi-label'].data
score_binary = net.blobs['binary-mask'].data


# In[ ]:


obj_mask = net.blobs['binary-mask'].data
x_offset_mask = 4 # offset to align output with original pic: due to padding
y_offset_mask = 4
masked_img = test_img.copy()
mask_grid_size = test_img.shape[0] / obj_mask.shape[2]
small_mask = obj_mask[0, 1, ...] * 255
resized_mask = cv2.resize(small_mask, (640, 480))
translationM = np.float32([[1, 0, x_offset_mask*mask_grid_size], [0, 1, y_offset_mask*mask_grid_size]])
resized_mask = cv2.warpAffine(resized_mask, translationM, (640, 480)) # translate (shift) the image


# In[ ]:


score_binary.min()


# In[ ]:


np.multiply(score_bb[0][0],255)


# In[ ]:


temp = score_binary[0][1]
mask = temp > 0
temp2 = np.multiply(mask*temp,255).astype(np.uint8)
temp2 = temp2.reshape((120,160,1))
other2 = np.zeros(shape=(120,160,2)).astype(np.uint8)
image_i = np.concatenate((temp2, other2), axis=2).astype(np.uint8)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
IPython.display.display(PIL.Image.fromarray(image_i))


# In[ ]:


for i in range(0,2):
    temp = score_binary[0][i]
    distance = np.max(temp)-np.min(temp)
#     temp = temp+abs(np.min(temp))
    temp = np.multiply(temp,255/distance)
    temp = temp.reshape((120,160,1))
    other = np.zeros(shape=(120,160,1))
    image_i = np.concatenate((other, temp, other), axis=2).astype(np.uint8)
    IPython.display.display(PIL.Image.fromarray(image_i))


# In[ ]:




