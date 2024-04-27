import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pkl

path = "Dataset1/TUSimple/train_set/"
files = ['label_data_0313.json']
train_samples = []
augmented_samples = []

for i in range(len(files)):
    json_gt = [json.loads(line) for line in open(path+files[i])]
      
    for i in range(len(json_gt)) :
        print(i)
        gt = json_gt[i]
        gt_lanes = gt['lanes']
        y_samples = gt['h_samples']
        raw_file = gt['raw_file']
        img = cv2.imread(path+raw_file)
        gt_lanes_vis = [[(x, y) for (x, y) in zip(lane, y_samples)
                    if x >= 0] for lane in gt_lanes]
        img_vis = img.copy()
        for lane in gt_lanes_vis:
                img_poly = cv2.polylines(img_vis, np.int32([lane]), isClosed= False,
                    color=(0,255,0), thickness=5)
        #cv2.imshow('',img_vis)
        #cv2.imwrite("images"+str(i)+".png",img_vis)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        mask = np.zeros_like(img)
        colors = [[255,0,0],[0,255,0],[0,0,255],[0,255,255]]
        for i in range(len(gt_lanes_vis)):
            img_mask = cv2.polylines(mask,np.int32([gt_lanes_vis[i]]),isClosed=False,color=colors[i],thickness=5)
        '''cv2.imshow('',img_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    '''
        label = np.zeros((720,1280),dtype = np.uint8)
        for i in range(len(colors)):
            label[np.where((mask == colors[i]).all(axis = 2))] = 1
        
        final_image = cv2.resize(img,(480,480))
        #print(final_image.shape)
        f_label = cv2.resize(label,(480,480))
        final_label = f_label.reshape(480,480,1)
        #print(final_label.shape)
        sample = {
            "image": final_image,
            "label": final_label
        }

        train_samples.append(sample)

with open(path + 'training_samples1.pkl', 'wb') as f:
      pkl.dump(train_samples, f)