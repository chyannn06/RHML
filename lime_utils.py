import os
import sys
try:
    import lime
except:
    sys.path.append(os.path.join('..', '..'))  # add the current directory
    import lime
from lime.wrappers.scikit_image import SegmentationAlgorithm
from lime import lime_image
import numpy as np


def lime_split(X, y, num_classes):
    X_lime = []
    y_lime = []
    for i in range(num_classes):
        X_i = []
        y_i = []
        for j in range(len(y)):
            if y[j] == i:
                X_i.append(X[j])
                y_i.append(i)
        X_lime.append(X_i)
        y_lime.append(y_i)
        
    return X_lime, y_lime
    

def cnn_lime(model, X, y, k, num_classes):
    explainer = lime_image.LimeImageExplainer(verbose=False)
    segmenter = SegmentationAlgorithm('slic', n_segments=100, compactness=1, sigma=1)
    
    n_features = 100
    n_samples = 1000
    
    importance_pic = []
    size = len(X[0])
    n_lime = int(len(y) / 10)
    for i in range(size):
        line = []
        for j in range(size):
            line.append(0)
        importance_pic.append(line)
        
    num_lime = 0
    for j in range(k):
        for i in range(n_lime):
            print(str(j) + '-' + str(i))
            explanation = explainer.explain_instance(X[i * 10].astype('float64'), 
                                                     classifier_fn=model[j].predict_proba,  
                                                     top_labels=num_classes, hide_color=0, num_samples=n_samples, segmentation_fn=segmenter)
            temp, mask = explanation.get_image_and_mask(y[i * 10], positive_only=True, num_features=n_features, hide_rest=False)
            
            num_lime += 1
            importance_pic = np.array(importance_pic) + np.array(mask)
    
    importance_pic = importance_pic / num_lime
    
    return importance_pic, explainer, segmenter


def lime_show(explainer, segmenter, model, X, y, k, pixel_map, num_classes):
    import matplotlib.pyplot as plt
    from skimage.color import label2rgb
    
    n_features = 100
    n_samples = 1000
    
    explanation = explainer.explain_instance(X, classifier_fn=model[0].predict_proba, 
                                             top_labels=num_classes, hide_color=0, num_samples=n_samples, segmentation_fn=segmenter)
    temp, mask = explanation.get_image_and_mask(y, positive_only=True, num_features=n_features, hide_rest=False)

    masks = mask
    for i in range(k):
        if i != 0:
            explanation = explainer.explain_instance(X, classifier_fn=model[i].predict_proba, 
                                                     top_labels=num_classes, hide_color=0, num_samples=n_samples, segmentation_fn=segmenter)
            temp, mask = explanation.get_image_and_mask(y, positive_only=True, num_features=n_features, hide_rest=False)

            masks = masks + mask
    
    signs = []
    for i in range(len(masks)):
        line = []
        for j in range(len(masks[0])):
            if masks[i][j] == 0:
                line.append([255, 255, 255])
            elif masks[i][j] == 1:
                line.append([211, 211, 211])
            elif masks[i][j] == 2:
                line.append([192, 192, 192])
            elif masks[i][j]== 3:
                line.append([128, 128, 128])
            elif masks[i][j] == 4:
                line.append([105, 105, 105])
            else:
                line.append([0, 0, 0])
        signs.append(line)
        
    signs = np.array(signs)
    
    import cv2
    img_a = signs.astype("uint8")
    img_b = temp.astype("uint8")
    
    lucency = 0.5
    img_a = cv2.resize(img_a, (img_b.shape[1], img_b.shape[0]))

    img_c = lucency * img_b + (1 - lucency) * img_a
    img_c = img_c.astype(np.uint8)
    img_c = np.clip(img_c, 0, 255)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.imshow(img_a)
    ax2.imshow(img_c)
    fig.show()
