import argparse
import numpy as np
import mdtraj as md
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabaz_score, davies_bouldin_score
from sklearn.model_selection import RepeatedKFold
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from sklearn import preprocessing
import functools
import os
import xlsxwriter
import math
import xlrd
import csv



def read_features(file_features):
    features = np.load(file_features)
    return features
    
def calculate_features(file_traj, file_top):
    #traj = md.load(file_traj, top=file_top)[:1000]
    top = md.load_topology(file_top)
    traj = md.load(file_traj, top=file_top)
    total_len = len(traj)
    # Calculate features using mdtraj
	  # Replace this with your actual feature calculation code
    atom_indices = top.select('backbone')
    sliced_trajectory = traj.atom_slice(atom_indices)
    reference_frame = sliced_trajectory[0]
    sliced_trajectory.superpose(reference_frame)
    features = md.rmsd(sliced_trajectory, reference_frame)  # Example: using rmse as features
    features = features.reshape(total_len, -1)
    return features

def run_clustering(features, n_clusters):
	km = KMeans(n_clusters=n_clusters)
	km.fit(features)
	labels = km.labels_
	centers = km.cluster_centers_
	return labels, centers


def compute_evaluation_metrics(features, labels):
    dbi = davies_bouldin_score(features, labels)
    chs = calinski_harabaz_score(features, labels)
    return dbi, chs
    
def find_true_centers(centers, features, n_clusters):
    cen_new = centers[:, np.newaxis]
    distances = np.sqrt(((features - cen_new)**2).sum(axis=2))
    distances = distances.tolist()
    min_pre = []
    for i in range(n_clusters):
        min_pre.append(distances[i].index(min(distances[i])))
    return min_pre

def calculate_ssr_sst1(features, labels, n_clusters): 
    mean = np.mean(features, axis=0)           
    sst = np.sum(np.linalg.norm(features - mean, axis=1)**2)
    
    sse = 0.0
    for i in range(n_clusters):
        x_t = []
        for j in range(len(labels)):
            if labels[j] == i:
                x_t.append(features[j])
        meani = np.mean(x_t)
        ssei = np.sum(np.linalg.norm(x_t - meani, axis=1)**2)
        sse += ssei
    ssr_sst_ratio = 1 - sse/sst
    ssr = sst - sse
    return ssr_sst_ratio

def data_encode(X_train, y_train, X_test, y_test, k, num_classes):
    for i in range(k):      
        y_train[i] = keras.utils.to_categorical(y_train[i], num_classes)
        y_test[i] = keras.utils.to_categorical(y_test[i], num_classes)
    
        X_train[i] = X_train[i].astype('float32')
        X_test[i] = X_test[i].astype('float32')

        X_train[i] /= 255
        X_test[i] /= 255
    
    return X_train, y_train, X_test, y_test

def xyz_to_rgb(xyz, trs=[[2.0413690, -0.5649464, -0.3446944], [-0.9692660, 1.8760108, 0.0415560], [0.0134474, -0.1183897, 1.0154096]]):
    rgb = []
    for i in range(3):
        tmp = xyz[0] * trs[i][0] + xyz[1] * trs[i][1] + xyz[2] * trs[i][2]
        rgb.append(tmp)
    return rgb

def sca_xyz(xyz, min=0, max=255):
    x = np.array(xyz)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(min, max))
    x_minmax = min_max_scaler.fit_transform(x)
    x_minmax2 = []
    for item in x_minmax:
        x_minmax2.append([int(item[0]), int(item[1]), int(item[2])])
    return x_minmax2

def traj_to_hex(traj):
    traj = traj.xyz
    traj1 = []
    for item in traj:
        item1 = []
        for atom in item:
            atom1 = xyz_to_rgb(atom)
            item1.append(atom1)
        traj1.append(item1)
    traj2 = []
    for item in traj1:
        item2 = sca_xyz(item, min=0, max=255)
        traj2.append(item2)
    pixel_map = traj2
    return pixel_map


def load_traj(file_nc, file_pdb):
    #traj = md.load(file_nc, top=file_pdb)[:1000]
    traj = md.load(file_nc, top=file_pdb)
    print('Info of traj:')
    print(traj)
    pixel_map = traj_to_hex(traj)
    print("Pixel-representation Start.")
    return pixel_map

def traj_to_pic(pixel):
    atom_n = len(pixel[0])
    size = math.ceil(atom_n ** 0.5)
    pixel_map = []
    for item in range(len(pixel)):
        for ti in range(size * size - atom_n):
            pixel[item].append([0, 0, 0])
        pic = []
        for i in range(size):
            line = []
            for j in range(size):
                line.append(pixel[item][i * size + j])
            pic.append(line)
        pixel_map.append(pic)
    return pixel_map

def kfold_split(k, All):
    random_state = 12883823

    rkf = RepeatedKFold(n_splits=k, n_repeats=1, random_state=random_state)
    
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    kfold_indices = rkf.split(All)

    for train_indices, test_indices in kfold_indices:
        Xk_train = []
        yk_train = []
        Xk_test = []
        yk_test = []

        for j in range(len(train_indices)):
            Xk_train.append(All[train_indices[j]][0])
            yk_train.append(All[train_indices[j]][1])
        X_train.append(np.array(Xk_train))
        y_train.append(np.array(yk_train))

        for j in range(len(test_indices)):
            Xk_test.append(All[test_indices[j]][0])
            yk_test.append(All[test_indices[j]][1])
        X_test.append(np.array(Xk_test))
        y_test.append(np.array(yk_test))

    return X_train, y_train, X_test, y_test

def split_by_group(X_all, y_all, group, cap, k):
    All = []
    for i in range(len(X_all)):
        line = [X_all[i], y_all[i]]
        All.append(line)
    
    split_all = []
    for i in range(group):
        line = []
        for j in range(cap):
            line.append(All[i * cap + j])
        
        split_all.append(line)
    
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for i in range(group):
        X_traint, y_traint, X_testt, y_testt = kfold_split(k, split_all[i])
        if i == 0:
            for w in range(k):
                X_train.append(list(X_traint[w]))
                y_train.append(list(y_traint[w]))
                X_test.append(list(X_testt[w]))
                y_test.append(list(y_testt[w]))
        else:
            for w in range(k):
                X_train[w].extend(list(X_traint[w]))
                y_train[w].extend(list(y_traint[w]))
                X_test[w].extend(list(X_testt[w]))
                y_test[w].extend(list(y_testt[w]))
    for w in range(k):
        X_train[w] = np.array(X_train[w])
        y_train[w] = np.array(y_train[w])
        X_test[w] = np.array(X_test[w])
        y_test[w] = np.array(y_test[w])
    
    return X_train, y_train, X_test, y_test

def read_classification_data(file_traj, file_top, labels, n_clusters):
    # Preprocess data
    pixel = load_traj(file_traj, file_top)
    X_all = traj_to_pic(pixel)
    X_all = np.array(X_all)
    y_all = labels 
    
    print("Preprocess Done.\n")

    group = int(n_clusters) * 10  # Number of groups
    cap = int(len(X_all) / group)  # Frames per group
    k = 5  # Cross-validation folds
    X_train, y_train, X_test, y_test = split_by_group(X_all, y_all, group, cap, k)
    
    X_train, y_train, X_test, y_test = data_encode(X_train, y_train, X_test, y_test, k, n_clusters)
    print('--------encode finish------')
    return X_all, y_all, X_train, y_train, X_test, y_test


def cnn_build(X_train, k, num_classes):
    model = []
    for i in range(k):
        modelt = Sequential()
        modelt.add(Conv2D(32, (3, 3), padding='same', input_shape=X_train[i][0].shape))
        modelt.add(Activation('relu'))
        modelt.add(Conv2D(32, (3, 3)))
        modelt.add(Activation('relu'))
        modelt.add(MaxPooling2D(pool_size=(2, 2)))
        modelt.add(Dropout(0.25))

        modelt.add(Conv2D(64, (3, 3), padding='same'))
        modelt.add(Activation('relu'))
        modelt.add(Conv2D(64, (3, 3)))
        modelt.add(Activation('relu'))
        modelt.add(MaxPooling2D(pool_size=(2, 2)))
        modelt.add(Dropout(0.25))

        modelt.add(Flatten())
        modelt.add(Dense(512))
        modelt.add(Activation('relu'))
        modelt.add(Dropout(0.5))
        modelt.add(Dense(num_classes))
        modelt.add(Activation('softmax'))
        
        modelt.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        model.append(modelt)
    return model


def cnn_train(model, X_train, y_train, X_test, y_test, k, batch_size, epochs ,verbose1='True'):  
    history = []
    verbose = 1
    if verbose1 == 'False':
	    verbose=0
    for i in range(k):
        historyt = model[i].fit(X_train[i], y_train[i],
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(X_test[i], y_test[i]),
                  shuffle=True, verbose=verbose)
        history.append(historyt)
    return model,history
    
            
def com_accuracy_cross(y1,y2):
    n = 0
    length_y = len(y1)
    
    for i in range(length_y):
        if y2[i][y1[i]]==1:
            n += 1           
    accuracy = n/length_y
    return accuracy

def eva_cross_acc_all(model,X_train,X_test,y_train,y_test):
    k = len(model)   
    for i in range(k):
        y_train_pre  = model[i].predict_classes(X_train[i])
        acc_train = com_accuracy_cross(y_train_pre, y_train[i])
        y_test_pre  = model[i].predict_classes(X_test[i])
        acc_test = com_accuracy_cross(y_test_pre, y_test[i])     
        print("The acc of model ",i,": train: ",acc_train," ,test: ",acc_test)
        
def res_score(atom_file, res_file, file0_2, c, fre):
    # Read file0_2
    f = open(file0_2, "r")
    data = f.readlines()
    f.close()

    # Remove unnecessary lines
    p = len(data) - 1
    data.pop(p)
    data.pop(p - 1)
    data.pop(0)

    # Extract atomic and residue information
    bond = []
    for line in data:
        if line != '\n':
            bond_line = line.split(' ')
            while '' in bond_line:
                bond_line.remove('')
            if bond_line[1] != '\n' and bond_line[4] != '\n':
                bond.append([int(bond_line[1]), int(bond_line[4])])

    # Read atomic importance scores
    rbook = xlrd.open_workbook(atom_file + '_cluster' + str(c) + '.xlsx')
    rsheet = rbook.sheet_by_index(0)
    i=0
    for row in rsheet.get_rows():
	if i==0:
            i=1
            continue
        if (int(row[0].value) - 1) < len(bond):
            bond[int(row[0].value) - 1].append(row[1].value / fre)

    # Compute residue scores
    res = []
    for i in range(len(bond)):
        res.append([bond[i][1], bond[i][2]])

    freq = []
    score = 0
    n = 0
    for i in range(len(res) - 1):
        if res[i][0] == res[i + 1][0]:
            score += res[i + 1][1]
            n += 1
        else:
            score += res[i + 1][1]
            n += 1
            freq.append([res[i][0], score / n])
            score = 0
            n = 0

    freq_sort = sorted(freq, key=lambda x: (x[1]))
    freq_sort.reverse()

    # Save residue scores to file
    workbook = xlsxwriter.Workbook(res_file + '_cluster' + str(c) + '.xlsx')
    worksheet = workbook.add_worksheet()
    worksheet.write(0, 0, 'residue')
    worksheet.write(0, 1, 'score')
    for i in range(len(freq_sort)):
        worksheet.write(i+1, 0, freq_sort[i][0])
        worksheet.write(i+1, 1, freq_sort[i][1])
    workbook.close()
    print("Residue-scores " + str(c) + " saved.\n")
