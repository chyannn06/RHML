import argparse
import numpy as np
import functools
import os
import xlsxwriter
import csv
from util import *
import lime_utils

# --need_clustering=True --need_cal_feature=True --traj_file='nore1.nc' --top_file='nore.pdb' --n_clusters=3 --batch_size=8 --epochs=10 --print_detail=1 --print_acc='True' --save_models=0 --atom_file='atom' --res_file='res'
parser = argparse.ArgumentParser(description='the IHDL framework')
parser.add_argument('--need_clustering', help='Whether to perform clustering processing')
parser.add_argument('--need_cal_feature', help='Whether to calculate clustering features')
parser.add_argument('--traj_file', help='The traj file')
parser.add_argument('--top_file', help='The top file')
parser.add_argument('--feature_file',help='The clustering feature file')
parser.add_argument('--labels_file',help='The labels file')
parser.add_argument('--n_clusters', help='The number of cluster')
parser.add_argument('--batch_size', help='cnn train batch_size')
parser.add_argument('--epochs', help='cnn train epochs')
parser.add_argument('--print_detail', help='Print details')
parser.add_argument('--print_acc', help='Print details')
parser.add_argument('--save_models')
parser.add_argument('--atom_file', help='Path to the atom file')
parser.add_argument('--res_file', help='Path to the result file')

args = parser.parse_args() 
 
if __name__ == '__main__':

    need_cal_feature = args.need_cal_feature
    need_clustering = args.need_clustering
    traj_file = args.traj_file
    top_file = args.top_file
    feature_file = args.feature_file
    n_clusters = int(args.n_clusters)
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    if_dtl = args.print_detail
    if_prt = args.print_acc
    if_save = args.save_models
    atom_file = args.atom_file
    res_file = args.res_file
    labels_file = args.labels_file
    
    features = []
    labels = []

    centers = []
	  # clustering processing
    if need_clustering == 'True':
        if need_cal_feature == 'True':
            features = calculate_features(traj_file,top_file)
        else:
            features = read_features(feature_file)
        labels ,centers = run_clustering(features, n_clusters)
        
        true_centers = find_true_centers(centers, features, n_clusters)  
           
        dbi, psf= compute_evaluation_metrics(features, labels)
        ssr_sst_ratio = calculate_ssr_sst1(features, labels, n_clusters)
        
        print('dbi:{}  psf:{}  ssr_sst_ratio:{}'.format(dbi, psf ,ssr_sst_ratio))
        save_la = np.array(labels)
        np.save("cluster_labels.npy",save_la)
        csv_file = open("cluster_result.csv",'w')
        writer = csv.writer(csv_file)
        writer.writerow(['centers','Representative frames','dbi', 'psf', 'ssr/sst'])
        writer.writerow([centers,true_centers,dbi,psf,ssr_sst_ratio])
        csv_file.close()
        print('----------cluster process done-----------')
    else:
        labels = np.load(labels_file)
    # end
    
    data_input = input('Whether to continue running?y/n:')
    if data_input == 'n' or data_input == 'N':
        print('Exiting the program.')
        sys.exit(0)
        
	  # class processing
    X_all, y_all, X_train, y_train, X_test, y_test = read_classification_data(traj_file, top_file, labels, n_clusters)
    
    print('--------class data process---------')
    model = cnn_build(X_train, 5, n_clusters)
    
    print('--------class data process---------')
    model, history = cnn_train(model, X_train, y_train, X_test, y_test, 5, batch_size, epochs, if_dtl)
    
    if if_prt == 'True':
        eva_cross_acc_all(model, X_train, X_test, y_train, y_test)
    if if_save == 'True':
        for t in range(5):
            model[t].save('model' + str(t) + '.h5')
            print("Models Saved.\n")
    
    data_input = input('Whether to continue running?y/n:')
    if data_input == 'n' or data_input == 'N':
        print('Exiting the program.')
        sys.exit(0)
    # LIME interpreter
    print("LIME Constructing.\n")

    X_lime, y_lime = lime_utils.lime_split(X_all, y_all, n_clusters)

    importance_pic = []
    for i in range(n_clusters):
        print("\nLIME of type" + str(i))
        importance_pic.append(lime_utils.cnn_lime(model, X_lime[i], y_lime[i], 5, n_clusters)[0])

    print("LIME Done.\n")

    # Output files

    for c in range(n_clusters):
        workbook = xlsxwriter.Workbook(atom_file + '_cluster' + str(c) + '.xlsx')  # Create the file
        worksheet = workbook.add_worksheet()
        size = len(importance_pic[c])
	worksheet.write(0, 0, 'atom')
        worksheet.write(0, 1, 'score')
        for i in range(size):
            for j in range(size):
                worksheet.write(i * size + j+1, 0, i * size + j + 1)
                worksheet.write(i * size + j+1, 1, importance_pic[c][i][j])
        workbook.close()
    print("Atom-scores Saved.\n")

    for c in range(n_clusters):
        res_score(atom_file, res_file, top_file, c, 1)

    print("Done.\n")
    eva_cross_acc_all(model, X_train, X_test, y_train, y_test)
