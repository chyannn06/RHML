# RHML
The user-friendly codes for aiding identification of the allosteric site and other MD studies associated with conformational changes.

# about RHML

The RHML model is developed by combining the unsupervised clustering and the interpretable CNN-based multi-classification models. Benefited from the technique advantages embedded in the framework like the auto-labeling capability of the unsupervised clustering, the pixel image representation of the conformation, powerful learning ability of CNN for images and locally linear approximation interpreter, the RHML model enables accurate conformation classification and identification of important residue deciding different conformational classes for any MD trajectory. 


# Requirements:
```
Tensorflow 1.14.0
sklearn
functools
Scikit-learn
numpy
keras
lime
mdtraj
xlrd
csv
XlsxWriter
```


# Detailed guidelines
The code offers customizable input options, automatically generating readable output files that include cluster categories and important residues deciding the classification. 

To use the code, you need to set certain arguments, as described below:

```
parser.add_argument('--need_clustering', help='Whether to perform clustering processing(True or False)')
parser.add_argument('--need_cal_feature', help='Whether to calculate clustering features(True or False)')
parser.add_argument('--traj_file', help='The trajectory file')
parser.add_argument('--top_file', help='The topology file')
parser.add_argument('--feature_file',help='The clustering feature file(Run clustering with your feature_file')
parser.add_argument('--labels_file',help='The labels file(not need to run clustering; train the classifacation model with your label_file)')
parser.add_argument('--n_clusters', help='The number of cluster')
parser.add_argument('--batch_size', help='CNN train batch size')
parser.add_argument('--epochs', help='CNN train epochs')
parser.add_argument('--print_detail', help='Print details ( True or False)')
parser.add_argument('--print_acc', help='Print accuracy (True or False)')
parser.add_argument('--save_models', help='Save models (True or False).')
parser.add_argument('--atom_file', help='Specify the filename for saving the importance scores of atoms')
parser.add_argument('--res_file', help='Specify the filename for saving the importance scores of residues')
```

Usage:

## Run clustering and classification in turn. Only need the topology file and the trajectory file.

```
python main_all.py --need_clustering=True --need_cal_feature=True --traj_file='your_traj.nc' --top_file='your_top.pdb' --n_clusters=cluster_number --batch_size=N --epochs=M --print_detail='True' --print_acc='True' --save_models='True' --atom_file='atom_name' --res_file='res_name'
```

## Run clustering with your feature file. Please provide the the topology file, the trajectory file, and the feature file.

```
python main_all.py --need_clustering=True --need_cal_feature=False --feature_file='your_feature' --traj_file='your_traj.nc' --top_file='your_traj.pdb' --n_clusters=cluster_number --batch_size=N --epochs=M --print_detail='True' --print_acc='True' --save_models='True' --atom_file='atom' --res_file='res'
```

## Train the classification model by self-provided label. Please provide the the topology file, the trajectory file, and the label file.

```
python main_all.py --need_clustering=False --need_cal_feature=False --labels_file='your_label.npy' --traj_file='your_traj.nc' --top_file='your_top.pdb' --n_clusters=cluster_number --batch_size=N --epochs=M --print_detail='True' --print_acc='True' --save_models='True' --atom_file='atom' --res_file='res'
```

## Example for the testing and tutorial usage. (Data for the tutorial can be found in the data/ folder)

```
python main_all.py --need_clustering=True --need_cal_feature=True --traj_file='test.nc' --top_file='test.pdb' --n_clusters=3 --batch_size=8 --epochs=10 --print_detail='True' --print_acc='True' --save_models='True' --atom_file='atom' --res_file='res'
```
