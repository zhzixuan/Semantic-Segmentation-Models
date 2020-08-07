# File Structure
## Code
prototxt files (network structure)

scripts when training and testing

scripts when applying in scenes of reality

Hit: The file_root of all scripts needs to be modified to real file directory.

## Result

There are some trained model (caffemodel) with parameters and  saved predicted segmentation (.png).

Due to limited size of uploaded file, I uploaded the required caffemodel to the Baidu web disk, which can be downloaded from [here] (https://pan.baidu.com/s/1zsG-Mos7nzcg3kL2dF1sig), password is ajqj.  

## Demo
A short demo video. Please click [here] (https://pan.baidu.com/s/1NbxCDoiXTRO-sKe-bJZUAw) to download, password is rf2m. 

# Environment
The code is developed or applied under the following configurations.

**Hardware**
2-8 GPUs (with at least 12G GPU memories)

**Software**
Ubuntu `16.04.3 LTS`, CUDA `8.0`, caffe, python, and OpenCV

**Dataset**
ADE20K 


# Run
## Train
Before training, please prepare model structure (.prototxt), hyper parameter setting file (solver.prototxt) and model training script (train.sh). Modify them to suit actual condition of the hardware.

Makesure the names of all training set image are including in the ade_sceneparsing_train_im2cate.txt.

Prepare the ImageNet-pre-trained caffemodel when fine-tuning. Download [here] (https://pan.baidu.com/s/1CNCAEG4iwsXFUq0eoXsiyQ), password is 8m8q.

Enter the command in the terminal to train a model:
```sh
sh train.sh
```

## Evaluate and Test
Prepare the corresponding deploy.prototxt (network structure when testing) and validation.txt (names of test dataset).

Enter the command in the terminal to evaluate the model on test set:
```sh
python evaluate_seg.py
```

It will produce a folder named ‘predict’, which contains the predicted segmentation result. 

To visualize the grayscale result in different colors, run:
```sh
python color.py
```

For determine the score of mIOU, pixel accuracy and mean accuracy, run:
```sh
python score.py
```


# Reference 
[ADE20K] (http://groups.csail.mit.edu/vision/datasets/ADE20K/)
[py-RFCN-priv] (https://github.com/zhzixuan/py-RFCN-priv)
