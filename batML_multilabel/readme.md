# Training and Running Multi-Label Classifiers


## Train Our Classifiers on Your Own Data

### Gather the Necessary Data
The data necessary for the detection can be found on Batdetective's website which is available [here](http://visual.cs.ucl.ac.uk/pubs/batDetective).
The data we used for the classification belongs to Natagora and is therefore not made available online. However, you can use your own labelled recordings to train our models. You will need to create a .npz file with the following fields: *train_files*, *train_durations*, *train_pos*, *train_class*, *test_files*, *test_durations*, *test_pos* and *test_class*.

- *train_files* is an array containing the filename of the recordings, without the extension.
- *train_durations* is an array containing the duration of each file present in *train_files*.
- *train_pos* is an array where each line is an array of the call positions in the respective file.
- *train_class* is an array where each line is an array containing the classes of the calls in the respective file.

The same holds for the test fields.

When two calls overlap at the same position, they need to be both inserted in the arrays as separate entries having the same position.

### Run Training and Evaluate the Model
The *run_training.py* file allows to train and evaluate any of our available architectures with your own data.
At the beginning of the main section of the file, you can define several parameters such as the name of the model you want to train. The trained model will be evaluated on the detection, the classification and on both tasks combined.
The performance is written in a text file and the trained model is saved in order to use it to classify new recordings.

The tuned hyperparameters we obtained when training our different architectures on Batdetective's and Natagora's data are available in the *data/* folder and will be used by default. If you want to change the hyperparameters you should modify the values in the .csv files of the *data/* folder.

Another important file where certain parameters need to be chosen is the *data_set_params.py* file.
- In this file, it is possible to perform tuning by setting the appropriate model variables to True and choosing the desired tuning time. Note that when the tuning is interrupted it will automatically resume to its last iteration when started again.
- To perform Hard Negative Mining during the training, the number of iterations should be indicated in the *num_hard_negative_mining* variable. By default no HNM is performed.
- To gain time, the *save_features_to_file* variable can be set to True so that the features are computed and saved once. The next time the features are needed by the models, the features will be loaded if the *load_features_from_file* variable is set to True.

To compile the .pyx files, run the *setup.py* file using the following command:
```
python setup.py build_ext --inplace
```

## Run Our Classifiers on Your Own Data
The *run_classifier.py* file allows to run an already trained model on new data to find calls and predict the corresponding species. At the beginning of the main section of the file, you can define various parameters such as the name of the model you want to use.

Our already trained models are available in the *data/models/* directory. Your own trained models will also be saved in this folder. To use one of your models, you have to change the value of the *date* and *hnm_iter* variables so that they correspond to the date present in the name of your model and to the number of HNM iterations used during training.


## System Requirements
The code takes between 5 minutes and 3.5 hours to run depending on the model. These measurements were made on a desktop with an i7-9800X CPU @ 3.80GHz, 32GB RAM and an RTX 2080 SUPER GPU on CentOS 8.1. The code has been designed to run using the following packages:
`Python 3.6`  
`CUDA 10.2`  
`cuDNN 7.6.5`  
`Cython 0.29.21`  
`hyperopt 0.2.5`  
`joblib 1.0.0`  
`numpy 1.16.4`  
`pandas 1.1.5`  
`scikit-image 0.17.2`  
`scikit-learn 0.24.1`  
`scikit-multilearn 0.2.0`  
`scipy 1.4.1`  
`tensorflow 2.1.0`  
`xgboost 1.4.0`  


## Acknowledgements
We would first like to thank our Master's thesis supervisors Mr. Siegfried Nijssen and Mr. Olivier Bonaventure for their guidance and valuable advice as well as their time.

This thesis could not have been completed without the work and datasets of [Batdetective](http://visual.cs.ucl.ac.uk/pubs/batDetective), which served as a starting point for our automated detection of bat species.

We would also like to express our gratitude to [Natagora](https://www.natagora.be/) and more specifically [Plecotus](https://plecotus.natagora.be/index.php?id=707) for the large amount of labelled bat call recordings they shared with us.

Our sincere thanks go to [Maxime Franco and Corrado Lipani](https://github.com/c-lipani/batmen) for the work they accomplished last year. We are especially thankful for their data preprocessing and their discussions with Natagora regarding the bat call recordings.

A special thanks to Maxime Piraux for his help on technical aspects as well as for his advice and time.

Last, but not least, many thanks to our families and friends for their support and encouragements.


## License
Everything that is made available on this GitHub repository can be used solely for non-commercial research purposes. Otherwise, the authors need to be contacted.
