import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from scipy.io import wavfile
import pyximport; pyximport.install()
from os import path
import time
import pickle

import nms_cnn2 as nms
from hyperopt import hp, tpe, fmin, space_eval, Trials
from spectrogram import compute_features_spectrogram
from cnn_helper import obj_func_cnn, network_fit



class NeuralNet:

    def __init__(self, params_):
        """
        Creates a CNN for detection and a CNN for classification.

        Parameters
        -----------
        params_ : DataSetParams
            Parameters of the model.
        """
        self.params = params_
        self.network_detect = None
        self.network_classif = None

    def train(self, positions, class_labels, files, durations):
        """
        Takes the file names and ground truth call positions and trains model.

        Parameters
        -----------
        positions : ndarray
            Training positions for each training file.
        class_labels : numpy array
            Class label for each training position.
        files : numpy array
            Names of the wav files used to train the model.        
        durations : numpy array
            Durations of the wav files used to train the model.
        """

        # compute or load the features of the training files and the associated class label.
        print("Compute or load features")
        tic = time.time()
        features_detect, labels_detect = self.features_labels_from_file(positions["detect"], class_labels["detect"], files["detect"],
                                                                        durations["detect"], "detection")
        features_classif, labels_classif = self.features_labels_from_file(positions["classif"], class_labels["classif"], files["classif"],
                                                                        durations["classif"], "classification")
        # the cnn has 7 classes from 0 to 6 but ours go from 1 to 7 => -1 for train and +1 in test
        labels_classif -= 1 
        toc = time.time()
        self.params.features_computation_time += toc-tic

        # tuning of the hyperparameters of the two CNNs
        if self.params.tune_cnn_2:
            print("Tune CNN detect")
            tic_cnn_2 = time.time()
            best_space_detect = self.tune_network(features_detect, labels_detect, self.params.trials_filename_1, goal="detection")
            toc_cnn_2 = time.time()
            while toc_cnn_2-tic_cnn_2 < self.params.tune_time:
                best_space_detect = self.tune_network(features_detect, labels_detect, self.params.trials_filename_1, goal="detection")
                toc_cnn_2 = time.time()
            print('total tuning time', round(toc_cnn_2-tic_cnn_2, 3), '(secs) =', round((toc_cnn_2-tic_cnn_2)/60,2), r"min \\")
        if self.params.tune_cnn_7:
            print("Tune CNN classif")
            tic_cnn_7 = time.time()
            best_space_classif = self.tune_network(features_classif, labels_classif, self.params.trials_filename_2, goal="classification")
            toc_cnn_7 = time.time()
            while toc_cnn_7-tic_cnn_7 < self.params.tune_time:
                best_space_classif = self.tune_network(features_classif, labels_classif, self.params.trials_filename_2, goal="classification")
                toc_cnn_7 = time.time()
            print('total tuning time', round(toc_cnn_7-tic_cnn_7, 3), '(secs) =', round((toc_cnn_7-tic_cnn_7)/60,2), r"min \\")
        
        # fit the two CNN
        print("Fit the two CNNs")
        self.network_detect, _ = network_fit(self.params, features_detect, labels_detect,  2, '_1')
        self.network_classif, _ = network_fit(self.params, features_classif, labels_classif, 7, '_2')

        if self.params.tune_cnn_2:
            print("best_space_detect =", best_space_detect)
        if self.params.tune_cnn_7:
            print("best_space_classif =", best_space_classif)

    def tune_network(self, features, labels, trials_filename, goal):
        """
        Tunes the network with hyperopt.

        Parameters
        -----------
        features : ndarray
            Array containing the spectrogram features for each window of the audio file.
        labels : numpy array
            Class label (0-7) for each training position.
        trials_filename : String
            Name of the file where the previous iterations of hyperopt are saved.
        goal : String
            Indicates whether the network needs to be tuned for detection or classification.
            Can be either "detection" or "classification".
        
        Returns
        --------
        best_space : dict
            Best hyperparameters found so far for the CNN.
        """
        
        space_cnn = { 'nb_conv_layers': hp.choice('nb_conv_layers', range(1,4)),
                    'nb_dense_layers': hp.choice('nb_dense_layers', range(1,5)),
                    'nb_filters': hp.choice('nb_filters', range(16, 65, 8)),
                    'filter_size': hp.choice('filter_size', range(2,6)),
                    'pool_size': 2,
                    'nb_dense_nodes': hp.choice('nb_dense_nodes', range(64, 513, 64)),
                    'dropout_proba': hp.choice('dropout_proba', np.arange(0.3, 0.8, 0.1)),
                    'learn_rate_adam': hp.choice('learn_rate_adam', np.logspace(-5, -2, num=15)),
                    'beta_1': hp.choice('beta_1', [0.8, 0.9, 0.95]),
                    'beta_2': hp.choice('beta_2', [0.95, 0.999]),
                    'epsilon': hp.choice('epsilon', [1e-8]),
                    'min_delta': hp.choice('min_delta', [0.00005, 0.0005, 0.005]),
                    'patience': hp.choice('patience', [5, 10, 15, 20]),
                    'batchsize': hp.choice('batchsize', range(64, 513, 64)),
                    'features': features,
                    'labels': labels,
                    'nb_output': (2 if goal=="detection" else 7)
                    }
        
        
        # load the saved trials
        try:
            trials = pickle.load(open(trials_filename+".hyperopt", "rb"))
            max_trials = len(trials.trials) + 1
        # create a new trials
        except:
            max_trials = 1
            trials = Trials()

        # optimise the objective function with the defined set of CNN parameters
        best_space_indices = fmin(obj_func_cnn, space_cnn, trials=trials,algo=tpe.suggest, max_evals=max_trials)
        best_space = space_eval(space_cnn, best_space_indices)
        best_space = {k: best_space[k] for k in best_space.keys() - {'features', 'labels'}}
        with open(trials_filename + ".hyperopt", "wb") as f:
            pickle.dump(trials, f)

        nb_cnn = (1 if goal=="detection" else 2)
        # CNN
        setattr(self.params, "nb_conv_layers_"+str(nb_cnn), best_space['nb_conv_layers'])
        setattr(self.params, "nb_dense_layers_"+str(nb_cnn), best_space['nb_dense_layers'])
        setattr(self.params, "nb_filters_"+str(nb_cnn), best_space['nb_filters'])
        setattr(self.params, "filter_size_"+str(nb_cnn), best_space['filter_size'])
        setattr(self.params, "pool_size_"+str(nb_cnn), 2)
        setattr(self.params, "nb_dense_nodes_"+str(nb_cnn), best_space['nb_dense_nodes'])
        setattr(self.params, "dropout_proba_"+str(nb_cnn), best_space['dropout_proba'])
        # Adam
        setattr(self.params, "learn_rate_adam_"+str(nb_cnn), best_space['learn_rate_adam'])
        setattr(self.params, "beta_1_"+str(nb_cnn), best_space['beta_1'])
        setattr(self.params, "beta_2_"+str(nb_cnn), best_space['beta_2'])
        setattr(self.params, "epsilon_"+str(nb_cnn), best_space['epsilon'])
        # early stopping
        setattr(self.params, "min_delta_"+str(nb_cnn), best_space['min_delta'])
        setattr(self.params, "patience_"+str(nb_cnn), best_space['patience'])
        # fit
        setattr(self.params, "batchsize_"+str(nb_cnn), best_space['batchsize'])

        return best_space


    def features_labels_from_file(self, positions, class_labels, files, durations, goal):
        """
        Computes or loads the features of each position of the files
        and indicates the associated class label.

        Parameters
        -----------
        positions : ndarray
            Training positions for each file.
        class_labels : numpy array
            Class label for each position.
        files : numpy array
            Names of the wav files.        
        durations : numpy array
            Durations of the wav files.
        goal : String
            Indicates whether the network needs to be tuned for detection or classification.
            Can be either "detection" or "classification".

        Returns
        --------
        features : ndarray
            Array containing the spectrogram features for each position of the audio files.
        labels : ndarray
            Class label (0-7) for each position of the audio files.
        """
        
        feats = []
        labs = []
        for ii, file_name in enumerate(files):
            if positions[ii].shape[0] > 0:
                local_feats = self.create_or_load_features(goal, file_name)
                # convert time in file to integer
                positions_ratio = positions[ii] / durations[ii]
                train_inds = (positions_ratio*float(local_feats.shape[0])).astype('int')
                feats.append(local_feats[train_inds, :, :, :])
                if goal == "detection": 
                    labs.append(class_labels[ii])
                elif goal == "classification":
                    labs.append(class_labels[ii])
        # flatten list of lists and set to correct output size
        features = np.vstack(feats)
        labels = np.hstack(labs)
        return features, labels
    
    def test(self, goal, file_name=None, file_duration=None, audio_samples=None, sampling_rate=None):
        """
        Makes a prediction on the position, probability and class of the calls present in an audio file.
        
        Parameters
        -----------
        goal : String
            Indicates whether the file needs to be tested for detection or classification.
            Can be either "detection" or "classification".
        file_name : String
            Name of the wav file used to make a prediction.
        file_duration : float
            Duration of the wav file used to make a prediction.
        audio_samples : numpy array
            Data read from wav file.
        sampling_rate : int
            Sample rate of wav file.

        Returns
        --------
        nms_pos : ndarray
            Predicted positions of calls for every test file.
        nms_prob : ndarray
            Confidence level of each prediction for every test file.
        pred_classes : ndarray
            Predicted class of each prediction for every test file.
        nb_windows : ndarray
            Number of windows for every test file.
        """

        # compute features and perform detection
        tic = time.time()
        features = self.create_or_load_features(goal, file_name, audio_samples, sampling_rate)
        toc=time.time()
        self.params.features_computation_time += toc - tic
        features = features.reshape(features.shape[0], features.shape[2], features.shape[3], 1)
        nb_windows = features.shape[0]
        tic = time.time()
        y_predictions_detect = self.network_detect.predict(features)
        toc=time.time()
        self.params.detect_time += toc - tic

        # smooth the output prediction per column so smooth each class prediction over time
        tic = time.time()
        if self.params.smooth_op_prediction:
            y_predictions_detect = gaussian_filter1d(y_predictions_detect, self.params.smooth_op_prediction_sigma, axis=0)
        
        # trying to get rid of rows with 0 highest
        call_predictions_bat = y_predictions_detect[:,1:]
        call_predictions_not_bat = y_predictions_detect[:,0]
        high_preds = np.array([np.max(x) for x in call_predictions_bat])[:, np.newaxis]
        pred_classes = np.array([np.argmax(x)+1 for x in call_predictions_bat])[:, np.newaxis]
        
        # perform non max suppression
        pos, prob, pred_classes, call_predictions_not_bat, features = nms.nms_1d(high_preds[:,0].astype(np.float), pred_classes, 
                                                                    call_predictions_not_bat, features, self.params.nms_win_size, file_duration)
        
        # remove pred that have a higher probability of not being a bat
        pos_bat = []
        prob_bat = []
        pred_classes_bat = []
        features_bat = []
        for i in range(len(pos)):
            if prob[i][0]>call_predictions_not_bat[i]:
                pos_bat.append(pos[i])
                prob_bat.append(prob[i])
                pred_classes_bat.append(pred_classes[i])
                features_bat.append(features[i])
        toc=time.time()
        self.params.nms_computation_time += toc - tic

        # perform classification
        tic = time.time()
        pred_proba = np.array([])
        pred_classes = np.array([])
        if np.array(features_bat).shape[0] != 0:
            y_predictions_classif = self.network_classif.predict(np.array(features_bat))
            pred_proba = np.array([np.max(x) for x in y_predictions_classif])[:, np.newaxis]
            pred_classes = np.array([np.argmax(x)+1 for x in y_predictions_classif])[:, np.newaxis]
        toc=time.time()
        self.params.classif_time += toc - tic

        nms_pos = np.array(pos_bat)
        nms_prob = pred_proba
        return  nms_pos, nms_prob, pred_classes, nb_windows

    def create_or_load_features(self, goal, file_name=None, audio_samples=None, sampling_rate=None):
        """
        Does 1 of 3 possible things
        1) computes feature from audio samples directly
        2) loads feature from disk OR
        3) computes features from file name

        Parameters
        -----------
        goal : String
            Indicates whether the features are used for detection or classification.
            Can be either "detection" or "classification".
        file_name : String
            Name of the wav file used to make a prediction.
        audio_samples : numpy array
            Data read from wav file.
        sampling_rate : int
            Sample rate of wav file.

        Returns
        --------
        features : ndarray
            Array containing the spectrogram features for each window of the audio file.
        """

        if goal == "detection":
            audio_dir = self.params.audio_dir_detect
            data_set = self.params.data_set_detect
        elif goal =="classification":
            audio_dir = self.params.audio_dir_classif
            data_set = self.params.data_set_classif

        # 1) computes feature from audio samples directly
        if file_name is None:
            features = compute_features_spectrogram(audio_samples, sampling_rate, self.params)
        else:
            # 2) loads feature from disk
            if self.params.load_features_from_file and path.exists(self.params.feature_dir + data_set + '_' + file_name.split("/")[-1] + '_spectrogram' + '.npy'):
                features = np.load(self.params.feature_dir + data_set + '_' + file_name.split("/")[-1] + '_spectrogram' + '.npy')
            # 3) computes features from file name
            else:
                if self.params.load_features_from_file: print("missing features have to be computed")
                sampling_rate, audio_samples = wavfile.read(audio_dir + file_name.split("/")[-1]  + '.wav')
                features = compute_features_spectrogram(audio_samples, sampling_rate, self.params)
                if self.params.save_features_to_file or self.params.load_features_from_file:
                    np.save(self.params.feature_dir + data_set + '_' + file_name.split("/")[-1] + '_spectrogram', features)
        return features

    def save_features(self, goal, files):
        """
        Computes and saves features to disk.

        Parameters
        ----------
        goal : String
            Indicates whether the features are computed for detection or classification.
            Can be either "detection" or "classification".
        files : String
            Name of the wav file used to make a prediction.
        """
        
        if goal == "detection":
            audio_dir = self.params.audio_dir_detect
            data_set = self.params.data_set_detect
        elif goal =="classification":
            audio_dir = self.params.audio_dir_classif
            data_set = self.params.data_set_classif

        for file_name in files:
            sampling_rate, audio_samples = wavfile.read(audio_dir + file_name.split("/")[-1] + '.wav')
            features = compute_features_spectrogram(audio_samples, sampling_rate, self.params)
            np.save(self.params.feature_dir + data_set + '_' + file_name.split("/")[-1] + '_spectrogram', features)
