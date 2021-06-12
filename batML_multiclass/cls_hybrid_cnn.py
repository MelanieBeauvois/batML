import numpy as np
from scipy.io import wavfile
import pyximport; pyximport.install()
from os import path
import time
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.utils import class_weight
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.preprocessing import MinMaxScaler
import gc
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Model

import nms as nms
from spectrogram import compute_features_spectrogram
from cnn_helper import network_fit, tune_network
from svm_xgb_helper import tune_svm, tune_xgb
from models_params_helper import params_to_dict


class NeuralNet:

    def __init__(self, params_):
        """
        Creates a CNN for features computation and an SVM or XGBoost model for detection and classification.

        Parameters
        -----------
        params_ : DataSetParams
            Parameters of the model.
        """

        self.params = params_
        self.network_features = None
        self.model_feat = None
        self.network_classif = None
        self.scaler = None

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

        # free memory
        if self.params.classification_model == "hybrid_cnn_xgboost" and self.network_classif is not None:
            self.network_classif._Booster.__del__()
            tf.keras.backend.clear_session()
            self.network_features = None
            self.model_feat = None
            self.network_classif = None
            self.scaler = None
            gc.collect()

        # compute or load the features of the training files and the associated class label.
        print("Compute or load features")
        feats = []
        labs = []
        tic = time.time()
        for ii, file_name in enumerate(files):
            if positions[ii].shape[0] > 0:
                local_feats = self.create_or_load_features("classification", file_name)
                # convert time in file to integer
                positions_ratio = positions[ii] / durations[ii]
                train_inds = (positions_ratio*float(local_feats.shape[0])).astype('int')
                feats.append(local_feats[train_inds, :, :, :])
                labs.append(class_labels[ii])
        # flatten list of lists and set to correct output size
        features = np.vstack(feats)
        labels = np.hstack(labs)
        toc = time.time()
        self.params.features_computation_time += toc-tic

        # tune the hyperparameters and fit the features CNN
        if self.params.tune_cnn_8:
            print("Tune features CNN")
            tic_cnn_8 = time.time()
            tune_network(self.params, features, labels, self.params.trials_filename_1)
            toc_cnn_8 = time.time()
            while toc_cnn_8-tic_cnn_8 < self.params.tune_time:
                tune_network(self.params, features, labels, self.params.trials_filename_1)
                toc_cnn_8 = time.time()
            print('total tuning time', round(toc_cnn_8-tic_cnn_8, 3), '(secs) =', round((toc_cnn_8-tic_cnn_8)/60,2), r"min \\")
        
        self.network_features, _ = network_fit(self.params, features, labels, 8)

        # extracting features from last layer of the features CNN
        self.model_feat = Model(inputs=self.network_features.input,
                                outputs=self.network_features.layers[len(self.network_features.layers)-3].output)
        features = features.reshape(features.shape[0], features.shape[2], features.shape[3], 1)
        feat_train = self.model_feat.predict(features)

        # train and tune the svm classification model
        if self.params.classification_model == "hybrid_cnn_svm":
            self.scaler = MinMaxScaler()
            feat_train = self.scaler.fit_transform(feat_train)
            if self.params.tune_svm_spectrogram:
                print("Tune SVM")
                tic_svm = time.time()
                tune_svm(self.params, feat_train, labels, self.params.trials_filename_2)
                toc_svm = time.time()
                while toc_svm-tic_svm < self.params.tune_time:
                    tune_svm(self.params, feat_train, labels, self.params.trials_filename_2)
                    toc_svm = time.time()
                print('total tuning time', round(toc_svm-tic_svm, 3), '(secs) =', round((toc_svm-tic_svm)/60,2), r"min \\")
            
            print("Fit SVM")
            tic = time.time()
            self.network_classif = SVC( kernel=self.params.kernel, C=self.params.C, degree=self.params.degree,
                                        gamma=self.params.gamma_svm, class_weight=self.params.class_weight,
                                        probability=True, verbose=False, max_iter=self.params.max_iter)
            self.network_classif.fit(feat_train, labels)
            toc = time.time()
            print('total SVM run time', round(toc-tic, 3), '(secs) =', round((toc-tic)/60,2), r"min \\")
            print("CNN and SVM params= ", params_to_dict(self.params))
        
        # train and tune the xgb classification model
        elif self.params.classification_model == "hybrid_cnn_xgboost":
            if self.params.tune_xgboost_spectrogram:
                print("Tune xgboost")
                tic_xgb = time.time()
                tune_xgb(self.params, feat_train, labels, self.params.trials_filename_2)
                toc_xgb = time.time()
                while toc_xgb-tic_xgb < self.params.tune_time:
                    tune_xgb(self.params, feat_train, labels, self.params.trials_filename_2)
                    toc_xgb = time.time()
                print('total tuning time', round(toc_xgb-tic_xgb, 3), '(secs) =', round((toc_xgb-tic_xgb)/60,2), r"min \\")
            
            print("Fit xgboost")
            tic = time.time()
            self.network_classif = xgb.XGBClassifier(eta=self.params.eta, min_child_weight=self.params.min_child_weight,
                                                    max_depth=self.params.max_depth, n_estimators=self.params.n_estimators,
                                                    gamma=self.params.gamma_xgb, subsample=self.params.subsample,
                                                    scale_pos_weight=self.params.scale_pos_weight, objective="multi:softprob",
                                                    tree_method='gpu_hist')
            train_feat, val_feat, train_labels, val_labels = train_test_split(feat_train, labels,
                                                        test_size=0.1, random_state=1, stratify=labels)
            class_weights = list(class_weight.compute_class_weight('balanced', np.unique(train_labels),train_labels))
            class_w = np.ones(train_labels.shape[0], dtype = 'float')
            for i, val in enumerate(train_labels):
                class_w[i] = class_weights[val]
            self.network_classif.fit(train_feat, train_labels, eval_metric=['mlogloss'], sample_weight=class_w,
                        early_stopping_rounds=self.params.n_estimators//10, eval_set=[(val_feat,val_labels)],
                        verbose=False)
            
            history_xgb = self.network_classif.evals_result()
            self.params.n_estimators = int(np.argmin(history_xgb['validation_0']['mlogloss']) + 1)
            toc = time.time()
            print('total xgboost run time', round(toc-tic, 3), '(secs) =', round((toc-tic)/60,2), r"min \\")
            print("XGBoost params= ", params_to_dict(self.params))

    
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

        # compute features
        tic = time.time()
        features = self.create_or_load_features(goal, file_name, audio_samples, sampling_rate)
        toc=time.time()
        self.params.features_computation_time += toc - tic
        features = features.reshape(features.shape[0], features.shape[2], features.shape[3], 1)
        tic = time.time()
        feat_test = self.model_feat.predict(features)
        toc=time.time()
        self.params.detect_time += toc - tic
        
        # perform classification
        tic = time.time()
        if self.params.classification_model == "hybrid_cnn_xgboost":
            y_predictions = self.network_classif.predict_proba(feat_test)
        elif self.params.classification_model == "hybrid_cnn_svm":
            feat_test = self.scaler.transform(feat_test)
            y_predictions = self.network_classif.predict_proba(feat_test)
        toc=time.time()
        self.params.classif_time += toc - tic

        # smooth the output prediction per column so smooth each class prediction over time
        tic = time.time()
        if self.params.smooth_op_prediction:
            y_predictions = gaussian_filter1d(y_predictions, self.params.smooth_op_prediction_sigma, axis=0)
        
        # trying to get rid of rows with 0 highest
        call_predictions_bat = y_predictions[:,1:]
        call_predictions_not_bat = y_predictions[:,0]
        high_preds = np.array([np.max(x) for x in call_predictions_bat])[:, np.newaxis]
        pred_classes = np.array([np.argmax(x)+1 for x in call_predictions_bat])[:, np.newaxis]
        
        # perform non max suppression
        pos, prob, pred_classes, call_predictions_not_bat = nms.nms_1d(high_preds[:,0].astype(np.float),
                                    pred_classes, call_predictions_not_bat, self.params.nms_win_size, file_duration)

        # remove pred that have a higher probability of not being a bat
        pos_bat = []
        prob_bat = []
        pred_classes_bat = []
        for i in range(len(pos)):
            if prob[i][0]>call_predictions_not_bat[i]:
                pos_bat.append(pos[i])
                prob_bat.append(prob[i])
                pred_classes_bat.append(pred_classes[i])
        toc=time.time()
        self.params.nms_computation_time += toc - tic

        nms_pos = np.array(pos_bat)
        nms_prob = np.array(prob_bat)
        pred_classes = np.array(pred_classes_bat)
        nb_windows = features.shape[0]
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
            if self.params.load_features_from_file and path.exists(self.params.feature_dir + data_set + '_' + file_name.split("/")[-1]  + '_spectrogram' + '.npy'):
                features = np.load(self.params.feature_dir + data_set + '_' + file_name.split("/")[-1] + '_spectrogram'  + '.npy')
            # 3) computes features from file name
            else:
                if self.params.load_features_from_file: print("missing features have to be computed ", self.params.feature_dir + data_set + '_' + file_name.split("/")[-1]  + '_spectrogram' )
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
            np.save(self.params.feature_dir + data_set + '_' + file_name.split("/")[-1]  + '_spectrogram', features)
