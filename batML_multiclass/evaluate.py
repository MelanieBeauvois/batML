import numpy as np
import math

def remove_end_preds(nms_pos_o, nms_prob_o, gt_pos_o, gt_classes_o, pred_classes_o, durations, win_size):
    """
    Filters out predictions and ground truth calls that are close to the end.

    Parameters
    -----------
    nms_pos_o : ndarray
        Predicted positions of calls for every file.
    nms_prob_o : ndarray
        Confidence level of each prediction for every file.
    gt_pos_o : ndarray
        Ground truth positions of the calls for every file.
    gt_classes_o : ndarray
        Ground truth class for each file.
    pred_classes_o : ndarray
        Predicted class of each prediction for every file.
    durations : numpy array
        Durations of the wav files. 
    win_size : float
        Size of a window.

    Returns
    --------
    nms_pos : ndarray
        Predicted positions of calls for every file without the ones to close to the end of the file.
    nms_prob : ndarray
        Confidence level of each prediction for every file without the ones to close to the end of the file.
    gt_pos : ndarray
        Ground truth positions of the calls for every file without the ones to close to the end of the file.
    gt_classes : ndarray
        Ground truth class for each file without the ones to close to the end of the file.
    pred_classes : ndarray
        Predicted class of each prediction for every file without the ones to close to the end of the file.        
    """
    
    nms_pos = []
    nms_prob = []
    gt_pos = []
    gt_classes = []
    pred_classes = []
    for ii in range(len(nms_pos_o)):
        valid_time = durations[ii] - win_size
        gt_cur = gt_pos_o[ii]
        if gt_cur.shape[0] > 0:
            valid_gt = gt_cur[:, 0] < valid_time
            gt_pos.append(gt_cur[:, 0][valid_gt][..., np.newaxis])
            gt_classes.append(gt_classes_o[ii][:, 0][valid_gt][..., np.newaxis])
        else:
            gt_pos.append(gt_cur)
            gt_classes.append(gt_classes_o[ii])

        if len(nms_pos_o[ii]) > 0:
            valid_preds = nms_pos_o[ii] < valid_time
            nms_pos.append(nms_pos_o[ii][valid_preds])
            nms_prob.append(nms_prob_o[ii][valid_preds, 0][..., np.newaxis])
            pred_classes.append(pred_classes_o[ii][valid_preds])
        else:
            nms_pos.append(nms_pos_o[ii])
            nms_prob.append(nms_prob_o[ii][..., np.newaxis])
            pred_classes.append(pred_classes_o[ii])
    return nms_pos, nms_prob, gt_pos, gt_classes, pred_classes


def prec_recall_1d(nms_pos_o, nms_prob_o, gt_pos_o, pred_classes_o, gt_classes_o, durations, detection_overlap, win_size, nb_windows, filename, remove_eof=True):
    """
    Computes and saves the performance for detection, classification and the combination of both.

    Parameters
    -----------
    nms_pos_o : ndarray
        Predicted positions of calls for every file.
    nms_prob_o : ndarray
        Confidence level of each prediction for every file.
    gt_pos_o : ndarray
        Ground truth positions of the calls for every file.
    pred_classes_o : ndarray
        Predicted class of each prediction for every file.
    gt_classes_o : ndarray
        Ground truth class for each file.
    durations : numpy array
        Durations of the wav files.
    detection_overlap : float
        Maximum distance between a prediction and a ground truth to be considered as overlapping. 
    win_size : float
        Size of a window.
    nb_windows : ndarray
        Number of windows for every test file.
    filename : String
        Name of the file in which the performance will be saved.
    remove_eof : bool
        True if the predictions and ground truth calls that are close to the end should be filtered out.
    """

    if remove_eof:
        # filter out the detections in both ground truth and predictions that are too
        # close to the end of the file - dont count them during eval
        nms_pos, _, gt_pos, gt_classes, pred_classes = remove_end_preds(nms_pos_o, nms_prob_o, gt_pos_o, gt_classes_o, pred_classes_o, durations, win_size)
    else:
        nms_pos = nms_pos_o
        gt_pos = gt_pos_o
        gt_classes = gt_classes_o
        pred_classes = pred_classes_o

    conf_matrix = np.zeros((8,8), dtype=int)
    conf_matrix_detect = np.zeros((2,2), dtype=int)
    conf_matrix_classif = np.zeros((7,7), dtype=int)

    # loop through each file
    for ii in range(len(nms_pos)):
        gt_classes_ii = np.array([]) if len(gt_classes[ii])==0 else gt_classes[ii][:,0]

        # check to make sure the file contains some predictions
        num_preds = nms_pos[ii].shape[0]
        if num_preds > 0:
            distance_to_gt = np.abs(gt_pos[ii].ravel()-nms_pos[ii].ravel()[:, np.newaxis])
            within_overlap = (distance_to_gt <= detection_overlap)
            # lines=pred pos, col=gt pos, inside=true if distance btw pred and gt pos is <= detection overlap

            # True if the gt_pos overlaps with a predicted call having the correct class
            gt_found_correct = [False] * gt_pos[ii].shape[0]
            # True if the gt_pos overlaps with a predicted call but not of the correct class
            gt_found_incorrect = [False] * gt_pos[ii].shape[0]

            # loop on the predictions
            for jj in range(num_preds):
                # get the indices of all gt pos that overlap with pred pos jj
                inds = np.where(within_overlap[jj,:])[0]
                # some gt overlap with the preds
                if inds.shape[0] > 0:
                    # correct timing but not correct species
                    if (gt_classes_ii[inds] == pred_classes[ii][jj]).sum() == 0:
                        unique = np.unique(gt_classes_ii[inds], return_counts=False)
                        unique = unique.astype('int')
                        conf_matrix[unique,pred_classes[ii][jj]] += 1
                        conf_matrix_detect[1][1] += 1
                        conf_matrix_classif[unique-1,pred_classes[ii][jj]-1] += 1
                        for i_overlap in inds:
                            gt_found_incorrect[i_overlap] = True
                    # correct timing and correct species
                    else:
                        for i_overlap in inds: # one pred can overlap with several gt pos
                            # do not add to conf matrix if the gt pos was already overlapped by another pred pos
                            if gt_classes_ii[i_overlap]==pred_classes[ii][jj] and not gt_found_correct[i_overlap]:
                                conf_matrix[pred_classes[ii][jj],pred_classes[ii][jj]] += 1
                                conf_matrix_detect[1][1] += 1
                                conf_matrix_classif[pred_classes[ii][jj]-1,pred_classes[ii][jj]-1] += 1
                                gt_found_correct[i_overlap] = True
                # a bat call is predicted while there is no call
                else:
                    conf_matrix[0][pred_classes[ii][jj]] += 1
                    conf_matrix_detect[0][1] += 1
                            
            # gt pos that were not overlapped by any pred
            for i_gt in range(len(gt_found_correct)):
                if (not gt_found_correct[i_gt]) and (not gt_found_incorrect[i_gt]):
                    conf_matrix[gt_classes_ii[i_gt]][0] += 1
                    conf_matrix_detect[1][0] += 1
                    
        # no calls predicted so for all gt pos we wrongly predicted that there is no call
        else:
            for gt_c in gt_classes_ii:
                conf_matrix[gt_c][0] += 1
                conf_matrix_detect[1][0] += 1
        
    # add to the conf matrix the TP of class 0 for the current file
    nb_tp_0 = sum(nb_windows) - conf_matrix.sum()
    conf_matrix[0][0] = nb_tp_0
    conf_matrix_detect[0][0] = nb_tp_0

    compute_perf('detect + classif', conf_matrix, filename)
    compute_perf('detect', conf_matrix_detect, filename)
    compute_perf('classif', conf_matrix_classif, filename)

def compute_perf(perf_type, conf_matrix, filename):
    """
    Computes the performance based on the confusion matrix. Saves and displays them.

    Parameters
    -----------
    perf_type : String
        Can be one of 'detect', 'classif', 'detect + classif' in function of the given confusion matrix.
    conf_matrix : numpy array
        Confusion matrix of the model.
    filename : String
        Name of the file in which the performance will be saved.
    """

    with open(filename,'a') as f:
        f.write('\n--------------\n')
        f.write('Confusion matrix '+perf_type+'\n')
        f.write('--------------\n')
        f.write(''+str(conf_matrix)+'\n')
        print('--------------')
        print('Confusion matrix', perf_type)
        print('--------------')
        print(conf_matrix)

        TP = np.diag(conf_matrix)
        FP = conf_matrix.sum(axis=0) - TP
        FN = conf_matrix.sum(axis=1) - TP
        TN = conf_matrix.sum() - (TP + FN + FP)
        AC = (TP+TN)/(TP+FP+FN+TN).astype(float)
        PRE = TP/(TP+FP).astype(float)
        REC = TP/(TP+FN).astype(float)
        F1 = 2*(PRE*REC)/(PRE+REC)
        SPEC = TN/(TN+FP)
        BCR = (REC+SPEC)/2
        
        for i in range(conf_matrix.shape[0]):
            print('--------------')
            if perf_type == 'classif': print('Class', i+1, perf_type)
            else: print('Class', i, perf_type)
            print('--------------')
            print('Accuracy', AC[i])
            print('Precision', PRE[i])
            print('Recall', REC[i])
            print('F1',F1[i])
            print('BCR',BCR[i])
            print( )
            if perf_type=="detect" and i==1:
                f.write('--------------\n')
                f.write('Class '+ str(i) + ' ' + str(perf_type)+"\n")
                f.write('--------------\n')
                f.write('Accuracy '+ str(AC[i])+'\n')
                f.write('Precision '+ str(PRE[i])+'\n')
                f.write('Recall '+ str(REC[i])+'\n')
                f.write('F1 '+str(F1[i])+'\n')
                f.write('BCR '+str(BCR[i])+'\n')
    
        print('--------------')
        print('GLOBAL', perf_type)
        print('--------------')
        print('Average Accuracy', np.mean(AC))
        print('Average Precision', np.mean(PRE))
        print('Average Recall', np.mean(REC))
        print('Average F1', np.mean(F1))
        print('Average BCR', np.mean(BCR))
        print('KAPPA', compute_KAPPA(conf_matrix), r"\\")
        print('CEN',compute_CEN(conf_matrix), r"\\")
        if perf_type!='detect':
            f.write('--------------\n')
            f.write('GLOBAL '+perf_type+'\n')
            f.write('--------------\n')
            f.write('Average Accuracy '+str(np.mean(AC))+'\n')
            f.write('Average Precision '+str(np.mean(PRE))+'\n')
            f.write('Average Recall '+str(np.mean(REC))+'\n')
            f.write('Average F1 '+str(np.mean(F1))+'\n')
            f.write('Average BCR '+str(np.mean(BCR))+'\n')
            f.write('KAPPA '+str(compute_KAPPA(conf_matrix))+'\n')
            f.write('CEN '+str(compute_CEN(conf_matrix))+'\n')


def compute_KAPPA(conf_matrix):
    """
    Computes the Cohen's Kappa score.

    Parameters
    -----------
    conf_matrix : numpy array
        Confusion matrix of the model.
    
    Returns
    --------
    KAPPA : float
        The Cohen's Kappa score.
    """
    tot = conf_matrix.sum()
    P_gt = np.sum(conf_matrix,axis=1) / tot
    P_pred = np.sum(conf_matrix,axis=0) / tot
    chanceAgree = np.sum(P_gt * P_pred)
    agree = np.trace(conf_matrix) / tot
    KAPPA = (agree - chanceAgree) / (1 - chanceAgree)
    return KAPPA

def compute_CEN(CM):
    """
    Computes the Confusion Entropy.

    Parameters
    -----------
    CM : numpy array
        Confusion matrix of the model.

    Returns
    --------
    CEN_total : float
        The Confusion Entropy of the model.
    """
    nb_classes = CM.shape[0]
    P_d = np.zeros((nb_classes, nb_classes), dtype=float)
    P_g = np.zeros((nb_classes, nb_classes), dtype=float)
    P = np.zeros((nb_classes), dtype=float)
    CEN = np.zeros((nb_classes), dtype=float)
    for i in range(nb_classes):
        for j in range(nb_classes):
            if i != j:
                P_d[i][j] = CM[i][j] / (np.sum(CM, axis=0)[j] + np.sum(CM, axis=1)[j])
                P_g[i][j] = CM[i][j] / (np.sum(CM, axis=0)[i] + np.sum(CM, axis=1)[i])
    for j in range(nb_classes):
        for k in range(nb_classes):
            if P_g[j][k]!=0:
                CEN[j] -= P_g[j][k]*math.log(P_g[j][k], 2*(nb_classes-1))
            if P_d[k][j]!=0 :
                CEN[j] -= P_d[k][j]*math.log(P_d[k][j], 2*(nb_classes-1))
        P[j] = (np.sum(CM, axis=0)[j] + np.sum(CM, axis=1)[j]) / (2*CM.sum())
    
    CEN_total = (P * CEN).sum()
    return CEN_total
