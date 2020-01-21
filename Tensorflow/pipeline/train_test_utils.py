import pandas as pd
import numpy as np

import graph_style

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def get_confusion_matrix(true_values, pred_values):
    '''
    Function to calculate metrics from model output

    INPUT:  true_values [pd.DataSeries/Array/List]  1D input of target 
                                                    values
            pred_values [pd.DataSeries/Array/List]  1D input of model 
                                                    prediction

    OUTPUT: df          [pd.DataFrame]  df with confusion matrix 
                                        (+extras) for different 
                                        thresholds
    '''
    df_conf_mat = pd.DataFrame()
    df_conf_mat['True'] = true_values
    df_conf_mat['Pred'] = pred_values
    df_thres_overall = pd.DataFrame()
    for threshold in [i/20 for i in range(0,20)]:
        df_thres = pd.DataFrame()
        TP = len(df_conf_mat[df_conf_mat['True']==1]
                [df_conf_mat['Pred']>=threshold])
        FP = len(df_conf_mat[df_conf_mat['True']==0]
                [df_conf_mat['Pred']>=threshold])
        TN = len(df_conf_mat[df_conf_mat['True']==0]
                [df_conf_mat['Pred']<threshold])
        FN = len(df_conf_mat[df_conf_mat['True']==1]
                [df_conf_mat['Pred']<threshold])
        df_thres['Threshold'] = [threshold]
        df_thres['TP'] = [TP]
        df_thres['FP'] = [FP]
        df_thres['TN'] = [TN]
        df_thres['FN'] = [FN]
        try:
            df_thres['Recall'] = [TP/(TP+FN)]
        except ZeroDivisionError:
            df_thres['Recall'] = -1
        try:
            df_thres['Precision'] = [TP/(TP+FP)]
        except ZeroDivisionError:
            df_thres['Precision'] = -1
        try:
            df_thres['MCC'] = (TP*TN-FP*FN) / \
                                ((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5
        except ZeroDivisionError:
            df_thres['MCC'] = 0
        df_thres_overall = pd.concat([df_thres_overall, df_thres])
    df_thres_overall['FP_Rate'] = df_thres_overall['FP'] / \
                            (df_thres_overall['FP'] + df_thres_overall['TN'])
    df_thres_overall['F1'] = (2* df_thres_overall['Recall']*
                            df_thres_overall['Precision'])/ \
                    (df_thres_overall['Recall']+df_thres_overall['Precision'])
    return df_thres_overall

def show_cm_graph(df_LSTM_cm):
    '''
    Function that displays three graphs: ROC, PR and how precision and
    recall change for different model prediction thresholds

    INPUT: df_LSTM_cm   [pd.DataFrame]      df with scores for various 
                                            thresholds

    OUTPUT: fig1        [plt.fig object]    figure object of ROC and PR 
                                            curves
            fig2        [plt.fig object]    figure object of PR 
                                            threshold bar chart
    '''
    fig1, (ax1,ax2) = plt.subplots(1,2, figsize=(16,8))

    ax1.plot(df_LSTM_cm['FP_Rate'] , df_LSTM_cm['Recall'] )
    ax1.plot([0,1], [0,1], ls=':')
    ax1.set_title('ROC', fontsize = 15)
    ax1.set_xlabel('False Positive Rate', fontsize = 15)
    ax1.set_ylabel('True Positive Rate', fontsize = 15)
    ax1.set_xlim(0,1)
    ax1.set_ylim(0,1)

    ax1.set_aspect('equal')

    # plt.show()

    ax2.plot(df_LSTM_cm[df_LSTM_cm['Precision']!=-1]['Recall'],
             df_LSTM_cm[df_LSTM_cm['Precision']!=-1]['Precision'] )
    ax2.set_title('PR', fontsize = 15)
    ax2.set_xlabel('Recall', fontsize = 15)
    ax2.set_ylabel('Precision', fontsize = 15)
    ax2.set_xlim(0,1)
    ax2.set_ylim(0,1)

    ax2.set_aspect('equal')
    plt.show()

    bar_index = [i for i in range(len(df_LSTM_cm))]

    values_precision = df_LSTM_cm['Precision']
    values_recall = df_LSTM_cm['Recall']
    values_F1 = df_LSTM_cm['F1']
    values_MCC = df_LSTM_cm['MCC'].fillna(0)

    ax1colour, ax2colour, ax3colour, ax4colour = \
                                        graph_style.colour_dict['default'][0:4]

    fig2,ax = plt.subplots(figsize=(16,8))
    bar_index = np.array(bar_index)
    precision_bar = ax.bar(bar_index-0.3, values_precision, 0.2,
                            facecolor = ax1colour)
    recall_bar = ax.bar(bar_index-0.1, values_recall, 0.2,facecolor = ax2colour)
    F1_bar = ax.bar(bar_index+0.1, values_F1, 0.2,facecolor = ax3colour)
    MCC_bar = ax.bar(bar_index+0.3, values_MCC, 0.2,facecolor = ax4colour)

    ax.set_ylabel("Proportion", weight = 'light',fontsize=15)
    ax.set_xlabel("Thresholds", weight = 'light',fontsize=15)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_xticks([i for i in range(0,len(df_LSTM_cm)+1,2)])
    ax.set_xticklabels((i/10 for i in range(int(len(df_LSTM_cm)/2)+1)),
                         weight = 'light',fontsize=15)
    ax.set_xlim(-0.5,len(df_LSTM_cm))
    ax.set_ylim(0,1)
    lgd = ax.legend((precision_bar[0], recall_bar[0],F1_bar[0], MCC_bar[0] ),
                     ('Precision', 'Recall', 'F1', 'MCC'),
                        fontsize=15, bbox_to_anchor=(1, 0.9))

    plt.show()
    return fig1, fig2


def get_tf_model(n_features, activation):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.layers import Input, Dense

    input_layer = Input(shape =(n_features,))
    hidden_layer_1 = Dense(10, activation = activation)(input_layer)
    hidden_layer_2 = Dense(20, activation = activation)(hidden_layer_1)
    hidden_layer_3 = Dense(10, activation = activation)(hidden_layer_2)
    output_layer = Dense(1, activation = 'sigmoid')(hidden_layer_3)

    model = keras.models.Model(input_layer, output_layer)

    return model
