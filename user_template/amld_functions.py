import pandas as pd
from io import StringIO
from azure.storage.blob import BlockBlobService
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, precision_score, recall_score)


def plot_confmat(y_true, y_pred, save=True):
    labels=['fraud', 'normal']
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7,7))
    sns.heatmap(conf_matrix,
                xticklabels=labels,
                yticklabels=labels,
                annot=True,
                fmt='.0f')
    plt.title('Confusion matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    if save:
        plt.savefig('user_files/confusion_matrix.png')


def get_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred, pos_label='fraud')
    recall = recall_score(y_true, y_pred, pos_label='fraud')
    return (700 * precision + 300 * recall)


def get_data(sas_token):
    df = pd.read_csv(__get_data_blob(sas_token))
    print(df)
    return (df.drop('class', axis=1).values,
            df['class'].values)


def __get_data_blob(sas_token):
    try:
        bs = BlockBlobService(
                account_name='dsdemostorage',
                sas_token=sas_token,
            )
        return StringIO(bs.get_blob_to_text('amld', 'live_data.csv').content)
    except:
        return open('test_data.csv')