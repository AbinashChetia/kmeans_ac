import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def splitTrainTest(x, y, train_ratio=0.8):
    '''
    Split data into training and testing sets.
    '''
    df_x = x.copy()
    df_y = y.copy()
    df_y = df_y.rename('y')
    df = pd.concat([df_x, df_y], axis=1)
    df = df.sample(frac=1).reset_index(drop=True)
    train_size = int(len(df) * train_ratio)
    train_x = df.iloc[:train_size, :-1].reset_index(drop=True)
    train_y = df.iloc[:train_size, -1].reset_index(drop=True)
    test_x = df.iloc[train_size:, :-1].reset_index(drop=True)
    test_y = df.iloc[train_size:, -1].reset_index(drop=True)
    return train_x, train_y, test_x, test_y

def get_performance_measure(y, pred):
    if np.unique(y).shape[0] == 2:
        tp, tn, fp, fn = 0, 0, 0, 0
        classes = np.unique(y)
        p_class = classes.max()
        n_class = classes.min()
        for i in range(len(y)):
            if y[i] == p_class and pred[i] == p_class:
                tp += 1
            elif y[i] == n_class and pred[i] == n_class:
                tn += 1
            elif y[i] == n_class and pred[i] == p_class:
                fp += 1
            elif y[i] == p_class and pred[i] == n_class:
                fn += 1
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        spec = tn / (tn + fp)
        f1 = 2 * precision * recall / (precision + recall)
        acc = (tp + tn) / (tp + tn + fp + fn)
        return {'tp': tp,
                'tn': tn,
                'fp': fp,
                'fn': fn,
                'precision': precision,
                'recall': recall,
                'spec': spec,
                'f1': f1,
                'acc': acc}
    elif np.unique(y).shape[0] > 2:
        acc = 0
        for i in range(len(y)):
            if y[i] == pred[i]:
                acc += 1
        acc /= len(y)
        return {'acc': acc}

def split_kfold(x, y, k=5):
    '''
    Split data into training and testing sets for k-fold cross validation.
    '''
    df_x = x.copy()
    df_y = y.copy()
    df_y = df_y.rename('y')
    df = pd.concat([df_x, df_y], axis=1)
    df = df.sample(frac=1).reset_index(drop=True)
    fold_size = int(len(df) / k)
    data_folds = []
    for i in range(k):
        if i != k - 1:
            data_folds.append(df.iloc[i * fold_size: (i + 1) * fold_size, :].reset_index(drop=True))
        else:
            data_folds.append(df.iloc[i * fold_size:, :].reset_index(drop=True))
    return data_folds
    

def normMinMax(df, mode='train', train_min=None, train_max=None):
    '''
    Perform min-max normalization on data.
    '''
    data = df.copy()
    if mode == 'train':
        train_max = {}
        train_min = {}
        for col in data.columns:
            train_max[col] = data[col].max()
            train_min[col] = data[col].min()
            data[col] = (data[col] - train_min[col]) / (train_max[col] - train_min[col])
        return data, train_min, train_max
    
    elif mode == 'test':
        if train_min is None or train_max is None:
            raise Exception('Pass train_min and/or train_max.')
        for col in data.columns:
            data[col] = (data[col] - train_min[col]) / (train_max[col] - train_min[col])
        return data
    
def disp_conf_mat(perf_m):
    gd1_cfm = [[perf_m['tn'], perf_m['fn']], [perf_m['fp'], perf_m['tp']]]
    _, ax = plt.subplots(figsize=(3, 3))
    ax.matshow(gd1_cfm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(len(gd1_cfm)):
        for j in range(len(gd1_cfm[i])):
            ax.text(x=j, y=i,s=gd1_cfm[i][j], va='center', ha='center', size='xx-large')
    plt.xlabel('Actuals', fontsize=10)
    plt.ylabel('Predictions', fontsize=10)
    plt.title('Confusion Matrix', fontsize=10)
    plt.show()