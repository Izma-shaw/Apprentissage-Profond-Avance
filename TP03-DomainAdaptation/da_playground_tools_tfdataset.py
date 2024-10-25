import numpy as np
from tensorflow.keras.callbacks import Callback

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_sample_blobs(blob_centers, blob_stds, blob_num_samples, shuffle=True):
    """
    Create X blobs of N_x elements of shape (D_0, ..., D_d), picked from different 
    normal distributions N(m_x, s_x).
    Note: this is a simplified version of the method `sklearn.datasets.make_blobs`
    :param blob_centers:     Array of shape (X, D_0, ..., D_d) defining the distribution means m_x
    :param blob_stds:        Array of shape (X, D_0, ..., D_d) defining the distribution STDs s_x
    :param blob_num_samples: Array of shape (X,) defining the numbers N_x of elements per blob
    :param shuffle:          Flag to shuffle the elements from all blobs
    :return:                 x: Samples array of shape (N_0 + ... + N_x, D_0, ..., D_d) 
                             y: Class array of shape (N_0 + ... + N_x,) 
    """
    num_blobs = len(blob_centers)
    input_size = len(blob_centers[0])
    x, y = [], []
    for i in range(num_blobs):
        x_i = np.random.normal(loc=blob_centers[i], scale=blob_stds[i], 
                               size=(blob_num_samples[i], input_size))
        y_i = [i] * blob_num_samples[i]
        
        x.append(x_i)
        y += y_i
    x = np.concatenate(x, axis=0)
    y = np.asarray(y,dtype='int32')
    if shuffle:
        shuffled_indices = np.random.permutation(len(y))
        x, y = x[shuffled_indices], y[shuffled_indices]
    return x, y

def plot_3d_source_and_target_blobs(x_s, y_s, x_t=None, y_t=None):
    """
    :param x_s: source features
    :param y_s: source labels
    :param x_t: target features
    :param y_t: target labels (optional)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x_s[..., 0], x_s[..., 1], x_s[..., 2], 
               c=[["skyblue", "salmon"][class_id] for class_id in y_s], 
               marker='o', s=20, alpha=.5)
    if x_t is not None:
        ax.scatter(x_t[..., 0], x_t[..., 1], x_t[..., 2],
               c=[["b", "r"][class_id] for class_id in y_t] if y_t is not None else "grey", 
               marker='^', s=20, alpha=.5)
    fig.show()

class AdditionalValidationSets(Callback):
    def __init__(self, validation_sets, verbose=0, updatelogs=1, batch_size=None):
        """
        :param validation_sets:
        a list of 3-tuples (validation_data, validation_targets, validation_set_name)
        or 4-tuples (validation_data, validation_targets, sample_weights, validation_set_name)
        :param verbose:
        verbosity mode, 1 or 0
        :param batch_size:
        batch size to be used when evaluating on the additional datasets
        """
        super(AdditionalValidationSets, self).__init__()
        self.validation_sets = validation_sets
        for validation_set in self.validation_sets:
            if len(validation_set) not in [2, 3, 4]:
                raise ValueError()
        self.epoch = []
        self.history = {}
        self.verbose = verbose
        self.updatelogs = updatelogs
        self.batch_size = batch_size

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)

        # record the same values as History() as well
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        # evaluate on the additional validation sets
        for validation_set in self.validation_sets:

            tf_dataset = None
            if len(validation_set) == 2:
                tf_dataset, validation_set_name = validation_set
            elif len(validation_set) == 3:
                validation_data, validation_targets, validation_set_name = validation_set
                sample_weights = None
            elif len(validation_set) == 4:
                validation_data, validation_targets, sample_weights, validation_set_name = validation_set
            else:
                raise ValueError()

            if tf_dataset is not None:
                results = self.model.evaluate(
                    tf_dataset,
                    verbose=self.verbose,
                    batch_size=self.batch_size)
            else:
                results = self.model.evaluate(x=validation_data,
                                          y=validation_targets,
                                          verbose=self.verbose,
                                          sample_weight=sample_weights,
                                          batch_size=self.batch_size)

            for metric, result in zip(self.model.metrics_names,results):
                valuename = validation_set_name + '_' + metric
                self.history.setdefault(valuename, []).append(result)
                if self.updatelogs:
                    logs[valuename]=result