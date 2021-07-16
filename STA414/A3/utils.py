import numpy as np
import matplotlib.pyplot as plt


def load_data(file_name):
    """ Loads the data.
    """
    npzfile = np.load(file_name)

    inputs_train = npzfile["inputs_train"].T / 255.0
    inputs_valid = npzfile["inputs_valid"].T / 255.0
    inputs_test = npzfile["inputs_test"].T / 255.0
    target_train = npzfile["target_train"].tolist()
    target_valid = npzfile["target_valid"].tolist()
    target_test = npzfile["target_test"].tolist()

    num_class = max(target_train + target_valid + target_test) + 1
    target_train_1hot = np.zeros([num_class, len(target_train)])
    target_valid_1hot = np.zeros([num_class, len(target_valid)])
    target_test_1hot = np.zeros([num_class, len(target_test)])

    for ii, xx in enumerate(target_train):
        target_train_1hot[xx, ii] = 1.0

    for ii, xx in enumerate(target_valid):
        target_valid_1hot[xx, ii] = 1.0

    for ii, xx in enumerate(target_test):
        target_test_1hot[xx, ii] = 1.0

    inputs_train = inputs_train.T
    inputs_valid = inputs_valid.T
    inputs_test = inputs_test.T
    target_train_1hot = target_train_1hot.T
    target_valid_1hot = target_valid_1hot.T
    target_test_1hot = target_test_1hot.T
    return inputs_train, inputs_valid, inputs_test, target_train_1hot, target_valid_1hot, target_test_1hot


def save(file_name, data):
    """ Saves the model to a numpy file.
    """
    print("Writing to " + file_name)
    np.savez_compressed(file_name, data)


def load(file_name):
    """ Loads the model from numpy file.
    """
    print("Loading from " + file_name)
    return dict(np.load(file_name,allow_pickle=True))['arr_0'].item()


def display_plot(train, valid, y_label, eta, bat_size, number=0):
    """ Displays training curve.
    :param train: Training statistics
    :param valid: Validation statistics
    :param y_label: Y-axis label of the plot
    :param number: The number of the plot
    :return: None
    """
    plt.figure(number)
    plt.clf()
    train = np.array(train)
    valid = np.array(valid)
    plt.plot(train[:, 0], train[:, 1], "b", label="Train")
    plt.plot(valid[:, 0], valid[:, 1], "g", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.title("{}: {} and {}: {}".format("Eta", eta, "Batch Size", bat_size))
    plt.legend()
    plt.draw()
    plt.pause(0.0001)
