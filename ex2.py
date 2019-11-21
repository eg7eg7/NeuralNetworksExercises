import tensorflow as tf
import numpy as np
import copy
import random
from datetime import datetime
import matplotlib.pyplot as plt

# EDEN DUPONT 204808596 EX2


MAX_EPOCHS = 40000
MIN_VALIDATION_LOSS = 0.2
MAX_MINIMAL_CHANGES = 10
MINIMAL_CHANGE = 0.0001
NUM_SUCCESSFUL_EXPERIMENTS = 10
temperature = 0.5

graph_mean_epoch = []
graph_epoch_std = []
graph_num_layers = []
graph_rate = []
graph_bridge = []


# INPUTS:
# [2^n x n] matrix
# K num of hidden layers
# bypass variable - implement bypass or not
# w and bias


# function requires an open file to print to console + file
def printf(f, _str):
    a = str(_str)
    print(a)
    try:
        f.write(a + "\n")
    except IOError:
        print("Could not print to file error")


def xor(file, training_data, validation_data, hidden, bridge, learning_rate):
    seed = random.seed(datetime.now())

    # input values
    x_training_data = training_data[0]
    x_validation_data = validation_data[0]

    # expected output values
    y_training = training_data[1]
    y_validation = validation_data[1]

    # number of inputs
    n = len(x_training_data[0])

    x_input_node = tf.compat.v1.placeholder(tf.float32, [None, n])
    target = tf.compat.v1.placeholder(tf.float32, [None, 1])

    w1 = tf.compat.v1.Variable(tf.random.uniform([n, hidden], -1, 1, seed=seed), dtype=tf.float32, name="weights1")
    b1 = tf.compat.v1.Variable(tf.random.uniform([1, hidden], -1, 1, seed=seed), dtype=tf.float32, name="bias1")
    w2 = tf.compat.v1.Variable(tf.random.uniform([(n + hidden if bridge else hidden), 1], -1, 1, seed=seed),
                               dtype=tf.float32, name="weights2")
    b2 = tf.compat.v1.Variable(tf.random.uniform([1, 1], -1, 1, seed=seed), dtype=tf.float32, name="bias2")

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    if bridge:
        z1 = tf.add(tf.matmul(x_input_node, w1), b1)
        z_sig1 = tf.sigmoid(z1 / temperature)
        z11 = tf.concat([z_sig1, x_input_node], 1)
        z2 = tf.matmul(z11, w2) + b2
        y = tf.sigmoid(z2 / temperature)
    else:
        z1 = tf.add(tf.matmul(x_input_node, w1), b1)
        z11 = tf.sigmoid(z1 / temperature)
        z2 = tf.matmul(z11, w2) + b2
        y = tf.sigmoid(z2 / temperature)

    prev_loss = np.inf
    counter = 0
    success_flag = False

    loss = tf.reduce_sum(-target * (tf.math.log(y)) - (1 - target) * (tf.math.log(1 - y)))
    train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    for epoch in range(0, MAX_EPOCHS):
        sess.run(train, {x_input_node: x_training_data, target: y_training})
        validation_loss = sess.run(loss, {x_input_node: x_validation_data, target: y_validation})

        if (prev_loss - validation_loss) < MINIMAL_CHANGE:
            counter += 1
            if counter >= MAX_MINIMAL_CHANGES and validation_loss < MIN_VALIDATION_LOSS:
                success_flag = True
                break
        else:
            counter = 0

        prev_loss = validation_loss

    curr_w1, curr_w2, curr_b1, curr_b2, curr_loss = sess.run([w1, w2, b1, b2, loss],
                                                             {x_input_node: x_validation_data, target: y_validation})
    train_loss = sess.run(loss, {x_input_node: x_training_data, target: y_training})
    if hidden == 1 and bridge is True:  # print output of hidden layer alongside the input
        hlayer_out = tf.sigmoid(tf.matmul(z_sig1 + b1, [w2[0]]) / temperature)
        hidden_output = sess.run(hlayer_out, {x_input_node: x_training_data})

        printf(file, "x_input: " + str(x_training_data))
        printf(file, "hlayer_output: " + str(hidden_output.tolist()))
    return curr_w1, curr_w2, curr_b1, curr_b2, curr_loss, success_flag, epoch, validation_loss, train_loss


def experiment(f, training, validation, hidden, bridge, rate, exp_counter):
    success_counter = 0
    failure_counter = 0
    epochs = []
    validation_losses = []
    train_losses = []
    while success_counter < NUM_SUCCESSFUL_EXPERIMENTS:
        curr_w1, curr_w2, curr_b1, curr_b2, curr_loss, success_flag, epoch, validation_loss, train_loss = \
            xor(f, training, validation, hidden, bridge, rate)
        if success_flag:
            success_counter += 1
            epochs.append(epoch)
            validation_losses.append(validation_loss)
            train_losses.append(train_loss)
        else:
            failure_counter += 1

    printf(f, "experiment " + str(exp_counter) + ": hidden: " + str(hidden) + ", bridge: " + str(bridge) + ", learning_rate: " +
           str(rate))
    epoch_mean = np.mean(epochs)
    epoch_std = np.std(epochs)
    val_loss_mean = np.mean(validation_losses)
    val_loss_std = np.std(validation_losses)
    train_loss_mean = np.mean(train_losses)
    train_loss_std = np.std(train_losses)

    graph_mean_epoch.append(float(epoch_mean))
    graph_bridge.append(int(1 if bridge else 0))
    graph_epoch_std.append(float(epoch_std))
    graph_num_layers.append(int(hidden))
    graph_rate.append(float(rate))

    printf(f, "mean_epochs : " + str(epoch_mean) + ", std/epoch : " + str(epoch_std) + " , Failures : " + str(failure_counter))
    printf(f, "mean_valid_loss : " + str(val_loss_mean) + ", std/valid_loss : " + str(val_loss_std))
    printf(f, "mean_train_loss : " + str(train_loss_mean) + ", std/train_loss : " + str(train_loss_std))


def hyper_print(f, training, validation):
    printf(f, "MAX EPOCHS = " + str(MAX_EPOCHS))
    printf(f, "MIN_VALIDATION_LOSS = " + str(MIN_VALIDATION_LOSS) + ", MAX_MINIMAL_CHANGES = " + str(MAX_MINIMAL_CHANGES))
    printf(f, "MINIMAL_CHANGE = " + str(MINIMAL_CHANGE))
    printf(f, "temperature = " + str(temperature))
    printf(f, "x_train = " + str(training[0]))
    printf(f, "y_train = " + str(training[1]))
    printf(f, "x_validation = " + str(validation[0]))
    printf(f, "y_validation = " + str(validation[1]))


def main():
    # preparing data
    # training validation contains both training and validation data
    data_training = [[[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [0]]]
    data_validation = [[[1, 0.1], [1, 0.9], [0.9, 0.9], [0.1, 0.9]],
                       [[1], [0], [0], [1]]]
    training_validation = copy.deepcopy(data_training)
    training_validation[0] += data_validation[0]
    training_validation[1] += data_validation[1]

    exp_counter = 1
    f = open("ex2_output.txt", "w+")
    hyper_print(f, data_training, training_validation)
    printf(f, "")

    for bridge in [True, False]:
        for hidden in [2, 4]:
            for rate in [0.1, 0.01]:
                experiment(f, data_training, training_validation, hidden, bridge, rate, exp_counter)
                exp_counter += 1
                printf(f, "-------------------------------------------------------------------------------------")
    print("Printing output of hidden layer: ")
    experiment(f, data_training, training_validation, 1, True, 0.1, exp_counter)
    exp_counter += 1
    printf(f, "-------------------------------------------------------------------------------------")
    printf(f, "OPTIONAL : TRY WITH 3 HIDDEN LAYERS (Q4) : ")
    experiment(f, data_training, training_validation, 3, False, 0.1, exp_counter)
    printf(f, "\nEden Dupont 204808596 Exercise 2 ")
    f.close()
    plot_graphs()


def plot_graphs():

    plt.subplot(3, 1, 1)
    plt.title('Graphs solutions')
    plt.plot(graph_mean_epoch, graph_num_layers, marker="8", linestyle='None')
    plt.ylabel('num_hlayer')
    plt.xlabel('mean_epochs')
    plt.subplot(3, 1, 2)
    plt.plot(graph_mean_epoch, graph_bridge, marker="8", linestyle='None')
    plt.ylabel('bridge (1-True, 0-False)')
    plt.xlabel('mean_epochs')
    plt.subplot(3, 1, 3)
    plt.plot(graph_epoch_std, graph_rate, marker="8", linestyle='None')
    plt.ylabel('learning rate')
    plt.xlabel('epochs_std')

    plt.show()


if __name__ == "__main__":
    main()
