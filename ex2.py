import tensorflow as tf
import copy
import math
# EDEN DUPONT 204808596 EX1

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

    temperature = 0.5
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

    w1 = tf.compat.v1.Variable(tf.random.uniform([n, hidden], -1, 1, seed=0), dtype=tf.float32, name="weights1")
    b1 = tf.compat.v1.Variable(tf.random.uniform([1, hidden], -1, 1, seed=0), dtype=tf.float32, name="bias1")
    w2 = tf.compat.v1.Variable(tf.random.uniform([(n + hidden if bridge else hidden), 1], -1, 1, seed=0),
                               dtype=tf.float32, name="weights2")
    b2 = tf.compat.v1.Variable(tf.random.uniform([1, 1], -1, 1, seed=0), dtype=tf.float32, name="bias2")

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
        z_sig1 = tf.sigmoid(z1 / temperature)
        z2 = tf.matmul(z_sig1, w2) + b2
        y = tf.sigmoid(z2 / temperature)

    prev_loss = 0
    counter = 0
    success_flag = False

    loss_mse = tf.reduce_sum(tf.square(y - target))
    loss_cross = tf.reduce_sum(-target*(tf.math.log(y))-(1-target)*(tf.math.log(1-y)))

    loss = loss_cross

    train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    for epoc in range(0, 40000):
        sess.run(train, {x_input_node: x_training_data, target: y_training})
        validation_loss = sess.run(loss, {x_input_node: x_validation_data, target: y_validation})
        # print(sess.run(z1, {x_input_node: x_training_data, target: y_training}))
        if epoc is 0:
            prev_loss = validation_loss

        flag1 = validation_loss < 0.2

        if (prev_loss - validation_loss) < 0.0001:
            counter += 1
            if counter >= 10 and flag1:
                success_flag = True
                break
        else:
            counter = 0
        prev_loss = validation_loss

    if success_flag:
        printf(file, "Successful training, stopped at epoc = " + str(epoc) + " with a validation loss of " + str(validation_loss))
    else:
        printf(file, "Failed training, reached 40,000 epocs with a validation loss of " + str(validation_loss))
    result_table = sess.run(y, {x_input_node: x_training_data, target: y_training})
    # printf(file, result_table)
    curr_w1, curr_w2, curr_b1, curr_b2, curr_loss = sess.run([w1, w2, b1, b2, loss],
                                                             {x_input_node: x_validation_data, target: y_validation})
    # print("W1: %s\nb1: %s\nW2: %s\nb2: %s\nloss: %s" % (curr_w1, curr_b1, curr_w2, curr_b2, curr_loss))

    # add training data
    # add validation data
    # make it run 10 times
    # add print to file
    # use cross entropy
    # use constants

    return curr_w1, curr_w2, curr_b1, curr_b2, curr_loss


def main():
    data_training = [[[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [0]]]
    data_validation = [[[1, 0.1], [1, 0.9], [0.9, 0.9], [0.1, 0.9], [0.2, 0.8], [0.8, 0.9]],
                       [[1], [0], [0], [1], [1], [0]]]
    training_validation = copy.deepcopy(data_training)
    training_validation[0] += data_validation[0]
    training_validation[1] += data_validation[1]
    f = open("ex2_output.txt", "w+")
    for bridge in [True, False]:
        for hidden in [2, 4]:
            for rate in [0.1, 0.01]:
                printf(f, "hidden: " + str(hidden) + ", bridge: " + str(bridge) + ", learning_rate: " + str(rate))
                for i in range(0, 3):
                    xor(f, data_training, training_validation, hidden, bridge, rate)
                printf(f, "-------------------------------------------------------------------------------------")
    xor(f, data_training, training_validation, hidden=1, bridge=True, learning_rate=0.01)
    # xor(data_training, training_validation, 1, True, 0.01)
    # xor(data, 2, False, 0.01)
    # xor(data, 4, False, 0.01)
    printf(f, " Eden Dupont 204808596 Exercise 2 ")
    f.close()


if __name__ == "__main__":
    main()


def pretty_print(f, result, loss, k):
    a = "\nK(Hidden neurons) = " + str(k)
    b = "\nRESULTS: \n" + str(result)
    c = "\nloss (RSS) : " + str(loss) + "\n\n------------------"

    f.write(a)
    f.write(b)
    f.write(c)
    print(a)
    print(b)
    print(c)
