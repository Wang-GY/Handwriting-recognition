import numpy as np
import pickle
import matplotlib.pyplot as plt


def softmax(m):
    x, y = np.shape(m)
    m_exp = np.exp(m)
    m_exp_row_sum = m_exp.sum(axis=1).reshape(x, 1)
    s = m_exp / m_exp_row_sum
    return s


def one_hot(value):
    n_values = np.max(value) + 1
    return np.eye(n_values)[value]


if __name__ == '__main__':

    # Definition of functions and parameters

    # NEPOCHis batch size; D_in is input dimension;
    # H1 is hidden layer1 dimension;
    # H2 is hidden layer2 dimension
    # D_out is output dimension.
    D_in = 784
    D_out = 10
    EPOCH = 100
    H1 = 300
    H2 = 100

    learning_rate = 0.1
    lam = 0.0005

    # Read all data from .pkl
    (train_images, train_labels, test_images, test_labels) = pickle.load(open('./mnist_data/data.pkl', 'rb'),
                                                                         encoding='latin1')

    # #1.Data preprocessing: normalize all pixels to [0,1) by dividing 256
    train_images = train_images / 256
    test_images = test_images / 256

    # #2. Weight initialization: Xavier
    w1 = (np.random.rand(D_in, H1) - 0.5) * (6 ** 0.5) / (H1 + D_in) ** 0.5
    w2 = (np.random.rand(H1, H2) - 0.5) * (6 ** 0.5) / (H2 + H1) ** 0.5
    w3 = (np.random.rand(H2, D_out) - 0.5) * (6 ** 0.5) / (H2 + D_out) ** 0.5

    b1 = (np.random.rand(H1, 1) - 0.5) * (6 ** 0.5) / (H1 + 1) ** 0.5
    b2 = (np.random.rand(H2, 1) - 0.5) * (6 ** 0.5) / (H2 + 1) ** 0.5
    b3 = (np.random.rand(D_out, 1) - 0.5) * (6 ** 0.5) / (D_out + 1) ** 0.5

    # #test set:
    x1 = test_images
    y1 = test_labels
    y1 = one_hot(y1)

    # #3. training of neural network

    accuracy = []
    loss = []
    n = int(len(train_images) / EPOCH)
    for epoch in range(0, n):
        # if epoch > 50:
        # learning_rate = 0.01
        # create input and output
        x = train_images[epoch * EPOCH:epoch * EPOCH + EPOCH]
        y = train_labels[epoch * EPOCH:epoch * EPOCH + EPOCH]
        y = one_hot(y)

        # Forward propagation
        h1 = (x.dot(w1).T + b1).T
        h1 = np.maximum(h1, 0)

        h2 = (h1.dot(w2).T + b2).T
        h2 = np.maximum(h2, 0)

        h3 = (h2.dot(w3).T + b3).T
        h3 = softmax(h3)

        e = -np.sum(y.dot(np.log(h3.T)))
        loss.append(e / EPOCH)
        l = e / EPOCH + 0.5 * lam * (np.sum(np.square(w1)) + np.sum(np.square(w2)) + np.sum(np.square(w3)))
        # Back propagation
        k3 = h3 - y
        h2_f = np.where(h2 > 0, 1, 0)
        k2 = k3.dot(w3.T) * h2_f
        h1_f = np.where(h1 > 0, 1, 0)
        k1 = k2.dot(w2.T) * h1_f

        # Gradient update
        w3 = w3 - learning_rate * h2.T.dot(k3) / EPOCH - learning_rate * lam * w3
        b3 = b3 - (learning_rate * np.sum(k3, axis=0) / EPOCH).reshape(D_out, 1)
        w2 = w2 - learning_rate * h1.T.dot(k2) / EPOCH - learning_rate * lam * w2
        b2 = b2 - (learning_rate * np.sum(k2, axis=0) / EPOCH).reshape(H2, 1)
        w1 = w1 - learning_rate * x.T.dot(k1) / EPOCH - learning_rate * lam * w1
        b1 = b1 - (learning_rate * np.sum(k1, axis=0) / EPOCH).reshape(H1, 1)

        # Testing for accuracy
        # input:x1
        # output:h3
        h1 = (x1.dot(w1).T + b1).T
        h1 = np.maximum(h1, 0)

        h2 = (h1.dot(w2).T + b2).T
        h2 = np.maximum(h2, 0)

        h3 = (h2.dot(w3).T + b3).T
        h3 = softmax(h3)

        same_num = 0
        a = np.zeros(np.shape(h3))
        for row in range(np.shape(h3)[0]):
            x = np.argmax(h3[row])
            a[row][x] = 1
            if y1[row][x] == 1:
                same_num = same_num + 1

        accuracy.append(same_num / np.shape(h3)[0])

    ### 4. Plot
    fig = plt.figure(figsize=(12, 5))
    fig.suptitle('accuracy')
    ax1 = plt.subplot(111)
    plt.plot(range(n), accuracy, label='accuracy')
    plt.xlabel('time')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend()
    plt.savefig('accuracy.pdf', dbi=600)
    plt.show()

    fig = plt.figure(figsize=(12, 5))
    fig.suptitle('loss')
    ax1 = plt.subplot(111)
    plt.plot(range(n), loss, label='loss')
    plt.xlabel('time')
    plt.ylabel('loss')
    plt.grid()
    plt.legend()
    plt.savefig('loss.pdf', dbi=600)
    plt.show()

    loss = np.array(loss)
    accuracy = np.array(accuracy)

    print(loss)
    print(accuracy)
    plt.close()
