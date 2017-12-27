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
    w1 = (np.random.rand(D_in, H1) - 0.5) * ((6 / (H1 + D_in)) ** 0.5) * 2
    w2 = (np.random.rand(H1, H2) - 0.5) * ((6 / (H1 + H2)) ** 0.5) * 2
    w3 = (np.random.rand(H2, D_out) - 0.5) * ((6 / (H2 + D_out)) ** 0.5) * 2

    b1 = (np.random.rand(H1, 1) - 0.5) * ((6 / (H1 + 1)) ** 0.5) * 2
    b2 = (np.random.rand(H2, 1) - 0.5) * ((6 / (H2 + 1)) ** 0.5) * 2
    b3 = (np.random.rand(D_out, 1) - 0.5) * ((6 / (1 + D_out)) ** 0.5) * 2
    '''
    b1 = np.zeros((H1, 1))
    b2 = np.zeros((H2, 1))
    b3 = np.zeros((D_out, 1))
    '''
    # #test set:
    x1 = test_images
    y1 = test_labels
    y1 = one_hot(y1)

    # #3. training of neural network

    accuracy = []
    loss = []
    n = int(len(train_images) / EPOCH)
    for epoch in range(0, n):
        if epoch > 50:
            learning_rate = 0.01
        for iter in range(0, n):
            # create input and output
            x = train_images[iter * EPOCH:iter * EPOCH + EPOCH]
            t = train_labels[iter * EPOCH:iter * EPOCH + EPOCH]
            t = one_hot(t)

            # Forward propagation
            a1 = (x.dot(w1).T + b1).T
            z1 = np.maximum(a1, 0)

            a2 = (z1.dot(w2).T + b2).T
            z2 = np.maximum(a2, 0)

            y = (z2.dot(w3).T + b3).T
            y = softmax(y)

            if iter == n - 1:
                e = -np.sum(t * np.log(y))
                l = e / EPOCH
                print('epoch%d loss: %f' % (epoch, l))
                loss.append(e / EPOCH)
            # e = -np.trace(t.dot(np.log(y.T)))
            # Back propagation
            k3 = y - t
            z2_f = np.where(a2 > 0, 1, 0)
            k2 = k3.dot(w3.T) * z2_f
            z1_f = np.where(a1 > 0, 1, 0)
            k1 = k2.dot(w2.T) * z1_f

            # Gradient update
            w3 = w3 - learning_rate * z2.T.dot(k3) / EPOCH - learning_rate * lam * w3
            b3 = b3 - (learning_rate * np.sum(k3, axis=0) / EPOCH).reshape(D_out, 1)
            w2 = w2 - learning_rate * z1.T.dot(k2) / EPOCH - learning_rate * lam * w2
            b2 = b2 - (learning_rate * np.sum(k2, axis=0) / EPOCH).reshape(H2, 1)
            w1 = w1 - learning_rate * x.T.dot(k1) / EPOCH - learning_rate * lam * w1
            b1 = b1 - (learning_rate * np.sum(k1, axis=0) / EPOCH).reshape(H1, 1)

        # Testing for accuracy
        # input:x1
        # output:h3
        z1 = (x1.dot(w1).T + b1).T
        z1 = np.maximum(z1, 0)

        z2 = (z1.dot(w2).T + b2).T
        z2 = np.maximum(z2, 0)

        y = (z2.dot(w3).T + b3).T

        y = softmax(y)

        same_num = 0
        a = np.zeros(np.shape(y))
        for row in range(np.shape(y)[0]):
            x = np.argmax(y[row])
            a[row][x] = 1
            if y1[row][x] == 1:
                same_num = same_num + 1
        ac = same_num / np.shape(y)[0]
        print('epoch%d accuracy: %f' % (epoch, ac))
        accuracy.append(ac)

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
