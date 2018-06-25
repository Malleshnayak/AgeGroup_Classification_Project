import tensorflow as tf
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import sklearn as sk 

df=pd.read_csv("fb_data_sheet_out_final.csv")
df2=df.head(3200)
dataset=df2.values
X=dataset[:,0:8]
Y=dataset[:,8]
print(len(dataset))

#df=pd.read_csv("fb_data_sheet_out_test.csv")
df3=df.tail(795)
dataset=df3.values
X_test=dataset[:,0:8]
Y_test=dataset[:,8]

# Parameters
learning_rate = 0.01
training_epochs = 50
batch_size = 100
display_step = 1


# Network Parameters
n_hidden_1 = 8 # 1st layer number of features
n_hidden_2 = 8 # 2nd layer number of features
n_input = 8 # Number of feature
n_classes = 2 # Number of classes to predict


#X = X.reshape((1800, 8, 1))

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
print('weights:')
#tf.Print(weights)


biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
print('biases')
#tf.Print(biases)

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


temp = Y.shape
Y = Y.reshape(temp[0], 1)
Y = np.concatenate((1-Y, Y), axis=1)
temp = Y_test.shape
Y_test = Y_test.reshape(temp[0], 1)
Y_test = np.concatenate((1-Y_test, Y_test), axis=1)




# Initializing the variables
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    print (sess.run(weights))
    print(sess.run(biases))
    
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(X)/batch_size)
        X_batches = np.array_split(X, total_batch)
        Y_batches = np.array_split(Y, total_batch)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = X_batches[i], Y_batches[i]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
        
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: X_test, y: Y_test}))
    global result 
    result = tf.argmax(pred, 1).eval({x: X_test, y: Y_test})

    y_p = tf.argmax(pred, 1)
    val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x:X_test, y:Y_test})

    print ("validation accuracy:")
    print(val_accuracy)
    y_true = np.argmax(Y_test,1)
    print ("Precision")
    print(sk.metrics.precision_score(y_true, y_pred))
    print ("Recall")
    print(sk.metrics.recall_score(y_true, y_pred))
    print ("f1_score")
    print(sk.metrics.f1_score(y_true, y_pred))
    print ("confusion_matrix")
    print (sk.metrics.confusion_matrix(y_true, y_pred))
    fpr, tpr, tresholds = sk.metrics.roc_curve(y_true, y_pred)
