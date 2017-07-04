from __future__ import print_function
from __future__ import division
from __future__ import absolute_import


import os
import numpy as np
import argparse
import tensorflow as tf
import time


#default for PSIPRED
NUM_CLASSES=2 #number of output classes
FLAGS={}

tf.logging.set_verbosity(tf.logging.INFO)


#DNN according to Shingh et al
def inference_singh(bins):
        
    #######################################
    #Parameter
    #Convolution
    k=FLAGS["conv"]
    Nout=FLAGS["Nout"]

    #Maxpooling
    m=FLAGS["mpool"]
    
    #Hidden layers
    dims=FLAGS["dims"]
    num_hidden=len(dims)
    ############################################
    
    #Get number of bins and histone modifications
    numberHists=FLAGS["numHistons"]
    numberBins=FLAGS["numBins"]
    
    #Mein versuch convolution networks zu verwenden
    with tf.name_scope("convolution"):
        weights_con = tf.Variable(tf.truncated_normal(shape=[k,numberHists, Nout], stddev=0.1), name="weights")
        biases_con = tf.constant(0.1, shape=[Nout], name="bias")
        #bin_image = tf.reshape(bins, [-1,numberBins,numberHists])
        #conv1=tf.nn.relu(tf.nn.conv1d(bin_image, weights_con, stride=1,padding='SAME')+biases_con)
        conv1=tf.nn.relu(tf.nn.conv1d(bins, weights_con, stride=1,padding='SAME')+biases_con)
    
    #2) max pooling
    with tf.name_scope("maxPooling"):
        maxPool=tf.nn.pool(conv1,window_shape=[m],pooling_type="MAX",strides=[m],padding='SAME')
            
    #3) drop out
    keep_prob = 0.5
    with tf.name_scope("dropOut"):
        dropOut=tf.nn.dropout(maxPool, keep_prob)
        
        #Reshape drop-out layer before starting the multilayer perceptron
        dropOut_flat = tf.reshape(dropOut, [-1, int(numberBins/m)*Nout])
    
    #4) multilayer perceptron
    dims = [int(numberBins/m)*Nout] + dims + [NUM_CLASSES]
    hiddens = [dropOut_flat]
    for i in range(num_hidden):            
        with tf.name_scope("hiddenLayer"+str(i)):
            weights = tf.Variable(tf.truncated_normal(shape=[dims[i], dims[i+1]], stddev=0.1, name="weights"))
            bias = tf.constant(0.1, shape=[dims[i+1]], name="bias")
            tmp_h = tf.nn.relu(tf.matmul(hiddens[i], weights)+bias)   
            hiddens.append(tmp_h)
        
    #softmax function    
    with tf.name_scope("softmax"):
        weights = tf.Variable(tf.truncated_normal(shape=[dims[-2], dims[-1]], stddev=0.1), name="weights")
        bias = tf.constant(0.1, shape=[2], name="bias")
        logits=tf.matmul(hiddens[-1], weights) + bias
        
    return logits


#define the loss function
#predictions
#real labels
def loss(logits, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits, name="xentropy")
    return tf.reduce_mean(cross_entropy, name="xentropy_mean")

#create training op 
def training(loss, learning_rate, momentum=None, global_step=None):
    # add loss function for summary plots
    tf.summary.scalar("loss", loss)

    optimizer = None
    if momentum == None:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name="adam_optimizer")
    else:
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, name="momentum_optimizer")

    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

#evalueate given predictions and labels
def evaluation(logits, labels):

    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #add accuracy for summary plots
    tf.summary.scalar("accuracy", accuracy)
    return accuracy


#run training on a given train and test data set
#chkptfile can restore saved checkpoint
def run_training(datasets, chkptfile=None):

    learning_rate = FLAGS["learnrate"]
    batchsize = FLAGS["batchsize"]
    numBins = FLAGS["numBins"]
    numHistons = FLAGS["numHistons"]
    
    niter = 50000
    kprob = 1.0

    print("starting to train")
    with tf.Graph().as_default():
        start_time = time.time()

        # build the graph
        bins = tf.placeholder(tf.float32, [None, numBins, numHistons])
        labels_ph = tf.placeholder(tf.float32, [None, NUM_CLASSES])
        keep_prob = tf.placeholder(tf.float32)
        global_step = tf.Variable(0, name="global_step", trainable=False)

        logits = inference_singh(bins)

        los = loss(logits, labels_ph)

        train_op = training(los, learning_rate=learning_rate, momentum=FLAGS["momentum"], global_step=global_step)

        eval_correct = evaluation(logits, labels_ph)

        # initilize summaries
        summary = tf.summary.merge_all()
      
        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

        sess = tf.Session()

        mod_dir = "mod_%.1e_%d_%.2f"%(learning_rate, niter, kprob)

        mod_dir = os.path.join(FLAGS["logdir"], mod_dir)
        summary_writer = tf.summary.FileWriter(mod_dir, sess.graph)

        sess.run(init)

        #restore trained data if chkptfile is given
        startv = 0
        if chkptfile is not None:
            print("Restoring model...")
            saver.restore(sess, chkptfile)
            startv = sess.run(global_step)

    
        for i in range(startv, startv+niter):
            #get data and corresponding labels for trainstep
            wins, labs = datasets["train"].get_batch(batchsize)

            #assign data to placeholders
            # and keep prob for dropout
            feed_dict = {bins : wins, labels_ph : labs, keep_prob : kprob}

            #not interested in the ouput of the optimizer
            _ , loss_value = sess.run([train_op, los], feed_dict=feed_dict)


            duration = time.time() - start_time

            #print current loss value
            if i%100 == 0:
                print('Step %d: loss = %.2f (%.3f sec)' % (i, loss_value, duration))
                feed_dict[keep_prob] = 1.0
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, i)
                summary_writer.flush()

            #check performance based on validation set
            if (i + 1) % 1000 == 0 or (i + 1) == niter:
                checkpoint_file = os.path.join(mod_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=global_step)
                print("Evaluation: ")
                tmp_feed_dict = {bins : datasets["validate"].getFlatWindow(), labels_ph : datasets["validate"].labels, keep_prob:1.0}
                print(eval_correct.eval(session=sess, feed_dict=tmp_feed_dict))


        # final evaluation on test set
        print("PERFORMANCE ON TESTSET:")
        print(datasets["test"].getFlatWindow().shape)
        tmp_feed_dict = {bins: datasets["test"].getFlatWindow(), labels_ph: datasets["test"].labels, keep_prob : 1.0}

        print(eval_correct.eval(session=sess, feed_dict=tmp_feed_dict))


#Split a data set random into two parts in a specific ratio
def splitRandom(ratio, windows, labels):
    n1 = int(ratio * float(len(windows)))
    set1 = np.ndarray(shape=(n1,)+windows.shape[1:])
    set2 = np.ndarray(shape=(windows.shape[0]-n1,)+windows.shape[1:])
    l1 = np.ndarray(shape=(n1,) + labels.shape[1:])
    l2 = np.ndarray(shape=(labels.shape[0]-n1,) + labels.shape[1:])

    perm = np.random.permutation(labels.shape[0])

    c1 = 0
    for i in perm[0:n1]:
        set1[c1] = windows[i]
        l1[c1] = labels[i]
        c1 += 1
        
    c2 = 0
    for i in perm[n1:]:
        set2[c2] = windows[i]
        l2[c2] = labels[i]
        c2 += 1
        
    return set1, l1, set2, l2



#Dataset class saving all training, validation or respectively test data
class Dataset(object):
    def __init__(self,windows, labels):
        self.windows = windows
        self.labels = labels
        self.counter = 0
        self.order = np.random.permutation(len(windows))

    #Gibt den ganzen
    def getFlatWindow(self):
        res = np.ndarray([len(self.windows), FLAGS["numBins"], FLAGS["numHistons"]])
        for i in range(len(self.windows)):
            #res[i] = self.windows[i].flatten()
            res[i] = self.windows[i]
        return res

    #Returns a feature batch and the corresponding label batch of size n
    def get_batch(self, n):
        res = np.ndarray([n, FLAGS["numBins"], FLAGS["numHistons"]])
        labs = np.ndarray((n,)+self.labels.shape[1:])
        lower = self.counter + n
        upper = lower + n
        self.counter += n

        c = 0
        #If the end of the data set is reached
        if upper > len(self.windows):
            for i in range(lower,len(self.windows)):
                res[c] = self.windows[self.order[i]]
                labs[c] = self.labels[self.order[i]]
                c += 1
                
            lower = 0
            upper = n - c
            #Permute data again
            self.order = np.random.permutation(len(self.windows))
            self.counter = upper
            
        for i in range(lower, upper):
            res[c] = self.windows[self.order[i]]
            labs[c] = self.labels[self.order[i]]
            c += 1
        return res,labs



if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--logdir", help = "log directory to save results of a run (used for tensorboard)",default="./logs")
    parser.add_argument("-i", help="Feature input data (bins)", dest= "data")
    parser.add_argument("-l", help="Label data", dest="label")
    parser.add_argument("-k", type = int,help="Size of the convolution filter",dest="conv", default=10)
    parser.add_argument("--Nout", type = int,help="Number of output channels after the convolution",default=50)
    parser.add_argument("-m", type = int, help="Pool size for the maxpooling step",dest="mpool",default=5)
    parser.add_argument("--hiddenDims",help="Hidden dimensions, comma separated e.g. \"1000,100\"",default="1000")
    parser.add_argument("--learnrate", default=0.005)
    parser.add_argument("--momentum", default=0.9)
    parser.add_argument("--batchsize", default=200)


    args = parser.parse_args()

    FLAGS["logdir"] = args.logdir
    FLAGS["learnrate"] = args.learnrate
    FLAGS["momentum"] = args.momentum
    FLAGS["batchsize"] = args.batchsize
    
    FLAGS["conv"]=args.conv
    FLAGS["Nout"]=args.Nout
    FLAGS["mpool"]=args.mpool
    
    #Get hidden dimensions
    dimensions=args.hiddenDims.split(",")
    dimensions=list(map(int,dimensions))
    
    FLAGS["dims"]=dimensions
    
    #Einlesen von unseren Input files und anschließend split in training und test
    labelFilename = args.label
    trainingData = args.data

    labelFile = open(labelFilename)
    labelDict = {}
    for line in labelFile.readlines():
        lineSplit=line.split()
        #Convert each label to a binary vector
        if(int(lineSplit[2])==0):
            labelDict[lineSplit[0]]=[1,0]
        elif(int(lineSplit[2])==1):
            labelDict[lineSplit[0]]=[0,1]
        else:
            print("Fehler beim Parsen des Input-Files.")
        

    trainset={}
    with open(trainingData) as featureFile:
        #Name of the data set (from the header)
        dataset=featureFile.readline().rstrip()[2:]
        #All modifications
        modifications=featureFile.readline().rstrip()[2:].split(" ")

        for line in featureFile.readlines():
            line=line.rstrip()
            if(line.startswith('#')):
                lineSplit=line.split(" ")
                geneID=lineSplit[0]
                #Remove the hashtag at the beginning of the line
                geneID=geneID[1:]
                trainset[geneID]=[]
            else:
                valueList=line.split(",")
                valueList=list(map(float,valueList))
                trainset[geneID].append(valueList)
    
    #Sort labels according to the feature list
    #Maybe for some genes no GENCODE entry could be found, these are only in the features list
    y=[]
    X=[]
    #If not all bins should be used
    for geneID in trainset:
        y.append(labelDict[geneID])
        valueMatrix=trainset[geneID]
        
        #Transpose matrix (first number Bins, then histone modifications)
        valueMatrix=list(map(list, zip(*valueMatrix)))
        X.append(valueMatrix)
    
    #Get number of histone modifications and number of bins from the dataset
    FLAGS["numHistons"]=len(valueMatrix[0])
    FLAGS["numBins"]=len(valueMatrix)
    
    #Konvertieren in ein numpy array
    X=np.array(X)
    y=np.array(y)
    
    print("finished reading in data!")
    
    ratio=0.1
    #split the dataset random in two parts and the labels with them
    test, testL, train, trainL = splitRandom(ratio,X,y)
    ratio=0.9
    train, trainL, valid, validL = splitRandom(ratio,train,trainL)
    datasets = {}
    datasets["train"]= Dataset(train,trainL)
    datasets["validate"] = Dataset(valid,validL)
    datasets["test"] = Dataset(test,testL)


    run_training(datasets)


