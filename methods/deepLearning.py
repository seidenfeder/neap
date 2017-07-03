from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

#from utils.data import *
#from utils.file import *
import os
import sys
import numpy as np
import argparse
import tensorflow as tf
import math
import time
#import matplotlib.pyplot as plt
#import seaborn as sns

#default for PSIPRED
NUM_CLASSES=2 #number of output classes
NUM_INPUT=5*160 #input dimension
FLAGS={}

tf.logging.set_verbosity(tf.logging.INFO)

#def conv2d(x, W):
#  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#
#def max_pool_2x2(x):
#  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
#                        strides=[1, 2, 2, 1], padding='SAME')

#def inf_conv(sequence_ph, keep_prob, num_conv, dims):
#    filter_length
#    for i in range(num_conv):
#        weights = tf.Variable(tf.truncated_normal(shape=[filter_length, input_dim, output_dim], 0.1))
#        bias = tf.constant(0.1, shape=[output_dim])
#        conv1 = tf.nn.relu(tf.nn.conv1D(input, weights, strides=[1,1,1], padding='SAME') + bias)



#Construct N fully connected layers with a softmax at the end
#placeholder input
#keep probability for dropout
#number of hidden layers
#number of neurons per layes
#dropouts = [True False] if layes should contain dropouts
#TODO: same for conv layers
def inference(sequence_ph, keep_prob, num_hidden=1, dims=[75],dropouts=None):

    if len(dims) != num_hidden:
        print("dimensions not matching")
        sys.exit()

    if dropouts == None:
        dropouts = [False for i in range(num_hidden)]

    dims = [NUM_INPUT] + dims + [NUM_CLASSES]
    hiddens = [sequence_ph]

    for i in range(num_hidden):

        with tf.name_scope("hidden_fc"+str(i)):
            weights = tf.Variable(tf.truncated_normal(shape=[dims[i], dims[i+1]], stddev=(1. / math.sqrt(dims[i]))),
                                  name="weights")
            bias = tf.constant(0.1, shape=[ dims[i+1] ], name="bias")
            tmp_h = tf.nn.relu(tf.matmul(hiddens[i], weights)+bias)
            if(dropouts[i]):
                hiddens.append(tf.nn.dropout(tmp_h, keep_prob))
                continue
            hiddens.append(tmp_h)

    with tf.name_scope("softmax"):
        weights = tf.Variable(tf.truncated_normal(shape=[dims[-2], dims[-1]], stddev=1.0/math.sqrt(dims[-2])), name="weights")
        bias = tf.constant(0.1, shape=[dims[-1]], name="bias")
        logits=tf.matmul(hiddens[-1], weights) + bias

    return logits

##Do 1d convolution
#def conv1d(x,W,k):
#    return 
#
#DNN according to Shingh et al
def inference_singh(bins,labels):
    k=10
    Nout=20
    numberHists=5
    numberBins=160
    
    #Mehrere convolutionary layers
    #i=1
    ##1) Create convolutionary network (mehrere Layer???)
    #with tf.name_scope("conv"+str(i)):
    #    #Stimmen zahlen ??? -> wsl eher nicht ...
    #    weights_con = tf.Variable(tf.truncated_normal(shape=[k, k], stddev=0.1), name="weights")
    #    biases_con = tf.constant(0.1, shape=[k], name="bias")
    #    conv1=tf.nn.relu(tf.nn.conv1d(bins, weights_con, strides=k)+biases_con)
    
    #Mein versuch convolution networks zu verwenden
    weights_con = tf.Variable(tf.truncated_normal(shape=[k,numberHists, Nout], stddev=0.1), name="weights")
    biases_con = tf.constant(0.1, shape=[Nout], name="bias")
    bin_image = tf.reshape(bins, [-1,numberBins,numberHists])
    conv1=tf.nn.relu(tf.nn.conv1d(bin_image, weights_con, stride=1,padding='SAME')+biases_con)

    #2) max pooling
    m=2
    maxPool=tf.nn.pool(conv1,window_shape=[m],pooling_type="MAX",strides=[m],padding='SAME') 
            
    #3) drop out
    keep_prob = 0.5
    dropOut=tf.nn.dropout(maxPool, keep_prob)
    print(dropOut)
    
    #4) multilayer perceptron
    #Wie viele hidden layer und welche dimensionen?
    #Wie alternierend linear und non-linear
    dims=[40]
    num_hidden=1
    dims = [int((numberBins-k+1)*keep_prob/m)] + dims + [NUM_CLASSES]
    hiddens = [dropOut]

    for i in range(num_hidden):

        with tf.name_scope("hidden_fc"+str(i)):
            weights = tf.Variable(tf.truncated_normal(shape=[dims[i], dims[i+1]], stddev=(1. / math.sqrt(dims[i]))),
                                  name="weights")
            bias = tf.constant(0.1, shape=[ dims[i+1] ], name="bias")
            tmp_h = tf.nn.relu(tf.matmul(hiddens[i], weights)+bias)
            hiddens.append(tmp_h)

    with tf.name_scope("softmax"):
        weights = tf.Variable(tf.truncated_normal(shape=[dims[-2], dims[-1]], stddev=1.0/math.sqrt(dims[-2])), name="weights")
        bias = tf.constant(0.1, shape=[dims[-1]], name="bias")
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
    #Eventuel besser irgendwo als Eingabewert übergeben
    niter = 50000
    kprob = 1.0

    print("starting to train")
    with tf.Graph().as_default():
        start_time = time.time()

        # build the graph
        sequences_ph = tf.placeholder(tf.float32, [None, NUM_INPUT])
        labels_ph = tf.placeholder(tf.float32, [None, NUM_CLASSES])
        keep_prob = tf.placeholder(tf.float32)
        global_step = tf.Variable(0, name="global_step", trainable=False)

        logits = inference_singh(sequences_ph, keep_prob)

        los = loss(logits, labels_ph)

        train_op = training(los, learning_rate=learning_rate, momentum=FLAGS["momentum"], global_step=global_step)

        eval_correct = evaluation(logits, labels_ph)


        # initilize summaries
        summary = tf.summary.merge_all()

        
        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

        sess = tf.Session()

        mod_dir = "mod_%.1e_%d_%.2f"%(learning_rate, niter, kprob)

        #print(mod_dir)
        #start_new =False
        #if not os.path.exists(os.path.join(FLAGS["logdir"], mod_dir)):
        #    start_new = True
        #    os.makedirs(os.path.join(FLAGS["logdir"], mod_dir))

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
            feed_dict = {sequences_ph : wins, labels_ph : labs, keep_prob : kprob}

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
                tmp_feed_dict = {sequences_ph : datasets["validate"].getFlatWindow(), labels_ph : datasets["validate"].labels, keep_prob:1.0}
                print(eval_correct.eval(session=sess, feed_dict=tmp_feed_dict))


        # final evaluation on test set
        print("PERFORMANCE ON TESTSET:")
        print(datasets["test"].getFlatWindow().shape)
        tmp_feed_dict = {sequences_ph: datasets["test"].getFlatWindow(), labels_ph: datasets["test"].labels, keep_prob : 1.0}

        print(eval_correct.eval(session=sess, feed_dict=tmp_feed_dict))



#def getWindowPWM(prot, start):
#    return prot.getWindow(at=start)
#
#def extractWindows(proteins, half_win):
#    n_winds = 0
#    for p in proteins:
#        n_winds += len(p.residues)
#
#    complete = np.ndarray([n_winds, 15,20])
#    structs = np.ndarray([n_winds, NUM_CLASSES])
#
#    c = 0
#    for p in proteins:
#        for i in range(p.length):
#            complete[c] = p.getPWMWindow(i,half_win)
#            structs[c] = p.residues[i].ss
#            c += 1
#            # complete.append(p.getPWMWindow(i,half_win))
#            # structs.append(p.residues[i].ss)
#
#    return (complete, structs)

#Wahrscheinlich uninteressant für uns
#def getStateDistribution(prots):
#    x = ["H", "B", "C"]
#    y = [0,0,0]
#    for p  in prots:
#        for r in p.residues:
#            if r.getSSLetter() == "H":
#                y[0] = y[0] + 1
#            elif r.getSSLetter() == "B":
#                y[1] = y[1] + 1
#            elif r.getSSLetter() == "C":
#                y[2] = y[2] + 1
#
#    plt.bar(np.arange(len(x)), y, align='center', alpha=0.5)
#    plt.xticks(np.arange(len(x)), x)
#    plt.ylabel('occurences')
#    plt.title('Secondary Structure state distribution')
#
#    plt.show()

def splitRandom(ratio, windows, labels):
    n1 = int(ratio * float(len(windows)))
    set1 = np.ndarray(shape=(n1,)+windows.shape[1:])
    set2 = np.ndarray(shape=(windows.shape[0]-n1,)+windows.shape[1:])
    l1 = np.ndarray(shape=(n1,) + labels.shape[1:])
    l2 = np.ndarray(shape=(labels.shape[0]-n1,) + labels.shape[1:])

    perm = np.random.permutation(labels.shape[0])

    #Verstaendlich, eventuell eleganter programmieren ...
    c1 = 0
    c2 = 0
    for i in perm:
        if c1 < n1:
            set1[c1] = windows[i]
            l1[c1] = labels[i]
            c1 += 1
        else:
            set2[c2] = windows[i]
            l2[c2] = labels[i]
            c2 += 1
    return set1, l1, set2, l2




class Dataset(object):
    def __init__(self,windows, labels):
        self.windows = windows
        self.labels = labels
        self.counter = 0
        self.order = np.random.permutation(len(windows))

    def getFlatWindow(self):
        res = np.ndarray([len(self.windows), NUM_INPUT])
        for i in range(len(self.windows)):
            res[i] = self.windows[i].flatten()
        return res

    def get_batch(self, n):
#        res = np.ndarray((n,)+self.windows.shape[1:])
        res = np.ndarray((n,)+(NUM_INPUT,))
        labs = np.ndarray((n,)+self.labels.shape[1:])
        lower = self.counter + n
        upper = lower + n
        self.counter += n

        c = 0
        #Verstaendlich, eventuell eleganter programmieren ...
        if upper > len(self.windows):
            upper = 0
            while(lower < len(self.windows)):
                res[c] = self.windows[self.order[lower]].flatten()
                labs[c] = self.labels[self.order[lower]]
                c += 1
                upper += 1
                lower += 1
            lower = 0
            upper = n - upper
            self.order = np.random.permutation(len(self.windows))
            print("reset coutner")
            self.counter = upper

        for i in range(lower, upper):
            res[c] = self.windows[self.order[i]].flatten()
            labs[c] = self.labels[self.order[i]]
            c += 1
        return res,labs



if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    #Help text hinzufügen?
    parser.add_argument("--logdir", default="./logs")
    parser.add_argument("-i", dest= "data")
    parser.add_argument("-l", dest="label")
    parser.add_argument("--plot", default=False)
    parser.add_argument("--nclasses", default=2)
    parser.add_argument("--ninput", default=800) #5 istone modifications * 160 bins
    parser.add_argument("--learnrate", default=0.005)
    parser.add_argument("--momentum", default=0.9)
    parser.add_argument("--batchsize", default=200)
    parser.add_argument("--npasses", default=5)
#    parser.add_argument("--fastdata", default=True)


    args = parser.parse_args()

    if args.nclasses is not None:
        FLAGS["nclasses"] = int(args.nclasses)

    if args.ninput is not None:
        FLAGS["ninput"] = int(args.ninput)

    FLAGS["logdir"] = args.logdir
    FLAGS["learnrate"] = args.learnrate
    FLAGS["momentum"] = args.momentum
    FLAGS["batchsize"] = args.batchsize
    FLAGS["npasses"] = args.npasses

    #Einlesen von unseren Input files und anschlißend split in training und test
    labelFilename = args.label
    trainingData = args.data


    #Einlesen von unseren Input files und anschlißend split in training und test
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
        X.append(valueMatrix)
    
    #Konvertieren in ein numpy array
    X=np.array(X)
    y=np.array(y)
    
    print("finished reading in data!")
    
    ratio=1/3
    #split the dataset random in two parts and the labels with them
    test, testL, train, trainL = splitRandom(ratio,X,y)
    ratio=0.5
    train, trainL, valid, validL = splitRandom(ratio,train,trainL)
    datasets = {}
    datasets["train"]= Dataset(train,trainL)
    datasets["validate"] = Dataset(valid,validL)
    datasets["test"] = Dataset(test,testL)


    run_training(datasets)


