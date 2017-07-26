from __future__ import print_function
from __future__ import division
from __future__ import absolute_import


import os
import numpy as np
import argparse
import tensorflow as tf
import time
import matplotlib.pyplot as plt


#default for PSIPRED
NUM_CLASSES=2 #number of output classes
FLAGS={}

tf.logging.set_verbosity(tf.logging.INFO)


#DNN according to Shingh et al
def inference_singh(bins, keep_prob):
        
    #######################################
    #Parameter
    #Convolution
    k=FLAGS["conv"]
    Nout=FLAGS["Nout"]
    num_convolution=len(k)
    
    #Maxpooling
    m=FLAGS["mpool"]
    
    #Hidden layers
    dims=FLAGS["dims"]
    num_hidden=len(dims)
    
    ############################################
    
    #Get number of bins and histone modifications
    numberHists=FLAGS["numHistons"]
            
    #Multiple stacked layers of convolution and maxpooling
    convDims=[numberHists]+Nout
    convolutionLayer=[bins]
    for i in range(num_convolution):
        #1) Convolution layer
        with tf.name_scope("convolution"+str(i)):
            weights_con = tf.Variable(tf.truncated_normal(shape=[k[i],convDims[i], convDims[i+1]], stddev=0.1), name="weights")
            biases_con = tf.constant(0.1, shape=[convDims[i+1]], name="bias")
            conv1=tf.nn.relu(tf.nn.conv1d(convolutionLayer[i], weights_con, stride=1,padding='SAME')+biases_con)
            
        
        #2) max pooling
        with tf.name_scope("maxPooling"+str(i)):
            maxPool=tf.nn.pool(conv1,window_shape=[m[i]],pooling_type="MAX",strides=[m[i]],padding='SAME')
            convolutionLayer.append(maxPool)
    
    if(FLAGS["batchnorm"]):
        #3) batch normalization
        with tf.name_scope("batch_normalization"):
            batch_mean, batch_var = tf.nn.moments(maxPool,[0])
            scale = tf.Variable(tf.ones([maxPool.get_shape()[1],maxPool.get_shape()[2]]))
            beta = tf.Variable(tf.zeros([maxPool.get_shape()[1],maxPool.get_shape()[2]]))
            batchNorm = tf.nn.batch_normalization(maxPool,batch_mean,batch_var, scale, beta, 0.000000001)
            batchNormShapes=batchNorm.get_shape()[1]
            batchNorm_flat = tf.reshape(batchNorm, [-1, batchNormShapes.value*Nout[-1]])
        
        #4) multilayer perceptron
        dims = [batchNormShapes.value*Nout[-1]] + dims + [NUM_CLASSES]
        hiddens = [batchNorm_flat]
        
    else:    
        #3) drop out
        with tf.name_scope("dropOut"):
            dropOut=tf.nn.dropout(maxPool, keep_prob)
    
            #Reshape drop-out layer before starting the multilayer perceptron
            dropOutShapes=dropOut.get_shape()[1]
            dropOut_flat = tf.reshape(dropOut, [-1, dropOutShapes.value*Nout[-1]])
        
        #4) multilayer perceptron
        dims = [dropOutShapes.value*Nout[-1]] + dims + [NUM_CLASSES]
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

    #Accuracy
    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)
    
    #AUC score
    a = tf.argmax(logits, 1)
    b = tf.argmax(labels,1)
#    if(dataset=='train'):
#        auc =tf.metrics.auc(a,b,name="auc_train")
#        
#    elif(dataset=='test'):
#       auc =tf.metrics.auc(a,b,name="auc_test")

    auc =tf.metrics.auc(a,b)
    
    tf.summary.scalar("auc",auc[1])      
    
    return auc


#run training on a given train and test data set
#chkptfile can restore saved checkpoint
def run_training(datasets, chkptfile=None):

    learning_rate = FLAGS["learnrate"]
    batchsize = FLAGS["batchsize"]
    numBins = FLAGS["numBins"]
    numHistons = FLAGS["numHistons"]
    
    niter = 5000
    kprob = 0.5

    #For naming the mode file
    k_toSTr = ",".join(map(str,FLAGS["conv"]))
    Nout_toStr = ",".join(map(str,FLAGS["Nout"]))
    m_toStr = ",".join(map(str,FLAGS["mpool"]))
    
    print("starting to train")
    with tf.Graph().as_default():
        start_time = time.time()

        # build the graph
        bins = tf.placeholder(tf.float32, [None, numBins, numHistons])
        labels_ph = tf.placeholder(tf.float32, [None, NUM_CLASSES])
        keep_prob = tf.placeholder(tf.float32)
        global_step = tf.Variable(0, name="global_step", trainable=False)

        logits = inference_singh(bins, keep_prob)

        los = loss(logits, labels_ph)

        train_op = training(los, learning_rate=learning_rate, momentum=FLAGS["momentum"], global_step=global_step)

        eval_correct = evaluation(logits, labels_ph)

        # initilize summaries
        summary = tf.summary.merge_all()

        saver = tf.train.Saver()

        sess = tf.Session()

        
        mod_dir = "mod_%.1e_%d_%.2f"%(learning_rate, niter, kprob) + "_c"+k_toSTr + "_Nout"+Nout_toStr + "_m" + m_toStr
        if(FLAGS["batchnorm"]):
            mod_dir = mod_dir + "_batchnorm"
            
        #write the summary to the folder logdir
        mod_dir = os.path.join(FLAGS["logdir"], mod_dir) 

        train_writer = tf.summary.FileWriter(mod_dir + '/train',
                                      sess.graph)
        test_writer = tf.summary.FileWriter(mod_dir + '/test')

        init = tf.global_variables_initializer()
        initLocal = tf.local_variables_initializer()
        sess.run(init)
        sess.run(initLocal)
       

        #Create an output file to store the AUC performance results for the website
        if FLAGS["outputfile"] != "None":
            outFile = open(FLAGS["outputfile"], 'a')
        
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
                
                sess.run(initLocal)
                summary_str, auc = sess.run([summary,eval_correct], feed_dict=feed_dict)
                
                   
                #Update the events file
                train_writer.add_summary(summary_str, i)
                train_writer.flush()

            #check performance based on validation set
            #if (i + 1) % 1000 == 0 or (i + 1) == niter:
                checkpoint_file = os.path.join(mod_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=global_step)
                
                tmp_feed_dict = {bins : datasets["validate"].getFlatWindow(), labels_ph : datasets["validate"].labels, keep_prob:1.0}
                sess.run(initLocal)
                summary_str, auc = sess.run([summary,eval_correct], feed_dict=tmp_feed_dict)
                
                print('AUC score at step %s: %s' % (i, auc[1]))
                
                   
                test_writer.add_summary(summary_str, i)
                test_writer.flush()



        # final evaluation on test set
        print("PERFORMANCE ON TESTSET:")
        print(datasets["test"].getFlatWindow().shape)
        tmp_feed_dict = {bins: datasets["test"].getFlatWindow(), labels_ph: datasets["test"].labels, keep_prob : 1.0}

        summary_str, auc = sess.run([summary,eval_correct], feed_dict=tmp_feed_dict)
        print('Auc score of %s' % (auc[1]))
        if FLAGS["outputfile"] != "None":
                    outFile.write(FLAGS["outputtag"] + "\t"+FLAGS["bin"]+"\t"+str(auc[1])+"\n")
                 
        test_writer.close()
        train_writer.close()
        
        if FLAGS["outputfile"] != "None":
            outFile.close()

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

#Plot distribution in training, validation and testset
def plotStateDistribution(trainL, validL, testL):
    x =["Train 0", "Train 1", "Valid 0", "Valid 1", "Test 0", "Test 1"]
#    y= [sum(trainL[:,0]), sum(trainL[:,1]), sum(validL[:,0]), sum(validL[:,1]), 
#        sum(testL[:,0]), sum(testL[:,1])]

    #Calculate numbers for each label and normalize through size of the data set
    lenTrain=len(trainL)
    lenValid=len(validL)
    lenTest=len(testL)
    y= [sum(trainL[:,0])/lenTrain, sum(trainL[:,1])/lenTrain, 
        sum(validL[:,0])/lenValid, sum(validL[:,1])/lenValid, 
        sum(testL[:,0])/lenTest, sum(testL[:,1])/lenTest]

    print(y)
    
    #Show bar plots with the results
    plt.bar(np.arange(len(x)), y, align='center', alpha=0.5)
    plt.xticks(np.arange(len(x)), x)
    plt.ylabel('frequency')
    plt.xlabel('data set - label')
    plt.title('Label distribution')

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--logdir", help = "log directory to save results of a run (used for tensorboard)",default="./logs")
    parser.add_argument("-i", help="Feature input data (bins)", dest= "data")
    parser.add_argument("-l", help="Label data", dest="label")
    parser.add_argument("-t", help="Additional test set iff you want a seperate test set", dest="test",default="")
    parser.add_argument("--labelsTest", help="Additional label file for the test set iff you want a seperate test set", dest="labelsTest",default="")
    parser.add_argument("-k", help="Sizes of all convolution filters, comma separated e.g. \"10,10\"",dest="conv", default="10,10")
    parser.add_argument("--Nout", help="Numbers of output channels after each convolution convolution step, comma separated e.g. \"20,50\"",default="20,50")
    parser.add_argument("-m", help="Pool size for each the maxpooling step, comma separated e.g. \"2,2\"",dest="mpool",default="2,2")
    parser.add_argument("--hiddenDims",help="Hidden dimensions, comma separated e.g. \"1000,100\"",default="200")
    parser.add_argument("--learnrate", type=float, default=0.005)
    parser.add_argument("--momentum", default=None)
    parser.add_argument("--batchsize", default=200)
    parser.add_argument("--fastdatadir", help="Directory to load preprocessed data in a fast way", default="")
    parser.add_argument("--saveFastdatadir", help="Directory to save parsed data, which is splitted into training, validation and test, for fast loading the next time (the directory need to exists already).",default="")
    parser.add_argument("--plot", help="Plot the distribution of the 0/1 labels in the training, validation and test set", action='store_true')
    parser.add_argument("--batchnorm", help="If set performs batch normalization instead of drop-out", action='store_true')
    parser.add_argument("--outputfile", help="Save the last auc score in this file",default ="None")
    parser.add_argument("-b", help="give a bin for the case you want to train only on one bin", dest="bin",default=-1)
    parser.add_argument("--outputtag", help="String to identify different runs in the output file.")

    args = parser.parse_args()

    #Directories to load and save the training and test data in numpy format
    #Empty string, if no fast data should be loaded / saved
    fastdatadir = args.fastdatadir
    saveFastdatadir = args.saveFastdatadir
    testfile = args.test
    labelsTest = args.labelsTest
    FLAGS["bin"]=args.bin
    oneBin = FLAGS["bin"]
    
    FLAGS["logdir"] = args.logdir
    FLAGS["learnrate"] = args.learnrate
    FLAGS["momentum"] = args.momentum
    FLAGS["batchsize"] = args.batchsize
    FLAGS["batchnorm"] = args.batchnorm
    FLAGS["outputfile"] = args.outputfile
    FLAGS["outputtag"] = args.outputtag
    
    FLAGS["mpool"]=args.mpool
    
    #Get convolution dimensions
    convDims = args.conv.split(",")
    convDims = list(map(int,convDims))
    FLAGS["conv"] = convDims
    
    #Get number of output channels after each convolution step
    noutDims = args.Nout.split(",")
    noutDims = list(map(int,noutDims))
    FLAGS["Nout"] = noutDims
    
    #Get number of pool for each maxpooling step
    mpoolsDims = args.mpool.split(",")
    mpoolsDims = list(map(int,mpoolsDims))
    FLAGS["mpool"] = mpoolsDims
    
    #Check if all three have the same length (otherwise print an error)
    if(len(convDims) != len(noutDims)):
        print("Anzahl der Convolution Filter und der Convolution Output Channel Zahlen stimmt nicht überein!")
        exit()
    
    if(len(convDims) != len(mpoolsDims)):
        print("Anzahl der Convolution Filter und der Maxpool Filter stimmt nicht überein!")
        exit()
        
    #Get hidden dimensions
    dimensions=args.hiddenDims.split(",")
    dimensions=list(map(int,dimensions))
    
    FLAGS["dims"]=dimensions
    
    #Read preparsed data from a directory
    if(len(fastdatadir)>0):
        #Read directory
        train = np.load(os.path.join(fastdatadir, "train1_win.npz.npy"))
        trainL = np.load(os.path.join(fastdatadir, "train1_lab.npz.npy"))
        valid = np.load(os.path.join(fastdatadir, "valid1_win.npz.npy"))
        validL = np.load(os.path.join(fastdatadir, "valid1_lab.npz.npy"))
        test = np.load(os.path.join(fastdatadir, "test1_win.npz.npy"))
        testL = np.load(os.path.join(fastdatadir, "test1_lab.npz.npy"))

        FLAGS["numHistons"]=len(train[0][0])
        FLAGS["numBins"]=len(train[0])
        
        print("finished reading in data!")
        
    else:
        #Einlesen von unseren Input files und anschließend split in training und test
        labelFilename = args.label
        trainingData = args.data
    
        labelFile = open(labelFilename)
        labelDict = {}
        for line in labelFile.readlines():
            if not line.startswith("##"):
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
        
        if(testfile=="" or labelsTest==""):
            ratio=0.1
            #split the dataset random in two parts and the labels with them
            test, testL, train, trainL = splitRandom(ratio,X,y)
            ratio=0.9
            train, trainL, valid, validL = splitRandom(ratio,train,trainL)
        else:
            testset={}
            #Einlesen von unseren Input files und anschließend split in training und test
            labelFilename = args.label
            trainingData = args.data
            
            labelFile = open(labelFilename)
            labelDict2 = {}
            for line in labelFile.readlines():
                if not line.startswith("##"):
                    lineSplit=line.split()
                    #Convert each label to a binary vector
                    if(int(lineSplit[2])==0):
                        labelDict2[lineSplit[0]]=[1,0]
                    elif(int(lineSplit[2])==1):
                        labelDict2[lineSplit[0]]=[0,1]
                    else:
                        print("Fehler beim Parsen des Input-Files.")
            with open(testfile) as test:
                #Name of the data set (from the header)
                datasetTest=test.readline().rstrip()[2:]
                #All modifications
                modificationsTest=test.readline().rstrip()[2:].split(" ")
        
                for line in test.readlines():
                    line=line.rstrip()
                    if(line.startswith('#')):
                        lineSplit=line.split(" ")
                        geneID=lineSplit[0]
                        #Remove the hashtag at the beginning of the line
                        geneID=geneID[1:]
                        testset[geneID]=[]
                    else:
                        valueList=line.split(",")
                        valueList=list(map(float,valueList))
                        testset[geneID].append(valueList)
            
            #Sort labels according to the feature list
            #Maybe for some genes no GENCODE entry could be found, these are only in the features list
            testL=[]
            test=[]
            #If not all bins should be used
            for geneID in testset:
                testL.append(labelDict2[geneID])
                valueMatrix=testset[geneID]
                
                #Transpose matrix (first number Bins, then histone modifications)
                valueMatrix=list(map(list, zip(*valueMatrix)))
                test.append(valueMatrix)
            
            #Konvertieren in ein numpy array
            test=np.array(test)
            testL=np.array(testL)
            ratio=0.9
            train, trainL, valid, validL = splitRandom(ratio,X,y)
            
        print("finished reading in data!")
        
        #Saved parsed data in a numpy file
        if(len(saveFastdatadir)>0):
            np.save(os.path.join(saveFastdatadir, "train1_win.npz.npy"), train)
            np.save(os.path.join(saveFastdatadir, "train1_lab.npz.npy"),trainL)
            np.save(os.path.join(saveFastdatadir, "valid1_win.npz.npy"),valid)
            np.save(os.path.join(saveFastdatadir, "valid1_lab.npz.npy"),validL)
            np.save(os.path.join(saveFastdatadir, "test1_win.npz.npy"),test)
            np.save(os.path.join(saveFastdatadir, "test1_lab.npz.npy"),testL)
            print("saving parsed data to the directory")
    # For the case you want only one bin extract that one bin
    if(int(oneBin)>-1):
        test= test[:,oneBin]
        train= train[:,oneBin]
        valid= valid[:,oneBin]
    datasets = {}
    datasets["train"]= Dataset(train,trainL)
    datasets["validate"] = Dataset(valid,validL)
    datasets["test"] = Dataset(test,testL)

    #Plot distribution of the labels in the three sets
    if(args.plot):
        print("Plotting the label distribution")
        plotStateDistribution(trainL, validL, testL)
    run_training(datasets)


