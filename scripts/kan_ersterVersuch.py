# import libraries
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam

import logging
import yaml
import sys
from datetime import datetime

from callbacks import TrainingPlot

from tfkan import layers
from tfkan.layers import DenseKAN


"""
Neural network implementation (architecture: KAN) used for classification (distinguishing between Farris- and Felsenstein-type trees).
It expects the path to a YAML config file as first argument (such as the ones specified in ./config). As second optional argument, the keyword 'train' can be given to start the training directly.
"""
class MLP:
    # initializes the multi-layer perceptron with the parameters as specified in the config file
    def __init__(self, path):
        try:
            with open(path, 'r') as stream:
                try:
                    data = yaml.safe_load(stream)
                    try:
                        # timestamp used for naming the network
                        timestamp = self.datetime_stamp()
                        # fraction of units to drop in dropout layer
                        self.dropout = data['dropout']
                        # list with node numbers for every layer including in- and output
                        self.layers = data['layers']
                        # learning rate for optimizer
                        self.learning_rate = data['learning_rate']
                        # size of batch
                        self.batch_size = data['batch_size']
                        # no of epochs
                        self.epochs = data['epochs']
                        # no of epochs it needs to process complete trainset 
                        self.no_epochs_dataset = data['no_epochs_dataset']
                        # fraction of dataset used for validation
                        self.perc_eval = data['perc_eval']
                        # path to training dataset saved as csv-file
                        self.data_file = data['data_file']
                        # number of datapoints in data set
                        self.data_length = data['data_length']
                        # model path
                        self.save_network_to = f"../models/{data['save_network_to']}_{timestamp}" if 'save_network_to' in data \
                            else None
                        # logfile path
                        self.output_file = f"../log/{data['save_network_to']}_{timestamp}.log" if 'save_network_to' in data \
                            else None
                        # initializer for weight matrices
                        self.weight_initializer = data['weight_initializer'] if 'weight_initializer' in data \
                            else 'xavier'
                        # activation function for interim layers
                        self.transfer_function = data['transfer_function'] if 'transfer_function' in data else 'relu'
                        # activation function for output layer
                        self.activation_function = data['activation_function'] if 'activation_function' in data \
                            else 'sigmoid'
                        # cost function on which network is optimized
                        self.cost_f = data['cost_function'] if 'cost_function' in data else 'cross-entropy'
                        # optimizer for the network training
                        self.opt = data['optimizer'] if 'optimizer' in data else 'adam'
                        # number of columns in dataset file to be ignored
                        self.offset = data['offset']
                        # number of points which are shuffled at same time
                        self.shuf_buffer = data['shuf_buffer'] if 'shuf_buffer' in data else 1000
                        # set tensorflow logging level
                        self.log_level = data['log_level'] if 'log_level' in data else 'info'
                        tf.compat.v1.logging.set_verbosity({'debug': 10, 'error': 40, 'fatal': 50, 'info': 20, 'warn': 30}
                                                 [data['log_level']] if 'log_level' in data else 20)
                        # seed for parameter initialization
                        self.seed_init = data['seed_init'] if 'seed_init' in data else None
                        # seed for shuffling
                        self.seed_shuffle = data['seed_shuffle'] if 'seed_shuffle' in data else None
                        # decay rates for adam optimizer
                        self.beta1 = data['beta1'] if 'beta1' in data else 0.9
                        self.beta2 = data['beta2'] if 'beta2' in data else 0.999
                        # use bias vector in linear transformation
                        self.use_bias = data['use_bias'] if 'use_bias' in data else True

                    except KeyError as e:
                        logging.error("Key error. Please refer to config file spec for more details", e)
                        exit(1)

                except yaml.scanner.ScannerError as e:
                    logging.error("yaml config is not valid. Please follow spec and provide valid yaml.", e)
                    exit(1)

        except FileNotFoundError:
            logging.error("File not found.")
            exit(1)

        # create logger and log all config parameters        
        self.logger = self.get_logger()
        self.logger.info('Parameters of Graph Neural Network:\n'
           + '\n'.join("%s: %s" % item for item in vars(self).items()) + '\n')


    # generate datetime stamp to append to model name
    def datetime_stamp(self):
        t = datetime.now()
        t = str(t)
        t = t.split('.')[0]
        t = t.replace(' ', '_')
        t = t.replace('-', '')
        t = t.replace(':', '')
        return t


    # set logger to log with date and time and to log to the set logfile and the stdout 
    def get_logger(self):

        logger = logging.getLogger('training')

        fh = logging.FileHandler(self.output_file)
        sh = logging.StreamHandler(sys.stdout)
        fmter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt = '%d.%m.%y %H:%M:%S')

        fh.setFormatter(fmter)
        sh.setFormatter(fmter)

        logger.addHandler(fh)
        logger.addHandler(sh)
        logger.setLevel(logging.INFO)

        return logger


    # set tensorflow record defaults (see https://www.tensorflow.org/api_docs/python/tf/io/decode_csv for details)
    def get_record_defaults(self):
        zeros = tf.zeros(shape = (1,), dtype = tf.float32)
        ones = tf.ones(shape = (1,), dtype = tf.float32)
        return [zeros] * (self.layers[0] + self.offset) + [ones]


    # parses row of input file and splits data into features and label
    def parse_row(self, tf_string):
        data = tf.io.decode_csv(tf.expand_dims(tf_string, axis = 0), self.get_record_defaults())

        # features are in all columns after the first offset columns and before the last column
        features = data[self.offset:-1]
        features = tf.stack(features, axis = -1)

        # class label is in the last column
        label = data[-1]
        features = tf.squeeze(features, axis = 0)
        label = tf.squeeze(label, axis = 0)
        return features, label


    # defines input pipeline for train data and returns dataset containing labelled features
    def create_dataset(self):
        # create TextLineDataset, ship header and shuffle it
        data = tf.data.TextLineDataset([self.data_file])
        data = data.skip(1).shuffle(buffer_size = self.shuf_buffer, seed = self.seed_shuffle)

        # the first perc_eval fraction of data is used for validation, the rest for training
        val_size = int(self.data_length * self.perc_eval)
        train_data = data.skip(val_size).batch(self.batch_size).repeat()
        val_data = data.take(val_size).batch(self.batch_size).repeat()

        # define which part of data are loaded as features and label
        train_data = train_data.map(self.parse_row, num_parallel_calls = 6)
        val_data = val_data.map(self.parse_row, num_parallel_calls = 6)

        # create an iterator for data
        iterator_train = iter(train_data)
        iterator_val = iter(val_data)

        return iterator_train, iterator_val


    # returns the cost function as specified in the config file (MSE or sigmoid cross-entropy)
    def cost_function(self):
        if self.cost_f == 'sigmoid cross-entropy':
            self.loss = 'binary_crossentropy'
        elif self.cost_f == 'MSE':
            self.loss = 'mean_squared_error'
        else:
            self.logger.error(f"Cost function {self.cost_f} not implemented.")
            exit(1)


    # returns the optimization algorithm used for training the network as specified in the config file (GSD or Adam)
    def optimizer(self):
        if self.opt == "adam":
            optm = Adam(learning_rate = self.learning_rate, beta_1 = self.beta1, beta_2 = self.beta2, amsgrad = False)
        else:
            self.logger.error(f"Optimizer {self.opt} not implemented.")
            exit(1)
        return optm
    

    # set activation function as defined by config file
    def get_activation(self):
        if self.activation_function == 'sigmoid':
            activation = tf.nn.sigmoid
        elif self.activation_function == 'softmax':
            activation = tf.nn.softmax
        elif self.activation_function == 'no':
            activation = None
        else:
            self.logger.error(f'Activation function {self.activation_function} is not implemented')
            exit(2)
        return activation

    
    # set transfer function as defined by config file
    def get_transfer(self):
        if self.transfer_function == 'sigmoid':
            transfer = tf.nn.sigmoid
        elif self.transfer_function == 'relu':
            transfer = tf.nn.relu
        else:
            self.logger.error(f'Transfer function {self.transfer_function} is not implemented')
            exit(2)
        return transfer


    # define network architecture as defined by config
    def get_model(self, w_init, b_init):

        #####################
        self.model = tf.keras.models.Sequential([
            DenseKAN(4),
            DenseKAN(1)
        ])
        model.build(input_shape=(None, 10)) 
        #####################
        
        # use activation, transfer function
        #activation = self.get_activation()
        #transfer = self.get_transfer()
        
        # build sequential architecture
        #self.model = keras.Sequential()

        # set input layer with as many nodes as first value in layer list (should be = #features)
        #self.model.add(Input(shape = (self.layers[0],)))

        # add interim layers with transfer function as activation and dropout
        #for l in range(1, len(self.layers) - 1):
        #    self.model.add(Dense(self.layers[l], activation = transfer, use_bias = self.use_bias, kernel_initializer = w_init, bias_initializer = b_init))
        #    self.model.add(Dropout(self.dropout))

        # add output layer with activation
        #self.model.add(Dense(self.layers[-1], activation = activation, use_bias = self.use_bias, kernel_initializer = w_init, bias_initializer = b_init))
    

    # initialize parameters
    def init_params(self):

        # initialize weight parameters by specified initializer 
        if self.weight_initializer == "xavier":
            w_init = tf.keras.initializers.GlorotNormal(seed = self.seed_init)  # Weight initializers
        else:
            self.logger.error(f"Weight initializer {self.weight_initializer} not implemented.")
            exit(1)

        # initialize bias parameters with 0's
        b_init = keras.initializers.Zeros()

        return w_init, b_init
    

    # return summary of model architecture
    def get_model_summary(self):
        summary = []
        self.model.summary(print_fn = lambda x: summary.append(x))
        return "\n".join(summary)
    

    # train neural network
    def train(self):

        # create datasets
        train_it, val_it = self.create_dataset()

        # init parameters and define model architecture
        w_init, b_init = self.init_params()
        self.get_model(w_init, b_init)
        self.logger.info(self.get_model_summary())

        # set optimizer, cost function and accuracy
        optm = self.optimizer()
        self.cost_function()
        bin_acc = keras.metrics.BinaryAccuracy(name = "binary_accuracy", dtype = None, threshold = 0.5)
        
        # compile model with optimizer, loss and accuracy
        #self.model.compile(loss = self.loss, optimizer = optm, metrics = [bin_acc])
        #########################
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
            loss='mse', metrics=['mae'])
        ############################
        # define that always the latest best model is saved on specified path according to validation accuracy
        model_checkpoint = ModelCheckpoint(filepath = self.save_network_to + '_{epoch:03d}-{binary_accuracy:.3f}-{val_binary_accuracy:.3f}', 
                                           monitor = 'val_binary_accuracy', 
                                           mode = 'max', 
                                           save_best_only = False, 
                                           verbose = 1)
        
        # define callback which reduces the lr after n epochs without improvement 
        reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.8, patience = 5, verbose = 1)
        # define callback which stops training after n epochs without improvement
        early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10, verbose = 1)
        # define callback which logs accuracy, loss and learning rate throughout the epochs
        plot_curves = TrainingPlot(self.save_network_to, self.logger)

        # set number of batches per epoch
        steps_epoch = int(self.data_length * (1 - self.perc_eval) / self.batch_size / self.no_epochs_dataset)
        val_steps_epoch = int(self.data_length * (self.perc_eval) / self.batch_size)

        # train model on training data and validate it after each epoch
        #self.model.fit(train_it, 
        #               steps_per_epoch = steps_epoch,
        #               epochs = self.epochs, 
        #               callbacks = [early_stopping, reduce_lr, model_checkpoint, plot_curves], 
        #               verbose = 1, 
        #               shuffle = True, 
        #               validation_data = val_it, 
        #               validation_steps = val_steps_epoch)
        #############################
        self.model.ft(train_it,
                      val_it, 
                      epochs = self.epochs, 
                      batch_size =self.batch_size)
        #############################

if __name__ == '__main__':
    try:
        # create MLP instance taking config path in first sys argument as input
        nn = MLP(sys.argv[1])
        # train network
        nn.train()
    except IndexError as e:
        logging.error("Please specify a path to a config file as first commandline argument")
        exit(1)
