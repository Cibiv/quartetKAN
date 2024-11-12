# import libraries
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense,Input,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import logging
import yaml
import sys
from datetime import datetime

from callbacks import TrainingPlot


"""
Neural network implementation (architecture: multi-layer perceptron) used for classification (distinguishing between Farris- and Felsenstein-type trees).
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
                        timestamp = self.datetime_stamp()
                        self.dropout = data['dropout']
                        self.layers = data['layers']
                        self.learning_rate = data['learning_rate']
                        self.batch_size = data['batch_size']
                        self.epochs = data['epochs']
                        self.no_epochs_dataset = data['no_epochs_dataset']
                        self.perc_eval = data['perc_eval']
                        self.multiple_data = data['multiple_data']
                        self.data_file = data['data_file']
                        self.data_length = data['data_length']
                        self.save_network_to = f"../models/{data['save_network_to']}_{timestamp}" if 'save_network_to' in data \
                            else None
                        self.output_file = f"../log/{data['save_network_to']}_{timestamp}.log" if 'save_network_to' in data \
                            else None
                        self.weight_initializer = data['weight_initializer'] if 'weight_initializer' in data \
                            else 'xavier'
                        self.transfer_function = data['transfer_function'] if 'transfer_function' in data else 'relu'
                        self.activation_function = data['activation_function'] if 'activation_function' in data \
                            else 'sigmoid'
                        self.cost_f = data['cost_function'] if 'cost_function' in data else 'cross-entropy'
                        self.opt = data['optimizer'] if 'optimizer' in data else 'adam'
                        self.offset = data['offset']
                        self.shuf_buffer = data['shuf_buffer'] if 'shuf_buffer' in data else 1000
                        self.log_level = data['log_level'] if 'log_level' in data else 'info'
                        tf.compat.v1.logging.set_verbosity({'debug': 10, 'error': 40, 'fatal': 50, 'info': 20, 'warn': 30}
                                                 [data['log_level']] if 'log_level' in data else 20)
                        self.seed_init = data['seed_init'] if 'seed_init' in data else None
                        self.seed_shuffle = data['seed_shuffle'] if 'seed_shuffle' in data else None
                        self.beta1 = data['beta1'] if 'beta1' in data else 0.9
                        self.beta2 = data['beta2'] if 'beta2' in data else 0.999
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
        
        self.logger = self.get_logger()
        self.logger.info('Parameters of Graph Neural Network:\n'
           +'\n'.join("%s: %s" % item for item in vars(self).items())+'\n')


    # generate datetime stamp to append to model name
    def datetime_stamp(self):
        t = datetime.now()
        t = str(t)
        t = t.split('.')[0]
        t = t.replace(' ','_')
        t = t.replace('-','')
        t = t.replace(':','')
        return t


    # writes network parameters to log file (as specified in the config file)
    def get_logger(self):

        logger = logging.getLogger('training')

        fh = logging.FileHandler(self.output_file)
        sh = logging.StreamHandler(sys.stdout)
        fmter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%d.%m.%y %H:%M:%S')

        fh.setFormatter(fmter)
        sh.setFormatter(fmter)

        logger.addHandler(fh)
        logger.addHandler(sh)
        logger.setLevel(logging.INFO)

        return logger


    # set tensorflow record defaults (see https://www.tensorflow.org/api_docs/python/tf/io/decode_csv for details)
    def get_record_defaults(self):
        zeros = tf.zeros(shape=(1,), dtype=tf.float32)
        ones = tf.ones(shape=(1,), dtype=tf.float32)
        return [zeros] * (self.layers[0] + self.offset) + [ones]


    # parses row of input file and splits data into features and label
    def parse_row(self, tf_string):
        data = tf.io.decode_csv(
            tf.expand_dims(tf_string, axis=0), self.get_record_defaults())
        features = data[self.offset:-1]
        features = tf.stack(features, axis=-1)
        label = data[-1]
        features = tf.squeeze(features, axis=0)
        label = tf.squeeze(label, axis=0)
        return features, label


    # defines input pipeline for train data and returns dataset containing labelled features
    def create_dataset(self):
        data = tf.data.TextLineDataset([self.data_file])
        data= data.skip(1).shuffle(buffer_size=self.shuf_buffer, seed=self.seed_shuffle)

        val_size = int(self.data_length * self.perc_eval)
        train_data = data.skip(val_size).batch(self.batch_size).repeat()
        val_data = data.take(val_size).batch(self.batch_size).repeat()

        train_data=train_data.map(self.parse_row, num_parallel_calls=6)
        val_data=val_data.map(self.parse_row, num_parallel_calls=6)

        iterator_train = iter(train_data)
        iterator_val = iter(val_data)

        return iterator_train, iterator_val


    # returns the cost function as specified in the config file (MSE or sigmoid cross-entropy)
    def cost_function(self):
        if self.cost_f=='sigmoid cross-entropy':
            self.loss='binary_crossentropy'
        elif self.cost_f == 'MSE':
            self.loss=tf.losses.mean_squared_error(y, self.prediction(x, False, True))
        else:
            self.logger.error(f"Cost function {self.cost_f} not implemented.")
            exit(1)


    # returns the optimization algorithm used for training the network as specified in the config file (GSD or Adam)
    def optimizer(self):
        if self.opt=="adam":
            optm = Adam(learning_rate=self.learning_rate, beta_1=self.beta1, beta_2=self.beta2, amsgrad=False)
        else:
            self.logger.error(f"Optimizer {self.opt} not implemented.")
            exit(1)
        return optm


    def get_activation(self):
        if self.activation_function == 'sigmoid':
            activation=tf.nn.sigmoid
        elif self.activation_function == 'softmax':
            activation=tf.nn.softmax
        elif self.activation_function == 'no':
            activation=None
        else:
            self.logger.error(f'Activation function {self.activation_function} is not implemented')
            exit(2)
        return activation

    
    def get_transfer(self):
        if self.transfer_function == 'sigmoid':
            transfer=tf.nn.sigmoid
        elif self.transfer_function == 'relu':
            transfer=tf.nn.relu
        else:
            self.logger.error(f'Transfer function {self.transfer_function} is not implemented')
            exit(2)
        return transfer


    def get_model(self, w_init, b_init):
        
        activation=self.get_activation()
        transfer = self.get_transfer()
        
        self.model = keras.Sequential()
        self.model.add(Input(shape=(self.layers[0],)))
        for l in range(1,len(self.layers)-1):
            self.model.add(Dense(self.layers[l], activation=transfer, use_bias=self.use_bias, kernel_initializer=w_init, bias_initializer=b_init))
            self.model.add(Dropout(self.dropout))
        self.model.add(Dense(self.layers[-1], activation=activation, use_bias=self.use_bias, kernel_initializer=w_init, bias_initializer=b_init))
    

    def init_params(self):

        if self.weight_initializer=="xavier":
            w_init = tf.keras.initializers.GlorotNormal(seed=self.seed_init)  # Weight initializers
        else:
            self.logger.error(f"Weight initializer {self.weight_initializer} not implemented.")
            exit(1)

        b_init = keras.initializers.Zeros()

        return w_init, b_init


    def get_model_summary(self):
        summary = []
        self.model.summary(print_fn=lambda x: summary.append(x))

        return "\n".join(summary)


    def train(self):

        train_it, val_it = self.create_dataset()

        w_init, b_init = self.init_params()
        self.get_model(w_init, b_init)
        self.logger.info(self.get_model_summary())

        optm = self.optimizer()
        self.cost_function()

        bin_acc = keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5)
        self.model.compile(loss=self.loss, optimizer=optm, metrics=[bin_acc])
        model_checkpoint = ModelCheckpoint(filepath = self.save_network_to+'_{epoch:03d}-{binary_accuracy:.3f}-{val_binary_accuracy:.3f}', 
                                           monitor = 'val_binary_accuracy', 
                                           mode = 'max', 
                                           save_best_only = False, 
                                           verbose = 1)

        reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.8, patience = 10, verbose = 1)
        early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10, verbose = 1)
        plot_curves = TrainingPlot(self.save_network_to, self.logger)

        steps_epoch = int(self.data_length*(1-self.perc_eval)/self.batch_size/self.no_epochs_dataset)
        val_steps_epoch = int(self.data_length*(self.perc_eval)/self.batch_size)

        self.model.fit(train_it, 
                       steps_per_epoch = steps_epoch,
                       epochs = self.epochs, 
                       callbacks = [early_stopping, reduce_lr, model_checkpoint, plot_curves], 
                       verbose = 1, 
                       shuffle = True, 
                       validation_data = val_it, 
                       validation_steps = val_steps_epoch)



if __name__ == '__main__':
    try:
        nn = MLP(sys.argv[1])
        nn.train()
    except IndexError as e:
        logging.error("Please specify a path to a config file as first commandline argument")
        exit(1)
