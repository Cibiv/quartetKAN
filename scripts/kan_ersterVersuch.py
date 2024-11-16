import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import logging
import yaml
import sys
from datetime import datetime

#tfkan muss im conda env installiert sein 
from tfkan.models import KAN  #importieren von KAN model aus tfkan
from tfkan.layers import GraphAttention, GraphConvolution  #layers aus tfkan
from callbacks import TrainingPlot  

"""
Wie in train.py wird als erstes Argument ein
 yaml config file uebergeben 
 (das Beispielfile hab ich in config gestellt). 
 train kann als zweites argument gegeben werden 
um das trainieren direkt zu starten
"""
class KANModel:
    def __init__(self, path):
        """
        KAN model wird mit den Parametern aus dem config file initialisiert
        """
        try:
            #config file laden:
            with open(path, 'r') as stream:
                try: 
                    data = yaml.safe_load(stream)
                    #parameter aus dem config file entnehmen:
                    try: 
                        timestamp = self.datetime_stamp()
                        self.dropout = data['dropout'] 
                        self.layers = data['layers']  
                        self.learning_rate = data['learning_rate']
                        self.batch_size = data['batch_size'] 
                        self.epochs = data['epochs'] 
                        self.no_epochs_dataset = data['no_epochs_dataset']  
                        self.perc_eval = data['perc_eval']  
                        self.data_file = data['data_file']  
                        self.data_length = data['data_length'] 
                        self.save_network_to = f"../models/{data['save_network_to']}_{timestamp}" if 'save_network_to' in data \
                                else None
                        self.output_file = f"../log/{data['save_network_to']}_{timestamp}.log" if 'save_network_to' in data \
                                else None
                        self.self.weight_initializer = data['weight_initializer'] if 'weight_initializer' in data \
                                else 'xavier'
                        self.transfer_function = data['transfer_function'] if 'transfer_function' in data else 'relu'
                        self.activation_function = data['activation_function'] if 'activation_function' in data \
                                else 'sigmoid'
                        self.cost_function = data.get('cost_function', 'binary_crossentropy') 
                        self.optimizer_type =  self.opt = data['optimizer'] if 'optimizer' in data else 'adam'
                        self.offset = data['offset']  
                        self.shuf_buffer = data['shuf_buffer'] if 'shuf_buffer' in data else 1000
                        self.seed_init = data['seed_init'] if 'seed_init' in data else None
                        self.seed_shuffle = data['seed_shuffle'] if 'seed_shuffle' in data else None
                        self.beta1 = data['beta1'] if 'beta1' in data else 0.9
                        self.beta2 = data['beta2'] if 'beta2' in data else 0.999
                        self.use_bias = data['use_bias'] if 'use_bias' in data else True
                        
                        self.log_level = data['log_level'] if 'log_level' in data else 'info'
                        tf.compat.v1.logging.set_verbosity({'debug': 10, 'error': 40, 'fatal': 50, 'info': 20, 'warn': 30}
                                                    [data['log_level']] if 'log_level' in data else 20)
                    except KeyError as e:
                            logging.error("Key error. Please refer to config file spec for more details", e)
                            exit(1)

                except yaml.scanner.ScannerError as e:
                    logging.error("yaml config is not valid. Please follow spec and provide valid yaml.", e)
                    exit(1)
        except FileNotFoundError:
            logging.error("Config file not found")
            exit(1)

        #logging
        self.logger = self.get_logger()
        self.logger.info("KAN initialized with parameters: " + str(data))

    def datetime_stamp(self):
        t = datetime.now()
        t = str(t)
        t = t.split('.')[0]
        t = t.replace(' ', '_')
        t = t.replace('-', '')
        t = t.replace(':', '')
        return t


    def get_logger(self):
       
        logger = logging.getLogger('training')
        fh = logging.FileHandler(self.output_file)
        sh = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt = '%d.%m.%y %H:%M:%S')
        fh.setFormatter(formatter)
        sh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(sh)
        logger.setLevel(logging.INFO)
        return logger

    def parse_row(self, tf_string):
        data = tf.io.decode_csv(tf.expand_dims(tf_string, axis = 0), self.get_record_defaults())
        features = data[self.offset:-1]
        features = tf.stack(features, axis = -1)
        label = data[-1]
        features = tf.squeeze(features, axis = 0)
        label = tf.squeeze(label, axis = 0)
        return features, label

    def create_dataset(self):
        data = tf.data.TextLineDataset([self.data_file])
        data = data.skip(1).shuffle(self.shuf_buffer, seed=self.seed_shuffle)
        
        val_size = int(self.data_length * self.perc_eval)
        train_data = data.skip(val_size).batch(self.batch_size).repeat()
        val_data = data.take(val_size).batch(self.batch_size).repeat()

        train_data = train_data.map(self.parse_row, num_parallel_calls=6)
        val_data = val_data.map(self.parse_row, num_parallel_calls=6)

        iterator_train = iter(train_data)
        iterator_val = iter(val_data)

        return iterator_train, iterator_val
    
    def cost_function(self):
        if self.cost_f == 'sigmoid cross-entropy':
            self.loss = 'binary_crossentropy'
        elif self.cost_f == 'MSE':
            self.loss = 'mean_squared_error'
        else:
            self.logger.error(f"Cost function {self.cost_f} not implemented.")
            exit(1)

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
    
    def get_transfer(self):
        if self.transfer_function == 'sigmoid':
            transfer = tf.nn.sigmoid
        elif self.transfer_function == 'relu':
            transfer = tf.nn.relu
        else:
            self.logger.error(f'Transfer function {self.transfer_function} is not implemented')
            exit(2)
        return transfer


    def build_kan_model(self):
        """
        KAN model mit tfkan bauen: 
        """
        self.model = KAN(
            layers=self.layers,
            activation = self.get_activation(),
            #activation=self.transfer_function,
            #output_activation=self.activation_function,
            dropout_rate=self.dropout,
            use_bias=True,
            kernel_initializer=self.weight_initializer,
            seed=self.seed_init
        )
        self.logger.info("KAN model architecture built")

    def compile_model(self):
        """
        Kompilieren des KAN models mit dem optimizer und loss function aus dem Config
        """
        if self.optimizer_type == "adam":
            optimizer = Adam(learning_rate=self.learning_rate, beta_1 = self.beta1, beta_2 = self.beta2, amsgrad = False)
        else:
            self.logger.error(f"Optimizer {self.optimizer_type} not implemented.")
            exit(1)

        self.model.compile(
            optimizer=optimizer,
            loss=self.cost_function,
            metrics=['accuracy']
        )
        self.logger.info("Model compiled with optimizer and loss function")


    def get_model_summary(self):
        summary = []
        self.model.summary(print_fn = lambda x: summary.append(x))
        return "\n".join(summary)

    def train(self):
        """
        trainieren mit den Daten 
        """
        train_it, val_it = self.create_dataset()
        
        #build and compile the model 
        self.build_kan_model()
        self.compile_model()

        model_checkpoint = ModelCheckpoint(filepath = self.save_network_to + '_{epoch:03d}-{binary_accuracy:.3f}-{val_binary_accuracy:.3f}', 
                                           monitor = 'val_binary_accuracy', 
                                           mode = 'max', 
                                           save_best_only = False, 
                                           verbose = 1)
        
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        plot_curves = TrainingPlot(self.save_network_to, self.logger)

        steps_per_epoch = int(self.data_length * (1 - self.perc_eval) / self.batch_size / self.no_epochs_dataset)
        val_steps = int(self.data_length * self.perc_eval / self.batch_size)

        
        self.model.fit(
            train_it,
            validation_data=val_it,
            epochs=self.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=val_steps,
            callbacks=[early_stopping, reduce_lr, model_checkpoint, plot_curves],
            verbose=1
        )
        self.logger.info("Training beendet")

if __name__ == '__main__':
    try:
        #read the config file path from command line arguments
        kan_model = KANModel(sys.argv[1])
        #train network
        kan_model.train()
    except IndexError as e:
        logging.error("Please specify a path to a config file as first commandline argument")
        exit(1)
