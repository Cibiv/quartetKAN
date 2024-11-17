import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import logging
import yaml
import sys
from datetime import datetime

# tfkan muss im conda env installiert sein
from tfkan import layers
from tfkan.layers import DenseKAN, Conv2DKAN
from callbacks import TrainingPlot


class KANModel:
    def __init__(self, path):
        """
        KAN model wird mit den Parametern aus dem config file initialisiert
        """
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
                        self.data_file = data['data_file']
                        self.data_length = data['data_length']
                        self.save_network_to = f"../models/{data['save_network_to']}_{timestamp}" if 'save_network_to' in data else None
                        self.output_file = f"../log/{data['save_network_to']}_{timestamp}.log" if 'save_network_to' in data else None
                        self.weight_initializer = data.get('weight_initializer', 'xavier')
                        self.transfer_function = data.get('transfer_function', 'relu')
                        self.activation_function = data.get('activation_function', 'sigmoid')
                        self.cost_function = data.get('cost_function', 'binary_crossentropy')
                        self.optimizer_type = data.get('optimizer', 'adam')
                        self.offset = data['offset']
                        self.shuf_buffer = data.get('shuf_buffer', 1000)
                        self.seed_init = data.get('seed_init', None)
                        self.seed_shuffle = data.get('seed_shuffle', None)
                        self.beta1 = data.get('beta1', 0.9)
                        self.beta2 = data.get('beta2', 0.999)
                        self.use_bias = data.get('use_bias', True)
                        
                        self.log_level = data.get('log_level', 'info')
                        tf.compat.v1.logging.set_verbosity({'debug': 10, 'error': 40, 'fatal': 50, 'info': 20, 'warn': 30}.get(data['log_level'], 20))
                    except KeyError as e:
                        logging.error("Key error. Please refer to config file spec for more details", e)
                        exit(1)
                except yaml.scanner.ScannerError as e:
                    logging.error("yaml config is not valid. Please provide valid yaml.", e)
                    exit(1)
        except FileNotFoundError:
            logging.error("Config file not found")
            exit(1)

        self.logger = self.get_logger()
        self.logger.info("KAN initialized with parameters: " + str(data))

    def datetime_stamp(self):
        t = datetime.now().strftime('%Y%m%d_%H%M%S')
        return t

    def get_logger(self):
        logger = logging.getLogger('training')
        fh = logging.FileHandler(self.output_file)
        sh = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%d.%m.%y %H:%M:%S')
        fh.setFormatter(formatter)
        sh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(sh)
        logger.setLevel(logging.INFO)
        return logger
    
    def get_record_defaults(self):
        zeros = tf.zeros(shape = (1,), dtype = tf.float32)
        ones = tf.ones(shape = (1,), dtype = tf.float32)
        return [zeros] * (self.layers[0] + self.offset) + [ones]
    
    def parse_row(self, tf_string):
        data = tf.io.decode_csv(tf.expand_dims(tf_string, axis=0), self.get_record_defaults())
        features = data[self.offset:-1]
        features = tf.stack(features, axis=-1)
        label = data[-1]
        features = tf.squeeze(features, axis=0)
        label = tf.squeeze(label, axis=0)
        return features, label

    def create_dataset(self):
        data = tf.data.TextLineDataset([self.data_file])
        data = data.skip(1).shuffle(self.shuf_buffer, seed=self.seed_shuffle)
        
        val_size = int(self.data_length * self.perc_eval)
        train_data = data.skip(val_size).batch(self.batch_size).repeat()
        val_data = data.take(val_size).batch(self.batch_size).repeat()

        train_data = train_data.map(self.parse_row, num_parallel_calls=6)
        val_data = val_data.map(self.parse_row, num_parallel_calls=6)

        return iter(train_data), iter(val_data)

    def get_activation(self):
        if self.activation_function == 'sigmoid':
            return tf.nn.sigmoid
        elif self.activation_function == 'softmax':
            return tf.nn.softmax
        elif self.activation_function == 'no':
            return None
        else:
            self.logger.error(f'Activation function {self.activation_function} not implemented')
            exit(2)

    def get_transfer(self):
        if self.transfer_function == 'sigmoid':
            return tf.nn.sigmoid
        elif self.transfer_function == 'relu':
            return tf.nn.relu
        else:
            self.logger.error(f'Transfer function {self.transfer_function} not implemented')
            exit(2)

    def build_kan_model(self):
        """
        KAN Modell bauen
        """
        self.model = keras.Sequential()

        for layer_config in self.layers:
            layer_type = layer_config['type']
            units = layer_config.get('units', 32)
            activation = layer_config.get('activation', self.transfer_function)

            if layer_type == 'dense':
                self.model.add(DenseKAN(
                    units=units,
                    activation=activation,
                    kernel_initializer=self.weight_initializer,
                    use_bias=self.use_bias
                ))

            elif layer_type == 'conv2d':
                filters = layer_config.get('filters', 32)
                kernel_size = layer_config.get('kernel_size', (3, 3))
                self.model.add(Conv2DKAN(
                    filters=filters,
                    kernel_size=kernel_size,
                    activation=activation,
                    kernel_initializer=self.weight_initializer,
                    use_bias=self.use_bias
                ))

            #optionales Dropout hinzufügen (brauche ich das?)
            if self.dropout > 0:
                self.model.add(keras.layers.Dropout(self.dropout))

        #ausgabeschicht
        self.model.add(DenseKAN(units=1, activation=self.activation_function))
        self.logger.info("KAN model architecture built with parameters from config file")

    def compile_model(self):
        if self.optimizer_type == "adam":
            optimizer = Adam(learning_rate=self.learning_rate, beta_1=self.beta1, beta_2=self.beta2, amsgrad=False)
        else:
            self.logger.error(f"Optimizer {self.optimizer_type} not implemented.")
            exit(1)

        self.model.compile(optimizer=optimizer, loss=self.cost_function, metrics=['accuracy'])
        self.logger.info("Model compiled")

    def train(self):
        train_it, val_it = self.create_dataset()
        self.build_kan_model()
        self.compile_model()

        model_checkpoint = ModelCheckpoint(filepath=self.save_network_to + '_{epoch:03d}-{accuracy:.3f}-{val_accuracy:.3f}', monitor='val_accuracy', mode='max', save_best_only=False, verbose=1)
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
        kan_model = KANModel(sys.argv[1])
        kan_model.train()
    except IndexError:
        logging.error("Please specify a path to a config file as the first command line argument")
        exit(1)
