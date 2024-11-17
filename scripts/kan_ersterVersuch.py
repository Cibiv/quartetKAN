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
        try:
            with open(path, 'r') as stream:
                data = yaml.safe_load(stream)
                timestamp = self.datetime_stamp()

                # Parameter aus der YAML-Datei laden
                self.dropout = data['dropout']
                self.layers = data['layers']
                self.learning_rate = data['learning_rate']
                self.batch_size = data['batch_size']
                self.epochs = data['epochs']
                self.no_epochs_dataset = data['no_epochs_dataset']
                self.perc_eval = data['perc_eval']
                self.data_file = data['data_file']
                self.data_length = data['data_length']
                self.save_network_to = f"../models/{data['save_network_to']}_{timestamp}"
                self.output_file = f"../log/{data['save_network_to']}_{timestamp}.log"
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

                # Eingabegröße bestimmen
                self.determine_input_size()
                
                self.logger = self.get_logger()
                self.logger.info("KAN initialized with parameters: " + str(data))

        except FileNotFoundError:
            logging.error("Config file not found")
            exit(1)

    def datetime_stamp(self):
        return datetime.now().strftime('%Y%m%d_%H%M%S')

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

    def determine_input_size(self):
        with open(self.data_file, 'r') as file:
            header = file.readline()
            line = file.readline().strip()
        num_columns = len(line.split(','))
        self.in_size = num_columns - self.offset - 1
        print(f"Bestimmte Eingabegröße (in_size): {self.in_size}")

    def get_record_defaults(self):
        zeros = tf.zeros(shape=(1,), dtype=tf.float32)
        ones = tf.ones(shape=(1,), dtype=tf.float32)
        return [zeros] * self.in_size + [ones]

    def parse_row(self, tf_string):
        data = tf.io.decode_csv(tf.expand_dims(tf_string, axis=0), self.get_record_defaults())
        features = data[:-1]
        features = tf.stack(features, axis=-1)
        label = data[-1]
        return tf.squeeze(features, axis=0), tf.squeeze(label, axis=0)

    def create_dataset(self):
        data = tf.data.TextLineDataset([self.data_file])
        data = data.skip(1).shuffle(buffer_size=self.shuf_buffer, seed=self.seed_shuffle)

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

    def build_kan_model(self):
        self.model = keras.Sequential()

        for layer_config in self.layers:
            layer_type = layer_config['type']
            units = layer_config.get('units', 32)
            activation = layer_config.get('activation', self.transfer_function)

            if layer_type == 'dense':
                self.model.add(DenseKAN(units=units, use_bias=self.use_bias))
                if activation:
                    self.model.add(keras.layers.Activation(activation))

            if self.dropout > 0:
                self.model.add(keras.layers.Dropout(self.dropout))

        self.model.add(DenseKAN(units=1))
        if self.activation_function:
            self.model.add(keras.layers.Activation(self.activation_function))

        self.logger.info("KAN model architecture built")

    def compile_model(self):
        optimizer = Adam(learning_rate=self.learning_rate, beta_1=self.beta1, beta_2=self.beta2)
        self.model.compile(optimizer=optimizer, loss=self.cost_function, metrics=['accuracy'])
        self.logger.info("Model compiled")

    def train(self):
        train_it, val_it = self.create_dataset()
        self.build_kan_model()
        self.compile_model()

        for features, label in train_it:
            print(f"Training features shape: {features.shape}")
            print(f"Training label shape: {label.shape}")
            break

        model_checkpoint = ModelCheckpoint(filepath=self.save_network_to + '_{epoch:03d}-{val_accuracy:.3f}', monitor='val_accuracy', save_best_only=True, verbose=1)
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
    if len(sys.argv) < 2:
        logging.error("Please specify a path to a config file")
        exit(1)
    kan_model = KANModel(sys.argv[1])
    kan_model.train()
