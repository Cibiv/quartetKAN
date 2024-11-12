import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tensorflow.python.keras import backend as K


class TrainingPlot(keras.callbacks.Callback):

    def __init__(self, save_network_to, logger, **kwargs):
        super(TrainingPlot, self).__init__()
        self.output = save_network_to.replace('models','results')
        self.epochs = []
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.lr = []
        self.logger = logger

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):

        epoch+=1

        self.logger.info(f'Epoch {epoch}: loss: {logs.get("loss"):.4g} - acc: {logs.get("binary_accuracy"):.4f} - val loss: {logs.get("val_loss"):.4g} - val acc: {logs.get("val_binary_accuracy"):.4f} - lr: {float(K.get_value(self.model.optimizer.lr)):.2g}')

        # Append the logs, losses and accuracies to the lists
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('binary_accuracy'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_binary_accuracy'))
        self.lr.append(float(K.get_value(self.model.optimizer.lr)))

        # Before plotting ensure at least 2 epochs have passed
        self.epochs.append(epoch)

        # You can chose the style of your preference
        # print(plt.style.available) to see the available options
        #plt.style.use("seaborn")

        # Plot train loss, train acc, val loss and val acc against epochs passed
        plt.figure()
        plt.plot(self.epochs, self.losses, label = "training loss")
        plt.plot(self.epochs, self.val_losses, label = "validation loss")
        plt.title("Losses")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.legend()
        # Make sure there exists a folder called output in the current directory
        # or replace 'output' with whatever direcory you want to put in the plots
        name = f'{self.output}_loss.png'
        plt.savefig(name)
        plt.close()

        plt.figure()
        plt.plot(self.epochs, self.acc, label = "training accuracy")
        plt.plot(self.epochs, self.val_acc, label = "validation accuracy")
        plt.title("Accuracies")
        plt.xlabel("Epoch #")
        plt.ylabel("Accuracy")
        plt.legend()
        # Make sure there exists a folder called output in the current directory
        # or replace 'output' with whatever direcory you want to put in the plots
        name = f'{self.output}_acc.png'
        plt.savefig(name)
        plt.close()

        plt.figure()
        plt.plot(self.epochs, self.lr)
        plt.title("Learning Rates")
        plt.xlabel("Epoch #")
        plt.ylabel("Learning Rate")
        # Make sure there exists a folder called output in the current directory
        # or replace 'output' with whatever direcory you want to put in the plots
        name = f'{self.output}_lr.png'
        plt.savefig(name)
        plt.close()
