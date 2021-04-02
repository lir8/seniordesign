from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, LSTM, Flatten
from keras.models import Sequential
from keras.utils import np_utils
import numpy as np
import os
import keras.backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping

import logging
modelDir = 'modelDir'
if not os.path.isdir(modelDir):
    os.makedirs(modelDir)
LOG = logging.getLogger('modelDir/lstm_results')


def loadTrainAndTestData():
    x = np.loadtxt(open("Xdata.csv"), dtype=np.float, delimiter=",")
    y = np.loadtxt(open("Ydata.csv"), dtype=np.int, delimiter=",")
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                        shuffle=True, random_state=42)

    return X_train, y_train, X_test, y_test


def loadData():
    X_train_raw, y_train, X_test_raw, y_test = loadTrainAndTestData()
    NUM_CLASS = 20

    X_train = X_train_raw.reshape(X_train_raw.shape[0], X_train_raw.shape[1], 1)
    X_test = X_test_raw.reshape(X_test_raw.shape[0], X_test_raw.shape[1], 1)
    y_train = np_utils.to_categorical(y_train, NUM_CLASS)
    y_test = np_utils.to_categorical(y_test, NUM_CLASS)

    return X_train, y_train, X_test, y_test, NUM_CLASS


#X_train, y_train, X_test, y_test, NUM_CLASS = loadData()
# print(X_train)
# print(X_train.shape)
# print(y_train)
# print(y_train.shape)
# print(X_test)
# print(X_test.shape)
# print(y_test)
# print(y_test.shape)


def generate_default_params():
    return {
            'optimizer': 'Adamax',
            'activation': 'tanh',
            'rc_act': 'hard_sigmoid',
            'dropout_rate1': 0.4,
            'dropout_rate2': 0.1,
            'batch_size': 121,
            'decay': 0.1,
            'epochs': 500,
            'data_dim': 1783,
            'layer1': 210,
            'layer2': 190,
            'dense': 70,
            'dense_act': 'selu',
            'kernel_ini': 'glorot_normal'
            }

class LSTM_Model():
    def __init__(self, params, name='lstm'):
        self.params = params
        self.name = name

    def create_model(self, NUM_CLASS):
        print ('Creating model...')
        layers = [LSTM(self.params['layer1'], activation=self.params['activation'], input_shape=(self.params['data_dim'], 1), return_sequences=True, recurrent_activation=self.params['rc_act'], kernel_initializer=self.params['kernel_ini']),
                  Dropout(rate=self.params['dropout_rate1']),

                  LSTM(self.params['layer2'], activation=self.params['activation'], return_sequences=True, recurrent_activation=self.params['rc_act'], kernel_initializer=self.params['kernel_ini']),
                  Dropout(rate=self.params['dropout_rate2']),

                  Flatten(),
                  Dense(self.params['dense'], activation=self.params['dense_act'], kernel_initializer=self.params['kernel_ini']),
                  Dense(NUM_CLASS, activation='softmax', kernel_initializer='glorot_normal')]

        model = Sequential(layers)

        print ('Compiling...')
        model.compile(loss='categorical_crossentropy',
                     optimizer=self.params['optimizer'],
                     metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, NUM_CLASS):
        model = self.create_model(NUM_CLASS)


        def lr_scheduler(epoch):
            if epoch % 20 == 0 and epoch != 0:
                lr = K.get_value(model.optimizer.lr)
                K.set_value(model.optimizer.lr, lr*self.params['decay'])
                print("lr changed to {}".format(lr*self.params['decay']))
            return K.get_value(model.optimizer.lr)

        modelPath = os.path.join(modelDir, 'lstm_weights_best.hdf5')
        checkpointer = ModelCheckpoint(filepath=modelPath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

        CallBacks = [checkpointer]
        if 'SGD' == self.params['optimizer']:
            scheduler = LearningRateScheduler(lr_scheduler)
            CallBacks.append(scheduler)
        CallBacks.append(EarlyStopping(monitor='val_acc', mode='max', patience=10))

        hist = model.fit(X_train, y_train,
                         batch_size=self.params['batch_size'],
                         epochs=self.params['epochs'],
                         validation_split = 0.2,
                         verbose=1,
                         callbacks=CallBacks)

        return modelPath

    def prediction(self, X_test, NUM_CLASS, modelPath):
        print ('Predicting results with best model...')
        model = self.create_model(NUM_CLASS)
        model.load_weights(modelPath)
        y_pred = model.predict(X_test)
        return y_pred

    def test(self, X_test, y_test, NUM_CLASS, modelPath):
        print ('Predicting results with best model...')
        model = self.create_model(NUM_CLASS)
        model.load_weights(modelPath)
        score, acc = model.evaluate(X_test, y_test, batch_size=100)

        tmpLine = ['Test score:'+str(score), 'Test accuracy:'+str(acc)]
        content = '\n'.join(tmpLine)
        print(content)
        return acc


def main():
    try:
        PARAMS = generate_default_params()
        X_train, y_train, X_test, y_test, NUM_CLASS = loadData()

        lstm = LSTM_Model(PARAMS)
        modelPath = lstm.train(X_train, y_train, NUM_CLASS)
        lstm.test(X_test, y_test, NUM_CLASS, modelPath)
    except Exception as e:
        LOG.exception(e)
        raise

if __name__ == '__main__':
    main()








