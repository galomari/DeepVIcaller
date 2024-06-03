import numpy as np
import os
from scipy.io import loadmat
import pandas as pd
import random
import sys
import itertools
import numbers
from collections import Counter
from warnings import warn
from abc import ABCMeta, abstractmethod
import tensorflow as tf
np.random.seed(1337)  # for reproducibility

from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv1D, MaxPooling1D, Activation, concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.regularizers import l2, l1, l1_l2
from tensorflow.keras import backend as K
from tensorflow.keras.utils import CustomObjectScope

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, roc_curve, auc, roc_auc_score, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt

class Attention(tf.keras.layers.Layer):
    def __init__(self, hidden, activation='linear', **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.hidden = hidden
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.input_length = input_shape[1]
        self.W0 = self.add_weight(shape=(input_dim, self.hidden), initializer='glorot_uniform', trainable=True, name='W0')
        self.W = self.add_weight(shape=(self.hidden, 1), initializer='glorot_uniform', trainable=True, name='W')
        self.b0 = self.add_weight(shape=(self.hidden,), initializer='zeros', trainable=True, name='b0')
        self.b = self.add_weight(shape=(1,), initializer='zeros', trainable=True, name='b')

    def call(self, x):
        attmap = self.activation(K.dot(x, self.W0) + self.b0)
        attmap = K.dot(attmap, self.W) + self.b
        attmap = K.reshape(attmap, (-1, self.input_length))
        attmap = tf.keras.activations.softmax(attmap)
        dense_representation = K.batch_dot(attmap, x, axes=(1, 1))
        out = K.concatenate([dense_representation, attmap])
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1] + input_shape[1])

class attention_flatten(tf.keras.layers.Layer):
    def __init__(self, keep_dim, **kwargs):
        super(attention_flatten, self).__init__(**kwargs)
        self.keep_dim = keep_dim

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.keep_dim)

    def call(self, x):
        return K.batch_flatten(x[:, :self.keep_dim])

def Build_model():
    seq_input_shape = (2000, 4)
    nb_filter = 64
    nb_filter1 = 128
    filter_length = 8
    filter_length1 = 7
    attentionhidden = 256

    seq_input = Input(shape=seq_input_shape, name='seq_input')
    convol_1 = Conv1D(filters=nb_filter, kernel_size=filter_length, padding='valid', activation='relu', kernel_constraint=MaxNorm(4))
    convol_2 = Conv1D(filters=nb_filter, kernel_size=filter_length, padding='valid', activation='relu', kernel_constraint=MaxNorm(4))
    convol_3 = Conv1D(filters=nb_filter1, kernel_size=filter_length1, padding='valid', activation='relu', kernel_constraint=MaxNorm(4), bias_constraint=MaxNorm(4))

    pooling_1 = MaxPooling1D(pool_size=3)
    pooling_2 = MaxPooling1D(pool_size=3)
    dropout_1 = Dropout(0.50)
    dropout_2 = Dropout(0.45)

    decoder = Attention(hidden=attentionhidden, activation='linear')

    dense_1 = Dense(1)
    dense_2 = Dense(1)

    output_1 = pooling_1(convol_2(convol_1(seq_input)))
    output_12 = pooling_2(convol_3(output_1))
    output_2 = dropout_1(output_12)
    output_3 = dense_1(dropout_2(Flatten()(output_2)))

    att_decoder = decoder(output_2)
    output_4 = attention_flatten(output_2.shape[2])(att_decoder)

    all_outp = concatenate([output_3, output_4])
    output_5 = dense_2(all_outp)

    output_f = Activation('sigmoid')(output_5)

    model = Model(inputs=seq_input, outputs=output_f)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])

    return model

def load_Data():
    vicaller_fastq_Data = np.load('VIcallerData/vicaller_fastq.npy')
    vicaller_fastq_Label = np.load('VIcallerData/fastqlable_Test_Label.npy')
    vicaller_fastq_Label = vicaller_fastq_Label.astype(int)
    vicaller_bam_Data = np.load('VIcallerData/vicaller_bam.npy')
    vicaller_bam_Label = np.load('VIcallerData/bam_label_Test_Label.npy')
    vicaller_bam_Label = vicaller_bam_Label.astype(int)
    return vicaller_fastq_Data, vicaller_fastq_Label, vicaller_bam_Data, vicaller_bam_Label

def train_model():
    model = Build_model()

    vicaller_fastq_Data, vicaller_fastq_Label, vicaller_bam_Data, vicaller_bam_Label = load_Data()

    # Combine fastq and bam data for training
    X_train = np.concatenate((vicaller_fastq_Data, vicaller_bam_Data), axis=0)
    y_train = np.concatenate((vicaller_fastq_Label, vicaller_bam_Label), axis=0)

    checkpoint = ModelCheckpoint('Model/DeepEBV_with_EBV_integration_sequences.keras', monitor='val_loss', save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')

    model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=128, callbacks=[checkpoint, early_stopping])

    model.save('Model/DeepEBV_with_EBV_integration_sequences.keras')

def test_model():
    model = Build_model()
    model.load_weights('Model/DeepEBV_with_EBV_integration_sequences.keras')
    model.summary()

    vicaller_fastq_Data, vicaller_fastq_Label, vicaller_bam_Data, vicaller_bam_Label = load_Data()

    print('Predicting...')
    vicaller_fastq_pred = model.predict(vicaller_fastq_Data)
    vicaller_bam_pred = model.predict(vicaller_bam_Data)

    vicaller_fastq_Acc_loss = model.evaluate(vicaller_fastq_Data, vicaller_fastq_Label, batch_size=128)
    vicaller_bam_Acc_loss = model.evaluate(vicaller_bam_Data, vicaller_bam_Label, batch_size=128)

    vicaller_fastq_rocauc_score = roc_auc_score(vicaller_fastq_Label, vicaller_fastq_pred)
    vicaller_bam_rocauc_score = roc_auc_score(vicaller_bam_Label, vicaller_bam_pred)

    vicaller_fastq_Test_Quantity = pd.Series(vicaller_fastq_Label).value_counts()
    vicaller_fastq_neg_Quantity = vicaller_fastq_Test_Quantity.values[0]
    vicaller_fastq_pos_Quantity = vicaller_fastq_Test_Quantity.values[1]
    vicaller_bam_Test_Quantity = pd.Series(vicaller_bam_Label).value_counts()
    vicaller_bam_neg_Quantity = vicaller_bam_Test_Quantity.values[0]
    vicaller_bam_pos_Quantity = vicaller_bam_Test_Quantity.values[1]

    vicaller_fastq_AveragePrecision_score = average_precision_score(vicaller_fastq_Label, vicaller_fastq_pred)
    vicaller_bam_AveragePrecision_score = average_precision_score(vicaller_bam_Label, vicaller_bam_pred)

    print('-------vicaller_fastq_test_result------------------')
    print('vicaller_fastq_Test_pos_Quantity:', vicaller_fastq_pos_Quantity)
    print('vicaller_fastq_Test_neg_Quantity:', vicaller_fastq_neg_Quantity)
    print('Test acc:', vicaller_fastq_Acc_loss[1])
    print('Test loss:', vicaller_fastq_Acc_loss[0])
    print('auroc:', vicaller_fastq_rocauc_score)
    print('aupr:', vicaller_fastq_AveragePrecision_score)

    print('-------vicaller_bam_test_result------------------')
    print('vicaller_bam_Test_pos_Quantity:', vicaller_bam_pos_Quantity)
    print('vicaller_bam_Test_neg_Quantity:', vicaller_bam_neg_Quantity)
    print('Test acc:', vicaller_bam_Acc_loss[1])
    print('Test loss:', vicaller_bam_Acc_loss[0])
    print('auroc:', vicaller_bam_rocauc_score)
    print('aupr:', vicaller_bam_AveragePrecision_score)

    # Save results
    output_directory = 'Test_Result/vicaller_Result/'
    os.makedirs(output_directory, exist_ok=True)

    model_test_results = pd.DataFrame(data=np.zeros((1, 6), dtype=float), index=[0],
                                      columns=['vicaller_Test_pos_Quantity', 'vicaller_Test_neg_Quantity', 
                                               'Test_acc','Test_loss','rocauc_score', 'AveragePrecision_score'])
    model_test_results['vicaller_Test_pos_Quantity'] = vicaller_fastq_pos_Quantity
    model_test_results['vicaller_Test_neg_Quantity'] = vicaller_fastq_neg_Quantity
    model_test_results['Test_acc'] = vicaller_fastq_Acc_loss[1]
    model_test_results['Test_loss'] = vicaller_fastq_Acc_loss[0]
    model_test_results['rocauc_score'] = vicaller_fastq_rocauc_score
    model_test_results['AveragePrecision_score'] = vicaller_fastq_AveragePrecision_score
    model_test_results.to_csv(output_directory + 'model_test_results_fastq.csv', index=False)

    model_test_results = pd.DataFrame(data=np.zeros((1, 6), dtype=float), index=[0],
                                      columns=['vicaller_Test_pos_Quantity', 'vicaller_Test_neg_Quantity', 
                                               'Test_acc','Test_loss', 'rocauc_score', 'AveragePrecision_score'])
    model_test_results['vicaller_Test_pos_Quantity'] = vicaller_bam_pos_Quantity
    model_test_results['vicaller_Test_neg_Quantity'] = vicaller_bam_neg_Quantity
    model_test_results['Test_acc'] = vicaller_bam_Acc_loss[1]
    model_test_results['Test_loss'] = vicaller_bam_Acc_loss[0]
    model_test_results['rocauc_score'] = vicaller_bam_rocauc_score
    model_test_results['AveragePrecision_score'] = vicaller_bam_AveragePrecision_score
    model_test_results.to_csv(output_directory + 'model_test_results_bam.csv', index=False)

if __name__ == '__main__':
    train_model()
    test_model()

