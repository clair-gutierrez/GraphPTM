#Import statements for this part
import numpy as np
import pickle
import tensorflow as tf
import csv
import pandas as pd
import os
import random

from spektral.models.gcn import GCN
from spektral.data import Dataset, DisjointLoader, Graph, BatchLoader, PackedBatchLoader
from spektral.layers import GCSConv, GlobalAvgPool, GlobalMaxPool, GlobalSumPool, GlobalAttentionPool, TAGConv, ARMAConv, GlobalAttnSumPool, GeneralConv, MessagePassing, AGNNConv, APPNPConv, EdgeConv, GATConv, GatedGraphConv, GCNConv, GINConv, GraphSageConv
from spektral.layers.pooling import TopKPool, SAGPool
from spektral.transforms.normalize_adj import NormalizeAdj

import scipy.sparse as sp
from scipy.sparse import csr_matrix

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class PhosDataset(Dataset):
    """    
    list_adj: list of numpy array of adjacencies
    labels: y_labels array
    list_features: array of node features
    """

    def __init__(self, list_adj, labels, list_features, **kwargs):
        self.list_adj=list_adj
        self.labels=labels
        self.list_features=list_features
        self.n_samples=len(list_adj)
       
        super().__init__(**kwargs)

    def read(self):
        def make_graph(index):
            #Adjacency matrix
            single_graph=self.list_adj[index]

            

            # Node features
            features=self.list_features[index][1]
            
            #features=features.astype('int32')
            #x = np.zeros((n, 21)) #There are 20 features in this dataset including amino acids, surface exp amino acids, cofactors, etc
            #x[np.arange(n), features] = 1 #One hot encoding
            x_list=convert_to_onehot(features)
            x=np.stack(x_list, axis=0)
            x=x.astype(np.float32)


            # Edges
            a=single_graph
            #a = csr_matrix(sparse_adj)

            # Labels
            y = np.zeros((2,))
            y[self.labels[index]]=1 #On hot encoding the output classes

            return Graph(x=x, a=a, y=y)

        # We must return a list of Graph objects
        return [make_graph(index) for index in range(self.n_samples)]

#From Alia
alphabet = "ARNDCEQGHILKMFPSTWYVXZU"

def convert_to_onehot(data):
    #Creates a dict, that maps to every char of alphabet an unique int based on position
    char_to_int = dict((c,i) for i,c in enumerate(alphabet))
    encoded_data = []
    #Replaces every char in data with the mapped int
    encoded_data.extend([char_to_int[char] for char in data])
    return encoded_data
    
from tqdm import tqdm

list_adj=np.load("/home/aliak/spektral_graph/r_15/CK2_adj.npy", allow_pickle=True) #type is numpy array
list_features=pickle.load(open("/home/aliak/spektral_graph/r_15/CK2_node_feat.pickle", "rb")) #type is list

print(list_adj.shape)

f = np.zeros(shape=(len(list_features),), dtype=object)
for i in tqdm(range(len(list_features)), desc="Loading...", position=0, leave=True):
    f[i] = list_features[i]
print(f.shape)

print(list_adj.shape)
list_adj=np.c_[list_adj, f]
print(list_adj.shape)

#negatives
list_n_adj=np.load("/home/aliak/spektral_graph/r_15/CK2_neg_adj.npy", allow_pickle=True)
list_n_features = pickle.load(open("/home/aliak/spektral_graph/r_15/CK2_neg_node_feat.pickle", "rb"))
print(list_n_adj.shape)

nf = np.zeros(shape=(len(list_n_features),), dtype=object)
for i in tqdm(range(len(list_n_features)), desc="Loading...", position=0, leave=True):
    nf[i] = list_n_features[i]
print(nf.shape)

print(list_n_adj.shape)
list_n_adj=np.c_[list_n_adj, nf]
print(list_n_adj.shape)

n=len(list_n_adj)
pos_adj=random.sample(list_adj.tolist(), n)
print(len(pos_adj))
pos=np.array(pos_adj, dtype=object)
pos=np.c_[pos[:,0], np.ones(len(pos), dtype=np.int32), pos[:,1]]
print(np.shape(pos))

neg=np.c_[list_n_adj[:,0], np.zeros(len(pos), dtype=np.int32), list_n_adj[:,1]]
print(np.shape(neg))

adj=np.vstack((pos, neg))
print(np.shape(adj))

np.random.shuffle(adj)
print(adj[0:10, 1])

labels=adj[:,1]
node_features=adj[:,2]
print(labels[0:10])
adj=adj[:,0]
print(adj.shape)

data = PhosDataset(list_adj=adj, labels=labels, list_features=node_features, transforms=NormalizeAdj())

list_adj=np.load("/home/aliak/spektral_graph/r_15/CK2_test_adj.npy", allow_pickle=True) #type is numpy array
list_features=pickle.load(open("/home/aliak/spektral_graph/r_15/CK2_test_node_feat.pickle", "rb")) #type is list

print(list_adj.shape)

f = np.zeros(shape=(len(list_features),), dtype=object)
for i in tqdm(range(len(list_features)), desc="Loading...", position=0, leave=True):
    f[i] = list_features[i]
print(f.shape)

print(list_adj.shape)
list_adj=np.c_[list_adj, f]
print(list_adj.shape)

#negatives
list_n_adj=np.load("/home/aliak/spektral_graph/r_15/CK2_test_neg_adj.npy", allow_pickle=True)
list_n_features = pickle.load(open("/home/aliak/spektral_graph/r_15/CK2_test_neg_node_feat.pickle", "rb"))
print(list_n_adj.shape)

nf = np.zeros(shape=(len(list_n_features),), dtype=object)
for i in tqdm(range(len(list_n_features)), desc="Loading...", position=0, leave=True):
    nf[i] = list_n_features[i]
print(nf.shape)

print(list_n_adj.shape)
list_n_adj=np.c_[list_n_adj, nf]
print(list_n_adj.shape)

n=len(list_n_adj)
pos_adj=random.sample(list_adj.tolist(), n)
print(len(pos_adj))
pos=np.array(pos_adj, dtype=object)
pos=np.c_[pos[:,0], np.ones(len(pos), dtype=np.int32), pos[:,1]]
print(np.shape(pos))

neg=np.c_[list_n_adj[:,0], np.zeros(len(pos), dtype=np.int32), list_n_adj[:,1]]
print(np.shape(neg))

adj=np.vstack((pos, neg))
print(np.shape(adj))

np.random.shuffle(adj)
print(adj[0:10, 1])

labels=adj[:,1]
node_features=adj[:,2]
print(labels[0:10])
adj=adj[:,0]
print(adj.shape)

data_te = PhosDataset(list_adj=adj, labels=labels, list_features=node_features, transforms=NormalizeAdj())

# Parameters
learning_rate = 5e-3  # Learning rate
epochs = 400  # Number of training epochs
es_patience = 15  # Patience for early stopping
batch_size = 31  # Batch size

F = data.n_node_features  # Dimension of node features
print(F)
n_out = data.n_labels  # Dimension of the target

# Train/valid/test split
idxs = np.random.permutation(len(data)) #1178 is the number of elements in the dataset
split_va = int(0.8 * len(data))
idx_tr, idx_va = np.split(idxs, [split_va])
data_tr = data[idx_tr]
data_va = data[idx_va]

loader_tr = DisjointLoader(data_tr, batch_size=batch_size, epochs=epochs)
loader_va = DisjointLoader(data_va, batch_size=batch_size)
loader_te = DisjointLoader(data_te, batch_size=batch_size)

#Model
###################################

X_in = Input(shape=(F,), name="X_in")
A_in = Input(shape=(None,), sparse=True)
I_in = Input(shape=(), name="segment_ids_in", dtype=tf.int32)

X_2 = GraphSageConv(32, activation='relu')([X_in, A_in])
X_3 = GlobalAvgPool()([X_2, I_in])
output = Dense(n_out, activation="softmax")(X_3)

####################################


# Build model
model = Model(inputs=[X_in, A_in, I_in], outputs=output)
opt = Adam(lr=learning_rate)
loss_fn = CategoricalCrossentropy()
model.summary()


#Train functions
@tf.function(input_signature=loader_tr.tf_signature(), experimental_relax_shapes=True)
def train_step(inputs, target):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(target, predictions)
        loss += sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    acc = tf.reduce_mean(categorical_accuracy(target, predictions))
    return loss, acc


def evaluate(loader):
    output = []
    step = 0
    while step < loader.steps_per_epoch:
        step += 1
        inputs, target = loader.__next__()
        pred = model(inputs, training=False)
        outs = (
            loss_fn(target, pred),
            tf.reduce_mean(categorical_accuracy(target, pred)), #Using categorical_accuracy to determine accuracy
                )
        output.append(outs)
    return np.mean(output, 0)


#Training
print("Fitting model")
progbar = tf.keras.utils.Progbar(loader_tr.steps_per_epoch)
current_batch = epoch = model_loss = model_acc = 0
best_val_loss = np.inf
best_weights = None
patience = es_patience

for batch in loader_tr:
    outs = train_step(*batch)

    model_loss += outs[0]
    model_acc += outs[1]
    current_batch += 1
    progbar.update(current_batch)
    
    if current_batch == loader_tr.steps_per_epoch:
        model_loss /= loader_tr.steps_per_epoch
        model_acc /= loader_tr.steps_per_epoch
        epoch += 1

        # Compute validation loss and accuracy
        val_loss, val_acc = evaluate(loader_va)
        print(
            "Ep. {} - Loss: {:.2f} - Acc: {:.2f} - Val loss: {:.2f} - Val acc: {:.2f}".format(
                epoch, model_loss, model_acc, val_loss, val_acc
            )
        )
        
        # Check if loss improved for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = es_patience
            print("New best val_loss {:.3f}".format(val_loss))
            best_weights = model.get_weights()
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping (best val_loss: {})".format(best_val_loss))
                break
        model_loss = 0
        model_acc = 0
        current_batch = 0

################################################################################
# EVALUATE MODEL
################################################################################
print("Testing model")
model.set_weights(best_weights)  # Load best model
test_loss, test_acc = evaluate(loader_te)
print("Done. Test loss: {:.4f}. Test acc: {:.2f}".format(test_loss, test_acc))
