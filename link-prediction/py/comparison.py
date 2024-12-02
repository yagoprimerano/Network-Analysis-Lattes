#!/usr/bin/env python
# coding: utf-8

# # Comparison of link prediction with random walks based node embedding

# <table><tr><td>Run the latest release of this notebook:</td><td><a href="https://mybinder.org/v2/gh/stellargraph/stellargraph/master?urlpath=lab/tree/demos/link-prediction/homogeneous-comparison-link-prediction.ipynb" alt="Open In Binder" target="_parent"><img src="https://mybinder.org/badge_logo.svg"/></a></td><td><a href="https://colab.research.google.com/github/stellargraph/stellargraph/blob/master/demos/link-prediction/homogeneous-comparison-link-prediction.ipynb" alt="Open In Colab" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg"/></a></td></tr></table>

# This demo notebook compares the link prediction performance of the embeddings learned by Node2Vec [1], Attri2Vec [2], GraphSAGE [3] and GCN [4] on the Cora dataset, under the same edge train-test-split setting. Node2Vec and Attri2Vec are learned by capturing the random walk context node similarity. GraphSAGE and GCN are learned in an unsupervised way by making nodes co-occurring in short random walks represented closely in the embedding space.
# 
# We're going to tackle link prediction as a supervised learning problem on top of node representations/embeddings. After obtaining embeddings, a binary classifier can be used to predict a link, or not, between any two nodes in the graph. Various hyperparameters could be relevant in obtaining the best link classifier - this demo demonstrates incorporating model selection into the pipeline for choosing the best binary operator to apply on a pair of node embeddings.
# 
# There are four steps:
# 
# 1. Obtain embeddings for each node
# 2. For each set of hyperparameters, train a classifier
# 3. Select the classifier that performs the best
# 4. Evaluate the selected classifier on unseen data to validate its ability to generalise
# 
# 
# **References:** 
# 
# [1] Node2Vec: Scalable Feature Learning for Networks. A. Grover, J. Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2016.
# 
# [2] Attributed Network Embedding via Subspace Discovery. D. Zhang, Y. Jie, X. Zhu and C. Zhang. Data Mining and Knowledge Discovery, 2019. 
# 
# [3] Inductive Representation Learning on Large Graphs. W.L. Hamilton, R. Ying, and J. Leskovec. Neural Information Processing Systems (NIPS), 2017.
# 
# [4] Graph Convolutional Networks (GCN): Semi-Supervised Classification with Graph Convolutional Networks. Thomas N. Kipf, Max Welling. International Conference on Learning Representations (ICLR), 2017

# In[1]:


# install StellarGraph if running on Google Colab
import sys
if 'google.colab' in sys.modules:
  get_ipython().run_line_magic('pip', 'install -q stellargraph[demos]==1.3.0b')


# In[2]:


# verify that we're using the correct version of StellarGraph for this notebook
import stellargraph as sg

try:
    sg.utils.validate_notebook_version("1.3.0b")
except AttributeError:
    raise ValueError(
        f"This notebook requires StellarGraph version 1.3.0b, but a different version {sg.__version__} is installed.  Please see <https://github.com/stellargraph/stellargraph/issues/1172>."
    ) from None


# In[3]:


import matplotlib.pyplot as plt
from math import isclose
from sklearn.decomposition import PCA
import os
import networkx as nx
import numpy as np
import pandas as pd
from stellargraph import StellarGraph, datasets
from stellargraph.data import EdgeSplitter
from collections import Counter
import multiprocessing
from IPython.display import display, HTML
from sklearn.model_selection import train_test_split

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load the dataset
# 
# The Cora dataset is a homogeneous network where all nodes are papers and edges between nodes are citation links, e.g. paper A cites paper B.

# (See [the "Loading from Pandas" demo](../basics/loading-pandas.ipynb) for details on how data can be loaded.)

# In[4]:


dataset = datasets.Cora()
display(HTML(dataset.description))
graph, _ = dataset.load(largest_connected_component_only=True, str_node_ids=True)


# In[5]:


print(graph.info())


# ## Construct splits of the input data
# 
# We have to carefully split the data to avoid data leakage and evaluate the algorithms correctly:
# 
# * For computing node embeddings, a **Train Graph** (`graph_train`)
# * For training classifiers, a classifier **Training Set** (`examples_train`) of positive and negative edges that weren't used for computing node embeddings
# * For choosing the best classifier, a **Model Selection Test Set** (`examples_model_selection`) of positive and negative edges that weren't used for computing node embeddings or training the classifier 
# * For the final evaluation, with the learned node embeddings from the **Train Graph** (`graph_train`), the chosen best classifier is applied to a **Test Set** (`examples_test`) of positive and negative edges not used for neither computing the node embeddings or for classifier training or model selection

# ###  Test Graph
# 
# We begin with the full graph and use the `EdgeSplitter` class to produce:
# 
# * Test Graph
# * Test set of positive/negative link examples
# 
# The Test Graph is the reduced graph we obtain from removing the test set of links from the full graph.

# In[6]:


# Define an edge splitter on the original graph:
edge_splitter_test = EdgeSplitter(graph)

# Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from graph, and obtain the
# reduced graph graph_test with the sampled links removed:
graph_test, examples_test, labels_test = edge_splitter_test.train_test_split(
    p=0.1, method="global"
)

print(graph_test.info())


# ### Train Graph
# 
# This time, we use the `EdgeSplitter` on the Test Graph, and perform a train/test split on the examples to produce:
# 
# * Train Graph
# * Training set of link examples
# * Set of link examples for model selection
# 

# In[7]:


# Do the same process to compute a training subset from within the test graph
edge_splitter_train = EdgeSplitter(graph_test)
graph_train, examples, labels = edge_splitter_train.train_test_split(
    p=0.1, method="global"
)
(
    examples_train,
    examples_model_selection,
    labels_train,
    labels_model_selection,
) = train_test_split(examples, labels, train_size=0.75, test_size=0.25)

print(graph_train.info())


# Below is a summary of the different splits that have been created in this section.

# In[8]:


pd.DataFrame(
    [
        (
            "Training Set",
            len(examples_train),
            "Train Graph",
            "Test Graph",
            "Train the Link Classifier",
        ),
        (
            "Model Selection",
            len(examples_model_selection),
            "Train Graph",
            "Test Graph",
            "Select the best Link Classifier model",
        ),
        (
            "Test set",
            len(examples_test),
            "Test Graph",
            "Full Graph",
            "Evaluate the best Link Classifier",
        ),
    ],
    columns=("Split", "Number of Examples", "Hidden from", "Picked from", "Use"),
).set_index("Split")


# ## Create random walker

# We define the helper function to generate biased random walks from the given graph with the fixed random walk parameters:
# 
# * `p` - Random walk parameter "p" that defines probability, "1/p", of returning to source node
# * `q` - Random walk parameter "q" that defines probability, "1/q", for moving to a node away from the source node

# In[9]:


from stellargraph.data import BiasedRandomWalk


def create_biased_random_walker(graph, walk_num, walk_length):
    # parameter settings for "p" and "q":
    p = 1.0
    q = 1.0
    return BiasedRandomWalk(graph, n=walk_num, length=walk_length, p=p, q=q)


# ## Parameter Settings

# We train Node2Vec, Attri2Vec, GraphSAGE, and GCN by following the same unsupervised learning procedure: we firstly generate a set of short random walks from the given graph and then learn node embeddings from batches of `target, context` pairs collected from random walks. For learning node embeddings, we need to specify the following parameters:

# * `dimension` - Dimensionality of node embeddings
# * `walk_number` - Number of walks from each node
# * `walk_length` - Length of each random walk
# * `epochs` - The number of epochs to train embedding learning model
# * `batch_size` - The batch size to train embedding learning model

# We consistently set the node embedding dimension to 128 for all algorithms. However, we use different hidden layers to learn node embeddings for different algorithms to exert their respective power. For the remaining parameters, we set them as:
# 
# |               | Node2Vec | Attri2Vec | GraphSAGE | GCN |
# |---------------|----------|-----------|-----------|-----|
# | `walk_number` |    20    |     4     |     1     |  1  |
# | `walk_length` |     5    |     5     |     5     |  5  |
# | `epochs`      |    6     |     6     |     6     |  6  |
# | `batch_size`  |    50    |     50    |    50     |  50 |

# As all algorithms use the same `walk_length`, `batch_size` and `epochs` values, we uniformly set them here:

# In[10]:


walk_length = 5


# In[11]:


epochs = 6


# In[12]:


batch_size = 50


# For different algorithms, users can find the best parameter setting with the `Model Selection` edge set.

# ## Node2Vec
# 
# We use Node2Vec [1], to calculate node embeddings. These embeddings are learned in such a way to ensure that nodes that are close in the graph remain close in the embedding space. We train Node2Vec with the Stellargraph Node2Vec components.

# In[13]:


from stellargraph.data import UnsupervisedSampler
from stellargraph.mapper import Node2VecLinkGenerator, Node2VecNodeGenerator
from stellargraph.layer import Node2Vec, link_classification
from tensorflow import keras


def node2vec_embedding(graph, name):

    # Set the embedding dimension and walk number:
    dimension = 128
    walk_number = 20

    print(f"Training Node2Vec for '{name}':")

    graph_node_list = list(graph.nodes())

    # Create the biased random walker to generate random walks
    walker = create_biased_random_walker(graph, walk_number, walk_length)

    # Create the unsupervised sampler to sample (target, context) pairs from random walks
    unsupervised_samples = UnsupervisedSampler(
        graph, nodes=graph_node_list, walker=walker
    )

    # Define a Node2Vec training generator, which generates batches of training pairs
    generator = Node2VecLinkGenerator(graph, batch_size)

    # Create the Node2Vec model
    node2vec = Node2Vec(dimension, generator=generator)

    # Build the model and expose input and output sockets of Node2Vec, for node pair inputs
    x_inp, x_out = node2vec.in_out_tensors()

    # Use the link_classification function to generate the output of the Node2Vec model
    prediction = link_classification(
        output_dim=1, output_act="sigmoid", edge_embedding_method="dot"
    )(x_out)

    # Stack the Node2Vec encoder and prediction layer into a Keras model, and specify the loss
    model = keras.Model(inputs=x_inp, outputs=prediction)
    model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-3),
        loss=keras.losses.binary_crossentropy,
        metrics=[keras.metrics.binary_accuracy],
    )

    # Train the model
    model.fit(
        generator.flow(unsupervised_samples),
        epochs=epochs,
        verbose=2,
        use_multiprocessing=False,
        workers=4,
        shuffle=True,
    )

    # Build the model to predict node representations from node ids with the learned Node2Vec model parameters
    x_inp_src = x_inp[0]
    x_out_src = x_out[0]
    embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)

    # Get representations for all nodes in ``graph``
    node_gen = Node2VecNodeGenerator(graph, batch_size).flow(graph_node_list)
    node_embeddings = embedding_model.predict(node_gen, workers=1, verbose=0)

    def get_embedding(u):
        u_index = graph_node_list.index(u)
        return node_embeddings[u_index]

    return get_embedding


# ## Attri2Vec
# 
# We use Attri2Vec [2] to calculate node embeddings. Attri2Vec learns node representations through performing a linear/non-linear mapping on node content attributes and simultaneously making nodes sharing similar context nodes in random walks have similar representations. With the node content features are used to learn node embeddings, we wish that Attri2Vec can achieve better link prediction performance than the only structure preserving network embedding algorithm Node2Vec.

# In[14]:


from stellargraph.mapper import Attri2VecLinkGenerator, Attri2VecNodeGenerator
from stellargraph.layer import Attri2Vec


def attri2vec_embedding(graph, name):

    # Set the embedding dimension and walk number:
    dimension = [128]
    walk_number = 4

    print(f"Training Attri2Vec for '{name}':")

    graph_node_list = list(graph.nodes())

    # Create the biased random walker to generate random walks
    walker = create_biased_random_walker(graph, walk_number, walk_length)

    # Create the unsupervised sampler to sample (target, context) pairs from random walks
    unsupervised_samples = UnsupervisedSampler(
        graph, nodes=graph_node_list, walker=walker
    )

    # Define an Attri2Vec training generator, which generates batches of training pairs
    generator = Attri2VecLinkGenerator(graph, batch_size)

    # Create the Attri2Vec model
    attri2vec = Attri2Vec(
        layer_sizes=dimension, generator=generator, bias=False, normalize=None
    )

    # Build the model and expose input and output sockets of Attri2Vec, for node pair inputs
    x_inp, x_out = attri2vec.in_out_tensors()

    # Use the link_classification function to generate the output of the Attri2Vec model
    prediction = link_classification(
        output_dim=1, output_act="sigmoid", edge_embedding_method="ip"
    )(x_out)

    # Stack the Attri2Vec encoder and prediction layer into a Keras model, and specify the loss
    model = keras.Model(inputs=x_inp, outputs=prediction)
    model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-3),
        loss=keras.losses.binary_crossentropy,
        metrics=[keras.metrics.binary_accuracy],
    )

    # Train the model
    model.fit(
        generator.flow(unsupervised_samples),
        epochs=epochs,
        verbose=2,
        use_multiprocessing=False,
        workers=1,
        shuffle=True,
    )

    # Build the model to predict node representations from node features with the learned Attri2Vec model parameters
    x_inp_src = x_inp[0]
    x_out_src = x_out[0]
    embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)

    # Get representations for all nodes in ``graph``
    node_gen = Attri2VecNodeGenerator(graph, batch_size).flow(graph_node_list)
    node_embeddings = embedding_model.predict(node_gen, workers=1, verbose=0)

    def get_embedding(u):
        u_index = graph_node_list.index(u)
        return node_embeddings[u_index]

    return get_embedding


# ## GraphSAGE
# 
# GraphSAGE [3] learns node embeddings for attributed graphs through aggregating neighboring node attributes. The aggregation parameters are learned by encouraging node pairs co-occurring in short random walks to have similar representations. As node attributes are also leveraged, GraphSAGE is expected to perform better than Node2Vec in link prediction.

# In[15]:


from stellargraph.mapper import GraphSAGELinkGenerator, GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE


def graphsage_embedding(graph, name):

    # Set the embedding dimensions, the numbers of sampled neighboring nodes and walk number:
    dimensions = [128, 128]
    num_samples = [10, 5]
    walk_number = 1

    print(f"Training GraphSAGE for '{name}':")

    graph_node_list = list(graph.nodes())

    # Create the biased random walker to generate random walks
    walker = create_biased_random_walker(graph, walk_number, walk_length)

    # Create the unsupervised sampler to sample (target, context) pairs from random walks
    unsupervised_samples = UnsupervisedSampler(
        graph, nodes=graph_node_list, walker=walker
    )

    # Define a GraphSAGE training generator, which generates batches of training pairs
    generator = GraphSAGELinkGenerator(graph, batch_size, num_samples)

    # Create the GraphSAGE model
    graphsage = GraphSAGE(
        layer_sizes=dimensions,
        generator=generator,
        bias=True,
        dropout=0.0,
        normalize="l2",
    )

    # Build the model and expose input and output sockets of GraphSAGE, for node pair inputs
    x_inp, x_out = graphsage.in_out_tensors()

    # Use the link_classification function to generate the output of the GraphSAGE model
    prediction = link_classification(
        output_dim=1, output_act="sigmoid", edge_embedding_method="ip"
    )(x_out)

    # Stack the GraphSAGE encoder and prediction layer into a Keras model, and specify the loss
    model = keras.Model(inputs=x_inp, outputs=prediction)
    model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-3),
        loss=keras.losses.binary_crossentropy,
        metrics=[keras.metrics.binary_accuracy],
    )

    # Train the model
    model.fit(
        generator.flow(unsupervised_samples),
        epochs=epochs,
        verbose=2,
        use_multiprocessing=False,
        workers=4,
        shuffle=True,
    )

    # Build the model to predict node representations from node features with the learned GraphSAGE model parameters
    x_inp_src = x_inp[0::2]
    x_out_src = x_out[0]
    embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)

    # Get representations for all nodes in ``graph``
    node_gen = GraphSAGENodeGenerator(graph, batch_size, num_samples).flow(
        graph_node_list
    )
    node_embeddings = embedding_model.predict(node_gen, workers=1, verbose=0)

    def get_embedding(u):
        u_index = graph_node_list.index(u)
        return node_embeddings[u_index]

    return get_embedding


# ## GCN
# 
# GCN [4] learns node embeddings through graph convolution. Traditional GCN relies on node labels as a supervision to perform training. Here, we consider the unsupervised link prediction setting and we try to learn informative GCN node embeddings by making nodes co-occurring in short random walks represented closely, as is performed in training GraphSAGE.

# In[16]:


from stellargraph.mapper import FullBatchLinkGenerator, FullBatchNodeGenerator
from stellargraph.layer import GCN, LinkEmbedding


def gcn_embedding(graph, name):

    # Set the embedding dimensions and walk number:
    dimensions = [128, 128]
    walk_number = 1

    print(f"Training GCN for '{name}':")

    graph_node_list = list(graph.nodes())

    # Create the biased random walker to generate random walks
    walker = create_biased_random_walker(graph, walk_number, walk_length)

    # Create the unsupervised sampler to sample (target, context) pairs from random walks
    unsupervised_samples = UnsupervisedSampler(
        graph, nodes=graph_node_list, walker=walker
    )

    # Define a GCN training generator, which generates the full batch of training pairs
    generator = FullBatchLinkGenerator(graph, method="gcn")

    # Create the GCN model
    gcn = GCN(
        layer_sizes=dimensions,
        activations=["relu", "relu"],
        generator=generator,
        dropout=0.3,
    )

    # Build the model and expose input and output sockets of GCN, for node pair inputs
    x_inp, x_out = gcn.in_out_tensors()

    # Use the dot product of node embeddings to make node pairs co-occurring in short random walks represented closely
    prediction = LinkEmbedding(activation="sigmoid", method="ip")(x_out)
    prediction = keras.layers.Reshape((-1,))(prediction)

    # Stack the GCN encoder and prediction layer into a Keras model, and specify the loss
    model = keras.Model(inputs=x_inp, outputs=prediction)
    model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-3),
        loss=keras.losses.binary_crossentropy,
        metrics=[keras.metrics.binary_accuracy],
    )

    # Train the model
    batches = unsupervised_samples.run(batch_size)
    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}/{epochs}")
        batch_iter = 1
        for batch in batches:
            samples = generator.flow(batch[0], targets=batch[1], use_ilocs=True)[0]
            [loss, accuracy] = model.train_on_batch(x=samples[0], y=samples[1])
            output = (
                f"{batch_iter}/{len(batches)} - loss:"
                + " {:6.4f}".format(loss)
                + " - binary_accuracy:"
                + " {:6.4f}".format(accuracy)
            )
            if batch_iter == len(batches):
                print(output)
            else:
                print(output, end="\r")
            batch_iter = batch_iter + 1

    # Get representations for all nodes in ``graph``
    embedding_model = keras.Model(inputs=x_inp, outputs=x_out)
    node_embeddings = embedding_model.predict(
        generator.flow(list(zip(graph_node_list, graph_node_list)))
    )
    node_embeddings = node_embeddings[0][:, 0, :]

    def get_embedding(u):
        u_index = graph_node_list.index(u)
        return node_embeddings[u_index]

    return get_embedding


# ## Train and evaluate the link prediction model
# 
# There are a few steps involved in using the learned embeddings to perform link prediction:
# 1. We calculate link/edge embeddings for the positive and negative edge samples by applying a binary operator on the embeddings of the source and target nodes of each sampled edge.
# 2. Given the embeddings of the positive and negative examples, we train a logistic regression classifier to predict a binary value indicating whether an edge between two nodes should exist or not.
# 3. We evaluate the performance of the link classifier for each of the 4 operators on the training data with node embeddings calculated on the **Train Graph** (`graph_train`), and select the best classifier.
# 4. The best classifier is then used to calculate scores on the test data with node embeddings trained on the **Train Graph** (`graph_train`).
# 
# Below are a set of helper functions that let us repeat these steps for each of the binary operators.

# In[17]:


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# 1. link embeddings
def link_examples_to_features(link_examples, transform_node, binary_operator):
    return [
        binary_operator(transform_node(src), transform_node(dst))
        for src, dst in link_examples
    ]


# 2. training classifier
def train_link_prediction_model(
    link_examples, link_labels, get_embedding, binary_operator
):
    clf = link_prediction_classifier()
    link_features = link_examples_to_features(
        link_examples, get_embedding, binary_operator
    )
    clf.fit(link_features, link_labels)
    return clf


def link_prediction_classifier(max_iter=5000):
    lr_clf = LogisticRegressionCV(Cs=10, cv=10, scoring="roc_auc", max_iter=max_iter)
    return Pipeline(steps=[("sc", StandardScaler()), ("clf", lr_clf)])


# 3. and 4. evaluate classifier
def evaluate_link_prediction_model(
    clf, link_examples_test, link_labels_test, get_embedding, binary_operator
):
    link_features_test = link_examples_to_features(
        link_examples_test, get_embedding, binary_operator
    )
    score = evaluate_roc_auc(clf, link_features_test, link_labels_test)
    return score


def evaluate_roc_auc(clf, link_features, link_labels):
    predicted = clf.predict_proba(link_features)

    # check which class corresponds to positive links
    positive_column = list(clf.classes_).index(1)
    return roc_auc_score(link_labels, predicted[:, positive_column])


# We consider 4 different operators: 
# 
# * *Hadamard*
# * $L_1$
# * $L_2$
# * *average*
# 
# The paper [1] provides a detailed description of these operators. All operators produce link embeddings that have equal dimensionality to the input node embeddings (128 dimensions for our example). 

# In[18]:


def operator_hadamard(u, v):
    return u * v


def operator_l1(u, v):
    return np.abs(u - v)


def operator_l2(u, v):
    return (u - v) ** 2


def operator_avg(u, v):
    return (u + v) / 2.0


def run_link_prediction(binary_operator, embedding_train):
    clf = train_link_prediction_model(
        examples_train, labels_train, embedding_train, binary_operator
    )
    score = evaluate_link_prediction_model(
        clf,
        examples_model_selection,
        labels_model_selection,
        embedding_train,
        binary_operator,
    )

    return {
        "classifier": clf,
        "binary_operator": binary_operator,
        "score": score,
    }


binary_operators = [operator_hadamard, operator_l1, operator_l2, operator_avg]


# ### Train and evaluate the link model with the specified embedding

# In[19]:


def train_and_evaluate(embedding, name):

    embedding_train = embedding(graph_train, "Train Graph")

    # Train the link classification model with the learned embedding
    results = [run_link_prediction(op, embedding_train) for op in binary_operators]
    best_result = max(results, key=lambda result: result["score"])
    print(
        f"\nBest result with '{name}' embeddings from '{best_result['binary_operator'].__name__}'"
    )
    display(
        pd.DataFrame(
            [(result["binary_operator"].__name__, result["score"]) for result in results],
            columns=("name", "ROC AUC"),
        ).set_index("name")
    )

    # Evaluate the best model using the test set
    test_score = evaluate_link_prediction_model(
        best_result["classifier"],
        examples_test,
        labels_test,
        embedding_train,
        best_result["binary_operator"],
    )

    return test_score


# ### Collect the link prediction results for Node2Vec, Attri2Vec, GraphSAGE and GCN

# #### Get Node2Vec link prediction result

# In[20]:


node2vec_result = train_and_evaluate(node2vec_embedding, "Node2Vec")


# #### Get Attri2Vec link prediction result

# In[21]:


attri2vec_result = train_and_evaluate(attri2vec_embedding, "Attri2Vec")


# #### Get GraphSAGE link prediction result

# In[22]:


graphsage_result = train_and_evaluate(graphsage_embedding, "GraphSAGE")


# #### Get GCN link prediction result

# In[23]:


gcn_result = train_and_evaluate(gcn_embedding, "GCN")


# #### Comparison between Node2Vec, Attri2Vec, GraphSAGE and GCN on the test set

# The ROC AUC scores on the test set of links of different embeddings with their corresponding best operators:

# In[24]:


pd.DataFrame(
    [
        ("Node2Vec", node2vec_result),
        ("Attri2Vec", attri2vec_result),
        ("GraphSAGE", graphsage_result),
        ("GCN", gcn_result),
    ],
    columns=("name", "ROC AUC"),
).set_index("name")


# ## Conclusion

# This example has demonstrated how to use the `stellargraph` library to build a link prediction algorithm for homogeneous graphs using the unsupervised embeddings learned by Node2Vec [1], Attri2Vec [2] and GraphSAGE [3] and GCN [4]. 
# 
# For more information about the link prediction process, all of these algorithms have specific demos with more details:
# 
# - [Node2Vec](node2vec-link-prediction.ipynb)
# - [Attri2Vec](attri2vec-link-prediction.ipynb)
# - [GraphSAGE](graphsage-link-prediction.ipynb)
# - [GCN](gcn-link-prediction.ipynb)

# <table><tr><td>Run the latest release of this notebook:</td><td><a href="https://mybinder.org/v2/gh/stellargraph/stellargraph/master?urlpath=lab/tree/demos/link-prediction/homogeneous-comparison-link-prediction.ipynb" alt="Open In Binder" target="_parent"><img src="https://mybinder.org/badge_logo.svg"/></a></td><td><a href="https://colab.research.google.com/github/stellargraph/stellargraph/blob/master/demos/link-prediction/homogeneous-comparison-link-prediction.ipynb" alt="Open In Colab" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg"/></a></td></tr></table>
