#!/usr/bin/env python
# coding: utf-8

# # Link prediction via inductive node representations with attri2vec

# <table><tr><td>Run the latest release of this notebook:</td><td><a href="https://mybinder.org/v2/gh/stellargraph/stellargraph/master?urlpath=lab/tree/demos/link-prediction/attri2vec-link-prediction.ipynb" alt="Open In Binder" target="_parent"><img src="https://mybinder.org/badge_logo.svg"/></a></td><td><a href="https://colab.research.google.com/github/stellargraph/stellargraph/blob/master/demos/link-prediction/attri2vec-link-prediction.ipynb" alt="Open In Colab" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg"/></a></td></tr></table>

# This demo notebook demonstrates how to perform link prediction for out-of-sample nodes through learning node representations inductively with attri2vec [1]. The implementation uses the stellargraph components.
# 
# <a name="refs"></a>
# **References:** 
# 
# [1] [Attributed Network Embedding via Subspace Discovery](https://link.springer.com/article/10.1007/s10618-019-00650-2). D. Zhang, Y. Jie, X. Zhu and C. Zhang. Data Mining and Knowledge Discovery, 2019. 
# 
# ## attri2vec
# 
# attri2vec learns node representations by performing a linear/non-linear mapping on node content attributes. To make the learned node representations respect structural similarity, [DeepWalk](https://dl.acm.org/citation.cfm?id=2623732)/[Node2Vec](https://snap.stanford.edu/node2vec) learning mechanism is used to make nodes sharing similar random walk context nodes represented closely in the subspace, which is achieved by maximizing the occurrence probability of context nodes conditioned on the representation of the target nodes. 
# 
# In this demo, we first train the attri2vec model on the in-sample subgraph and obtain a mapping function from node attributes to node representations, then apply the mapping function to the content attributes of out-of-sample nodes and obtain the representations of out-of-sample nodes. We evaluate the quality of inferred out-of-sample node representations by using it to predict the links of out-of-sample nodes.

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


import networkx as nx
import pandas as pd
import numpy as np
import os
import random

import stellargraph as sg
from stellargraph.data import UnsupervisedSampler
from stellargraph.mapper import Attri2VecLinkGenerator, Attri2VecNodeGenerator
from stellargraph.layer import Attri2Vec, link_classification

from tensorflow import keras

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# ## Loading DBLP network data

# This demo uses a DBLP citation network, a subgraph extracted from [DBLP-Citation-network V3](https://aminer.org/citation). To form this subgraph, papers from four subjects are extracted according to their venue information: *Database, Data Mining, Artificial Intelligence and Computer Vision*, and papers with no citations are removed. The DBLP network contains 18,448 papers and 45,661 citation relations. From paper titles, we construct 2,476-dimensional binary node feature vectors, with each element indicating the presence/absence of the corresponding word. By ignoring the citation direction, we take the DBLP subgraph as an undirected network.
# 
# As papers in DBLP are attached with publication year, the DBLP network with the dynamic property can be used to study the problem of out-of-sample node representation learning. From the DBLP network, we construct four in-sample subgraphs using papers published before 2006, 2007, 2008 and 2009, and denote the four subgraphs as DBLP2006, DBLP2007, DBLP2008, and DBLP2009. For each subgraph, the remaining papers are taken as out-of-sample nodes. We consider the case where new coming nodes have no links. We predict the links of out-of-sample nodes using the learned out-of-sample node representations and compare its performance with the node content feature baseline.
# 
# The dataset used in this demo can be downloaded from https://www.kaggle.com/daozhang/dblp-subgraph.
# The following is the description of the dataset:
# 
# > The `content.txt` file contains descriptions of the papers in the following format:
# >
# >     <paper_id> <word_attributes> <class_label> <publication_year>
# >
# > The first entry in each line contains the unique integer ID (ranging from 0 to 18,447) of the paper followed by binary values indicating whether each word in the vocabulary is present (indicated by 1) or absent (indicated by 0) in the paper. Finally, the last two entries in the line are the class label and the publication year of the paper.
# > The `edgeList.txt` file contains the citation relations. Each line describes a link in the following format:
# >
# >     <ID of paper1> <ID of paper2>
# >
# > Each line contains two paper IDs, with paper2 citing paper1 or paper1 citing paper2.
# 
# 
# Download and unzip the `dblp-subgraph.zip` file to a location on your computer and set the `data_dir` variable to
# point to the location of the dataset (the `DBLP` directory containing `content.txt` and `edgeList.txt`).

# In[4]:


data_dir = "~/data/DBLP"


# Load the graph from the edge list.

# In[5]:


edgelist = pd.read_csv(
    os.path.join(data_dir, "edgeList.txt"),
    sep="\t",
    header=None,
    names=["source", "target"],
)
edgelist["label"] = "cites"  # set the edge type


# Load paper content features, subjects and publishing years.

# In[6]:


feature_names = ["w_{}".format(ii) for ii in range(2476)]
node_column_names = feature_names + ["subject", "year"]
node_data = pd.read_csv(
    os.path.join(data_dir, "content.txt"), sep="\t", header=None, names=node_column_names
)


# Construct the whole graph from edge list.

# In[7]:


G_all_nx = nx.from_pandas_edgelist(edgelist, edge_attr="label")


# Specify node types.

# In[8]:


nx.set_node_attributes(G_all_nx, "paper", "label")


# Get node features.

# In[9]:


all_node_features = node_data[feature_names]


# Create the Stellargraph with node features.

# In[10]:


G_all = sg.StellarGraph.from_networkx(G_all_nx, node_features=all_node_features)


# In[11]:


print(G_all.info())


# ## Get DBLP Subgraph 
# ### with papers published before a threshold year

# Get the edge list connecting in-sample nodes.

# In[12]:


year_thresh = 2006  # the threshold year for in-sample and out-of-sample set split, which can be 2007, 2008 and 2009
subgraph_edgelist = []
for ii in range(len(edgelist)):
    source_index = edgelist["source"][ii]
    target_index = edgelist["target"][ii]
    source_year = int(node_data["year"][source_index])
    target_year = int(node_data["year"][target_index])
    if source_year < year_thresh and target_year < year_thresh:
        subgraph_edgelist.append([source_index, target_index])
subgraph_edgelist = pd.DataFrame(
    np.array(subgraph_edgelist), columns=["source", "target"]
)
subgraph_edgelist["label"] = "cites"  # set the edge type


# Construct the network from the selected edge list.

# In[13]:


G_sub_nx = nx.from_pandas_edgelist(subgraph_edgelist, edge_attr="label")


# Specify node types.

# In[14]:


nx.set_node_attributes(G_sub_nx, "paper", "label")


# Get the ids of the nodes in the selected subgraph.

# In[15]:


subgraph_node_ids = sorted(list(G_sub_nx.nodes))


# Get the node features of the selected subgraph.

# In[16]:


subgraph_node_features = node_data[feature_names].reindex(subgraph_node_ids)


# Create the Stellargraph with node features.

# In[17]:


G_sub = sg.StellarGraph.from_networkx(G_sub_nx, node_features=subgraph_node_features)


# In[18]:


print(G_sub.info())


# ## Train attri2vec on the DBLP Subgraph

# Specify the other optional parameter values: root nodes, the number of walks to take per node, the length of each walk.

# In[19]:


nodes = list(G_sub.nodes())
number_of_walks = 2
length = 5


# Create the UnsupervisedSampler instance with the relevant parameters passed to it.

# In[20]:


unsupervised_samples = UnsupervisedSampler(
    G_sub, nodes=nodes, length=length, number_of_walks=number_of_walks
)


# Set the batch size and the number of epochs. 

# In[21]:


batch_size = 50
epochs = 6


# Define an attri2vec training generator, which generates a batch of (feature of target node, index of context node, label of node pair) pairs per iteration.

# In[22]:


generator = Attri2VecLinkGenerator(G_sub, batch_size)


# Building the model: a 1-hidden-layer node representation ('input embedding') of the `target` node and the parameter vector ('output embedding') for predicting the existence of `context node` for each `(target context)` pair, with a link classification layer performed on the dot product of the 'input embedding' of the `target` node and the 'output embedding' of the `context` node.
# 
# attri2vec part of the model, with a 128-dimension hidden layer, no bias term and no normalization. (Normalization can be set to 'l2'). 

# In[23]:


layer_sizes = [128]
attri2vec = Attri2Vec(
    layer_sizes=layer_sizes, generator=generator, bias=False, normalize=None
)


# In[24]:


# Build the model and expose input and output sockets of attri2vec, for node pair inputs:
x_inp, x_out = attri2vec.in_out_tensors()


# Use the link_classification function to generate the prediction, with the `ip` edge embedding generation method and the `sigmoid` activation, which actually performs the dot product of the 'input embedding' of the target node and the 'output embedding' of the context node followed by a sigmoid activation. 

# In[25]:


prediction = link_classification(
    output_dim=1, output_act="sigmoid", edge_embedding_method="ip"
)(x_out)


# Stack the attri2vec encoder and prediction layer into a Keras model, and specify the loss.

# In[26]:


model = keras.Model(inputs=x_inp, outputs=prediction)

model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-2),
    loss=keras.losses.binary_crossentropy,
    metrics=[keras.metrics.binary_accuracy],
)


# Train the model.

# In[27]:


history = model.fit(
    generator.flow(unsupervised_samples),
    epochs=epochs,
    verbose=2,
    use_multiprocessing=False,
    workers=1,
    shuffle=True,
)


# ## Predicting links of out-of-sample nodes with the learned attri2vec model

# Build the node based model for predicting node representations from node content attributes with the learned parameters. Below a Keras model is constructed, with `x_inp[0]` as input and `x_out[0]` as output. Note that this model's weights are the same as those of the corresponding node encoder in the previously trained node pair classifier.

# In[28]:


x_inp_src = x_inp[0]
x_out_src = x_out[0]
embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)


# Get the node embeddings, for both in-sample and out-of-sample nodes, by applying the learned mapping function to node content features.

# In[29]:


node_ids = node_data.index
node_gen = Attri2VecNodeGenerator(G_all, batch_size).flow(node_ids)
node_embeddings = embedding_model.predict(node_gen, workers=4, verbose=1)


# Get the positive and negative edges for in-sample nodes and out-of-sample nodes. The edges of the in-sample nodes only include the edges between in-sample nodes, and the edges of out-of-sample nodes are referred to all the edges linked to out-of-sample nodes, including the edges connecting in-sample and out-of-sample edges.

# In[30]:


year_thresh = 2006
in_sample_edges = []
out_of_sample_edges = []
for ii in range(len(edgelist)):
    source_index = edgelist["source"][ii]
    target_index = edgelist["target"][ii]
    if source_index > target_index:  # neglect edge direction for the undirected graph
        continue
    source_year = int(node_data["year"][source_index])
    target_year = int(node_data["year"][target_index])
    if source_year < year_thresh and target_year < year_thresh:
        in_sample_edges.append([source_index, target_index, 1])  # get the positive edge
        negative_target_index = unsupervised_samples.random.choices(
            node_data.index.tolist(), k=1
        )  # generate negative node
        in_sample_edges.append(
            [source_index, negative_target_index[0], 0]
        )  # get the negative edge
    else:
        out_of_sample_edges.append(
            [source_index, target_index, 1]
        )  # get the positive edge
        negative_target_index = unsupervised_samples.random.choices(
            node_data.index.tolist(), k=1
        )  # generate negative node
        out_of_sample_edges.append(
            [source_index, negative_target_index[0], 0]
        )  # get the negative edge
in_sample_edges = np.array(in_sample_edges)
out_of_sample_edges = np.array(out_of_sample_edges)


# Construct the edge features from the learned node representations with l2 normed difference, where edge features are the element-wise square of the difference between the embeddings of two head nodes. Other strategy like element-wise product can also be used to construct edge features.

# In[31]:


in_sample_edge_feat_from_emb = (
    node_embeddings[in_sample_edges[:, 0]] - node_embeddings[in_sample_edges[:, 1]]
) ** 2
out_of_sample_edge_feat_from_emb = (
    node_embeddings[out_of_sample_edges[:, 0]]
    - node_embeddings[out_of_sample_edges[:, 1]]
) ** 2


# Train the Logistic Regression classifier from in-sample edges with the edge features constructed from attri2vec embeddings. 

# In[32]:


clf_edge_pred_from_emb = LogisticRegression(
    verbose=0, solver="lbfgs", multi_class="auto", max_iter=500
)
clf_edge_pred_from_emb.fit(in_sample_edge_feat_from_emb, in_sample_edges[:, 2])


# Predict the edge existence probability with the trained Logistic Regression classifier.

# In[33]:


edge_pred_from_emb = clf_edge_pred_from_emb.predict_proba(
    out_of_sample_edge_feat_from_emb
)


# Get the positive class index of `edge_pred_from_emb`.

# In[34]:


if clf_edge_pred_from_emb.classes_[0] == 1:
    positive_class_index = 0
else:
    positive_class_index = 1


# Evaluate the AUC score for the prediction with attri2vec embeddings.

# In[35]:


roc_auc_score(out_of_sample_edges[:, 2], edge_pred_from_emb[:, positive_class_index])


# As the baseline, we also investigate the performance of node content features in predicting the edges of out-of-sample nodes. Firstly, we construct edge features from node content features with the same strategy.

# In[36]:


in_sample_edge_rep_from_feat = (
    node_data[feature_names].values[in_sample_edges[:, 0]]
    - node_data[feature_names].values[in_sample_edges[:, 1]]
) ** 2
out_of_sample_edge_rep_from_feat = (
    node_data[feature_names].values[out_of_sample_edges[:, 0]]
    - node_data[feature_names].values[out_of_sample_edges[:, 1]]
) ** 2


# Then we train the Logistic Regression classifier from in-sample edges with the edge features constructed from node content features.

# In[37]:


clf_edge_pred_from_feat = LogisticRegression(
    verbose=0, solver="lbfgs", multi_class="auto", max_iter=500
)
clf_edge_pred_from_feat.fit(in_sample_edge_rep_from_feat, in_sample_edges[:, 2])


# Predict the edge existence probability with the trained Logistic Regression classifier.

# In[38]:


edge_pred_from_feat = clf_edge_pred_from_feat.predict_proba(
    out_of_sample_edge_rep_from_feat
)


# Get positive class index of `clf_edge_pred_from_feat`.

# In[39]:


if clf_edge_pred_from_feat.classes_[0] == 1:
    positive_class_index = 0
else:
    positive_class_index = 1


# Evaluate the AUC score for the prediction with node content features.

# In[40]:


roc_auc_score(out_of_sample_edges[:, 2], edge_pred_from_feat[:, positive_class_index])


# attri2vec can inductively infer the representations of out-of-sample nodes from their content attributes. As the inferred node representations well capture both structure and node content information, they perform much better than node content features in predicting the links of out-of-sample nodes.

# <table><tr><td>Run the latest release of this notebook:</td><td><a href="https://mybinder.org/v2/gh/stellargraph/stellargraph/master?urlpath=lab/tree/demos/link-prediction/attri2vec-link-prediction.ipynb" alt="Open In Binder" target="_parent"><img src="https://mybinder.org/badge_logo.svg"/></a></td><td><a href="https://colab.research.google.com/github/stellargraph/stellargraph/blob/master/demos/link-prediction/attri2vec-link-prediction.ipynb" alt="Open In Colab" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg"/></a></td></tr></table>
