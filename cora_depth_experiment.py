#!/usr/bin/env python
# coding: utf-8

# # Cora main comparison template
# 
# The batch runner sets NUM_LAYERS=16 and supplies the seed. All current model and training logic is preserved.
# 
# Checkpoint selection uses validation accuracy; test is evaluated only after restoring the best checkpoint.
# 

# In[ ]:


DATASET = "cora"

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.manifold import TSNE
import numpy as np
import tensorflow as tf
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import urllib.request
import tarfile
from Linear_Separability_numpy_mul import *
import networkx as nx 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.sparse
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

SEED = int(os.environ.get("GCN_SEED", "42"))
np.random.seed(SEED)
tf.random.set_seed(SEED)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20
HIDDEN_DIM = 16
NUM_LAYERS = int(os.environ.get("GCN_LAYERS", "16"))
NUM_EPOCHS = 300
EARLY_STOPPING = False
BEST_MODEL_MONITOR = 'val_acc'
PLAIN_LEARNING_RATE = 0.01
PLAIN_WEIGHT_DECAY = 5e-4
LS_WEIGHT_LEARNING_RATE = 0.01
LS_GRAPH_LEARNING_RATE = 0.001
LS_LAMBDA = 0.2
LS_EVAL_EVERY = 10
CANDIDATE_K = 3
EDGE_INIT_ON_LOGIT = 2.0
EDGE_INIT_OFF_LOGIT = -2.0

DROPEDGE_REFERENCE_PARAMS = {
    'SEED': 42,
    'TRAIN_RATIO': 0.8,
    'HIDDEN_DIM': 16,
    'NUM_LAYERS': 4,
    'DROPOUT_RATE': 0.5,
    'DROP_EDGE_RATE': 0.5,
    'NUM_EPOCHS': 200,
    'EARLY_STOPPING': False,
    'LEARNING_RATE': 0.01,
    'WEIGHT_DECAY': 5e-4,
}


# In[ ]:


# Load data
def download_and_extract_cora():
    url = 'https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz'
    filename = 'cora.tgz'
    if not os.path.exists('cora'):
        print("Downloading Cora dataset...")
        urllib.request.urlretrieve(url, filename)
        print("Extracting Cora dataset...")
        with tarfile.open(filename, 'r:gz') as tar_ref:
            tar_ref.extractall()
        os.remove(filename)
    else:
        print("Cora dataset already exists.")

def load_data():
    download_and_extract_cora()
    print("Loading and preprocessing data...")

    content = []
    with open("cora/cora.content", 'r') as f:
        for line in f:
            content.append(line.strip().split())

    node_ids = [int(line[0]) for line in content]
    id_map = {j: i for i, j in enumerate(node_ids)}

    features = np.array([list(map(float, line[1:-1])) for line in content])
    labels = [line[-1] for line in content]

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)



    edges = []
    with open("cora/cora.cites", 'r') as f:
        for line in f:
            src, dst = map(int, line.strip().split())
            if src in id_map and dst in id_map:
                edges.append((id_map[src], id_map[dst]))

    num_nodes = len(node_ids)
    adj = sparse.lil_matrix((num_nodes, num_nodes))
    for i, j in edges:
        adj[i, j] = 1
        adj[j, i] = 1

    adj = preprocess_adj(adj)
    return features, adj, labels

def preprocess_adj(adj):
    adj = sparse.coo_matrix(adj)
    adj_normalized = adj
    # adj_ = adj + sparse.eye(adj.shape[0])
    # rowsum = np.array(adj_.sum(1))
    # d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    # d_mat_inv_sqrt = sparse.diags(d_inv_sqrt)
    # adj_normalized = adj_.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return adj_normalized
def filter_data_by_labels(features, labels, adj, target_labels=None):

    if target_labels is None or target_labels == 'all':
        return features, labels, adj

    mask = np.isin(labels, target_labels)

    features_filtered = features[mask]
    labels_filtered = labels[mask]


    adj_csr = adj.tocsr()
    node_indices = np.where(mask)[0]
    adj_filtered = adj_csr[node_indices, :][:, node_indices]

    return features_filtered, labels_filtered, adj_filtered


# In[ ]:


def build_candidate_edges(features_np, adj, candidate_k=CANDIDATE_K):
    """Build sparse candidate structure: original edges plus top-k two-hop possible new edges."""
    num_nodes = features_np.shape[0]
    adj_coo = adj.tocoo()
    original_edges = {
        (min(int(i), int(j)), max(int(i), int(j)))
        for i, j in zip(adj_coo.row, adj_coo.col)
        if int(i) != int(j)
    }

    candidate_edges = set(original_edges)
    adj_csr = adj.tocsr().astype(np.float32)
    two_hop = (adj_csr @ adj_csr).tolil()
    for src in range(num_nodes):
        two_hop[src, src] = 0
        for dst in adj_csr[src].indices:
            two_hop[src, int(dst)] = 0
        row = two_hop.getrow(src).tocoo()
        if row.nnz == 0:
            continue
        top_local = np.argsort(row.data)[-candidate_k:]
        for dst in row.col[top_local]:
            candidate_edges.add((min(src, int(dst)), max(src, int(dst))))

    candidate_edges = sorted(candidate_edges)
    edge_init_logits = np.array([
        EDGE_INIT_ON_LOGIT if edge in original_edges else EDGE_INIT_OFF_LOGIT
        for edge in candidate_edges
    ], dtype=np.float32)

    rows = np.array([i for i, j in candidate_edges] + [j for i, j in candidate_edges], dtype=np.int64)
    cols = np.array([j for i, j in candidate_edges] + [i for i, j in candidate_edges], dtype=np.int64)
    values = np.ones(len(rows), dtype=np.float32)
    candidate_adj = tf.SparseTensor(
        indices=np.column_stack((rows, cols)),
        values=values,
        dense_shape=adj.shape,
    )
    return tf.sparse.reorder(candidate_adj), tf.constant(edge_init_logits, dtype=tf.float32), len(original_edges), len(candidate_edges)


features, adj, labels_np = load_data()
# print("adj is None?", adj is None)
features, labels_np, adj = filter_data_by_labels(features, labels_np, adj)
features_np = features.astype(np.float32)
candidate_adj_tf, candidate_edge_init_tf, num_original_edges, num_candidate_edges = build_candidate_edges(features_np, adj)
features = tf.constant(features_np, dtype=tf.float32)
labels = tf.constant(labels_np, dtype=tf.int32)

adj = adj.tocoo()
indices = np.column_stack((adj.row, adj.col))
adj_tf = tf.SparseTensor(indices=indices, values=adj.data.astype(np.float32), dense_shape=adj.shape)
adj_tf = tf.sparse.reorder(adj_tf)
print(f'Candidate edges: original={num_original_edges}, total={num_candidate_edges}, added={num_candidate_edges - num_original_edges}')

num_nodes = features.shape[0]
rng = np.random.default_rng(SEED)
all_indices = np.arange(num_nodes, dtype=np.int32)
idx_train_val, idx_test = train_test_split(
    all_indices,
    test_size=TEST_RATIO,
    random_state=SEED,
    stratify=labels_np,
)
relative_val_ratio = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
idx_train, idx_val = train_test_split(
    idx_train_val,
    test_size=relative_val_ratio,
    random_state=SEED,
    stratify=labels_np[idx_train_val],
)
idx_train = np.asarray(idx_train, dtype=np.int32)
idx_val = np.asarray(idx_val, dtype=np.int32)
idx_test = np.asarray(idx_test, dtype=np.int32)
idx_train_tf = tf.constant(idx_train, dtype=tf.int32)
idx_val_tf = tf.constant(idx_val, dtype=tf.int32)
idx_test_tf = tf.constant(idx_test, dtype=tf.int32)

partial_indices = np.random.choice(idx_train, 500, replace=False)
partial_indices_tf = tf.constant(partial_indices, dtype=tf.int32)
partial_labels = labels.numpy()[partial_indices]
partial_labels_tf = tf.constant(partial_labels, dtype=tf.int32)
idx_train.shape


# In[ ]:


def adj_preprocess(adj):
    """Normalize sparse adjacency as D^{-1/2} A D^{-1/2} without densifying."""
    if not isinstance(adj, tf.SparseTensor):
        adj = tf.sparse.from_dense(adj)

    adj = tf.sparse.reorder(adj)
    num_nodes = tf.cast(adj.dense_shape[0], tf.int64)
    loop_nodes = tf.range(num_nodes, dtype=tf.int64)
    loop_indices = tf.stack([loop_nodes, loop_nodes], axis=1)

    indices = tf.concat([tf.cast(adj.indices, tf.int64), loop_indices], axis=0)
    values = tf.concat([
        tf.cast(adj.values, tf.float32),
        tf.ones([tf.shape(loop_nodes)[0]], dtype=tf.float32),
    ], axis=0)

    rows = indices[:, 0]
    cols = indices[:, 1]
    degree = tf.math.unsorted_segment_sum(values, rows, num_nodes)
    d_inv_sqrt = tf.math.rsqrt(tf.maximum(degree, 1e-12))
    norm_values = values * tf.gather(d_inv_sqrt, rows) * tf.gather(d_inv_sqrt, cols)

    return tf.sparse.reorder(tf.SparseTensor(indices, norm_values, adj.dense_shape))


# In[ ]:


def to_sparse_tensor(x):
    """Convert input to SparseTensor with stable index/value dtypes."""
    if not isinstance(x, tf.SparseTensor):
        x = tf.sparse.from_dense(x)

    x = tf.SparseTensor(
        indices=tf.cast(x.indices, tf.int64),
        values=tf.cast(x.values, tf.float32),
        dense_shape=tf.cast(x.dense_shape, tf.int64)
    )
    return x


# In[ ]:


def d_deal(d):
    D_tanh = (d + tf.transpose(d))

    D_discrete = D_tanh - tf.linalg.diag(tf.linalg.diag_part(D_tanh))
    return D_discrete


# In[ ]:


def A_deal(d, adj, hi=0.6, lo=0.4, temperature=12.0, make_symmetric=True, remove_self_loops=True):
    """Legacy sparse graph update kept for compatibility; the fast GCN path uses edge vectors."""
    d_sp = to_sparse_tensor(d)
    adj_sp = to_sparse_tensor(adj)
    summed = tf.sparse.reorder(tf.sparse.add(adj_sp, d_sp))

    if remove_self_loops:
        keep = summed.indices[:, 0] != summed.indices[:, 1]
        indices = tf.boolean_mask(summed.indices, keep)
        values = tf.boolean_mask(summed.values, keep)
    else:
        indices = summed.indices
        values = summed.values

    values = tf.sigmoid(temperature * (values - lo))
    if make_symmetric:
        reverse_indices = tf.stack([indices[:, 1], indices[:, 0]], axis=1)
        indices = tf.concat([indices, reverse_indices], axis=0)
        values = tf.concat([values, values], axis=0)

    return tf.sparse.reorder(tf.SparseTensor(indices, values, adj_sp.dense_shape))


# In[ ]:


class GCN(tf.keras.Model):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        adj,
        candidate_adj=None,
        edge_init_logits=None,
        num_layers=NUM_LAYERS,
        lambda_reg=1e-4,
        lambda_l21=1e-4,
        lambda_l1=1e-4
    ):
        super(GCN, self).__init__()

        self.adj = adj
        self.num_layers = num_layers
        self.num_hidden_layers = num_layers - 1

        self.lambda_l1 = lambda_l1
        self.lambda_reg = lambda_reg
        self.lambda_l21 = lambda_l21
        self.dtype_ = tf.float32

        # =====================================================
        # Sparse adjacency
        # =====================================================
        self.adj = tf.sparse.reorder(adj)

        self.candidate_adj = tf.sparse.reorder(
            candidate_adj
            if candidate_adj is not None
            else adj
        )

        self.dense_shape = tf.cast(
            self.adj.dense_shape,
            tf.int64
        )

        # =====================================================
        # Keep only the upper triangle of undirected candidate edges.
        # Learn one parameter per undirected edge.
        # =====================================================
        adj_indices = tf.cast(
            self.candidate_adj.indices,
            tf.int64
        )

        upper_edge_mask = (
            adj_indices[:, 0]
            <
            adj_indices[:, 1]
        )

        self.upper_edge_indices = tf.boolean_mask(
            adj_indices,
            upper_edge_mask
        )

        self.num_edges = int(
            tf.shape(
                self.upper_edge_indices
            )[0].numpy()
        )

        # =====================================================
        # Initial edge logits
        # =====================================================
        if edge_init_logits is None:
            edge_init_logits = (
                tf.ones(
                    [self.num_edges],
                    dtype=self.dtype_
                )
                * EDGE_INIT_ON_LOGIT
            )

        self.edge_init_logits = tf.cast(
            edge_init_logits,
            self.dtype_
        )

        # Distinguish between:
        # Original edges / new candidate edges
        self.original_edge_mask = (
            self.edge_init_logits > 0.0
        )

        # =====================================================
        # Assign one d_i to each hidden layer.
        # =====================================================
        self.d_params = []

        for i in range(
            self.num_hidden_layers
        ):
            self.d_params.append(
                self.add_weight(
                    name=f'd{i + 1}',
                    shape=(self.num_edges,),
                    initializer='zeros',
                    trainable=True
                )
            )

        # =====================================================
        # Hidden layers
        # =====================================================
        self.hidden_layers = [
            tf.keras.layers.Dense(
                hidden_dim,
                activation='relu',
                name=f'gcn_layer_{i + 1}'
            )
            for i in range(
                self.num_hidden_layers
            )
        ]

        # =====================================================
        # Output layer
        # =====================================================
        self.output_layer = (
            tf.keras.layers.Dense(
                output_dim,
                name='output_layer'
            )
        )

        # =====================================================
        # Build layers
        # =====================================================
        prev_dim = input_dim

        for layer in self.hidden_layers:
            layer.build(
                (None, prev_dim)
            )
            prev_dim = hidden_dim

        self.output_layer.build(
            (None, hidden_dim)
        )

        # Preserve the existing layer_groups name.
        self.layer_groups = [
            [self.d_params[i]]
            +
            self.hidden_layers[
                i
            ].trainable_variables
            for i in range(
                self.num_hidden_layers
            )
        ] + [
            self.output_layer.trainable_variables
        ]

    # =========================================================
    # Forward
    # =========================================================
    def call(
        self,
        x,
        return_intermediate=False
    ):

        intermediate_outputs = {}

        # Start from the initial logits.
        current_logits = (
            self.edge_init_logits
        )

        current_adj = self.adj

        # =====================================================
        # Hidden layers
        #
        # Layer 1:
        # z1 = z0 + d1
        #
        # Layer 2:
        # z2 = z1 + d2
        #    = z0 + d1 + d2
        #
        # ...
        # =====================================================
        for i, layer in enumerate(
            self.hidden_layers
        ):

            current_adj, current_logits = (
                self.learned_sparse_adj(
                    self.d_params[i],
                    current_logits
                )
            )

            D = adj_preprocess(
                current_adj
            )

            x = (
                tf.sparse
                .sparse_dense_matmul(
                    D,
                    x
                )
            )

            x = layer(x)

            intermediate_outputs[
                f'gcn_layer_{i + 1}'
            ] = x

        # =====================================================
        # Output layer
        #
        # Use the graph after the final accumulation.
        # =====================================================
        D = adj_preprocess(
            current_adj
        )

        x = (
            tf.sparse
            .sparse_dense_matmul(
                D,
                x
            )
        )

        x = self.output_layer(x)

        intermediate_outputs[
            'output_layer'
        ] = x

        if return_intermediate:
            return (
                x,
                intermediate_outputs
            )

        return x

    # =========================================================
    # Learnable sparse adjacency
    # =========================================================
    def learned_sparse_adj(
        self,
        edge_delta,
        prev_logits,
        threshold=0.5
    ):

        # Core update:
        # Current logits = previous-layer logits + current-layer d
        logits = (
            prev_logits
            +
            edge_delta
        )

        edge_prob = tf.sigmoid(
            logits
        )

        # Use hard binary values (0/1) in the forward pass.
        hard_gate = tf.cast(
            edge_prob > threshold,
            tf.float32
        )

        # Straight-Through Estimator
        #
        # forward:
        #   hard_gate
        #
        # backward:
        #   edge_prob
        upper_values = (
            edge_prob
            +
            tf.stop_gradient(
                hard_gate
                -
                edge_prob
            )
        )

        # Undirected graph:
        # Add reverse edges.
        reverse_indices = tf.stack(
            [
                self.upper_edge_indices[:, 1],
                self.upper_edge_indices[:, 0]
            ],
            axis=1
        )

        indices = tf.concat(
            [
                self.upper_edge_indices,
                reverse_indices
            ],
            axis=0
        )

        values = tf.concat(
            [
                upper_values,
                upper_values
            ],
            axis=0
        )

        adj = tf.SparseTensor(
            indices=indices,
            values=values,
            dense_shape=self.dense_shape
        )

        return (
            tf.sparse.reorder(adj),
            logits
        )

    # =========================================================
    # Graph structure statistics
    # =========================================================
    def structure_counts(
        self,
        layer_index=None,
        threshold=0.5
    ):

        # Match the forward pass exactly.
        logits = self.edge_init_logits

        if layer_index is None:
            params = self.d_params

        else:
            params = (
                self.d_params[
                    :layer_index + 1
                ]
            )

        # Accumulate.
        for d in params:
            logits = (
                logits
                +
                d
            )

        active = (
            tf.sigmoid(logits)
            >
            threshold
        )

        original = (
            self.original_edge_mask
        )

        candidate = (
            tf.logical_not(
                original
            )
        )

        return {

            'kept_original_edges':
                int(
                    tf.reduce_sum(
                        tf.cast(
                            tf.logical_and(
                                active,
                                original
                            ),
                            tf.int32
                        )
                    ).numpy()
                ),

            'dropped_original_edges':
                int(
                    tf.reduce_sum(
                        tf.cast(
                            tf.logical_and(
                                tf.logical_not(
                                    active
                                ),
                                original
                            ),
                            tf.int32
                        )
                    ).numpy()
                ),

            'added_candidate_edges':
                int(
                    tf.reduce_sum(
                        tf.cast(
                            tf.logical_and(
                                active,
                                candidate
                            ),
                            tf.int32
                        )
                    ).numpy()
                ),

            'inactive_candidate_edges':
                int(
                    tf.reduce_sum(
                        tf.cast(
                            tf.logical_and(
                                tf.logical_not(
                                    active
                                ),
                                candidate
                            ),
                            tf.int32
                        )
                    ).numpy()
                )
        }

    # =========================================================
    # Regularization
    # =========================================================
    def l21_regularization(self):

        return (
            self.lambda_l21
            *
            tf.add_n(
                [
                    tf.norm(
                        d,
                        ord=2
                    )
                    for d in self.d_params
                ]
            )
        )

    def nuclear_norm_regularization(self):

        regs = [
            tf.nn.l2_loss(d)
            for d in self.d_params
        ]

        return (
            self.lambda_reg
            *
            tf.add_n(regs)
        )

    def nuclear_norm_regularization_for(
        self,
        d_param
    ):

        return (
            self.lambda_reg
            *
            tf.nn.l2_loss(
                d_param
            )
        )

    def l1_regularization(self):

        regs = [
            tf.reduce_sum(
                tf.abs(d)
            )
            for d in self.d_params
        ]

        return (
            self.lambda_l1
            *
            tf.add_n(regs)
        )

    # Preserve the existing function name.
    def apply_upper_triangle_mask(
        self
    ):
        return


# In[ ]:


class PlainGCN(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, output_dim, adj, num_layers=NUM_LAYERS):
        super().__init__()
        self.adj_norm = adj_preprocess(adj)
        self.num_layers = num_layers
        self.num_hidden_layers = num_layers - 1
        self.hidden_layers = [
            tf.keras.layers.Dense(hidden_dim, activation='relu', name=f'gcn_layer_{i + 1}')
            for i in range(self.num_hidden_layers)
        ]
        self.output_layer = tf.keras.layers.Dense(output_dim, name='output_layer')

        prev_dim = input_dim
        for layer in self.hidden_layers:
            layer.build((None, prev_dim))
            prev_dim = hidden_dim
        self.output_layer.build((None, hidden_dim))

    def call(self, x, return_intermediate=False):
        intermediate_outputs = {}
        for i, layer in enumerate(self.hidden_layers):
            x = tf.sparse.sparse_dense_matmul(self.adj_norm, x)
            x = layer(x)
            intermediate_outputs[f'gcn_layer_{i + 1}'] = x

        x = tf.sparse.sparse_dense_matmul(self.adj_norm, x)
        x = self.output_layer(x)
        intermediate_outputs['output_layer'] = x
        return (x, intermediate_outputs) if return_intermediate else x


def ce_loss(labels_sub, logits_sub):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_sub, logits=logits_sub))


def masked_ce_loss(labels, logits, indices, model=None, weight_decay=0.0):
    logits_sub = tf.gather(logits, indices)
    labels_sub = tf.gather(labels, indices)
    loss = ce_loss(labels_sub, logits_sub)
    if model is not None and weight_decay > 0:
        kernels = [tf.nn.l2_loss(v) for v in model.trainable_variables if 'kernel' in v.name]
        if kernels:
            loss += weight_decay * tf.add_n(kernels)
    return loss


def evaluate_logits(logits, indices):
    logits_sub = tf.gather(logits, indices)
    labels_sub = tf.gather(labels, indices)
    loss = ce_loss(labels_sub, logits_sub)
    pred = tf.argmax(logits_sub, axis=1)
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(pred, labels_sub.dtype), labels_sub), tf.float32))
    return float(loss.numpy()), float(acc.numpy())


def get_layer_names(num_layers=NUM_LAYERS):
    return [f'gcn_layer_{i + 1}' for i in range(num_layers - 1)] + ['output_layer']


def init_layer_ls_history(layer_names):
    return {
        layer_name: {'LS1': [], 'LS2': [], 'Jw': [], 'LDA': []}
        for layer_name in layer_names
    }


def append_layer_ls(history, layer_names, intermediate_outputs, indices_for_ls, labels_for_ls, compute=True):
    for layer_name in layer_names:
        metrics = history['layer_ls'][layer_name]
        if compute:
            partial_output = tf.gather(intermediate_outputs[layer_name], indices_for_ls)
            ls1, ls2, jw, lda = W(partial_output, labels_for_ls.reshape(-1, 1))
            values = {'LS1': float(ls1), 'LS2': float(ls2), 'Jw': float(jw), 'LDA': float(lda)}
        else:
            values = {key: (metrics[key][-1] if metrics[key] else np.nan) for key in metrics}
        for key, value in values.items():
            metrics[key].append(value)


def build_history(layer_names):
    return {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'ls2_output': [],
        'layer_ls': init_layer_ls_history(layer_names),
    }


def monitor_value_from_history(history, monitor_name):
    if monitor_name not in history:
        raise ValueError(f'Unknown monitor: {monitor_name}')
    return history[monitor_name][-1]


def is_better_monitor(value, best_value, monitor_name):
    if best_value is None:
        return True
    if monitor_name.endswith('acc') or monitor_name.startswith('ls'):
        return value > best_value
    return value < best_value


def final_layer_ls(intermediate_outputs, layer_names, indices_for_ls, labels_for_ls):
    final_ls = {}
    for layer_name in layer_names:
        partial_output = tf.gather(intermediate_outputs[layer_name], indices_for_ls)
        ls1, ls2, jw, lda = W(partial_output, labels_for_ls.reshape(-1, 1))
        final_ls[layer_name] = {'LS1': float(ls1), 'LS2': float(ls2), 'Jw': float(jw), 'LDA': float(lda)}
    return final_ls


def train_plain_gcn(name=None, num_epochs=NUM_EPOCHS):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    if name is None:
        name = f'Plain-GCN-{NUM_LAYERS}-layer'
    model = PlainGCN(input_dim, HIDDEN_DIM, output_dim, adj_tf, num_layers=NUM_LAYERS)
    optimizer = tf.keras.optimizers.Adam(learning_rate=PLAIN_LEARNING_RATE)
    layer_names = get_layer_names(NUM_LAYERS)
    history = build_history(layer_names)
    for epoch in range(num_epochs):
        with tf.GradientTape() as tape:
            logits = model(features)
            loss = masked_ce_loss(labels, logits, idx_train_tf, model, PLAIN_WEIGHT_DECAY)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        compute_ls = (epoch % LS_EVAL_EVERY == 0) or (epoch == num_epochs - 1)
        if compute_ls:
            output, intermediate_outputs = model(features, return_intermediate=True)
        else:
            output = model(features)
            intermediate_outputs = None
        train_loss, train_acc = evaluate_logits(output, idx_train_tf)
        val_loss, val_acc = evaluate_logits(output, idx_val_tf)
        append_layer_ls(history, layer_names, intermediate_outputs, partial_indices_tf, partial_labels, compute=compute_ls)
        ls2_output = history['layer_ls']['output_layer']['LS2'][-1]

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['ls2_output'].append(ls2_output)

        if epoch % 20 == 0 or epoch == num_epochs - 1:
            print(
                f'[{name}] epoch={epoch:03d} loss={float(loss.numpy()):.4f} '
                f'train={train_acc:.4f} val={val_acc:.4f} LS2_out={ls2_output:.4f}'
            )
    output, intermediate_outputs = model(features, return_intermediate=True)
    _, final_test = evaluate_logits(output, idx_test_tf)
    return {
        'name': name,
        'model': model,
        'history': history,
        'final_test': final_test,
        'final_ls': final_layer_ls(intermediate_outputs, layer_names, partial_indices_tf, partial_labels),
        'params': {
            'learning_rate': PLAIN_LEARNING_RATE,
            'weight_decay': PLAIN_WEIGHT_DECAY,
            'hidden_dim': HIDDEN_DIM,
            'num_layers': NUM_LAYERS,
            'num_epochs': num_epochs,
            'early_stopping': EARLY_STOPPING,
        },
    }


# In[ ]:


def differentiable_LS2(H, labels_sub, num_classes, eps=1e-8):
    """Differentiable version of W(...)[1] from Linear_Separability_numpy_mul.py.

    For each class c, this keeps the original one-vs-rest LS2 structure:
      M = {a_i - b_j}
      w = sum(M) / ||sum(M)||
      LS2 = abs(sum(w @ M.T)) / sum(abs(w @ M.T))

    To avoid materializing the full M feature matrix, we project A and B first,
    then form only the pairwise projected differences.
    """
    H = tf.cast(H, tf.float32)
    labels_sub = tf.reshape(tf.cast(labels_sub, tf.int32), [-1])
    n_total = tf.cast(tf.shape(H)[0], tf.float32)

    total_ls2 = tf.constant(0.0, dtype=tf.float32)
    total_weight = tf.constant(0.0, dtype=tf.float32)

    for c in range(num_classes):
        mask_a = tf.equal(labels_sub, c)
        mask_b = tf.logical_not(mask_a)
        H_a = tf.boolean_mask(H, mask_a)
        H_b = tf.boolean_mask(H, mask_b)
        r1 = tf.cast(tf.shape(H_a)[0], tf.float32)
        r2 = tf.cast(tf.shape(H_b)[0], tf.float32)

        def class_ls2():
            sum_m = r2 * tf.reduce_sum(H_a, axis=0) - r1 * tf.reduce_sum(H_b, axis=0)
            w = sum_m / (tf.norm(sum_m) + eps)
            p_a = tf.linalg.matvec(H_a, w)
            p_b = tf.linalg.matvec(H_b, w)
            pairwise_proj = tf.expand_dims(p_a, 1) - tf.expand_dims(p_b, 0)
            numerator = tf.abs(tf.reduce_sum(pairwise_proj))
            denominator = tf.reduce_sum(tf.abs(pairwise_proj)) + eps
            weight = r1 / (n_total + eps)
            return (numerator / denominator) * weight, weight

        def empty_ls2():
            return tf.constant(0.0, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32)

        class_value, class_weight = tf.cond(
            tf.logical_and(tf.shape(H_a)[0] > 0, tf.shape(H_b)[0] > 0),
            class_ls2,
            empty_ls2,
        )
        total_ls2 += class_value
        total_weight += class_weight

    return total_ls2 / (total_weight + eps)


def loss_lsm(labels, logits, model, H_partial, labels_partial, num_classes, lambda_ls=0.1, active_d=None):
    ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    ls2 = differentiable_LS2(H_partial, labels_partial, num_classes)
    if active_d is None:
        nuclear_reg = model.nuclear_norm_regularization()
    else:
        nuclear_reg = model.nuclear_norm_regularization_for(active_d)
    loss = ce + nuclear_reg + lambda_ls * (1.0 - ls2)
    return loss


# In[ ]:


def train_ls_gcn(
    name=None,
    num_epochs=NUM_EPOCHS
):

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    if name is None:
        name = (
            f'LS-aware-GCN-'
            f'{NUM_LAYERS}-layer'
        )

    # =========================================================
    # Model
    # =========================================================
    model = GCN(
        input_dim,
        HIDDEN_DIM,
        output_dim,
        adj_tf,
        candidate_adj_tf,
        candidate_edge_init_tf,
        num_layers=NUM_LAYERS
    )

    # =========================================================
    # Optimizers
    # =========================================================
    optimizer = (
        tf.keras.optimizers.Adam(
            learning_rate=
            LS_WEIGHT_LEARNING_RATE
        )
    )

    optimizer_d = (
        tf.keras.optimizers.Adam(
            learning_rate=
            LS_GRAPH_LEARNING_RATE
        )
    )

    # Run a forward pass first.
    # Ensure model variables are initialized before building the optimizer.
    _ = model(features)

    optimizer.build(
        model.trainable_variables
    )

    optimizer_d.build(
        model.trainable_variables
    )

    # =========================================================
    # d parameters
    # =========================================================
    d_params = model.d_params

    d_param_names = {
        var.name
        for var in d_params
    }

    # Preserve the existing layer_names.
    layer_names = get_layer_names(
        NUM_LAYERS
    )

    history = build_history(
        layer_names
    )

    # =========================================================
    # Best model
    # =========================================================
    best_monitor_value = None
    best_epoch = -1
    best_weights = None
    best_metrics = {}

    # =========================================================
    # Training
    # =========================================================
    for epoch in range(
        num_epochs
    ):

        # =====================================================
        # Step 1
        # Update the structure parameters d_i.
        # =====================================================
        for i, d_param in enumerate(
            d_params
        ):

            with tf.GradientTape() as tape:

                logits, intermediate_outputs_train = (
                    model(
                        features,
                        return_intermediate=True
                    )
                )

                # Current hidden layer
                H_partial = tf.gather(
                    intermediate_outputs_train[
                        layer_names[i]
                    ],
                    partial_indices_tf
                )

                # =============================================
                # CE loss
                # =============================================
                ce = tf.reduce_mean(
                    tf.nn
                    .sparse_softmax_cross_entropy_with_logits(
                        labels=tf.gather(
                            labels,
                            idx_train_tf
                        ),
                        logits=tf.gather(
                            logits,
                            idx_train_tf
                        )
                    )
                )

                # =============================================
                # LS2 for the current hidden layer
                # =============================================
                ls2_hidden = (
                    differentiable_LS2(
                        H_partial,
                        partial_labels_tf,
                        output_dim
                    )
                )

                # =============================================
                # Regularization for the current d_i
                # =============================================
                nuclear_reg = (
                    model
                    .nuclear_norm_regularization_for(
                        d_param
                    )
                )

                # =============================================
                # Final d
                #
                # Jointly optimize:
                # LS for the final hidden layer
                # +
                # output LS
                # =============================================
                if (
                    i
                    ==
                    len(d_params) - 1
                ):

                    H_output_partial = (
                        tf.gather(
                            intermediate_outputs_train[
                                'output_layer'
                            ],
                            partial_indices_tf
                        )
                    )

                    ls2_output_train = (
                        differentiable_LS2(
                            H_output_partial,
                            partial_labels_tf,
                            output_dim
                        )
                    )

                    ls_loss = (
                        0.5
                        *
                        (
                            1.0
                            -
                            ls2_hidden
                        )
                        +
                        0.5
                        *
                        (
                            1.0
                            -
                            ls2_output_train
                        )
                    )

                else:

                    ls_loss = (
                        1.0
                        -
                        ls2_hidden
                    )

                # =============================================
                # Structure loss
                #
                # Note:
                # Do not call loss_lsm() again here.
                # Doing so would overwrite the output LS computed above.
                # =============================================
                loss_d = (
                    ce
                    +
                    nuclear_reg
                    +
                    LS_LAMBDA
                    *
                    ls_loss
                )

            # =============================================
            # Update only d_i at this step.
            # =============================================
            grads_d = tape.gradient(
                loss_d,
                d_param
            )

            if grads_d is None:

                print(
                    f'Warning: gradient for '
                    f'd_params[{i}] is None'
                )

            else:

                optimizer_d.apply_gradients(
                    [
                        (
                            grads_d,
                            d_param
                        )
                    ]
                )

            model.apply_upper_triangle_mask()

        # =====================================================
        # Step 2
        # Update the GCN network weights.
        # =====================================================
        w_params = [
            var
            for var in model.trainable_variables
            if var.name
            not in d_param_names
        ]

        with tf.GradientTape() as tape:

            logits = model(
                features
            )

            loss_w = masked_ce_loss(
                labels,
                logits,
                idx_train_tf,
                model,
                PLAIN_WEIGHT_DECAY
            )

        grads_w = tape.gradient(
            loss_w,
            w_params
        )

        grads_and_vars = [
            (g, v)
            for g, v in zip(
                grads_w,
                w_params
            )
            if g is not None
        ]

        if not grads_and_vars:

            print(
                'Warning: all gradients '
                'for w_params are None'
            )

        else:

            optimizer.apply_gradients(
                grads_and_vars
            )

        # =====================================================
        # Evaluation
        # =====================================================
        compute_ls = (
            epoch % LS_EVAL_EVERY == 0
            or
            epoch == num_epochs - 1
        )

        if compute_ls:

            output, intermediate_outputs = (
                model(
                    features,
                    return_intermediate=True
                )
            )

        else:

            output = model(
                features
            )

            intermediate_outputs = None

        train_loss, train_acc = (
            evaluate_logits(
                output,
                idx_train_tf
            )
        )

        val_loss, val_acc = (
            evaluate_logits(
                output,
                idx_val_tf
            )
        )

        # =====================================================
        # LS evaluation
        # =====================================================
        append_layer_ls(
            history,
            layer_names,
            intermediate_outputs,
            partial_indices_tf,
            partial_labels,
            compute=compute_ls
        )

        ls2_output = (
            history[
                'layer_ls'
            ][
                'output_layer'
            ][
                'LS2'
            ][-1]
        )

        # =====================================================
        # History
        # =====================================================
        history[
            'train_loss'
        ].append(
            train_loss
        )

        history[
            'train_acc'
        ].append(
            train_acc
        )

        history[
            'val_loss'
        ].append(
            val_loss
        )

        history[
            'val_acc'
        ].append(
            val_acc
        )

        history[
            'ls2_output'
        ].append(
            ls2_output
        )

        # =====================================================
        # Best model
        # =====================================================
        monitor_value = (
            monitor_value_from_history(
                history,
                BEST_MODEL_MONITOR
            )
        )

        improved = (
            is_better_monitor(
                monitor_value,
                best_monitor_value,
                BEST_MODEL_MONITOR
            )
        )

        if improved:

            best_monitor_value = (
                monitor_value
            )

            best_epoch = epoch

            best_weights = (
                model.get_weights()
            )

            best_metrics = {

                'epoch':
                    best_epoch,

                'monitor':
                    BEST_MODEL_MONITOR,

                'monitor_value':
                    best_monitor_value,

                'train_loss':
                    train_loss,

                'train_acc':
                    train_acc,

                'val_loss':
                    val_loss,

                'val_acc':
                    val_acc,

                'ls2_output':
                    ls2_output
            }

        # =====================================================
        # Print
        # =====================================================
        print(
            f'[{name}] '
            f'epoch={epoch:03d} '
            f'loss_w='
            f'{float(loss_w.numpy()):.4f} '
            f'loss_d='
            f'{float(loss_d.numpy()):.4f} '
            f'train='
            f'{train_acc:.4f} '
            f'val='
            f'{val_acc:.4f} '
            f'LS2_out='
            f'{ls2_output:.4f}'
        )

        if improved:

            print(
                f'[{name}] '
                f'new best '
                f'{BEST_MODEL_MONITOR}='
                f'{best_monitor_value:.4f} '
                f'at epoch='
                f'{best_epoch:03d}'
            )

    # =========================================================
    # Restore best
    # =========================================================
    if best_weights is not None:

        model.set_weights(
            best_weights
        )

        print(
            f'[{name}] '
            f'restored best '
            f'epoch={best_epoch:03d}, '
            f'{BEST_MODEL_MONITOR}='
            f'{best_monitor_value:.4f}'
        )

    # =========================================================
    # Final evaluation
    # =========================================================
    output, intermediate_outputs = (
        model(
            features,
            return_intermediate=True
        )
    )

    _, final_test = evaluate_logits(
        output,
        idx_test_tf
    )

    # =========================================================
    # Return
    #
    # Preserve all existing field names.
    # =========================================================
    return {

        'name':
            name,

        'model':
            model,

        'history':
            history,

        'final_test':
            final_test,

        'final_ls':
            final_layer_ls(
                intermediate_outputs,
                layer_names,
                partial_indices_tf,
                partial_labels
            ),

        'structure_counts':
            model.structure_counts(),

        'best_metrics':
            best_metrics,

        'params': {

            'weight_learning_rate':
                LS_WEIGHT_LEARNING_RATE,

            'graph_learning_rate':
                LS_GRAPH_LEARNING_RATE,

            'lambda_ls':
                LS_LAMBDA,

            'best_model_monitor':
                BEST_MODEL_MONITOR,

            'best_epoch':
                best_epoch,

            'hidden_dim':
                HIDDEN_DIM,

            'num_layers':
                NUM_LAYERS,

            'num_epochs':
                num_epochs,

            'early_stopping':
                EARLY_STOPPING
        }
    }


# ## 1. Proposed LS-aware GCN
# 

# In[ ]:


input_dim = features.shape[1]
output_dim = len(np.unique(labels_np))
experiment_params = {
    'SEED': SEED,
    'TRAIN_RATIO': TRAIN_RATIO,
    'VAL_RATIO': VAL_RATIO,
    'TEST_RATIO': TEST_RATIO,
    'HIDDEN_DIM': HIDDEN_DIM,
    'NUM_LAYERS': NUM_LAYERS,
    'NUM_EPOCHS': NUM_EPOCHS,
    'EARLY_STOPPING': EARLY_STOPPING,
    'BEST_MODEL_MONITOR': BEST_MODEL_MONITOR,
    'PLAIN_LEARNING_RATE': PLAIN_LEARNING_RATE,
    'PLAIN_WEIGHT_DECAY': PLAIN_WEIGHT_DECAY,
    'LS_WEIGHT_LEARNING_RATE': LS_WEIGHT_LEARNING_RATE,
    'LS_GRAPH_LEARNING_RATE': LS_GRAPH_LEARNING_RATE,
    'LS_LAMBDA': LS_LAMBDA,
    'LS_EVAL_EVERY': LS_EVAL_EVERY,
    'CANDIDATE_K': CANDIDATE_K,
    'EDGE_INIT_ON_LOGIT': EDGE_INIT_ON_LOGIT,
    'EDGE_INIT_OFF_LOGIT': EDGE_INIT_OFF_LOGIT,
    'DROPEDGE_REFERENCE_PARAMS': DROPEDGE_REFERENCE_PARAMS,
}

print(f'LS comparison params: layers={NUM_LAYERS}, hidden_dim={HIDDEN_DIM}')
for key, value in experiment_params.items():
    print(f'  {key}: {value}')
print(f'Train/Val/Test: {len(idx_train)}/{len(idx_val)}/{len(idx_test)}')

# Only run the proposed model here; Plain-GCN is run once with the common baselines below.
ls_result = train_ls_gcn(f'LS-aware-GCN-{NUM_LAYERS}-layer')
ls_comparison_results = [ls_result]



# ## 2. Common baselines
# 
# The baseline code reloads Cora with the same seed and split, so it does not overwrite the already-computed proposed-model result.
# 

# In[ ]:


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tarfile
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy import sparse
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from Linear_Separability_numpy_mul import W

try:
    get_ipython().run_line_magic("matplotlib", "inline")
except NameError:
    pass

SEED = int(os.environ.get("GCN_SEED", "42"))
np.random.seed(SEED)
tf.random.set_seed(SEED)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20
HIDDEN_DIM = 16
NUM_LAYERS = int(os.environ.get("GCN_LAYERS", "16"))
NUM_EPOCHS = 300
DROPOUT_RATE = 0.5
DROP_EDGE_RATE = 0.2
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4

EARLY_STOPPING = False
EARLY_STOPPING_PATIENCE = 40
EARLY_STOPPING_MIN_DELTA = 1e-4
EARLY_STOPPING_MONITOR = "val_acc"

LS_EVAL_EVERY = 5
LS_SAMPLE_SIZE = 500
PAIRNORM_SCALE = 1.0
GCNII_ALPHA = 0.1
GCNII_LAMBDA = 0.5


def download_and_extract_cora():
    url = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
    filename = "cora.tgz"
    if os.path.exists("cora"):
        print("Cora dataset already exists.")
        return

    print("Downloading Cora dataset...")
    urllib.request.urlretrieve(url, filename)
    with tarfile.open(filename, "r:gz") as tar_ref:
        tar_ref.extractall()
    os.remove(filename)


def load_data():
    download_and_extract_cora()

    content = []
    with open("cora/cora.content", "r") as f:
        for line in f:
            content.append(line.strip().split())

    node_ids = [int(line[0]) for line in content]
    id_map = {node_id: i for i, node_id in enumerate(node_ids)}
    features = np.array([list(map(float, line[1:-1])) for line in content], dtype=np.float32)
    labels = LabelEncoder().fit_transform([line[-1] for line in content])

    edges = []
    with open("cora/cora.cites", "r") as f:
        for line in f:
            src, dst = map(int, line.strip().split())
            if src in id_map and dst in id_map:
                edges.append((id_map[src], id_map[dst]))

    n_nodes = len(node_ids)
    adj = sparse.lil_matrix((n_nodes, n_nodes), dtype=np.float32)
    for i, j in edges:
        adj[i, j] = 1.0
        adj[j, i] = 1.0

    return features, sparse.coo_matrix(adj), labels


def make_split(labels_np, seed=SEED):
    all_indices = np.arange(labels_np.shape[0], dtype=np.int32)
    idx_train_val, idx_test = train_test_split(
        all_indices,
        test_size=TEST_RATIO,
        random_state=seed,
        stratify=labels_np,
    )
    relative_val_ratio = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    idx_train, idx_val = train_test_split(
        idx_train_val,
        test_size=relative_val_ratio,
        random_state=seed,
        stratify=labels_np[idx_train_val],
    )
    return (
        np.asarray(idx_train, dtype=np.int32),
        np.asarray(idx_val, dtype=np.int32),
        np.asarray(idx_test, dtype=np.int32),
    )


def sparse_tensor_from_scipy(adj_coo):
    indices = np.column_stack([adj_coo.row, adj_coo.col]).astype(np.int64)
    values = adj_coo.data.astype(np.float32)
    shape = np.array(adj_coo.shape, dtype=np.int64)
    adj_tf = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    return tf.sparse.reorder(adj_tf), indices


def make_undirected_edge_groups(indices):
    undirected = np.sort(indices, axis=1)
    _, inverse = np.unique(undirected, axis=0, return_inverse=True)
    return tf.constant(inverse.astype(np.int32)), int(inverse.max() + 1)


def dropedge(adj_sparse, drop_rate, training, edge_group=None, num_edge_groups=None):
    if (not training) or drop_rate <= 0:
        return adj_sparse

    keep_prob = 1.0 - drop_rate
    if edge_group is None:
        mask = tf.random.uniform(tf.shape(adj_sparse.values), dtype=tf.float32) < keep_prob
    else:
        group_keep = tf.random.uniform([num_edge_groups], dtype=tf.float32) < keep_prob
        mask = tf.gather(group_keep, edge_group)

    indices = tf.boolean_mask(adj_sparse.indices, mask)
    values = tf.boolean_mask(adj_sparse.values, mask)
    dropped = tf.SparseTensor(indices=indices, values=values, dense_shape=adj_sparse.dense_shape)
    return tf.sparse.reorder(dropped)


def normalize_adj_with_self_loop(adj_sparse):
    n = tf.cast(adj_sparse.dense_shape[0], tf.int64)
    eye_idx = tf.stack([tf.range(n, dtype=tf.int64), tf.range(n, dtype=tf.int64)], axis=1)
    eye_val = tf.ones([n], dtype=tf.float32)
    eye = tf.SparseTensor(eye_idx, eye_val, adj_sparse.dense_shape)

    adj_hat = tf.sparse.add(adj_sparse, eye)
    adj_hat = tf.sparse.reorder(adj_hat)

    degree = tf.sparse.reduce_sum(adj_hat, axis=1)
    d_inv_sqrt = tf.pow(tf.maximum(degree, 1e-8), -0.5)
    row = tf.cast(adj_hat.indices[:, 0], tf.int32)
    col = tf.cast(adj_hat.indices[:, 1], tf.int32)
    scale = tf.gather(d_inv_sqrt, row) * tf.gather(d_inv_sqrt, col)

    normed = tf.SparseTensor(
        indices=adj_hat.indices,
        values=adj_hat.values * scale,
        dense_shape=adj_hat.dense_shape,
    )
    return tf.sparse.reorder(normed)


def pair_norm(x, scale=PAIRNORM_SCALE, eps=1e-6):
    x = x - tf.reduce_mean(x, axis=0, keepdims=True)
    row_norm_mean = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(x), axis=1) + eps))
    return scale * x / (row_norm_mean + eps)


class DeepGCN(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=NUM_LAYERS, variant="plain"):
        super().__init__()
        self.variant = variant
        self.hidden_layers = [
            tf.keras.layers.Dense(hidden_dim, activation="relu", name=f"gcn_layer_{i + 1}")
            for i in range(num_layers - 1)
        ]
        self.out = tf.keras.layers.Dense(output_dim, name="output_layer")
        self.layer_names = [f"gcn_layer_{i + 1}" for i in range(num_layers - 1)] + ["output_layer"]

    def call(
        self,
        x,
        adj_sparse,
        drop_rate=0.0,
        training=False,
        return_intermediate=False,
        edge_group=None,
        num_edge_groups=None,
    ):
        inter = {}
        adj_used = dropedge(adj_sparse, drop_rate, training, edge_group, num_edge_groups)
        adj_norm = normalize_adj_with_self_loop(adj_used)

        for i, layer in enumerate(self.hidden_layers, start=1):
            x = tf.sparse.sparse_dense_matmul(adj_norm, x)
            x = layer(x)
            if self.variant == "pairnorm":
                x = pair_norm(x)
            inter[f"gcn_layer_{i}"] = x
            if training:
                x = tf.nn.dropout(x, rate=DROPOUT_RATE)

        x = tf.sparse.sparse_dense_matmul(adj_norm, x)
        logits = self.out(x)
        inter["output_layer"] = logits
        return (logits, inter) if return_intermediate else logits


class GCNII(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=NUM_LAYERS, alpha=GCNII_ALPHA, lamda=GCNII_LAMBDA):
        super().__init__()
        self.num_layers = num_layers
        self.alpha = alpha
        self.lamda = lamda
        self.input_proj = tf.keras.layers.Dense(hidden_dim, activation="relu", name="input_projection")
        self.hidden_layers = [
            tf.keras.layers.Dense(hidden_dim, use_bias=False, name=f"gcnii_layer_{i + 1}")
            for i in range(num_layers)
        ]
        self.out = tf.keras.layers.Dense(output_dim, name="output_layer")
        self.layer_names = [f"gcn_layer_{i + 1}" for i in range(num_layers)] + ["output_layer"]

    def call(
        self,
        x,
        adj_sparse,
        drop_rate=0.0,
        training=False,
        return_intermediate=False,
        edge_group=None,
        num_edge_groups=None,
    ):
        inter = {}
        adj_used = dropedge(adj_sparse, drop_rate, training, edge_group, num_edge_groups)
        adj_norm = normalize_adj_with_self_loop(adj_used)

        h0 = self.input_proj(x)
        h = h0
        if training:
            h = tf.nn.dropout(h, rate=DROPOUT_RATE)

        for i, layer in enumerate(self.hidden_layers, start=1):
            beta = np.log(self.lamda / i + 1.0)
            propagated = tf.sparse.sparse_dense_matmul(adj_norm, h)
            support = (1.0 - self.alpha) * propagated + self.alpha * h0
            h = (1.0 - beta) * support + beta * layer(support)
            h = tf.nn.relu(h)
            inter[f"gcn_layer_{i}"] = h
            if training:
                h = tf.nn.dropout(h, rate=DROPOUT_RATE)

        logits = self.out(h)
        inter["output_layer"] = logits
        return (logits, inter) if return_intermediate else logits

def ce_loss(labels_sub, logits_sub):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_sub, logits=logits_sub))


def masked_ce_loss(labels, logits, indices, model=None, weight_decay=WEIGHT_DECAY):
    logits_sub = tf.gather(logits, indices)
    labels_sub = tf.gather(labels, indices)
    loss = ce_loss(labels_sub, logits_sub)
    if model is not None and weight_decay > 0:
        kernels = [tf.nn.l2_loss(v) for v in model.trainable_variables if "kernel" in v.name]
        if kernels:
            loss += weight_decay * tf.add_n(kernels)
    return loss


def evaluate(model, adj_tf, idx):
    logits, inter = model(features, adj_tf, drop_rate=0.0, training=False, return_intermediate=True)
    idx_tf = tf.constant(idx, dtype=tf.int32)
    loss = ce_loss(tf.gather(labels, idx_tf), tf.gather(logits, idx_tf))
    pred = tf.argmax(tf.gather(logits, idx_tf), axis=1).numpy()
    acc = accuracy_score(labels_np[idx], pred)
    return float(loss.numpy()), float(acc), logits, inter


def get_layer_names(num_layers=NUM_LAYERS):
    return [f"gcn_layer_{i + 1}" for i in range(num_layers - 1)] + ["output_layer"]


def init_layer_ls_history(layer_names):
    return {layer_name: {"LS1": [], "LS2": [], "Jw": [], "LDA": []} for layer_name in layer_names}


def build_history(layer_names):
    return {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "ls2_output": [],
        "layer_ls": init_layer_ls_history(layer_names),
    }


def append_layer_ls(history, layer_names, inter, compute=True):
    for layer_name in layer_names:
        metrics = history["layer_ls"][layer_name]
        if compute:
            partial_output = tf.gather(inter[layer_name], partial_indices_tf).numpy()
            ls1, ls2, jw, lda = W(partial_output, partial_labels.reshape(-1, 1))
            values = {"LS1": float(ls1), "LS2": float(ls2), "Jw": float(jw), "LDA": float(lda)}
        else:
            values = {key: (metrics[key][-1] if metrics[key] else np.nan) for key in metrics}
        for key, value in values.items():
            metrics[key].append(value)


def final_layer_ls(inter, layer_names):
    final_ls = {}
    for layer_name in layer_names:
        partial_output = tf.gather(inter[layer_name], partial_indices_tf).numpy()
        ls1, ls2, jw, lda = W(partial_output, partial_labels.reshape(-1, 1))
        final_ls[layer_name] = {"LS1": float(ls1), "LS2": float(ls2), "Jw": float(jw), "LDA": float(lda)}
    return final_ls


def is_better(value, best, monitor):
    if best is None:
        return True
    if monitor.endswith("acc") or monitor.startswith("ls"):
        return value > best + EARLY_STOPPING_MIN_DELTA
    return value < best - EARLY_STOPPING_MIN_DELTA


def run_experiment(name, model_factory, drop_rate=0.0, num_epochs=NUM_EPOCHS):
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    model = model_factory()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    layer_names = getattr(model, "layer_names", get_layer_names(NUM_LAYERS))
    history = build_history(layer_names)

    best_monitor = None
    best_epoch = -1
    best_weights = None
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        with tf.GradientTape() as tape:
            logits = model(
                features,
                adj_tf,
                drop_rate=drop_rate,
                training=True,
                edge_group=edge_group,
                num_edge_groups=num_edge_groups,
            )
            loss = masked_ce_loss(labels, logits, idx_train_tf, model)

        grads = tape.gradient(loss, model.trainable_variables)
        grads_and_vars = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
        optimizer.apply_gradients(grads_and_vars)

        train_loss, train_acc, _, _ = evaluate(model, adj_tf, idx_train)
        val_loss, val_acc, _, inter = evaluate(model, adj_tf, idx_val)
        compute_ls = (epoch % LS_EVAL_EVERY == 0) or (epoch == num_epochs - 1)
        append_layer_ls(history, layer_names, inter, compute=compute_ls)
        ls2_output = history["layer_ls"]["output_layer"]["LS2"][-1]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["ls2_output"].append(float(ls2_output))

        monitor_value = history[EARLY_STOPPING_MONITOR][-1]
        if is_better(monitor_value, best_monitor, EARLY_STOPPING_MONITOR):
            best_monitor = monitor_value
            best_epoch = epoch
            best_weights = model.get_weights()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 20 == 0 or epoch == num_epochs - 1:
            print(
                f"[{name}] epoch={epoch:03d} loss={float(loss.numpy()):.4f} "
                f"train={train_acc:.4f} val={val_acc:.4f} LS2_out={ls2_output:.4f}"
            )

        if EARLY_STOPPING and epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"[{name}] early stopping at epoch={epoch:03d}; best epoch={best_epoch:03d}")
            break

    if best_weights is not None:
        model.set_weights(best_weights)

    _, final_test, _, final_inter = evaluate(model, adj_tf, idx_test)
    return {
        "name": name,
        "model": model,
        "history": history,
        "final_test": final_test,
        "final_ls": final_layer_ls(final_inter, layer_names),
        "best_metrics": {
            "monitor": EARLY_STOPPING_MONITOR,
            "monitor_value": best_monitor,
            "best_epoch": best_epoch,
        },
        "params": {
            "num_layers": NUM_LAYERS,
            "hidden_dim": HIDDEN_DIM,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "dropout_rate": DROPOUT_RATE,
            "drop_rate": drop_rate,
            "ls_eval_every": LS_EVAL_EVERY,
            "ls_sample_size": len(partial_indices),
        },
    }


def plot_results(results):
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    for result in results:
        plt.plot(result["history"]["train_loss"], label=result["name"])
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    for result in results:
        plt.plot(result["history"]["val_acc"], label=result["name"])
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 3, 3)
    for result in results:
        plt.plot(result["history"]["ls2_output"], label=result["name"])
    plt.title("Output-layer LS2")
    plt.xlabel("Epoch")
    plt.ylabel("LS2")
    plt.legend()

    plt.tight_layout()
    plt.close()


def plot_layer_ls2(results):
    for result in results:
        history = result["history"]
        x = np.arange(len(history["train_loss"]))
        plt.figure(figsize=(9, 5))
        for layer_name, metrics in history["layer_ls"].items():
            plt.plot(x, metrics["LS2"], label=layer_name)
        plt.title(f"{result['name']} Layer-wise LS2")
        plt.xlabel("Epoch")
        plt.ylabel("LS2")
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.close()


def print_layer_ls_at_epoch(result, epoch=-1, metric_name="LS2"):
    history = result["history"]
    if epoch < 0:
        epoch = len(history["train_loss"]) + epoch
    print(f"{result['name']} epoch={epoch} {metric_name}")
    for layer_name, metrics in history["layer_ls"].items():
        print(f"  {layer_name}: {metrics[metric_name][epoch]:.4f}")


def print_final_summary(results):
    for result in results:
        print(f"\n{result['name']} final test acc: {result['final_test']:.4f}")
        print(f"  best: {result['best_metrics']}")
        print(f"  params: {result['params']}")
        for layer_name, metric in result["final_ls"].items():
            print(
                f"  {layer_name}: LS1={metric['LS1']:.4f}, "
                f"LS2={metric['LS2']:.4f}, "
                f"Jw={metric['Jw']:.4f}, "
                f"LDA={metric['LDA']:.4f}"
            )


features_np, adj_coo, labels_np = load_data()
num_classes = len(np.unique(labels_np))
features = tf.constant(features_np, dtype=tf.float32)
labels = tf.constant(labels_np, dtype=tf.int32)

idx_train, idx_val, idx_test = make_split(labels_np)
idx_train_tf = tf.constant(idx_train, dtype=tf.int32)
idx_val_tf = tf.constant(idx_val, dtype=tf.int32)
idx_test_tf = tf.constant(idx_test, dtype=tf.int32)

rng = np.random.default_rng(SEED)
ls_sample_size = min(LS_SAMPLE_SIZE, len(idx_train))
partial_indices = rng.choice(idx_train, ls_sample_size, replace=False).astype(np.int32)
partial_indices_tf = tf.constant(partial_indices, dtype=tf.int32)
partial_labels = labels_np[partial_indices]

adj_tf, adj_indices_np = sparse_tensor_from_scipy(adj_coo)
edge_group, num_edge_groups = make_undirected_edge_groups(adj_indices_np)
input_dim = features_np.shape[1]
output_dim = num_classes

experiment_params = {
    "SEED": SEED,
    "TRAIN_RATIO": TRAIN_RATIO,
    "VAL_RATIO": VAL_RATIO,
    "TEST_RATIO": TEST_RATIO,
    "HIDDEN_DIM": HIDDEN_DIM,
    "NUM_LAYERS": NUM_LAYERS,
    "NUM_EPOCHS": NUM_EPOCHS,
    "DROPOUT_RATE": DROPOUT_RATE,
    "DROP_EDGE_RATE": DROP_EDGE_RATE,
    "LEARNING_RATE": LEARNING_RATE,
    "WEIGHT_DECAY": WEIGHT_DECAY,
    "PAIRNORM_SCALE": PAIRNORM_SCALE,
    "GCNII_ALPHA": GCNII_ALPHA,
    "GCNII_LAMBDA": GCNII_LAMBDA,
    "LS_EVAL_EVERY": LS_EVAL_EVERY,
    "LS_SAMPLE_SIZE": len(partial_indices),
}

print(
    f"Nodes: {features_np.shape[0]}, Classes: {num_classes}, Edges: {adj_coo.nnz}, "
    f"Train/Val/Test: {len(idx_train)}/{len(idx_val)}/{len(idx_test)}, "
    f"LS sample: {len(partial_indices)}"
)
print(f"{NUM_LAYERS}-layer baseline comparison params")
for key, value in experiment_params.items():
    print(f"  {key}: {value}")

experiments = [
    (f"Plain-GCN-{NUM_LAYERS}-layer", lambda: DeepGCN(input_dim, HIDDEN_DIM, output_dim, NUM_LAYERS, variant="plain"), 0.0),
    (f"DropEdge-GCN-{NUM_LAYERS}-layer-{DROP_EDGE_RATE}", lambda: DeepGCN(input_dim, HIDDEN_DIM, output_dim, NUM_LAYERS, variant="plain"), DROP_EDGE_RATE),
    (f"PairNorm-GCN-{NUM_LAYERS}-layer", lambda: DeepGCN(input_dim, HIDDEN_DIM, output_dim, NUM_LAYERS, variant="pairnorm"), 0.0),
    (f"GCNII-{NUM_LAYERS}-layer", lambda: GCNII(input_dim, HIDDEN_DIM, output_dim, NUM_LAYERS), 0.0),
]

baseline_comparison_results = []
for name, factory, drop_rate in experiments:
    baseline_comparison_results.append(run_experiment(name, factory, drop_rate=drop_rate))

plot_results(baseline_comparison_results)
plot_layer_ls2(baseline_comparison_results)
print_final_summary(baseline_comparison_results)

print("\nLayer-wise LS2 at final epoch")
for result in baseline_comparison_results:
    print_layer_ls_at_epoch(result, epoch=-1, metric_name="LS2")


# ## 3. Overall comparison and export
# 

# In[ ]:


overall_results = ls_comparison_results + baseline_comparison_results

def print_overall_final_summary(results):
    print('\nOverall comparison models:')
    for result in results:
        history = result['history']
        print(
            f"  {result['name']}: final_test_acc={result['final_test']:.4f}, "
            f"best_val_acc={float(np.nanmax(history['val_acc'])):.4f} "
            f"@ epoch {int(np.nanargmax(history['val_acc']))}"
        )
        if result.get('structure_counts'):
            print(f"    learned structure: {result['structure_counts']}")

plot_results(overall_results)
plot_layer_ls2(overall_results)
print_overall_final_summary(overall_results)

print('\nLayer-wise LS2 at final epoch')
for result in overall_results:
    print_layer_ls_at_epoch(result, epoch=-1, metric_name='LS2')


# In[ ]:


import pickle
from datetime import datetime
from pathlib import Path

def build_overall_paper_metrics(results):
    packed_results = []
    for result in results:
        history = result['history']
        packed_results.append({
            'name': result['name'],
            'final_test_acc': float(result['final_test']),
            'best_metrics': result.get('best_metrics', {}),
            'structure_counts': result.get('structure_counts'),
            'params': result.get('params', {}),
            'final_ls': result.get('final_ls', {}),
            'best_val_acc': float(np.nanmax(history['val_acc'])),
            'best_val_acc_epoch': int(np.nanargmax(history['val_acc'])),
            'history': history,
        })
    return {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'experiment_name': f'overall_comparison_cora_layers{NUM_LAYERS}',
        'models': [item['name'] for item in packed_results],
        'results': packed_results,
    }

save_dir = Path('results_pkls')
save_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
overall_save_path = save_dir / f'overall_comparison_cora_layers{NUM_LAYERS}_{timestamp}.pkl'
with open(overall_save_path, 'wb') as f:
    pickle.dump(build_overall_paper_metrics(overall_results), f, protocol=pickle.HIGHEST_PROTOCOL)
print(f'Saved overall metrics to: {overall_save_path.resolve()}')


# In[ ]:




