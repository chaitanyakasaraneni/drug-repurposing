"""
AntiViralDL: A Graph Neural Network and Self-Supervised Learning Approach
for Virus-Drug Association Prediction

Based on the IEEE CIACON 2025 Conference Paper
Author: Chaitanya Krishna Kasaraneni
"""

import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import matplotlib.pyplot as plt


class LightGCN(tf.keras.Model):
    """
    Light Graph Convolutional Network for embedding generation
    Simplified GCN without feature transformation and nonlinear activation
    """
    
    def __init__(self, num_viruses, num_drugs, embedding_dim, num_layers, 
                 virus_features=None, drug_features=None):
        """
        Args:
            num_viruses: Number of virus nodes
            num_drugs: Number of drug nodes
            embedding_dim: Dimension of embeddings
            num_layers: Number of GCN layers
            virus_features: Initial virus features (optional)
            drug_features: Initial drug features (optional)
        """
        super(LightGCN, self).__init__()
        
        self.num_viruses = num_viruses
        self.num_drugs = num_drugs
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # Initialize embeddings
        if virus_features is not None:
            # Use provided features
            self.virus_embedding = tf.Variable(
                tf.constant(virus_features, dtype=tf.float32),
                trainable=True,
                name='virus_embedding'
            )
        else:
            # Random initialization
            self.virus_embedding = tf.Variable(
                tf.random.normal([num_viruses, embedding_dim], stddev=0.1),
                trainable=True,
                name='virus_embedding'
            )
        
        if drug_features is not None:
            self.drug_embedding = tf.Variable(
                tf.constant(drug_features, dtype=tf.float32),
                trainable=True,
                name='drug_embedding'
            )
        else:
            self.drug_embedding = tf.Variable(
                tf.random.normal([num_drugs, embedding_dim], stddev=0.1),
                trainable=True,
                name='drug_embedding'
            )
    
    def build_adjacency_matrix(self, interactions):
        """
        Build normalized adjacency matrix for bipartite graph
        
        Args:
            interactions: List of (virus_id, drug_id) tuples
        
        Returns:
            Normalized adjacency matrix
        """
        num_nodes = self.num_viruses + self.num_drugs
        adj_matrix = np.zeros((num_nodes, num_nodes))
        
        # Build bipartite graph
        for virus_id, drug_id in interactions:
            # Virus to Drug edge
            adj_matrix[virus_id, self.num_viruses + drug_id] = 1
            # Drug to Virus edge (symmetric)
            adj_matrix[self.num_viruses + drug_id, virus_id] = 1
        
        # Normalize adjacency matrix: D^(-1/2) * A * D^(-1/2)
        degree = np.sum(adj_matrix, axis=1)
        degree_inv_sqrt = np.power(degree, -0.5)
        degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.
        degree_matrix = np.diag(degree_inv_sqrt)
        
        normalized_adj = degree_matrix @ adj_matrix @ degree_matrix
        
        return tf.sparse.from_dense(normalized_adj)
    
    def propagate(self, adj_matrix):
        """
        Multi-hop graph propagation using LightGCN
        
        Args:
            adj_matrix: Normalized adjacency matrix
        
        Returns:
            Final virus and drug embeddings
        """
        # Concatenate virus and drug embeddings
        all_embeddings = tf.concat([self.virus_embedding, self.drug_embedding], axis=0)
        
        embeddings_list = [all_embeddings]
        
        # Multi-layer propagation
        for layer in range(self.num_layers):
            all_embeddings = tf.sparse.sparse_dense_matmul(adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        
        # Average embeddings across all layers (including layer 0)
        final_embeddings = tf.reduce_mean(tf.stack(embeddings_list, axis=0), axis=0)
        
        # Split back into virus and drug embeddings
        virus_final = final_embeddings[:self.num_viruses, :]
        drug_final = final_embeddings[self.num_viruses:, :]
        
        return virus_final, drug_final
    
    def call(self, adj_matrix):
        """Forward pass"""
        return self.propagate(adj_matrix)


class ContrastiveLoss(tf.keras.layers.Layer):
    """
    Contrastive learning loss with temperature parameter
    """
    
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
    
    def compute_similarity(self, virus_emb, drug_emb):
        """
        Compute cosine similarity between virus and drug embeddings
        
        Args:
            virus_emb: Virus embeddings [batch_size, embedding_dim]
            drug_emb: Drug embeddings [batch_size, embedding_dim]
        
        Returns:
            Cosine similarity scores
        """
        # Normalize embeddings
        virus_emb_norm = tf.nn.l2_normalize(virus_emb, axis=1)
        drug_emb_norm = tf.nn.l2_normalize(drug_emb, axis=1)
        
        # Compute cosine similarity
        similarity = tf.reduce_sum(virus_emb_norm * drug_emb_norm, axis=1)
        
        return similarity
    
    def call(self, virus_emb, pos_drug_emb, all_drug_emb):
        """
        Compute contrastive loss
        
        Args:
            virus_emb: Virus embeddings [batch_size, embedding_dim]
            pos_drug_emb: Positive drug embeddings [batch_size, embedding_dim]
            all_drug_emb: All drug embeddings [num_drugs, embedding_dim]
        
        Returns:
            Contrastive loss value
        """
        # Positive similarity
        pos_sim = self.compute_similarity(virus_emb, pos_drug_emb) / self.temperature
        
        # Negative similarities (all drugs)
        virus_emb_norm = tf.nn.l2_normalize(virus_emb, axis=1)
        all_drug_emb_norm = tf.nn.l2_normalize(all_drug_emb, axis=1)
        
        neg_sim = tf.matmul(virus_emb_norm, all_drug_emb_norm, transpose_b=True) / self.temperature
        
        # Compute contrastive loss
        # LCL = -log(exp(pos_sim) / sum(exp(neg_sim)))
        pos_exp = tf.exp(pos_sim)
        neg_exp_sum = tf.reduce_sum(tf.exp(neg_sim), axis=1)
        
        loss = -tf.reduce_mean(tf.math.log(pos_exp / (neg_exp_sum + 1e-8)))
        
        return loss


class AntiViralDL:
    """
    Main AntiViralDL model combining LightGCN and Contrastive Learning
    """
    
    def __init__(self, num_viruses, num_drugs, embedding_dim=128, num_layers=2,
                 learning_rate=0.01, lambda_cl=0.5, temperature=0.1,
                 virus_features=None, drug_features=None):
        """
        Args:
            num_viruses: Number of virus nodes
            num_drugs: Number of drug nodes
            embedding_dim: Dimension of embeddings (default: 128)
            num_layers: Number of GCN layers (default: 2)
            learning_rate: Learning rate (default: 0.01)
            lambda_cl: Weight for contrastive loss (default: 0.5)
            temperature: Temperature parameter for contrastive loss (default: 0.1)
            virus_features: Initial virus features
            drug_features: Initial drug features
        """
        self.num_viruses = num_viruses
        self.num_drugs = num_drugs
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.lambda_cl = lambda_cl
        self.temperature = temperature
        
        # Initialize LightGCN model
        self.lightgcn = LightGCN(
            num_viruses, num_drugs, embedding_dim, num_layers,
            virus_features, drug_features
        )
        
        # Initialize contrastive loss
        self.contrastive_loss = ContrastiveLoss(temperature)
        
        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        
        # Metrics storage
        self.train_losses = []
        self.val_aucs = []
        self.val_auprs = []
    
    def add_noise_to_embeddings(self, embeddings, noise_std=0.1):
        """
        Add random Gaussian noise to embeddings for data augmentation
        
        Args:
            embeddings: Input embeddings
            noise_std: Standard deviation of Gaussian noise
        
        Returns:
            Noisy embeddings
        """
        noise = tf.random.normal(shape=tf.shape(embeddings), mean=0.0, stddev=noise_std)
        return embeddings + noise
    
    def compute_bpr_loss(self, virus_emb, pos_drug_emb, neg_drug_emb):
        """
        Compute Bayesian Personalized Ranking (BPR) loss
        
        Args:
            virus_emb: Virus embeddings
            pos_drug_emb: Positive drug embeddings
            neg_drug_emb: Negative drug embeddings
        
        Returns:
            BPR loss
        """
        pos_scores = tf.reduce_sum(virus_emb * pos_drug_emb, axis=1)
        neg_scores = tf.reduce_sum(virus_emb * neg_drug_emb, axis=1)
        
        bpr_loss = -tf.reduce_mean(tf.nn.log_sigmoid(pos_scores - neg_scores))
        
        return bpr_loss
    
    def train_step(self, adj_matrix, pos_interactions, neg_interactions):
        """
        Single training step
        
        Args:
            adj_matrix: Normalized adjacency matrix
            pos_interactions: Positive (virus, drug) pairs
            neg_interactions: Negative (virus, drug) pairs
        
        Returns:
            Total loss value
        """
        with tf.GradientTape() as tape:
            # Get embeddings from LightGCN
            virus_emb, drug_emb = self.lightgcn(adj_matrix)
            
            # Extract positive and negative samples
            pos_virus_ids = pos_interactions[:, 0]
            pos_drug_ids = pos_interactions[:, 1]
            neg_virus_ids = neg_interactions[:, 0]
            neg_drug_ids = neg_interactions[:, 1]
            
            pos_virus_emb = tf.gather(virus_emb, pos_virus_ids)
            pos_drug_emb = tf.gather(drug_emb, pos_drug_ids)
            neg_virus_emb = tf.gather(virus_emb, neg_virus_ids)
            neg_drug_emb = tf.gather(drug_emb, neg_drug_ids)
            
            # Compute BPR loss
            bpr_loss = self.compute_bpr_loss(pos_virus_emb, pos_drug_emb, neg_drug_emb)
            
            # Data augmentation: Add noise to embeddings
            virus_emb_aug = self.add_noise_to_embeddings(pos_virus_emb)
            drug_emb_aug = self.add_noise_to_embeddings(pos_drug_emb)
            
            # Compute contrastive loss
            cl_loss = self.contrastive_loss(virus_emb_aug, drug_emb_aug, drug_emb)
            
            # Total loss
            total_loss = bpr_loss + self.lambda_cl * cl_loss
            
            # L2 regularization
            l2_loss = tf.reduce_sum([tf.nn.l2_loss(var) for var in self.lightgcn.trainable_variables])
            total_loss += 1e-5 * l2_loss
        
        # Compute gradients and update
        gradients = tape.gradient(total_loss, self.lightgcn.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.lightgcn.trainable_variables))
        
        return total_loss
    
    def predict(self, adj_matrix, virus_ids, drug_ids):
        """
        Predict association scores for virus-drug pairs
        
        Args:
            adj_matrix: Normalized adjacency matrix
            virus_ids: Virus node IDs
            drug_ids: Drug node IDs
        
        Returns:
            Prediction scores
        """
        virus_emb, drug_emb = self.lightgcn(adj_matrix)
        
        virus_emb_selected = tf.gather(virus_emb, virus_ids)
        drug_emb_selected = tf.gather(drug_emb, drug_ids)
        
        # Inner product for prediction
        scores = tf.reduce_sum(virus_emb_selected * drug_emb_selected, axis=1)
        
        return tf.nn.sigmoid(scores).numpy()
    
    def evaluate(self, adj_matrix, test_interactions, all_interactions):
        """
        Evaluate model performance using AUC and AUPR
        
        Args:
            adj_matrix: Normalized adjacency matrix
            test_interactions: Test set interactions
            all_interactions: All known interactions (for negative sampling)
        
        Returns:
            AUC and AUPR scores
        """
        virus_emb, drug_emb = self.lightgcn(adj_matrix)
        
        # Create set of known interactions for efficient lookup
        known_interactions = set(map(tuple, all_interactions))
        
        y_true = []
        y_scores = []
        
        # Positive samples
        for virus_id, drug_id in test_interactions:
            virus_emb_i = virus_emb[virus_id:virus_id+1]
            drug_emb_j = drug_emb[drug_id:drug_id+1]
            
            score = tf.reduce_sum(virus_emb_i * drug_emb_j).numpy()
            y_true.append(1)
            y_scores.append(score)
        
        # Negative samples (same number as positive)
        num_neg_samples = len(test_interactions)
        neg_count = 0
        
        while neg_count < num_neg_samples:
            virus_id = np.random.randint(0, self.num_viruses)
            drug_id = np.random.randint(0, self.num_drugs)
            
            if (virus_id, drug_id) not in known_interactions:
                virus_emb_i = virus_emb[virus_id:virus_id+1]
                drug_emb_j = drug_emb[drug_id:drug_id+1]
                
                score = tf.reduce_sum(virus_emb_i * drug_emb_j).numpy()
                y_true.append(0)
                y_scores.append(score)
                neg_count += 1
        
        # Compute metrics
        auc = roc_auc_score(y_true, y_scores)
        aupr = average_precision_score(y_true, y_scores)
        
        return auc, aupr, y_true, y_scores


def generate_negative_samples(pos_interactions, num_viruses, num_drugs, num_neg_per_pos=1):
    """
    Generate negative samples for training
    
    Args:
        pos_interactions: Positive interactions
        num_viruses: Total number of viruses
        num_drugs: Total number of drugs
        num_neg_per_pos: Number of negative samples per positive sample
    
    Returns:
        Negative interactions
    """
    pos_set = set(map(tuple, pos_interactions))
    neg_interactions = []
    
    for virus_id, drug_id in pos_interactions:
        neg_count = 0
        while neg_count < num_neg_per_pos:
            neg_drug_id = np.random.randint(0, num_drugs)
            if (virus_id, neg_drug_id) not in pos_set:
                neg_interactions.append([virus_id, neg_drug_id])
                neg_count += 1
    
    return np.array(neg_interactions)
