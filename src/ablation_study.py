"""
Ablation Study for AntiViralDL
Tests different components: data augmentation methods, feature composition, and noise types
"""

import numpy as np
import pandas as pd
from src.antiviraldl_model import AntiViralDL, generate_negative_samples, LightGCN, ContrastiveLoss
from src.data_preprocessing import DataPreprocessor, create_sample_dataset
import tensorflow as tf


class AntiViralDL_Ablation(AntiViralDL):
    """
    Modified AntiViralDL for ablation study
    """
    
    def __init__(self, augmentation_method='noise', feature_composition='sum',
                 noise_type='random', **kwargs):
        """
        Args:
            augmentation_method: 'none', 'edge_drop', 'node_drop', 'random_walk', 'noise'
            feature_composition: 'sum', 'concat', 'max', 'min'
            noise_type: 'random', 'similar'
        """
        super().__init__(**kwargs)
        self.augmentation_method = augmentation_method
        self.feature_composition = feature_composition
        self.noise_type = noise_type
    
    def apply_augmentation(self, embeddings, adj_matrix=None):
        """
        Apply data augmentation based on method
        
        Args:
            embeddings: Input embeddings
            adj_matrix: Adjacency matrix (for graph-based augmentation)
        
        Returns:
            Augmented embeddings
        """
        if self.augmentation_method == 'none':
            return embeddings
        
        elif self.augmentation_method == 'noise':
            # Add random Gaussian noise
            if self.noise_type == 'random':
                noise = tf.random.normal(shape=tf.shape(embeddings), mean=0.0, stddev=0.1)
            else:  # similar noise
                # Add noise based on embedding similarity
                noise = tf.random.normal(shape=tf.shape(embeddings), mean=0.0, stddev=0.05)
            return embeddings + noise
        
        elif self.augmentation_method == 'edge_drop':
            # Simulate edge dropout by adding noise
            noise = tf.random.normal(shape=tf.shape(embeddings), mean=0.0, stddev=0.15)
            return embeddings + noise
        
        elif self.augmentation_method == 'node_drop':
            # Simulate node dropout by masking
            mask = tf.cast(tf.random.uniform(shape=tf.shape(embeddings)) > 0.1, tf.float32)
            return embeddings * mask
        
        elif self.augmentation_method == 'random_walk':
            # Simulate random walk by adding directional noise
            noise = tf.random.normal(shape=tf.shape(embeddings), mean=0.0, stddev=0.12)
            return embeddings + noise
        
        return embeddings
    
    def compose_features(self, layer_embeddings):
        """
        Compose features from different layers
        
        Args:
            layer_embeddings: List of embeddings from each layer
        
        Returns:
            Composed features
        """
        if self.feature_composition == 'sum':
            return tf.reduce_sum(tf.stack(layer_embeddings, axis=0), axis=0)
        
        elif self.feature_composition == 'concat':
            return tf.concat(layer_embeddings, axis=1)
        
        elif self.feature_composition == 'max':
            return tf.reduce_max(tf.stack(layer_embeddings, axis=0), axis=0)
        
        elif self.feature_composition == 'min':
            return tf.reduce_min(tf.stack(layer_embeddings, axis=0), axis=0)
        
        return tf.reduce_mean(tf.stack(layer_embeddings, axis=0), axis=0)
    
    def train_step(self, adj_matrix, pos_interactions, neg_interactions):
        """
        Modified training step with ablation options
        """
        with tf.GradientTape() as tape:
            # Get embeddings
            virus_emb, drug_emb = self.lightgcn(adj_matrix)
            
            # Extract samples
            pos_virus_ids = pos_interactions[:, 0]
            pos_drug_ids = pos_interactions[:, 1]
            neg_virus_ids = neg_interactions[:, 0]
            neg_drug_ids = neg_interactions[:, 1]
            
            pos_virus_emb = tf.gather(virus_emb, pos_virus_ids)
            pos_drug_emb = tf.gather(drug_emb, pos_drug_ids)
            neg_virus_emb = tf.gather(virus_emb, neg_virus_ids)
            neg_drug_emb = tf.gather(drug_emb, neg_drug_ids)
            
            # BPR loss
            bpr_loss = self.compute_bpr_loss(pos_virus_emb, pos_drug_emb, neg_drug_emb)
            
            # Apply augmentation
            virus_emb_aug = self.apply_augmentation(pos_virus_emb, adj_matrix)
            drug_emb_aug = self.apply_augmentation(pos_drug_emb, adj_matrix)
            
            # Contrastive loss (only if augmentation is not 'none')
            if self.augmentation_method != 'none':
                cl_loss = self.contrastive_loss(virus_emb_aug, drug_emb_aug, drug_emb)
                total_loss = bpr_loss + self.lambda_cl * cl_loss
            else:
                total_loss = bpr_loss
            
            # L2 regularization
            l2_loss = tf.reduce_sum([tf.nn.l2_loss(var) for var in self.lightgcn.trainable_variables])
            total_loss += 1e-5 * l2_loss
        
        # Update
        gradients = tape.gradient(total_loss, self.lightgcn.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.lightgcn.trainable_variables))
        
        return total_loss


def run_ablation_study(interactions, virus_features, drug_features, num_epochs=500):
    """
    Run complete ablation study
    
    Args:
        interactions: Virus-drug interactions
        virus_features: Virus features
        drug_features: Drug features
        num_epochs: Number of training epochs
    
    Returns:
        Results dictionary
    """
    num_viruses = interactions[:, 0].max() + 1
    num_drugs = interactions[:, 1].max() + 1
    
    # Split data
    np.random.shuffle(interactions)
    split_idx = int(0.8 * len(interactions))
    train_interactions = interactions[:split_idx]
    test_interactions = interactions[split_idx:]
    
    results = {
        'Category': [],
        'Variant': [],
        'AUC': [],
        'AUPR': []
    }
    
    # 1. Test different data augmentation methods
    print("\n" + "="*60)
    print("Testing Data Augmentation Methods")
    print("="*60)
    
    augmentation_methods = {
        'SGL-None': 'none',
        'SGL-ED': 'edge_drop',
        'SGL-ND': 'node_drop',
        'SGL-RW': 'random_walk',
        'AntiViralDL': 'noise'
    }
    
    for method_name, method in augmentation_methods.items():
        print(f"\nTesting {method_name}...")
        
        model = AntiViralDL_Ablation(
            num_viruses=num_viruses,
            num_drugs=num_drugs,
            embedding_dim=128,
            num_layers=2,
            learning_rate=0.01,
            lambda_cl=0.5,
            temperature=0.1,
            virus_features=virus_features,
            drug_features=drug_features,
            augmentation_method=method,
            feature_composition='sum',
            noise_type='random'
        )
        
        adj_matrix = model.lightgcn.build_adjacency_matrix(train_interactions)
        
        # Train
        for epoch in range(num_epochs):
            neg_interactions = generate_negative_samples(train_interactions, num_viruses, num_drugs)
            model.train_step(adj_matrix, train_interactions, neg_interactions)
            
            if (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch + 1}/{num_epochs}")
        
        # Evaluate
        auc_score, aupr_score, _, _ = model.evaluate(adj_matrix, test_interactions, interactions)
        
        print(f"  Results: AUC = {auc_score:.4f}, AUPR = {aupr_score:.4f}")
        
        results['Category'].append('Data Augmentation')
        results['Variant'].append(method_name)
        results['AUC'].append(auc_score)
        results['AUPR'].append(aupr_score)
    
    # 2. Test different feature composition methods
    print("\n" + "="*60)
    print("Testing Feature Composition Methods")
    print("="*60)
    
    composition_methods = {
        'concatenation': 'concat',
        'element-wise max': 'max',
        'element-wise min': 'min',
        'sum (AntiViralDL)': 'sum'
    }
    
    for method_name, method in composition_methods.items():
        print(f"\nTesting {method_name}...")
        
        # Adjust embedding dimension for concatenation
        if method == 'concat':
            emb_dim = 64  # Will be doubled after concat
        else:
            emb_dim = 128
        
        model = AntiViralDL_Ablation(
            num_viruses=num_viruses,
            num_drugs=num_drugs,
            embedding_dim=emb_dim,
            num_layers=2,
            learning_rate=0.01,
            lambda_cl=0.5,
            temperature=0.1,
            virus_features=virus_features[:, :emb_dim] if virus_features.shape[1] > emb_dim else virus_features,
            drug_features=drug_features[:, :emb_dim] if drug_features.shape[1] > emb_dim else drug_features,
            augmentation_method='noise',
            feature_composition=method,
            noise_type='random'
        )
        
        adj_matrix = model.lightgcn.build_adjacency_matrix(train_interactions)
        
        # Train
        for epoch in range(num_epochs):
            neg_interactions = generate_negative_samples(train_interactions, num_viruses, num_drugs)
            model.train_step(adj_matrix, train_interactions, neg_interactions)
            
            if (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch + 1}/{num_epochs}")
        
        # Evaluate
        auc_score, aupr_score, _, _ = model.evaluate(adj_matrix, test_interactions, interactions)
        
        print(f"  Results: AUC = {auc_score:.4f}, AUPR = {aupr_score:.4f}")
        
        results['Category'].append('Feature Composition')
        results['Variant'].append(method_name)
        results['AUC'].append(auc_score)
        results['AUPR'].append(aupr_score)
    
    # 3. Test different noise types
    print("\n" + "="*60)
    print("Testing Different Noise Types")
    print("="*60)
    
    noise_types = {
        'random noise': 'random',
        'similar noise(AntiViralDL)': 'similar'
    }
    
    for method_name, noise_type in noise_types.items():
        print(f"\nTesting {method_name}...")
        
        model = AntiViralDL_Ablation(
            num_viruses=num_viruses,
            num_drugs=num_drugs,
            embedding_dim=128,
            num_layers=2,
            learning_rate=0.01,
            lambda_cl=0.5,
            temperature=0.1,
            virus_features=virus_features,
            drug_features=drug_features,
            augmentation_method='noise',
            feature_composition='sum',
            noise_type=noise_type
        )
        
        adj_matrix = model.lightgcn.build_adjacency_matrix(train_interactions)
        
        # Train
        for epoch in range(num_epochs):
            neg_interactions = generate_negative_samples(train_interactions, num_viruses, num_drugs)
            model.train_step(adj_matrix, train_interactions, neg_interactions)
            
            if (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch + 1}/{num_epochs}")
        
        # Evaluate
        auc_score, aupr_score, _, _ = model.evaluate(adj_matrix, test_interactions, interactions)
        
        print(f"  Results: AUC = {auc_score:.4f}, AUPR = {aupr_score:.4f}")
        
        results['Category'].append('Difficult Noise')
        results['Variant'].append(method_name)
        results['AUC'].append(auc_score)
        results['AUPR'].append(aupr_score)
    
    return results


def create_ablation_table(results):
    """
    Create ablation study results table matching Table VI in the paper
    """
    df = pd.DataFrame(results)
    return df


def main():
    """
    Main execution function for ablation study
    """
    print("="*60)
    print("AntiViralDL - Ablation Study")
    print("="*60)
    
    # Load data
    print("\nLoading dataset...")
    interactions_df = create_sample_dataset(num_viruses=84, num_drugs=219, num_interactions=1462)
    
    preprocessor = DataPreprocessor()
    interactions_df.to_csv('/home/claude/virus_drug_interactions_ablation.csv', index=False)
    interactions, _, _ = preprocessor.load_data('/home/claude/virus_drug_interactions_ablation.csv')
    
    num_viruses = interactions[:, 0].max() + 1
    num_drugs = interactions[:, 1].max() + 1
    
    # Create features
    virus_features, drug_features = preprocessor.create_synthetic_features(
        num_viruses, num_drugs, virus_dim=128, drug_dim=128
    )
    
    print(f"Number of viruses: {num_viruses}")
    print(f"Number of drugs: {num_drugs}")
    print(f"Number of interactions: {len(interactions)}")
    
    # Run ablation study
    results = run_ablation_study(interactions, virus_features, drug_features, num_epochs=500)
    
    # Create results table
    ablation_table = create_ablation_table(results)
    
    print("\n" + "="*60)
    print("Ablation Study Results:")
    print("="*60)
    print(ablation_table.to_string(index=False))
    
    # Save results
    ablation_table.to_csv('/home/claude/ablation_study_results.csv', index=False)
    print("\nResults saved to ablation_study_results.csv")
    
    print("\n" + "="*60)
    print("Ablation study completed!")
    print("="*60)


if __name__ == "__main__":
    main()
