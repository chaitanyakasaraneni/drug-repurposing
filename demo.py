"""
Quick Demo of AntiViralDL
Runs a small-scale example to verify the implementation
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from src.antiviraldl_model import AntiViralDL, generate_negative_samples
from src.data_preprocessing import DataPreprocessor, create_sample_dataset

print("="*60)
print("AntiViralDL - Quick Demo")
print("="*60)
print("\nThis demo runs a small-scale experiment to verify the implementation.")
print("For full results, run train_antiviraldl.py")
print("="*60)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create small sample dataset
print("\n1. Creating sample dataset...")
num_viruses = 20
num_drugs = 30
num_interactions = 100

interactions_df = create_sample_dataset(
    num_viruses=num_viruses,
    num_drugs=num_drugs,
    num_interactions=num_interactions
)

print(f"   - Viruses: {num_viruses}")
print(f"   - Drugs: {num_drugs}")
print(f"   - Interactions: {num_interactions}")

# Process data
print("\n2. Processing data...")
preprocessor = DataPreprocessor()
interactions_df.to_csv('/home/claude/demo_interactions.csv', index=False)
interactions, _, _ = preprocessor.load_data('/home/claude/demo_interactions.csv')

# Create synthetic features
virus_features, drug_features = preprocessor.create_synthetic_features(
    num_viruses, num_drugs, virus_dim=16, drug_dim=18
)

print(f"   - Virus features: {virus_features.shape}")
print(f"   - Drug features: {drug_features.shape}")

# Split data
print("\n3. Splitting data (80/20 train/test)...")
np.random.shuffle(interactions)
split_idx = int(0.8 * len(interactions))
train_interactions = interactions[:split_idx]
test_interactions = interactions[split_idx:]

print(f"   - Training samples: {len(train_interactions)}")
print(f"   - Test samples: {len(test_interactions)}")

# Initialize model
print("\n4. Initializing AntiViralDL model...")
model = AntiViralDL(
    num_viruses=num_viruses,
    num_drugs=num_drugs,
    embedding_dim=64,  # Smaller for demo
    num_layers=2,
    learning_rate=0.01,
    lambda_cl=0.5,
    temperature=0.1,
    virus_features=virus_features,
    drug_features=drug_features
)

print("   Model initialized successfully!")
print(f"   - Embedding dimension: 64")
print(f"   - GCN layers: 2")
print(f"   - Lambda: 0.5")

# Build adjacency matrix
print("\n5. Building bipartite graph...")
adj_matrix = model.lightgcn.build_adjacency_matrix(train_interactions)
print("   Adjacency matrix built!")

# Training
print("\n6. Training model (100 epochs for demo)...")
num_epochs = 100

for epoch in range(num_epochs):
    # Generate negative samples
    neg_interactions = generate_negative_samples(
        train_interactions, num_viruses, num_drugs
    )
    
    # Training step
    loss = model.train_step(adj_matrix, train_interactions, neg_interactions)
    
    if (epoch + 1) % 20 == 0:
        print(f"   Epoch {epoch + 1}/{num_epochs}, Loss: {loss.numpy():.4f}")

print("\n   Training completed!")

# Evaluation
print("\n7. Evaluating on test set...")
auc_score, aupr_score, y_true, y_scores = model.evaluate(
    adj_matrix, test_interactions, interactions
)

print(f"\n{'='*60}")
print("Demo Results:")
print(f"{'='*60}")
print(f"AUC:  {auc_score:.4f}")
print(f"AUPR: {aupr_score:.4f}")
print(f"{'='*60}")

# Make predictions
print("\n8. Making sample predictions...")
print("\nTop 5 virus-drug predictions:")
print(f"{'Virus':<10} {'Drug':<10} {'Score':<10}")
print("-" * 30)

virus_emb, drug_emb = model.lightgcn(adj_matrix)
predictions = []

for virus_id in range(min(5, num_viruses)):
    for drug_id in range(num_drugs):
        if (virus_id, drug_id) not in set(map(tuple, train_interactions)):
            virus_emb_i = virus_emb[virus_id:virus_id+1]
            drug_emb_j = drug_emb[drug_id:drug_id+1]
            score = tf.reduce_sum(virus_emb_i * drug_emb_j).numpy()
            score = 1 / (1 + np.exp(-score))  # Sigmoid
            predictions.append((virus_id, drug_id, score))

# Sort and show top 5
predictions.sort(key=lambda x: x[2], reverse=True)
for i, (v_id, d_id, score) in enumerate(predictions[:5], 1):
    print(f"Virus_{v_id:<4} Drug_{d_id:<5} {score:.4f}")

print("\n" + "="*60)
print("Demo completed successfully!")
print("="*60)
print("\nNext steps:")
print("1. Run 'python train_antiviraldl.py' for full 5-fold CV")
print("2. Run 'python parameter_sensitivity.py' for parameter analysis")
print("3. Run 'python ablation_study.py' for ablation experiments")
print("4. Run 'python drug_repurposing.py' for COVID-19 predictions")
print("="*60)
