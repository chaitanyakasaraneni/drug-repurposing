"""
Main training script for AntiViralDL
Implements 5-fold cross-validation as described in the paper
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

from src.antiviraldl_model import AntiViralDL, generate_negative_samples
from src.data_preprocessing import (
    DataPreprocessor, CrossValidationSplitter,
    create_sample_dataset, create_sample_features
)


def train_antiviraldl(interactions, virus_features=None, drug_features=None,
                      embedding_dim=128, num_layers=2, learning_rate=0.01,
                      lambda_cl=0.5, temperature=0.1, num_epochs=1000,
                      batch_size=1024, verbose=True):
    """
    Train AntiViralDL model with 5-fold cross-validation
    
    Args:
        interactions: Virus-drug interaction pairs
        virus_features: Virus feature matrix
        drug_features: Drug feature matrix
        embedding_dim: Embedding dimension (default: 128)
        num_layers: Number of GCN layers (default: 2)
        learning_rate: Learning rate (default: 0.01)
        lambda_cl: Contrastive learning weight (default: 0.5)
        temperature: Temperature parameter (default: 0.1)
        num_epochs: Number of training epochs (default: 1000)
        batch_size: Batch size for training (default: 1024)
        verbose: Print progress (default: True)
    
    Returns:
        results: Dictionary containing cross-validation results
    """
    # Get number of viruses and drugs
    num_viruses = interactions[:, 0].max() + 1
    num_drugs = interactions[:, 1].max() + 1
    
    # Initialize cross-validation
    cv_splitter = CrossValidationSplitter(n_splits=5, shuffle=True, random_state=42)
    
    # Storage for results
    fold_results = {
        'auc': [],
        'aupr': [],
        'roc_curves': [],
        'fold_models': []
    }
    
    # 5-fold cross-validation
    for fold_idx, (train_interactions, test_interactions) in enumerate(cv_splitter.split(interactions)):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Fold {fold_idx + 1}/5")
            print(f"{'='*60}")
            print(f"Train samples: {len(train_interactions)}")
            print(f"Test samples: {len(test_interactions)}")
        
        # Initialize model
        model = AntiViralDL(
            num_viruses=num_viruses,
            num_drugs=num_drugs,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            learning_rate=learning_rate,
            lambda_cl=lambda_cl,
            temperature=temperature,
            virus_features=virus_features,
            drug_features=drug_features
        )
        
        # Build adjacency matrix from training data
        adj_matrix = model.lightgcn.build_adjacency_matrix(train_interactions)
        
        # Training loop
        for epoch in range(num_epochs):
            # Shuffle training data
            np.random.shuffle(train_interactions)
            
            # Generate negative samples
            neg_interactions = generate_negative_samples(
                train_interactions, num_viruses, num_drugs, num_neg_per_pos=1
            )
            
            # Mini-batch training
            num_batches = len(train_interactions) // batch_size + 1
            epoch_loss = 0.0
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(train_interactions))
                
                if start_idx >= len(train_interactions):
                    break
                
                batch_pos = train_interactions[start_idx:end_idx]
                batch_neg = neg_interactions[start_idx:end_idx]
                
                # Training step
                loss = model.train_step(adj_matrix, batch_pos, batch_neg)
                epoch_loss += loss.numpy()
            
            # Print progress
            if verbose and (epoch + 1) % 100 == 0:
                avg_loss = epoch_loss / num_batches
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        # Evaluate on test set
        auc_score, aupr_score, y_true, y_scores = model.evaluate(
            adj_matrix, test_interactions, interactions
        )
        
        if verbose:
            print(f"\nFold {fold_idx + 1} Results:")
            print(f"AUC: {auc_score:.4f}")
            print(f"AUPR: {aupr_score:.4f}")
        
        # Store results
        fold_results['auc'].append(auc_score)
        fold_results['aupr'].append(aupr_score)
        
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        fold_results['roc_curves'].append((fpr, tpr))
        fold_results['fold_models'].append(model)
    
    # Compute average results
    avg_auc = np.mean(fold_results['auc'])
    avg_aupr = np.mean(fold_results['aupr'])
    
    if verbose:
        print(f"\n{'='*60}")
        print("5-Fold Cross-Validation Results:")
        print(f"{'='*60}")
        for i, (auc_val, aupr_val) in enumerate(zip(fold_results['auc'], fold_results['aupr'])):
            print(f"Fold {i+1}: AUC = {auc_val:.4f}, AUPR = {aupr_val:.4f}")
        print(f"{'='*60}")
        print(f"Average AUC: {avg_auc:.4f}")
        print(f"Average AUPR: {avg_aupr:.4f}")
        print(f"{'='*60}")
    
    results = {
        'fold_auc': fold_results['auc'],
        'fold_aupr': fold_results['aupr'],
        'avg_auc': avg_auc,
        'avg_aupr': avg_aupr,
        'roc_curves': fold_results['roc_curves'],
        'models': fold_results['fold_models']
    }
    
    return results


def plot_roc_curves(results, save_path='roc_curves.png'):
    """
    Plot ROC curves for all folds
    
    Args:
        results: Results dictionary from train_antiviraldl
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    
    for i, (fpr, tpr) in enumerate(results['roc_curves']):
        auc_score = results['fold_auc'][i]
        plt.plot(fpr, tpr, color=colors[i], lw=2, 
                 label=f'ROC fold {i+1} (AUC = {auc_score:.4f})')
    
    # Plot mean ROC curve
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(mean_fpr)
    
    for fpr, tpr in results['roc_curves']:
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
    
    mean_tpr /= len(results['roc_curves'])
    mean_auc = results['avg_auc']
    
    plt.plot(mean_fpr, mean_tpr, color='black', lw=3, linestyle='--',
             label=f'Mean ROC (AUC = {mean_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Five-fold cross-validation ROC curves of AntiViralDL', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curves saved to {save_path}")
    
    plt.close()


def create_results_table(results):
    """
    Create results table matching Table III in the paper
    
    Args:
        results: Results dictionary from train_antiviraldl
    
    Returns:
        DataFrame with results
    """
    data = {
        'Validation set': [i+1 for i in range(5)] + ['Average'],
        'AUC': results['fold_auc'] + [results['avg_auc']],
        'AUPR': results['fold_aupr'] + [results['avg_aupr']]
    }
    
    df = pd.DataFrame(data)
    
    return df


def main():
    """
    Main execution function
    """
    print("AntiViralDL: AI-Driven Drug Repurposing")
    print("A Graph Neural Network and Self-Supervised Learning Approach")
    print("="*60)
    
    # Load or create sample data
    print("\n1. Loading dataset...")
    
    # Create sample dataset matching paper statistics
    # In practice, load from actual DrugVirus2 and FDA datasets
    interactions_df = create_sample_dataset(num_viruses=84, num_drugs=219, num_interactions=1462)
    
    # Process data
    preprocessor = DataPreprocessor()
    
    # Save sample data
    interactions_df.to_csv('/home/claude/virus_drug_interactions.csv', index=False)
    
    # Load interactions
    interactions, _, _ = preprocessor.load_data('/home/claude/virus_drug_interactions.csv')
    
    num_viruses = interactions[:, 0].max() + 1
    num_drugs = interactions[:, 1].max() + 1
    
    print(f"Number of viruses: {num_viruses}")
    print(f"Number of drugs: {num_drugs}")
    print(f"Number of interactions: {len(interactions)}")
    
    # Create synthetic features (16 virus features, 18 drug features as per paper)
    virus_features, drug_features = preprocessor.create_synthetic_features(
        num_viruses, num_drugs, virus_dim=16, drug_dim=18
    )
    
    print(f"Virus features shape: {virus_features.shape}")
    print(f"Drug features shape: {drug_features.shape}")
    
    # Train model with 5-fold cross-validation
    print("\n2. Training AntiViralDL with 5-fold cross-validation...")
    print("Parameters:")
    print("  - Embedding dimension: 128")
    print("  - GCN layers: 2")
    print("  - Learning rate: 0.01")
    print("  - Lambda (contrastive): 0.5")
    print("  - Temperature: 0.1")
    print("  - Epochs: 1000")
    
    results = train_antiviraldl(
        interactions=interactions,
        virus_features=virus_features,
        drug_features=drug_features,
        embedding_dim=128,
        num_layers=2,
        learning_rate=0.01,
        lambda_cl=0.5,
        temperature=0.1,
        num_epochs=1000,
        batch_size=1024,
        verbose=True
    )
    
    # Generate visualizations
    print("\n3. Generating visualizations...")
    plot_roc_curves(results, save_path='/home/claude/roc_curves.png')
    
    # Create results table
    results_table = create_results_table(results)
    print("\n4. Results Table:")
    print(results_table.to_string(index=False))
    
    # Save results
    results_table.to_csv('/home/claude/cross_validation_results.csv', index=False)
    print("\nResults saved to cross_validation_results.csv")
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
