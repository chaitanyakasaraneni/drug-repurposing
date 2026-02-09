"""
Parameter Sensitivity Analysis for AntiViralDL
Analyzes the effect of different GCN layers, embedding sizes, and lambda values
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.antiviraldl_model import AntiViralDL, generate_negative_samples
from src.data_preprocessing import DataPreprocessor, create_sample_dataset


def analyze_gcn_layers(interactions, virus_features, drug_features, 
                       layer_values=[1, 2, 3, 4], num_epochs=500):
    """
    Analyze the effect of different GCN layer configurations
    
    Args:
        interactions: Virus-drug interactions
        virus_features: Virus features
        drug_features: Drug features
        layer_values: List of layer values to test
        num_epochs: Number of training epochs
    
    Returns:
        Results dictionary
    """
    print("\nAnalyzing effect of different GCN layers...")
    
    num_viruses = interactions[:, 0].max() + 1
    num_drugs = interactions[:, 1].max() + 1
    
    results = {'layers': [], 'auc': [], 'aupr': []}
    
    # Split data into train/test (80/20)
    np.random.shuffle(interactions)
    split_idx = int(0.8 * len(interactions))
    train_interactions = interactions[:split_idx]
    test_interactions = interactions[split_idx:]
    
    for num_layers in layer_values:
        print(f"\nTesting with {num_layers} layer(s)...")
        
        # Initialize model
        model = AntiViralDL(
            num_viruses=num_viruses,
            num_drugs=num_drugs,
            embedding_dim=128,
            num_layers=num_layers,
            learning_rate=0.01,
            lambda_cl=0.5,
            temperature=0.1,
            virus_features=virus_features,
            drug_features=drug_features
        )
        
        # Build adjacency matrix
        adj_matrix = model.lightgcn.build_adjacency_matrix(train_interactions)
        
        # Train
        for epoch in range(num_epochs):
            neg_interactions = generate_negative_samples(
                train_interactions, num_viruses, num_drugs
            )
            model.train_step(adj_matrix, train_interactions, neg_interactions)
            
            if (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch + 1}/{num_epochs}")
        
        # Evaluate
        auc_score, aupr_score, _, _ = model.evaluate(
            adj_matrix, test_interactions, interactions
        )
        
        print(f"  Results: AUC = {auc_score:.4f}, AUPR = {aupr_score:.4f}")
        
        results['layers'].append(num_layers)
        results['auc'].append(auc_score)
        results['aupr'].append(aupr_score)
    
    return results


def analyze_embedding_sizes(interactions, virus_features, drug_features,
                            embedding_sizes=[16, 32, 64, 128, 256], num_epochs=500):
    """
    Analyze the effect of different embedding sizes
    
    Args:
        interactions: Virus-drug interactions
        virus_features: Virus features
        drug_features: Drug features
        embedding_sizes: List of embedding sizes to test
        num_epochs: Number of training epochs
    
    Returns:
        Results dictionary
    """
    print("\nAnalyzing effect of different embedding sizes...")
    
    num_viruses = interactions[:, 0].max() + 1
    num_drugs = interactions[:, 1].max() + 1
    
    results = {'embedding_size': [], 'auc': [], 'aupr': []}
    
    # Split data
    np.random.shuffle(interactions)
    split_idx = int(0.8 * len(interactions))
    train_interactions = interactions[:split_idx]
    test_interactions = interactions[split_idx:]
    
    for emb_size in embedding_sizes:
        print(f"\nTesting with embedding size {emb_size}...")
        
        # Adjust features to match embedding size if needed
        if virus_features.shape[1] < emb_size:
            # Pad features
            virus_feat = np.pad(virus_features, 
                               ((0, 0), (0, emb_size - virus_features.shape[1])), 
                               'constant')
            drug_feat = np.pad(drug_features,
                              ((0, 0), (0, emb_size - drug_features.shape[1])),
                              'constant')
        elif virus_features.shape[1] > emb_size:
            # Truncate features
            virus_feat = virus_features[:, :emb_size]
            drug_feat = drug_features[:, :emb_size]
        else:
            virus_feat = virus_features
            drug_feat = drug_features
        
        # Initialize model
        model = AntiViralDL(
            num_viruses=num_viruses,
            num_drugs=num_drugs,
            embedding_dim=emb_size,
            num_layers=2,
            learning_rate=0.01,
            lambda_cl=0.5,
            temperature=0.1,
            virus_features=virus_feat,
            drug_features=drug_feat
        )
        
        # Build adjacency matrix
        adj_matrix = model.lightgcn.build_adjacency_matrix(train_interactions)
        
        # Train
        for epoch in range(num_epochs):
            neg_interactions = generate_negative_samples(
                train_interactions, num_viruses, num_drugs
            )
            model.train_step(adj_matrix, train_interactions, neg_interactions)
            
            if (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch + 1}/{num_epochs}")
        
        # Evaluate
        auc_score, aupr_score, _, _ = model.evaluate(
            adj_matrix, test_interactions, interactions
        )
        
        print(f"  Results: AUC = {auc_score:.4f}, AUPR = {aupr_score:.4f}")
        
        results['embedding_size'].append(emb_size)
        results['auc'].append(auc_score)
        results['aupr'].append(aupr_score)
    
    return results


def analyze_lambda_values(interactions, virus_features, drug_features,
                          lambda_values=[0.1, 0.3, 0.5, 0.7, 0.9], num_epochs=500):
    """
    Analyze the effect of different lambda (contrastive learning weight) values
    
    Args:
        interactions: Virus-drug interactions
        virus_features: Virus features
        drug_features: Drug features
        lambda_values: List of lambda values to test
        num_epochs: Number of training epochs
    
    Returns:
        Results dictionary
    """
    print("\nAnalyzing effect of different lambda values...")
    
    num_viruses = interactions[:, 0].max() + 1
    num_drugs = interactions[:, 1].max() + 1
    
    results = {'lambda': [], 'auc': [], 'aupr': []}
    
    # Split data
    np.random.shuffle(interactions)
    split_idx = int(0.8 * len(interactions))
    train_interactions = interactions[:split_idx]
    test_interactions = interactions[split_idx:]
    
    for lambda_val in lambda_values:
        print(f"\nTesting with lambda = {lambda_val}...")
        
        # Initialize model
        model = AntiViralDL(
            num_viruses=num_viruses,
            num_drugs=num_drugs,
            embedding_dim=128,
            num_layers=2,
            learning_rate=0.01,
            lambda_cl=lambda_val,
            temperature=0.1,
            virus_features=virus_features,
            drug_features=drug_features
        )
        
        # Build adjacency matrix
        adj_matrix = model.lightgcn.build_adjacency_matrix(train_interactions)
        
        # Train
        for epoch in range(num_epochs):
            neg_interactions = generate_negative_samples(
                train_interactions, num_viruses, num_drugs
            )
            model.train_step(adj_matrix, train_interactions, neg_interactions)
            
            if (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch + 1}/{num_epochs}")
        
        # Evaluate
        auc_score, aupr_score, _, _ = model.evaluate(
            adj_matrix, test_interactions, interactions
        )
        
        print(f"  Results: AUC = {auc_score:.4f}, AUPR = {aupr_score:.4f}")
        
        results['lambda'].append(lambda_val)
        results['auc'].append(auc_score)
        results['aupr'].append(aupr_score)
    
    return results


def plot_parameter_sensitivity(layer_results, emb_results, lambda_results, 
                               save_path='/home/claude/parameter_sensitivity.png'):
    """
    Plot parameter sensitivity analysis results
    
    Args:
        layer_results: Results from GCN layer analysis
        emb_results: Results from embedding size analysis
        lambda_results: Results from lambda analysis
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot GCN layers effect
    ax1 = axes[0]
    ax1.plot(layer_results['layers'], layer_results['auc'], 'o-', 
             color='blue', label='AUC', linewidth=2, markersize=8)
    ax1.plot(layer_results['layers'], layer_results['aupr'], 's-', 
             color='red', label='AUPR', linewidth=2, markersize=8)
    ax1.set_xlabel('Layers of GCN', fontsize=12)
    ax1.set_ylabel('Performance', fontsize=12)
    ax1.set_title('(a) Effect of different GCN layers', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(layer_results['layers'])
    
    # Plot embedding size effect
    ax2 = axes[1]
    ax2.plot(emb_results['embedding_size'], emb_results['auc'], 'o-',
             color='blue', label='AUC', linewidth=2, markersize=8)
    ax2.plot(emb_results['embedding_size'], emb_results['aupr'], 's-',
             color='red', label='AUPR', linewidth=2, markersize=8)
    ax2.set_xlabel('Embedding sizes', fontsize=12)
    ax2.set_ylabel('Performance', fontsize=12)
    ax2.set_title('(b) Effect of different Embedding Sizes', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    ax2.set_xticks(emb_results['embedding_size'])
    ax2.set_xticklabels(emb_results['embedding_size'])
    
    # Plot lambda effect
    ax3 = axes[2]
    ax3.plot(lambda_results['lambda'], lambda_results['auc'], 'o-',
             color='blue', label='AUC', linewidth=2, markersize=8)
    ax3.plot(lambda_results['lambda'], lambda_results['aupr'], 's-',
             color='red', label='AUPR', linewidth=2, markersize=8)
    ax3.set_xlabel('Lambda', fontsize=12)
    ax3.set_ylabel('Performance', fontsize=12)
    ax3.set_title('(c) Effect of lambda', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(lambda_results['lambda'])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nParameter sensitivity plots saved to {save_path}")
    plt.close()


def create_sensitivity_table(layer_results, emb_results, lambda_results):
    """
    Create table matching Table VII in the paper
    
    Args:
        layer_results: Results from GCN layer analysis
        emb_results: Results from embedding size analysis
        lambda_results: Results from lambda analysis
    
    Returns:
        DataFrame with sensitivity results
    """
    data = {
        'Parameter': [],
        'Value': [],
        'AUC': [],
        'AUPR': []
    }
    
    # Add layer results
    for i, layer in enumerate(layer_results['layers']):
        data['Parameter'].append('Layer')
        data['Value'].append(layer)
        data['AUC'].append(layer_results['auc'][i])
        data['AUPR'].append(layer_results['aupr'][i])
    
    # Add embedding size results
    for i, size in enumerate(emb_results['embedding_size']):
        data['Parameter'].append('Embedding Sizes')
        data['Value'].append(size)
        data['AUC'].append(emb_results['auc'][i])
        data['AUPR'].append(emb_results['aupr'][i])
    
    # Add lambda results
    for i, lam in enumerate(lambda_results['lambda']):
        data['Parameter'].append('Lambda')
        data['Value'].append(lam)
        data['AUC'].append(lambda_results['auc'][i])
        data['AUPR'].append(lambda_results['aupr'][i])
    
    df = pd.DataFrame(data)
    
    return df


def main():
    """
    Main execution function for parameter sensitivity analysis
    """
    print("="*60)
    print("AntiViralDL - Parameter Sensitivity Analysis")
    print("="*60)
    
    # Load data
    print("\nLoading dataset...")
    interactions_df = create_sample_dataset(num_viruses=84, num_drugs=219, num_interactions=1462)
    
    preprocessor = DataPreprocessor()
    interactions_df.to_csv('/home/claude/virus_drug_interactions_temp.csv', index=False)
    interactions, _, _ = preprocessor.load_data('/home/claude/virus_drug_interactions_temp.csv')
    
    num_viruses = interactions[:, 0].max() + 1
    num_drugs = interactions[:, 1].max() + 1
    
    # Create features
    virus_features, drug_features = preprocessor.create_synthetic_features(
        num_viruses, num_drugs, virus_dim=16, drug_dim=18
    )
    
    print(f"Number of viruses: {num_viruses}")
    print(f"Number of drugs: {num_drugs}")
    print(f"Number of interactions: {len(interactions)}")
    
    # Run sensitivity analyses
    layer_results = analyze_gcn_layers(interactions, virus_features, drug_features)
    emb_results = analyze_embedding_sizes(interactions, virus_features, drug_features)
    lambda_results = analyze_lambda_values(interactions, virus_features, drug_features)
    
    # Plot results
    plot_parameter_sensitivity(layer_results, emb_results, lambda_results)
    
    # Create results table
    sensitivity_table = create_sensitivity_table(layer_results, emb_results, lambda_results)
    print("\n" + "="*60)
    print("Parameter Sensitivity Results:")
    print("="*60)
    print(sensitivity_table.to_string(index=False))
    
    # Save results
    sensitivity_table.to_csv('/home/claude/parameter_sensitivity_results.csv', index=False)
    print("\nResults saved to parameter_sensitivity_results.csv")
    
    print("\n" + "="*60)
    print("Parameter sensitivity analysis completed!")
    print("="*60)


if __name__ == "__main__":
    main()
