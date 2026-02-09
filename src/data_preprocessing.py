"""
Data preprocessing and loading utilities for AntiViralDL
Handles DrugVirus2 and FDA-approved drug datasets
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold


class DataPreprocessor:
    """
    Preprocess virus-drug interaction data
    """
    
    def __init__(self):
        self.virus_feature_scaler = StandardScaler()
        self.drug_feature_scaler = StandardScaler()
        self.virus_id_map = {}
        self.drug_id_map = {}
        self.id_virus_map = {}
        self.id_drug_map = {}
    
    def load_data(self, virus_drug_file, virus_features_file=None, drug_features_file=None):
        """
        Load virus-drug interaction data and features
        
        Args:
            virus_drug_file: Path to virus-drug associations CSV
            virus_features_file: Path to virus features CSV (optional)
            drug_features_file: Path to drug features CSV (optional)
        
        Returns:
            interactions, virus_features, drug_features
        """
        # Load interactions
        interactions_df = pd.read_csv(virus_drug_file)
        
        # Create ID mappings
        unique_viruses = interactions_df['virus'].unique()
        unique_drugs = interactions_df['drug'].unique()
        
        self.virus_id_map = {virus: idx for idx, virus in enumerate(unique_viruses)}
        self.drug_id_map = {drug: idx for idx, drug in enumerate(unique_drugs)}
        self.id_virus_map = {idx: virus for virus, idx in self.virus_id_map.items()}
        self.id_drug_map = {idx: drug for drug, idx in self.drug_id_map.items()}
        
        # Map to IDs
        interactions = []
        for _, row in interactions_df.iterrows():
            virus_id = self.virus_id_map[row['virus']]
            drug_id = self.drug_id_map[row['drug']]
            interactions.append([virus_id, drug_id])
        
        interactions = np.array(interactions)
        
        # Load features if provided
        virus_features = None
        drug_features = None
        
        if virus_features_file:
            virus_features_df = pd.read_csv(virus_features_file)
            virus_features = self._process_virus_features(virus_features_df)
        
        if drug_features_file:
            drug_features_df = pd.read_csv(drug_features_file)
            drug_features = self._process_drug_features(drug_features_df)
        
        return interactions, virus_features, drug_features
    
    def _process_virus_features(self, virus_features_df):
        """
        Process virus features
        
        Args:
            virus_features_df: DataFrame with virus features
        
        Returns:
            Processed virus features array
        """
        # Extract numerical features
        # According to paper: 16 virus features including genome sequence, 
        # host type, family, pathogenicity
        
        feature_columns = [col for col in virus_features_df.columns if col != 'virus']
        
        virus_features = []
        for virus in [self.id_virus_map[i] for i in range(len(self.virus_id_map))]:
            if virus in virus_features_df['virus'].values:
                features = virus_features_df[virus_features_df['virus'] == virus][feature_columns].values[0]
            else:
                # Default features if virus not found
                features = np.zeros(len(feature_columns))
            virus_features.append(features)
        
        virus_features = np.array(virus_features)
        
        # Normalize features
        virus_features = self.virus_feature_scaler.fit_transform(virus_features)
        
        return virus_features
    
    def _process_drug_features(self, drug_features_df):
        """
        Process drug features
        
        Args:
            drug_features_df: DataFrame with drug features
        
        Returns:
            Processed drug features array
        """
        # According to paper: 18 drug features including SMILES, 
        # ATC classification, molecular weight, drug type
        
        feature_columns = [col for col in drug_features_df.columns if col != 'drug']
        
        drug_features = []
        for drug in [self.id_drug_map[i] for i in range(len(self.drug_id_map))]:
            if drug in drug_features_df['drug'].values:
                features = drug_features_df[drug_features_df['drug'] == drug][feature_columns].values[0]
            else:
                # Default features if drug not found
                features = np.zeros(len(feature_columns))
            drug_features.append(features)
        
        drug_features = np.array(drug_features)
        
        # Normalize features
        drug_features = self.drug_feature_scaler.fit_transform(drug_features)
        
        return drug_features
    
    def create_synthetic_features(self, num_viruses, num_drugs, 
                                   virus_dim=16, drug_dim=18):
        """
        Create synthetic features when real features are not available
        
        Args:
            num_viruses: Number of viruses
            num_drugs: Number of drugs
            virus_dim: Dimension of virus features (default: 16)
            drug_dim: Dimension of drug features (default: 18)
        
        Returns:
            virus_features, drug_features
        """
        virus_features = np.random.randn(num_viruses, virus_dim)
        drug_features = np.random.randn(num_drugs, drug_dim)
        
        virus_features = self.virus_feature_scaler.fit_transform(virus_features)
        drug_features = self.drug_feature_scaler.fit_transform(drug_features)
        
        return virus_features, drug_features


class CrossValidationSplitter:
    """
    K-fold cross-validation splitter for virus-drug associations
    """
    
    def __init__(self, n_splits=5, shuffle=True, random_state=42):
        """
        Args:
            n_splits: Number of folds (default: 5)
            shuffle: Whether to shuffle data (default: True)
            random_state: Random seed (default: 42)
        """
        self.n_splits = n_splits
        self.kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    def split(self, interactions):
        """
        Split interactions into train and test sets for each fold
        
        Args:
            interactions: Array of virus-drug interactions
        
        Yields:
            train_interactions, test_interactions for each fold
        """
        for train_idx, test_idx in self.kfold.split(interactions):
            train_interactions = interactions[train_idx]
            test_interactions = interactions[test_idx]
            
            yield train_interactions, test_interactions


def create_sample_dataset(num_viruses=84, num_drugs=219, num_interactions=1462):
    """
    Create a sample dataset matching the paper's statistics
    
    Args:
        num_viruses: Number of virus types (default: 84)
        num_drugs: Number of drug compounds (default: 219)
        num_interactions: Number of known associations (default: 1462)
    
    Returns:
        Sample dataset as DataFrame
    """
    np.random.seed(42)
    
    # Generate virus names
    virus_names = [f"Virus_{i}" for i in range(num_viruses)]
    
    # Generate drug names
    drug_names = [f"Drug_{i}" for i in range(num_drugs)]
    
    # Generate random interactions
    interactions = []
    interaction_set = set()
    
    while len(interactions) < num_interactions:
        virus_idx = np.random.randint(0, num_viruses)
        drug_idx = np.random.randint(0, num_drugs)
        
        if (virus_idx, drug_idx) not in interaction_set:
            interactions.append({
                'virus': virus_names[virus_idx],
                'drug': drug_names[drug_idx]
            })
            interaction_set.add((virus_idx, drug_idx))
    
    df = pd.DataFrame(interactions)
    
    return df


def create_sample_features(num_viruses=84, num_drugs=219):
    """
    Create sample features for viruses and drugs
    
    Args:
        num_viruses: Number of viruses
        num_drugs: Number of drugs
    
    Returns:
        virus_features_df, drug_features_df
    """
    np.random.seed(42)
    
    # Virus features (16 features as per paper)
    virus_names = [f"Virus_{i}" for i in range(num_viruses)]
    virus_features = {
        'virus': virus_names,
        **{f'feature_{i}': np.random.randn(num_viruses) for i in range(16)}
    }
    virus_features_df = pd.DataFrame(virus_features)
    
    # Drug features (18 features as per paper)
    drug_names = [f"Drug_{i}" for i in range(num_drugs)]
    drug_features = {
        'drug': drug_names,
        **{f'feature_{i}': np.random.randn(num_drugs) for i in range(18)}
    }
    drug_features_df = pd.DataFrame(drug_features)
    
    return virus_features_df, drug_features_df


def load_drugvirus2_dataset():
    """
    Load DrugVirus2 dataset
    Note: This is a placeholder - actual implementation would load from the real database
    
    Returns:
        DataFrame with virus-drug associations
    """
    # According to paper Table II:
    # DrugVirus2: 153 viruses, 231 drugs, 1519 associations
    return create_sample_dataset(num_viruses=153, num_drugs=231, num_interactions=1519)


def load_fda_dataset():
    """
    Load FDA-approved virus-drug dataset
    Note: This is a placeholder - actual implementation would load from the real database
    
    Returns:
        DataFrame with virus-drug associations
    """
    # According to paper Table II:
    # US FDA: 16 viruses, 111 drugs, 142 associations
    return create_sample_dataset(num_viruses=16, num_drugs=111, num_interactions=142)


def merge_datasets(drugvirus2_df, fda_df):
    """
    Merge DrugVirus2 and FDA datasets
    
    Args:
        drugvirus2_df: DrugVirus2 dataset
        fda_df: FDA dataset
    
    Returns:
        Merged dataset
    """
    merged_df = pd.concat([drugvirus2_df, fda_df], ignore_index=True)
    merged_df = merged_df.drop_duplicates(subset=['virus', 'drug'])
    
    return merged_df
