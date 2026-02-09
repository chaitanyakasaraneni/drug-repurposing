# AntiViralDL: AI-Driven Drug Repurposing

A Graph Neural Network and Self-Supervised Learning Approach for Virus-Drug Association Prediction

**Paper**:[**"AI-Driven Drug Repurposing: A Graph Neural Network and Self-Supervised Learning Approach"**](https://doi.org/10.1109/CIACON65473.2025.11189545)
**Conference**: IEEE CIACON 2025  
**Author**: Chaitanya Krishna Kasaraneni

## Overview

AntiViralDL is a novel graph contrastive learning framework for predicting virus-drug associations, enabling rapid antiviral drug development and repurposing. The method combines:

- **LightGCN**: Lightweight Graph Convolutional Network for embedding learning
- **Contrastive Learning**: Self-supervised learning to handle sparse virus-drug associations
- **Noise-based Augmentation**: Novel approach using random noise instead of graph structure perturbation

## Key Features

- 5-fold cross-validation achieving **0.8450 AUC** and **0.8494 AUPR**
- Bipartite graph representation of virus-drug interactions
- Self-supervised learning with contrastive loss
- Parameter sensitivity analysis for optimal configuration
- Drug repurposing predictions for COVID-19 and other viruses

## Installation

### Requirements

- Python 3.7+
- TensorFlow 2.4+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

### Setup

```bash
# Install dependencies
pip install tensorflow==2.4.0 numpy pandas scikit-learn matplotlib

# Clone or download the code
# No additional installation required
```

## Project Structure

```
antiviraldl/
├── antiviraldl_model.py          # Core model implementation
├── data_preprocessing.py          # Data loading and preprocessing
├── train_antiviraldl.py          # Main training script
├── parameter_sensitivity.py      # Parameter analysis
├── ablation_study.py             # Ablation experiments
├── drug_repurposing.py           # COVID-19 case study
└── README.md                     # This file
```

## Usage

### 1. Basic Training with 5-Fold Cross-Validation

```bash
python train_antiviraldl.py
```

This will:
- Load the virus-drug interaction dataset
- Perform 5-fold cross-validation
- Generate ROC curves
- Output AUC and AUPR metrics

**Expected Output**:
```
Average AUC: 0.8450
Average AUPR: 0.8494
```

### 2. Parameter Sensitivity Analysis

```bash
python parameter_sensitivity.py
```

Analyzes the effect of:
- GCN layers (1, 2, 3, 4)
- Embedding sizes (16, 32, 64, 128, 256)
- Lambda values (0.1, 0.3, 0.5, 0.7, 0.9)

**Optimal Parameters** (from paper):
- Layers: 2
- Embedding size: 128
- Lambda: 0.5

### 3. Ablation Study

```bash
python ablation_study.py
```

Tests different components:
- **Data Augmentation**: None, Edge Drop, Node Drop, Random Walk, Noise
- **Feature Composition**: Concatenation, Max, Min, Sum
- **Noise Types**: Random, Similar

### 4. Drug Repurposing for COVID-19

```bash
python drug_repurposing.py
```

Predicts top-10 candidate drugs for COVID-19 treatment.

## Model Architecture

### LightGCN Component

```python
# Initialize model
model = AntiViralDL(
    num_viruses=84,
    num_drugs=219,
    embedding_dim=128,
    num_layers=2,
    learning_rate=0.01,
    lambda_cl=0.5,
    temperature=0.1
)
```

### Training Process

1. **Build bipartite graph** from virus-drug interactions
2. **Multi-hop propagation** using LightGCN (2 layers)
3. **BPR loss** for recommendation
4. **Contrastive loss** with noise augmentation
5. **Combine losses** with weight λ = 0.5

### Loss Function

```
L_total = L_BPR + λ × L_CL + L_reg

where:
L_BPR = Bayesian Personalized Ranking loss
L_CL = Contrastive Learning loss with temperature τ = 0.1
L_reg = L2 regularization
```

## Dataset Information

### Merged Dataset Statistics
- **Viruses**: 84 types
- **Drugs**: 219 compounds
- **Known Associations**: 1,462

### Features
- **Virus features** (16 dimensions): Genome sequence, host type, family, pathogenicity
- **Drug features** (18 dimensions): SMILES, ATC classification, molecular weight, drug type

### Data Sources
- DrugVirus2 database
- FDA-approved virus-drug associations

## Results

### Cross-Validation Performance

| Fold | AUC    | AUPR   |
|------|--------|--------|
| 1    | 0.8645 | 0.8595 |
| 2    | 0.8428 | 0.8469 |
| 3    | 0.8090 | 0.8237 |
| 4    | 0.8507 | 0.8602 |
| 5    | 0.8578 | 0.8570 |
| **Average** | **0.8450** | **0.8494** |

### Comparison with Baselines

| Method | AUC | AUPR |
|--------|-----|------|
| VDA-KATZ | 0.7991 | 0.7496 |
| IRNMF | 0.8122 | 0.7610 |
| DRRS | 0.8214 | 0.8172 |
| VDA-DLCMNMF | 0.8372 | 0.8318 |
| **AntiViralDL** | **0.8450** | **0.8494** |

### Training Time
- Average: 12.3 minutes per fold on NVIDIA RTX 3090 GPU
- Faster than VDA-DLCMNMF (17.2 min) and IRNMF (14.5 min)

## Customization

### Using Your Own Dataset

```python
from data_preprocessing import DataPreprocessor

# Load your data
preprocessor = DataPreprocessor()
interactions, virus_features, drug_features = preprocessor.load_data(
    virus_drug_file='your_interactions.csv',
    virus_features_file='your_virus_features.csv',
    drug_features_file='your_drug_features.csv'
)

# Train model
from train_antiviraldl import train_antiviraldl

results = train_antiviraldl(
    interactions=interactions,
    virus_features=virus_features,
    drug_features=drug_features
)
```

### Adjusting Hyperparameters

```python
model = AntiViralDL(
    num_viruses=your_num_viruses,
    num_drugs=your_num_drugs,
    embedding_dim=256,        # Increase for larger datasets
    num_layers=3,             # More layers for complex graphs
    learning_rate=0.005,      # Adjust based on convergence
    lambda_cl=0.7,           # Higher weight for contrastive learning
    temperature=0.05         # Lower for harder negatives
)
```

## Key Innovations

1. **Noise-based Augmentation**: Instead of dropping edges/nodes, adds Gaussian noise to embeddings
2. **Self-supervised Learning**: Addresses sparsity in virus-drug associations
3. **LightGCN Architecture**: Simplified GCN without feature transformation
4. **Bipartite Graph**: Natural representation of virus-drug interactions

## COVID-19 Drug Predictions

Top predicted drugs (from case study):

1. **Baloxavir marboxil** (0.99) - Clinical trial
2. **Laninamivir octanoate** (0.96) - Unconfirmed
3. **Arbidol** (0.96) - Phase 4
4. **Peramivir** (0.96) - Clinical trial
5. **Zanamivir** (0.94) - Unconfirmed

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{kasaraneni2025antiviraldl,
  title={AI-Driven Drug Repurposing: A Graph Neural Network and Self-Supervised Learning Approach},
  author={Kasaraneni, Chaitanya Krishna},
  booktitle={2025 International Conference on Computing, Intelligence, and Application (CIACON)},
  year={2025},
  organization={IEEE}
}
```

## License

This code is provided for research and educational purposes.

## Contact

Chaitanya Krishna Kasaraneni  
Software Engineer, Egen Solutions  
Email: kc.kasaraneni@ieee.org

## Troubleshooting

### Common Issues

**Issue**: Out of memory during training
```python
# Solution: Reduce batch size
results = train_antiviraldl(..., batch_size=512)
```

**Issue**: Slow convergence
```python
# Solution: Increase learning rate or reduce epochs
model = AntiViralDL(..., learning_rate=0.05)
```

**Issue**: Poor performance
```python
# Solution: Adjust lambda and temperature
model = AntiViralDL(..., lambda_cl=0.7, temperature=0.05)
```

## Future Enhancements

- Transfer learning for new virus families
- Integration of heterogeneous node features
- Larger-scale datasets (10,000+ associations)
- Attention mechanisms for interpretability
- Multi-task learning for side effect prediction

## Acknowledgments

- DrugVirus2 database
- FDA drug approval database
- IEEE CIACON 2025 conference
- Egen Solutions

---

**Last Updated**: February 2026  
**Version**: 1.0
