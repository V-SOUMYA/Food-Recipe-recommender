# Food Recipe Recommendation System

A comprehensive machine learning project that implements and evaluates multiple recommendation system approaches for food recipes using the Food.com dataset. The project explores content-based filtering, collaborative filtering, and hybrid recommendation techniques across four distinct phases.

##  Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Models Implemented](#models-implemented)
- [Evaluation Metrics](#evaluation-metrics)
- [Results Summary](#results-summary)
- [Requirements](#requirements)
- [Usage](#usage)
- [Project Phases](#project-phases)

##  Overview

This project implements a multi-phase recommendation system for food recipes, comparing various recommendation algorithms including:

- **Content-Based Filtering (CBF)**: Uses recipe features (name, description, category, keywords, ingredients)
- **Collaborative Filtering (CF)**: User-based and item-based approaches
- **Matrix Factorization**: Traditional MF and Probabilistic Matrix Factorization (PMF)
- **Hybrid Models**: Multiple strategies to combine different recommendation approaches

The goal is to predict user ratings for recipes and provide personalized recipe recommendations.

## Motivation
Recommender systems often struggle with sparsity and cold-start problems in real-world datasets.
This project explores how hybrid approaches can combine content-based and collaborative signals to improve recommendation quality, robustness, and interpretability at scale.

##  Dataset

The project uses the Food.com dataset containing:

- **Reviews Dataset**: 1,401,982 reviews with 8 columns
  - ReviewId, RecipeId, AuthorId, AuthorName, Rating, Review, DateSubmitted, DateModified
  
- **Recipes Dataset**: 522,517 recipes with 28 columns
  - RecipeId, Name, Description, RecipeCategory, Keywords, RecipeIngredientParts, and nutritional information

### Data Preprocessing

- Filtered recipes with at least 20 reviews
- Filtered users with at least 20 reviews
- Final dataset: 276,559 reviews from 4,416 users and 10,268 recipes
- Train/Test split: 80/20 (221,247 training, 55,312 test samples)

##  Project Structure

```
├── ML_project_food_com_final_submission.ipynb  # Main notebook with all code
└── README.md                                    # This file
```

##  Methodology

### Feature Engineering

- **Meta Text Creation**: Combines recipe Name, Description, RecipeCategory, Keywords, and RecipeIngredientParts into a single text feature
- **TF-IDF Vectorization**: Converts recipe text features into TF-IDF vectors (max_features=40,000)
- **Dimensionality Reduction**: Uses TruncatedSVD to reduce TF-IDF features to 100 dimensions for content embeddings

### Similarity Computation

- **Content Similarity**: Cosine similarity on TF-IDF vectors (Top-K=200)
- **User Similarity**: Cosine similarity on user-centered rating vectors
- **Item Similarity**: Cosine similarity on item-centered rating vectors (Top-K=100)

##  Models Implemented

### Phase 1: Basic Recommendation Models

1. **Content-Based Filtering (CBF)**
   - Uses TF-IDF-based item similarity
   - Predicts ratings based on similarity to user's previously rated items

2. **Collaborative Filtering (CF)**
   - User-based collaborative filtering
   - Uses top-K similar users (K=50)

3. **Hybrid Model (Phase 1)**
   - Weighted combination of CBF and CF (α=0.5)

### Phase 2: Content-Based Machine Learning Models

1. **Ridge Regression**
   - Linear model with L2 regularization
   - Features: SVD-reduced content embeddings + user/item statistics

2. **Bayesian Ridge Regression**
   - Bayesian approach to Ridge regression

3. **Approximate Kernel Ridge Regression**
   - Uses RBF feature approximation (150 components)
   - Trained on subsample of 120,000 samples

### Phase 3: Advanced Collaborative Filtering

1. **Item-Item Collaborative Filtering**
   - Item-based approach using item-centered ratings
   - Top-K=100 similar items

2. **Matrix Factorization (MF)**
   - Traditional MF with biases
   - 20 latent factors, 10 epochs
   - Learning rate: 0.01, Regularization: 0.05

3. **Probabilistic Matrix Factorization (PMF)**
   - PMF-style factorization without biases
   - 20 latent factors, 8 epochs
   - Learning rate: 0.008, Regularization: 0.08

### Phase 4: Hybrid Models

1. **H1: 2-Way Weighted Blend**
   - Combines CBF (Ridge) and MF
   - Optimal weight: α=0.6 (CBF) vs 0.4 (MF)

2. **H2: 3-Way Blend**
   - Combines CBF, MF, and PMF
   - Optimal weights: (0.20, 0.80, 0.00)

3. **H3: Switching Hybrid**
   - Uses CBF for cold-start cases (users/items with <20 ratings)
   - Uses MF for warm-start cases

4. **H4: Meta-Learner**
   - Ridge regression on base model predictions
   - Features: predictions from CBF, MF, PMF, ItemCF

##  Evaluation Metrics

The project evaluates models using multiple metrics:

### Rating Prediction Metrics
- **RMSE** (Root Mean Squared Error): Measures prediction accuracy
- **MAE** (Mean Absolute Error): Average absolute prediction error
- **FCP** (Fraction of Concordant Pairs): Measures ranking quality

### Ranking Metrics
- **Precision@K**: Fraction of top-K recommendations that are relevant (rating ≥ 5)
- **Recall@K**: Fraction of relevant items found in top-K
- **NDCG@K**: Normalized Discounted Cumulative Gain at K

##  Results Summary

### Best Performing Models (Test Set)

**Rating Prediction:**
- Best RMSE: **H1_2Way** (0.864) - Hybrid of CBF and MF
- Best MAE: **H2_3Way** (0.494) - 3-way blend
- Best FCP: **H4_Meta** (0.606) - Meta-learner

**Ranking Performance (K=10):**
- Best Precision@10: **H2_3Way** (0.822)
- Best Recall@10: **H2_3Way** (0.630)
- Best NDCG@10: **MF** (0.864)

### Phase-by-Phase Highlights

**Phase 1:**
- Hybrid model outperformed individual CBF and CF models
- Test RMSE: 0.902 (Hybrid) vs 0.996 (CBF) vs 0.916 (CF)

**Phase 2:**
- Ridge and Bayesian Ridge performed similarly
- Test RMSE: ~0.869 for both Ridge models
- Kernel Ridge: 0.881 (slightly worse)

**Phase 3:**
- Matrix Factorization achieved best overall performance
- Test RMSE: 0.860 (MF) vs 0.935 (ItemCF) vs 0.897 (PMF)

**Phase 4:**
- H2 (3-way blend) showed best balance across metrics
- Meta-learner (H4) achieved best FCP but higher RMSE

##  Requirements

### Python Libraries

```python
numpy
pandas
scikit-learn
matplotlib
```

### Data Requirements

- `reviews.parquet`: Reviews dataset from Food.com
- `recipes.parquet`: Recipes dataset from Food.com

**Note**: The notebook is configured for Google Colab with data stored in Google Drive. Adjust the `DATA_DIR` path for local execution.

## Reproducibility Notes
- Raw Food.com datasets are not included due to size and licensing constraints.
- Users must download the dataset separately and update the `DATA_DIR` path.
- Experiments were run on Google Colab due to memory and compute requirements.

##  Usage

### Running the Notebook

1. **Google Colab Setup** (as configured):
   ```python
   # Mount Google Drive
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Local Setup**:
   - Update `DATA_DIR` to point to your local data directory
   - Ensure parquet files are accessible

3. **Execute Cells Sequentially**:
   - The notebook is designed to run cells in order
   - Each phase builds upon previous phases
   - Results are stored in variables for later comparison

### Key Parameters

- **Train/Test Split**: 80/20 (random_state=42)
- **TF-IDF**: max_features=40,000, stop_words="english"
- **SVD Components**: 100 dimensions
- **Top-K Similarities**: 200 (content), 100 (item CF), 50 (user CF)
- **MF Factors**: 20 latent factors
- **Relevance Threshold**: Rating ≥ 5 for ranking metrics

##  Project Phases

### Phase 1: Foundation Models
- Basic CBF and CF implementations
- Simple hybrid combination
- Establishes baseline performance

### Phase 2: Content-Based ML
- Machine learning models on content features
- Explores linear and non-linear approaches
- Feature engineering with SVD

### Phase 3: Advanced CF
- Item-based collaborative filtering
- Matrix factorization techniques
- Latent factor models

### Phase 4: Hybrid Strategies
- Multiple hybrid combination methods
- Meta-learning approach
- Context-aware switching

##  Key Insights

1. **Hybrid models consistently outperform individual approaches**
2. **Matrix Factorization provides strong baseline for collaborative filtering**
3. **Content-based features are valuable for cold-start scenarios**
4. **Meta-learning can improve ranking quality (FCP) but may increase RMSE**
5. **The 3-way blend (H2) offers best overall performance balance**

##  Notes

- The project uses a filtered dataset to ensure sufficient data for each user and recipe
- Some models use subsampling for computational efficiency (Kernel Ridge, Meta-Learner)
- All models are evaluated on the same test set for fair comparison
- Ranking metrics use K values: [1, 3, 5, 10, 20, 50]

##  Future Work

- Refactor the notebook into modular Python scripts
- Add unit tests for core recommender components
- Evaluate performance on true cold-start users and items
- Explore neural and sequential recommender models

## 📄 License
This project is for educational/research purposes. Please ensure proper attribution when using the Food.com dataset.

