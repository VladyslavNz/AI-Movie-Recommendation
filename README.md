# Movie Recommendation System Using Neural Networks

This project implements a movie recommendation system based on artificial neural networks. The system recommends movies to users by analyzing their preferences and the content of the movies.

## Features

- Personalized movie recommendations based on user preferences and movie content
- Deep neural network model that uses user and movie embeddings along with bias terms
- Enhanced recommendation explanations that consider:
  - User genre preferences with average ratings
  - Tag analysis using TF-IDF
  - Similar users' ratings and preferences
  - Viewing history patterns
- Visual analytics of user preferences
- Classification of recommendations as familiar or novel content
- Interactive confidence ratings for recommendations (🌟 Strong match, ✅ Good match)
- Expandable detailed explanations for each recommendation
- User preference visualization through interactive charts
- Advanced filtering options for recommendations
- Model retraining capabilities with performance visualization

## Requirements

- **Python 3.11** is required.
- It is **highly recommended** to use a virtual environment to ensure isolated and consistent dependencies.
- Install required Python libraries via `pip`:

```bash
pip install numpy pandas tensorflow matplotlib scikit-learn requests streamlit plotly pytest
```

## DATASET

- **Source:** [MovieLens 20M Dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset?resource=download)
- **Used files:** `movies.csv`, `ratings.csv`, `tags.csv`

**Start Project**

```bash
streamlit run app.py
```

**Start tests**

```
python -m pytest tests/  
```