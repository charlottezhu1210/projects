# Data Science Portfolio: Charlotte Zhu

Data science and analytics projects in Python.

| Project | Summary | Techniques |
|---|---|---|
| [**Online Retail: Customer Segmentation & Recommender**](online-retail-analysis/) | Segmented 4,383 wholesale customers with RFM + K-means (silhouette 0.47, validated against PAM and Ward), and built a user-based collaborative-filtering recommender tuned on a time-based validation window with NDCG@20, and benchmarked it against popularity and "buy again" baselines on a held-out test window. | Clustering, PCA, TF-IDF, collaborative filtering, NDCG, bootstrap CIs |
| [**Fragrance Market Exploration**](perfume-market-analysis/) | Analyzed 59K perfumes and 2K eBay listings to map launch trends, brand e-commerce positioning, and consumer ratings by accord. Network analysis and Louvain community detection reveal three "sensory families" of accords. | EDA, data cleaning, network analysis, community detection |

Each project folder has its own README, notebook (rendered with outputs) and `requirements.txt`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r online-retail-analysis/requirements.txt -r perfume-market-analysis/requirements.txt
```
