import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, Normalizer
from sklearn.feature_extraction.text import TfidfTransformer

class RecommenderSystem:
    def __init__(self, data_path, test_size=0.2, top_n = 100):
        """Initialize the recommender system"""
        self.data_path = data_path
        self.test_size = test_size
        self.top_n = top_n
        self.data = None
        self.train_data = None
        self.test_data = None
        self.user_item_matrix = None
        self.user_features = None
        self.user_similarity = None
        self.item_desc_map = None
        self.user_ids = None
        self.user_country_map = None

    def preprocess_data(self):
        """Preprocess data"""
        # Load and clean data
        self.data = pd.read_excel(self.data_path)
        self.data = self.data.dropna(subset=['Customer ID', 'StockCode', 'Description'])
        self.data['Customer ID'] = self.data['Customer ID'].astype(int)
        self.data = self.data[(self.data['Quantity'] > 0) & (self.data['Price'] > 0)]
        self.data['Amount'] = self.data['Quantity'] * self.data['Price']
        self.data['InvoiceDate'] = pd.to_datetime(self.data['InvoiceDate'])
        
        # Create mappings for item descriptions and user countries
        self.item_desc_map = self.data.drop_duplicates('StockCode').set_index('StockCode')['Description'].to_dict()
        self.user_country_map = self.data.groupby('Customer ID')['Country'].first().to_dict()
        
        # Filter out special stock codes
        exclude_codes = ['M', 'ADJUST', 'ADJUST2', 'POST']
        self.data = self.data[~self.data['StockCode'].isin(exclude_codes)]
        
        return self.data

    def split_data(self):
        """Split data into train and test sets chronologically"""
        # Split based on date quantile
        cutoff_date = self.data['InvoiceDate'].quantile(1 - self.test_size)
        self.train_data = self.data[self.data['InvoiceDate'] < cutoff_date]
        self.test_data = self.data[self.data['InvoiceDate'] >= cutoff_date]
        
        # Ensure test users exist in training data
        train_users = set(self.train_data['Customer ID'].unique())
        self.test_data = self.test_data[self.test_data['Customer ID'].isin(train_users)]
        return self.train_data, self.test_data

    def build_user_features(self):
        """Build user feature matrix"""
        # Behavioral features: User-item consumption matrix
        self.user_item_matrix = self.train_data.pivot_table(
            index='Customer ID', columns='StockCode', values='Amount', fill_value=0
        )
        self.user_ids = self.user_item_matrix.index.tolist()
        user_item_sparse = csr_matrix(self.user_item_matrix.values)
        
        # Apply TF-IDF to reduce impact of high-consumption items
        tfidf = TfidfTransformer()
        user_item_sparse = csr_matrix(self.user_item_matrix.values)
        user_item_tfidf = tfidf.fit_transform(user_item_sparse)
        
        # Normalize features
        normalizer = Normalizer(norm='l2')
        user_item_normalized = normalizer.fit_transform(user_item_tfidf)

        # Geographic features: One-hot encoded countries
        user_country = pd.DataFrame({
            'Customer ID': self.user_ids,
            'Country': [self.user_country_map.get(uid, 'Unknown') for uid in self.user_ids]
        })
        encoder = OneHotEncoder()
        country_features = encoder.fit_transform(user_country[['Country']])
        country_normalized = normalizer.fit_transform(country_features)

        # Combine features with weighted importance
        self.user_features = hstack([
            user_item_normalized * 0.8,  # 80% weight to behavior
            country_normalized * 0.2     # 20% weight to geography
        ])

        return self.user_features

    def compute_similarity(self):
        """Compute user similarity matrix"""
        # Calculate cosine similarity
        self.user_similarity = cosine_similarity(self.user_features)
        
        # Apply exponential transform to enhance discriminative power
        self.user_similarity = np.exp(self.user_similarity * 3)
    
        # Set self-similarity to zero
        np.fill_diagonal(self.user_similarity, 0)

        return self.user_similarity

    def recommend_items(self, user_id, k=15):
        """Generate recommendations for a user"""
        if user_id not in self.user_ids:
            return []
        
        user_idx = self.user_ids.index(user_id)
        
        # 1. Vectorized selection of top-k similar users
        similarity_row = self.user_similarity[user_idx]
        similar_indices = np.argsort(similarity_row)[-k:][::-1]
        sim_users_items = self.user_item_matrix.iloc[similar_indices]
        sim_weights = similarity_row[similar_indices].reshape(-1, 1)
        
        # 2. Vectorized normalization of purchase amounts
        max_per_user = np.max(sim_users_items.values, axis=1, keepdims=True)
        max_per_user = np.where(max_per_user == 0, 1, max_per_user) 
        normalized_amounts = sim_users_items.values / max_per_user
        
        # 3. Vectorized score calculation
        item_scores = np.sum(normalized_amounts * sim_weights, axis=0)
        
        # 4. Select top-N recommendations
        top_indices = np.argsort(item_scores)[::-1][:self.top_n]
        
        # 5. Format recommendation results
        recommendations = []
        for idx in top_indices:
            stock_code = self.user_item_matrix.columns[idx]
            score = item_scores[idx]
            desc = self.item_desc_map.get(stock_code, "Unknown")
            recommendations.append((stock_code, score, desc))
        
        return recommendations

    def evaluate(self, k_values):
        """Evaluate recommendation system performance"""
        test_users = self.test_data['Customer ID'].unique()
        results = {}
        
        for k in k_values:
            precision_sum = 0
            recall_sum = 0
            f1_sum = 0
            user_count = 0
            
            # Evaluate for each test user
            for user_id in test_users:
                # Generate recommendations
                recommendations = self.recommend_items(user_id, k=k)
                recommended_items = [item[0] for item in recommendations]
                recommended_count = len(recommended_items)
                
                # Get actual purchased items
                actual_items = set(
                    self.test_data[self.test_data['Customer ID'] == user_id]['StockCode'].unique()
                )
                actual_count = len(actual_items)
                
                # Calculate evaluation metrics
                relevant_items = set(recommended_items).intersection(actual_items)
                relevant_count = len(relevant_items)
                precision = relevant_count / recommended_count if recommended_count > 0 else 0
                recall = relevant_count / actual_count if actual_count > 0 else 0
                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    f1 = 0

                # Aggregate metrics
                precision_sum += precision
                recall_sum += recall
                f1_sum += f1
                user_count += 1
            
            # Calculate average metrics
            results[k] = {
                'precision': precision_sum / user_count,
                'recall': recall_sum / user_count,
                'f1score': f1_sum / user_count
            }
                
        return results

    def plot_metrics(self, metrics):
        """Plot metric trends over K values"""
        # Configure plot for Chinese characters (kept as is)
        plt.rcParams["font.family"] = ["Heiti TC"]
        
        # Extract metrics data
        k_values = sorted(metrics.keys())
        precision = [metrics[k]['precision'] for k in k_values]
        recall = [metrics[k]['recall'] for k in k_values]
        f1 = [metrics[k]['f1score'] for k in k_values]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, precision, 'o-', label='Precision', color='#5DA5DA', linewidth=2)
        plt.plot(k_values, recall, 's-', label='Recall', color='#FAA43A', linewidth=2)
        plt.plot(k_values, f1, '^-', label='F1-Score', color='#60BD68', linewidth=2)
        
        # Add plot elements
        plt.title('Recommendation System Metric Trends', fontsize=16)
        plt.xlabel('K (Number of Similar Users)', fontsize=14)
        plt.ylabel('Metric Value', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.xticks(k_values, rotation=45)
        plt.tight_layout()
        
        plt.savefig('recommendation_metrics.png', dpi=300)
        plt.show()

    def run_pipeline(self):
        """Run the full recommendation system workflow"""
        print("Starting data preprocessing...")
        self.preprocess_data()
        
        print("Splitting into train and test sets...")
        self.split_data()
        
        print("Building user feature matrix...")
        self.build_user_features()
        
        print("Calculating user similarity...")
        self.compute_similarity()
        
        print("Evaluating system performance...")
        metrics = self.evaluate(k_values=range(11, 31))
        
        # Generate visualization
        print("Generating metrics visualization...")
        self.plot_metrics(metrics)
        
        # Output best K metrics
        k_values = sorted(metrics.keys())
        max_f1_idx = np.argmax([metrics[k]['f1score'] for k in k_values])
        best_k = k_values[max_f1_idx]
        best_metrics = metrics[best_k]
        
        print(f"\nOptimal K value: {best_k}")
        print(f"Precision: {best_metrics['precision']:.4f}")
        print(f"Recall: {best_metrics['recall']:.4f}")
        print(f"F1-Score: {best_metrics['f1score']:.4f}")
        
        # Generate example recommendations
        sample_user = self.user_ids[0]
        print(f"\nSample recommendations for user {sample_user} (K={best_k}):")
        recommendations = self.recommend_items(sample_user, k=best_k)
        
        for i, (item, score, desc) in enumerate(recommendations, 1):
            print(f"{i}. {desc} (Score: {score:.4f})")
            
        return metrics, best_k

if __name__ == "__main__":
    # Run the recommender system
    data_path = "online_retail_II.xlsx"
    recommender = RecommenderSystem(data_path)
    metrics, best_k = recommender.run_pipeline()