library(tidyverse)
library(cluster)
library(sqldf)
library(readxl)
library(showtext)
library(naniar)
library(factoextra) 
library(microbenchmark)
library(dendextend)
library(patchwork)

# Part I. Data Processing
# 1) File reading
transactions <- read_excel("online_retail_II.xlsx")
head(transactions)

# 2) Missing value
colSums(is.na(transactions))
colMeans(is.na(transactions))
missing_summary <- miss_var_summary(transactions)
ggplot(missing_summary, aes(x = reorder(variable, -n_miss), y = n_miss)) +
  geom_col() +
  labs(title = "Missing Values by Variable", x = "Variable", y = "Number of Missing Values") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1),
        plot.title = element_text(hjust = 0.5)) 
transactions <- na.omit(transactions)

# 3) Factor conversion
transactions <- transactions %>%
  mutate(across(c(Invoice, StockCode, Description, Country, `Customer ID`), as.factor))



# Part II. Variable Description
str(transactions)
summary(transactions$Price)
summary(transactions$Quantity)
summary(transactions$InvoiceDate)

# 1) Quantity 
p1 <- ggplot(transactions, aes(x = Quantity)) +
  geom_histogram(bins = 30, fill = "steelblue", alpha = 0.7) +
  scale_x_continuous(limits = quantile(transactions$Quantity, c(0.01, 0.99))) +
  labs(title = "Quantity Distribution",
       subtitle = "Excluding extreme values") +
  theme_minimal()

p2 <- ggplot(transactions, aes(y = Quantity)) +
  geom_boxplot(fill = "lightblue", alpha = 0.7) +
  scale_y_log10() +
  labs(subtitle = "Log distribution of all data",
       y = "Quantity (log scale)") +
  theme_minimal()

p1 + p2 + plot_layout(ncol = 2, heights = c(1, 0.5))

# 2) Price
p3 <- ggplot(transactions, aes(x = Price)) +
  geom_histogram(bins = 30, fill = "salmon", alpha = 0.7) +
  scale_x_continuous(limits = quantile(transactions$Quantity, c(0.01, 0.99))) +
  labs(title = "Price Distribution",
       subtitle = "Excluding extreme values") +
  theme_minimal()

p4 <- ggplot(transactions, aes(y = Price)) +
  geom_boxplot(fill = "peachpuff") +
  scale_y_log10(labels = scales::label_number()) +
  labs(subtitle = "Log distribution of all data",
       y = "Price (log scale)") +
  theme_minimal()

p3 + p4 + plot_layout(ncol = 2, heights = c(1, 0.5))

# 3) Date
transactions$InvoiceDate <- as.Date(transactions$InvoiceDate)
ggplot(transactions, aes(x = InvoiceDate)) +
  geom_histogram(binwidth = 30, fill = "skyblue") +
  scale_x_date(date_breaks = "1 month", date_labels = "%Y-%m") +
  labs(title = "Invoice Date Distribution",
       x = "Date", 
       y = "Transaction Count") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))



# Part III. Descriptive Analysis
# 1) Transaction Type Distribution
by_type <- sqldf("
      SELECT CASE WHEN Invoice LIKE 'C%' THEN 'Cancelled' ELSE 'Normal' END AS order_type, 
             COUNT(DISTINCT Invoice) AS order_cnt
      FROM transactions
      GROUP BY CASE WHEN Invoice LIKE 'C%' THEN 'Cancelled' ELSE 'Normal' END
")

ggplot(data = by_type) +
  geom_bar(mapping = aes(x = order_type, y = order_cnt, fill = order_type), stat = "identity") +
  scale_fill_manual(values = c("Normal" = "skyblue", "Cancelled" = "salmon")) +
  scale_y_continuous(labels = scales::comma) +
  labs(title = "Transaction Type Distribution", 
       x = "Transaction Type", y = "Transaction Count") +
  theme_minimal() +
  theme(legend.position = "none", plot.title = element_text(hjust = 0.5))

# 2) Product Type Distribution
by_product <- sqldf("
      SELECT CASE 
               WHEN StockCode = 'M' THEN 'Manual'
               WHEN StockCode IN ('TEST001','TEST002') THEN 'Test Product'
               WHEN StockCode IN ('ADJUST','ADJUST2') THEN 'Adjustment'
               WHEN StockCode = 'POST' THEN 'Postage' 
               ELSE 'Regular Product' END AS product_type,
             SUM(Quantity * Price) AS Amount
      FROM transactions
      GROUP BY CASE 
               WHEN StockCode = 'M' THEN 'Manual'
               WHEN StockCode IN ('TEST001','TEST002') THEN 'Test Product'
               WHEN StockCode IN ('ADJUST','ADJUST2') THEN 'Adjustment'
               WHEN StockCode = 'POST' THEN 'Postage' END
      ORDER BY Amount DESC
")

ggplot(data = by_product) +
  geom_bar(mapping = aes(x = reorder(product_type, -Amount), y = Amount, fill = product_type), stat = "identity") +
  scale_y_continuous(labels = scales::comma) +
  scale_fill_brewer(palette = 1) +
  labs(title = "Product Type Distribution", 
       x = "Product Type", y = "Total Amount") +
  theme_minimal() +
  theme(legend.position = "none", plot.title = element_text(hjust = 0.5))

# 3) Customer Spending Distribution
by_customer <- transactions %>%
  group_by(`Customer ID`) %>%
  summarise(
    transaction_cnt = n_distinct(Invoice),
    total_spent = sum(Quantity * Price),
    avg_spent = total_spent / transaction_cnt,
    .groups = "drop"
  )
by_customer$sign <- ifelse(by_customer$total_spent >= 0, "Positive Amount", "Negative Amount")

ggplot(by_customer, aes(x = total_spent, fill = sign)) +
  geom_histogram(aes(color = sign), position = "identity", bins = 30, alpha = 0.7) +
  facet_grid(sign ~ ., scales = "free_y") +
  scale_fill_manual(values = c("Positive Amount" = "#3498DB", "Negative Amount" = "#E74C3C")) +
  scale_color_manual(values = c("Positive Amount" = "#1A5276", "Negative Amount" = "#922B21")) +
  scale_x_continuous(
    trans = scales::pseudo_log_trans(sigma = 1000, base = 10),
    breaks = c(-100000, -10000, -1000, 0, 1000, 10000, 100000, 1000000),
    labels = scales::comma
  ) +
  labs(title = "Customer Spending Distribution", x = "Total Spending", y = "Customer Count", fill = "Amount Type") +
  theme_minimal() +
  theme(legend.position = "none", panel.spacing = unit(1.5, "lines"), plot.title = element_text(hjust = 0.5))

# 4) Customer Country Distribution
by_country <- transactions %>%
  group_by(Country) %>%
  summarise(
    total_spent = sum(Quantity * Price),
    .groups = "drop"
  ) %>%
  arrange(desc(total_spent))

ggplot(head(by_country, 15)) +
  geom_col(aes(x = reorder(Country, total_spent), y = total_spent), fill = "skyblue") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Top 15 Markets by Spending", x = NULL, y = "Total Transaction Amount") +
  theme(plot.title = element_text(hjust = 0.5))

# Part IV. RFM Clustering
# 1) feature engineering
transactions$days_ago <- as.numeric(as.Date("2010-12-10") - as.Date(transactions$InvoiceDate))
customers <- sqldf("
    SELECT `Customer ID`,
           MIN(days_ago) AS recency,
           COUNT(DISTINCT Invoice) AS frequency,
           SUM(Quantity * Price) AS monetary
    FROM transactions
    GROUP BY `Customer ID`
")

# Remove Customer ID
cluster_data <- select(customers, -`Customer ID`)
row.names(cluster_data) <- customers$`Customer ID`

# Apply pseudo-log transformation to Monetary
pseudo_log <- function(x) {
  sign(x) * log1p(abs(x))
}
cluster_data$monetary <- pseudo_log(cluster_data$monetary)
ggplot(cluster_data) +
  geom_histogram(aes(x = monetary), fill = "steelblue") +
  labs(title = "Monetary Value Distribution", x = "Pseudo-log(Monetary)", y = "Customer Count") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# Z-score standardization for all metrics
cluster_data <- scale(cluster_data)

# 2) Determine optimal k using elbow method
set.seed(33)
wcss_result <- sapply(1:10, function(k){kmeans(cluster_data, centers = k, nstart = 5)$tot.withinss})
wcss_df <- data.frame(cluster = 1:10, wcss = wcss_result)
ggplot(wcss_df, aes(x = cluster, y = wcss)) +
  geom_line() +
  geom_point() +
  labs(title = "Elbow Method", x = "Number of Clusters", y = "Within-Cluster Sum of Squares (WCSS)") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5)) 

# 3) Perform clustering with selected k
set.seed(33)
k <- 5
km_result <- kmeans(cluster_data, centers = k, nstart = 20)
cluster_data <- as.data.frame(cluster_data) %>% 
  mutate(cluster = as.factor(km_result$cluster))
segment_result <- cluster_data %>%
  group_by(cluster) %>%
  summarise(
    avg_Recency = mean(recency), 
    avg_Frequency = mean(frequency), 
    avg_Monetary = mean(monetary),
    ttl_Customer = n()
  ) %>%
  arrange(cluster) 
print(segment_result)

# 4) K-means clustering evaluation
cluster_data_std <- cluster_data[, c("recency", "frequency", "monetary")]
km_sil <- silhouette(km_result$cluster, dist(cluster_data_std))
km_sil_avg <- summary(km_sil)$avg.width
km_visual <- fviz_cluster(km_result, data = cluster_data_std, 
                          ellipse.type = "norm", 
                          main = "K-means Clustering Results",
                          palette = "uchicago",
                          pointsize = 0,    
                          shape = 0,           
                          alpha = 0.2,
                          labelsize = 0) +
  annotate("text", x = Inf, y = -Inf, hjust = 1, vjust = -0.5,
           label = sprintf("Silhouette Score: %.3f", km_sil_avg),
           size = 4, color = "black") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))
km_visual

# 5) K-medoids evaluation
set.seed(33)
pam_result <- pam(cluster_data_std, k = k)
pam_sil <- silhouette(pam_result)
pam_sil_avg <- summary(pam_sil)$avg.width
pam_visual <- fviz_cluster(pam_result, data = cluster_data_std, 
                           ellipse.type = "euclid", 
                           main = "K-medoids Clustering Results",
                           palette = "uchicago",
                           pointsize = 0,
                           shape = 0,    
                           alpha = 0.2,
                           labelsize = 0) +
  annotate("text", x = Inf, y = -Inf, hjust = 1, vjust = -0.5,
           label = sprintf("Silhouette Score: %.3f", pam_sil_avg),
           size = 4, color = "black") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))
pam_visual

# 6) Hierarchical clustering evaluation
set.seed(33)
dist_matrix <- dist(cluster_data_std, method = "euclidean")
hc <- hclust(dist_matrix, method = "ward.D2")
hc_cut <- cutree(hc, k = k)
sil_hc <- silhouette(hc_cut, dist_matrix)
hc_sil_avg <- mean(sil_hc[, "sil_width"])
hc_for_plot <- list(
  data = cluster_data_std,
  cluster = hc_cut
)
hc_visual <- fviz_cluster(hc_for_plot,    
                          ellipse.type = "convex", 
                          main = "Hierarchical Clustering Results",
                          palette = "uchicago",
                          pointsize = 0,  
                          alpha = 0.2,
                          labelsize = 0) +
  annotate("text", x = Inf, y = -Inf, hjust = 1, vjust = -0.5,
           label = sprintf("Silhouette Score: %.3f", hc_sil_avg),
           size = 4.5, fontface = "bold", color = "#333333") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))
hc_visual
