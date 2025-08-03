# Online-Retail-Customer-Segmentation-and-Product-Recommendation-System
This project focuses on leveraging data science techniques to derive actionable business insights from a comprehensive online retail transaction dataset. The core objective was to develop two interconnected and practical tools: a customer segmentation model and a product recommendation system.  
The project began with a rigorous data preprocessing phase, where raw transaction data was cleaned to handle missing values, filter out canceled orders, and prepare the dataset for analysis.

Using the Recency, Frequency, and Monetary (RFM) analysis framework, customers were segmented into distinct groups such as 'High-Value,' 'At-Risk,' and 'Occasional' buyers. This was achieved through a K-Means clustering algorithm, providing the business with a clear understanding of its customer base and enabling targeted marketing strategies.

Additionally, an item-based collaborative filtering approach was implemented to build a robust product recommendation system. By computing a cosine similarity matrix based on customer purchase behavior, the system can dynamically suggest up to five highly similar products for any given item, enhancing the user experience and potentially increasing sales.

The final solution is deployed as a user-friendly Streamlit web application, which loads the pre-trained models and provides an interactive interface for real-time customer segmentation and product recommendations.
