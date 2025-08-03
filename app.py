import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Load saved models and data ---
try:
    with open('kmeans_model.pkl', 'rb') as f:
        kmeans_model = pickle.load(f)
    with open('rfm_scaler.pkl', 'rb') as f:
        rfm_scaler = pickle.load(f)
    with open('product_similarity.pkl', 'rb') as f:
        product_similarity_df = pickle.load(f)
    with open('stock_code_to_desc.pkl', 'rb') as f:
        stock_code_to_desc = pickle.load(f)
    with open('desc_to_stock_code.pkl', 'rb') as f:
        desc_to_stock_code = pickle.load(f)
    with open('cluster_map.pkl', 'rb') as f:
        cluster_map = pickle.load(f)
    
    st.success("Models and data loaded successfully!")
except FileNotFoundError:
    st.error("Error: Could not find the necessary model files.")
    st.info("Please run the provided Colab code first to generate and save the models.")
    st.stop()


# --- Recommendation System Function ---
def get_recommendations(product_name, similarity_matrix, desc_to_stock_code_map, stock_code_to_desc_map, n=5):
    """
    Finds and returns product recommendations based on similarity.
    """
    if product_name not in desc_to_stock_code_map:
        return ["Product not found in the database. Please try another name."]
    
    stock_code = desc_to_stock_code_map[product_name]
    
    if stock_code not in similarity_matrix.index:
        return ["Product not found in the similarity matrix."]
    
    similar_products = similarity_matrix[stock_code].sort_values(ascending=False)
    top_n_recs = similar_products[1:n+1].index
    
    recommended_products = [stock_code_to_desc_map.get(sc, "Unknown Product") for sc in top_n_recs]
    return recommended_products

# --- Prediction Function for Customer Segmentation ---
def predict_customer_cluster(recency, frequency, monetary, scaler, kmeans, cluster_map):
    """
    Predicts the customer cluster based on RFM values.
    """
    # Create a DataFrame with the input values
    input_data = pd.DataFrame([[recency, frequency, monetary]], columns=['Recency', 'Frequency', 'Monetary'])
    
    # Log transform and scale the input data
    input_log = input_data.apply(lambda x: np.log1p(x))
    input_scaled = scaler.transform(input_log)
    
    # Predict the cluster
    predicted_cluster = kmeans.predict(input_scaled)[0]
    
    # Map the cluster number to the segment label
    return cluster_map.get(predicted_cluster, "Unknown Segment")

# --- Streamlit App UI ---
st.title("Online Retail Analytics App")
st.markdown("""
This app provides a customer segmentation analysis and a product recommendation system
based on the online retail dataset.
""")

# --- Tab-based layout ---
tab1, tab2 = st.tabs(["Product Recommendation", "Customer Segmentation"])

with tab1:
    st.header("🎯 Product Recommendation Module")
    st.markdown("""
    Enter a product name to get 5 recommendations based on customer purchase history.
    """)
    
    # Get a list of unique product names for the dropdown
    unique_products = sorted(list(desc_to_stock_code.keys()))
    
    product_name_input = st.selectbox(
        "Select a product from the list:",
        unique_products
    )
    
    if st.button("Get Recommendations"):
        if product_name_input:
            with st.spinner("Generating recommendations..."):
                recommendations = get_recommendations(
                    product_name_input, 
                    product_similarity_df, 
                    desc_to_stock_code, 
                    stock_code_to_desc, 
                    n=5
                )
            
            st.subheader("Recommended Products:")
            if recommendations:
                for rec in recommendations:
                    st.write(f"- {rec}")
        else:
            st.warning("Please select a product name.")

with tab2:
    st.header("🎯 Customer Segmentation Module")
    st.markdown("""
    Enter the RFM (Recency, Frequency, Monetary) values for a customer
    to predict their cluster segment.
    """)
    
    # Input fields for Recency, Frequency, and Monetary
    recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=365, value=30)
    frequency = st.number_input("Frequency (number of purchases)", min_value=1, value=5)
    monetary = st.number_input("Monetary (total spend)", min_value=0.0, value=250.0)
    
    if st.button("Predict Customer Segment"):
        with st.spinner("Predicting segment..."):
            predicted_segment = predict_customer_cluster(
                recency, 
                frequency, 
                monetary, 
                rfm_scaler, 
                kmeans_model, 
                cluster_map
            )
        st.subheader("Predicted Customer Segment:")
        st.success(f"**{predicted_segment}**")