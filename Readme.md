# Customer Segmentation using K-Means Clustering

A Python-based **Customer Segmentation** project that uses exploratory
data analysis, feature engineering, **K-Means clustering**,
dimensionality reduction with **PCA**, and a **Streamlit** application
to classify customers into different segments.

The project is built around customer demographic, spending,
purchasing-channel, and recency information.

------------------------------------------------------------------------

## 📌 Project Overview

Customer segmentation is the process of grouping customers with similar
characteristics and behaviors.

In this project, customer data is cleaned and transformed, relevant
features are engineered, and **K-Means Clustering** is applied to divide
customers into **6 clusters**.

The project also includes a Streamlit web application where a user can
enter customer details and receive a predicted cluster.

### Main objectives

-   Clean and preprocess customer data
-   Perform exploratory data analysis (EDA)
-   Create useful customer-level features
-   Analyze relationships between customer attributes
-   Select an appropriate number of clusters using the **Elbow Method**
-   Apply **K-Means Clustering**
-   Analyze and interpret cluster characteristics
-   Visualize clusters using **PCA**
-   Save the trained model and scaler using **Joblib**
-   Build an interactive **Streamlit** prediction app

------------------------------------------------------------------------

## 🛠️ Technologies & Tools

  Tool / Library         Purpose
  ---------------------- -----------------------------------------------------
  **Python**             Core programming language
  **Pandas**             Data loading, cleaning, transformation and analysis
  **NumPy**              Numerical operations
  **Matplotlib**         Data visualization
  **Seaborn**            Statistical visualization
  **Scikit-learn**       Scaling, K-Means clustering and PCA
  **Joblib**             Saving and loading trained models
  **Streamlit**          Interactive web application
  **Jupyter Notebook**   EDA, experimentation and model development
  **VS Code**            Development environment
  **Git/GitHub**         Version control and project sharing

------------------------------------------------------------------------

## 📂 Project Structure

``` text
CUSTOMER_SEGMENTATION_PROJECT/
│
├── customer_segmentation.csv     # Customer dataset
├── analysis_model.ipynb           # EDA, feature engineering and clustering
├── segmentation.py                # Streamlit application
├── kmeans_model.pkl               # Trained K-Means model
├── scaler.pkl                     # Fitted StandardScaler
└── README.md                      # Project documentation
```

------------------------------------------------------------------------

## 📊 Dataset & Features

The project works with customer information including demographic and
purchasing behavior.

The final clustering model uses these **7 features**:

``` text
Age
Income
Total_Spending
NumWebPurchases
NumStorePurchases
NumWebVisitsMonth
Recency
```

### Feature descriptions

  -----------------------------------------------------------------------
  Feature                             Description
  ----------------------------------- -----------------------------------
  `Age`                               Customer age

  `Income`                            Customer income

  `Total_Spending`                    Total spending across the selected
                                      product categories

  `NumWebPurchases`                   Number of purchases made through
                                      the web

  `NumStorePurchases`                 Number of purchases made through
                                      stores

  `NumWebVisitsMonth`                 Number of web visits per month

  `Recency`                           Days since the customer's last
                                      purchase
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 🔄 Data Preparation & Feature Engineering

The notebook performs several preprocessing and feature-engineering
steps.

### 1. Missing-value handling

Missing values are checked and removed before analysis.

``` python
df.dropna(inplace=True)
```

### 2. Date conversion

The customer date field is converted into a datetime format.

``` python
df["Dt_Customer"] = pd.to_datetime(
    df["Dt_Customer"],
    dayfirst=True
)
```

### 3. Age calculation

Age is derived from the birth year.

``` python
df["Age"] = 2026 - df["Year_Birth"]
```

### 4. Total children

Children and teenagers in the household are combined.

``` python
df["Total_Children"] = df["Kidhome"] + df["Teenhome"]
```

### 5. Total spending

Spending across multiple product categories is combined into one
feature.

``` python
spend_cols = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds"
]

df["Total_Spending"] = df[spend_cols].sum(axis=1)
```

------------------------------------------------------------------------

## 🔎 Exploratory Data Analysis

EDA is performed to understand customer behavior before applying
clustering.

The analysis includes:

-   Age distribution
-   Income distribution
-   Total spending distribution
-   Income by education level
-   Spending by marital status
-   Correlation analysis
-   Average income by education and marital status
-   Average spending by education
-   Campaign acceptance rate by marital status
-   Average income by age group

### Correlation Analysis

The following variables are analyzed using a correlation matrix:

``` text
Income
Age
Recency
Total_Spending
NumWebPurchases
NumStorePurchases
```

Visualizations are created using **Matplotlib** and **Seaborn**.

------------------------------------------------------------------------

## 🤖 Machine Learning Approach

### Algorithm: K-Means Clustering

K-Means is an **unsupervised machine learning algorithm** that groups
observations into clusters based on their feature similarity.

### Step 1 --- Select features

``` python
features = [
    "Age",
    "Income",
    "Total_Spending",
    "NumWebPurchases",
    "NumStorePurchases",
    "NumWebVisitsMonth",
    "Recency"
]

X = df[features].copy()
```

### Step 2 --- Feature scaling

Because the features have different numerical ranges, **StandardScaler**
is used.

``` python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Step 3 --- Find the number of clusters

The **Elbow Method** is used by calculating WCSS for different values of
`K`.

``` python
wcss = []

for i in range(2, 10):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
```

The resulting WCSS values are visualized to determine a suitable number
of clusters.

### Step 4 --- Train K-Means

The final model uses **6 clusters**.

``` python
kmeans = KMeans(n_clusters=6)

df["Cluster"] = kmeans.fit_predict(X_scaled)
```

------------------------------------------------------------------------

## 📈 Cluster Analysis

After clustering, the average values of the selected features are
calculated for each cluster.

``` python
cluster_summary = df.groupby("Cluster")[features].mean()
```

This allows the characteristics of each customer segment to be compared.

The Streamlit application currently describes the six clusters as:

  Cluster         Segment Description
  --------------- ---------------------------------------------
  **Cluster 0**   High budget, web visitors
  **Cluster 1**   High Spending
  **Cluster 2**   Web Visitors
  **Cluster 3**   High Spending, web visitors, store visitors
  **Cluster 4**   Premium Customers
  **Cluster 5**   Budget Customer

> These labels are descriptive interpretations used by the application;
> the K-Means algorithm itself produces numerical cluster IDs.

------------------------------------------------------------------------

## 📉 PCA Visualization

**Principal Component Analysis (PCA)** is used to reduce the
seven-dimensional feature space to two dimensions for visualization.

``` python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pca_data = pca.fit_transform(X_scaled)

df["PCA1"] = pca_data[:, 0]
df["PCA2"] = pca_data[:, 1]
```

The resulting PCA coordinates are plotted against the assigned clusters
to visually inspect customer segmentation.

------------------------------------------------------------------------

## 💾 Model Saving

The trained K-Means model and scaler are saved using **Joblib**.

``` python
import joblib

joblib.dump(kmeans, "kmeans_model.pkl")
joblib.dump(scaler, "scaler.pkl")
```

These files allow the Streamlit application to reuse the trained model
without retraining it every time.

------------------------------------------------------------------------

# 🌐 Streamlit Application

The project includes an interactive customer segmentation application.

The application loads:

``` python
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
```

Users can enter:

-   Age
-   Income
-   Total Spending
-   Number of Web Purchases
-   Number of Store Purchases
-   Number of Web Visits per Month
-   Recency

The entered data is converted into a DataFrame, scaled using the saved
scaler, and passed to the trained K-Means model.

``` python
input_scaled = scaler.transform(input_data)

cluster = kmeans.predict(input_scaled)[0]
```

The application then displays the predicted customer segment.

------------------------------------------------------------------------

## ▶️ How to Run the Project

### 1. Clone the repository

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd CUSTOMER_SEGMENTATION_PROJECT
```

### 2. Install dependencies

``` bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit jupyter
```

### 3. Run the Streamlit application

If the `streamlit` command is available:

``` bash
streamlit run segmentation.py
```

If your terminal shows **`streamlit: command not found`**, run:

``` bash
python -m streamlit run segmentation.py
```

### 4. Open the application

Streamlit will provide a local URL, normally similar to:

``` text
http://localhost:8501
```

Open that address in your browser.

------------------------------------------------------------------------

## 🧠 Project Workflow

``` text
Customer Dataset
       ↓
Data Cleaning
       ↓
Feature Engineering
       ↓
Exploratory Data Analysis
       ↓
Feature Selection
       ↓
StandardScaler
       ↓
Elbow Method
       ↓
K-Means Clustering
       ↓
Cluster Analysis
       ↓
PCA Visualization
       ↓
Save Model + Scaler
       ↓
Streamlit Application
       ↓
Predicted Customer Segment
```

------------------------------------------------------------------------

## 📌 Key Concepts Demonstrated

This project demonstrates practical experience with:

-   Data Cleaning
-   Data Preprocessing
-   Feature Engineering
-   Exploratory Data Analysis
-   Statistical Visualization
-   Correlation Analysis
-   Feature Scaling
-   Unsupervised Machine Learning
-   K-Means Clustering
-   Elbow Method
-   WCSS
-   PCA
-   Model Persistence
-   Streamlit App Development
-   Python Data Analysis

------------------------------------------------------------------------

## 🚀 Future Improvements

Possible improvements include:

-   Add cluster-wise interactive visualizations to Streamlit
-   Display customer segment characteristics dynamically
-   Add downloadable prediction reports
-   Add more clustering evaluation metrics such as Silhouette Score
-   Compare K-Means with other clustering algorithms
-   Add interactive PCA/cluster plots
-   Deploy the Streamlit application online
-   Add a requirements.txt file
-   Add automated data validation

------------------------------------------------------------------------

## 👨‍💻 Author

**Aditya Singh**

B.Tech CSE --- Data Science

📧 **Email:** adityasinghdmr@gmail.com

🔗 **LinkedIn:** [Linkedin](https://www.linkedin.com/in/aditya-singh-068552267/)

💻 **GitHub:** [GitHub](https://github.com/levyyy1)

------------------------------------------------------------------------

## ⭐ Project Summary

This project demonstrates an end-to-end machine learning workflow for
customer segmentation, starting from raw customer data and exploratory
analysis through feature engineering, K-Means clustering, PCA
visualization, model serialization, and deployment through an
interactive Streamlit application.

If you find this project useful, consider giving the repository a ⭐ on
GitHub.
