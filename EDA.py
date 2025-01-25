import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
customer = pd.read_csv("N:\\zeotap\\customer.csv.csv")
product = pd.read_csv("N:\\zeotap\\product.csv.csv")
transaction = pd.read_csv("N:\\zeotap\\transaction.csv.csv")

# 1. Customer Data Summary
customer_summary = customer.describe(include='all')
print("Customer Data Summary:")
print(customer_summary)

# Save customer summary
customer_summary.to_csv("N:\\zeotap\\Customer_Summary.csv", index=True)

# 2. Product Data Summary
product_summary = product.describe(include='all')
print("\nProduct Data Summary:")
print(product_summary)

# Save product summary
product_summary.to_csv("N:\\zeotap\\Product_Summary.csv", index=True)

# 3. Transaction Data Summary
transaction_summary = transaction.describe()
print("\nTransaction Data Summary:")
print(transaction_summary)

# Save transaction summary
transaction_summary.to_csv("N:\\zeotap\\Transaction_Summary.csv", index=True)

# 4. Check for missing values
missing_customer = customer.isnull().sum()
missing_product = product.isnull().sum()
missing_transaction = transaction.isnull().sum()

print("\nMissing values in Customer Data:")
print(missing_customer)

print("\nMissing values in Product Data:")
print(missing_product)

print("\nMissing values in Transaction Data:")
print(missing_transaction)

# Save missing value information
missing_customer.to_csv("N:\\zeotap\\Missing_Customer.csv", header=True)
missing_product.to_csv("N:\\zeotap\\Missing_Product.csv", header=True)
missing_transaction.to_csv("N:\\zeotap\\Missing_Transaction.csv", header=True)

# 5. Visualizations
# Distribution of customers by region
plt.figure(figsize=(10, 6))
sns.countplot(data=customer, x='Region')
plt.title('Number of Customers by Region')
plt.xticks(rotation=45)
plt.savefig("N:\\zeotap\\Customers_By_Region.png")
plt.show()

# Distribution of products by category
plt.figure(figsize=(10, 6))
sns.countplot(data=product, x='Category')
plt.title('Product Distribution by Category')
plt.xticks(rotation=45)
plt.savefig("N:\\zeotap\\Products_By_Category.png")
plt.show()

# Price distribution of products
plt.figure(figsize=(10, 6))
sns.histplot(product['Price'], kde=True)
plt.title('Price Distribution of Products')
plt.savefig("N:\\zeotap\\Price_Distribution.png")
plt.show()

# Transaction Value distribution
plt.figure(figsize=(10, 6))
sns.histplot(transaction['TotalValue'], kde=True)
plt.title('Transaction Value Distribution')
plt.savefig("N:\\zeotap\\Transaction_Value_Distribution.png")
plt.show()

# 6. Correlation heatmap (only for numeric columns in product data)
numeric_product = product.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_product.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Product Data')
plt.savefig("N:\\zeotap\\Product_Correlation_Heatmap.png")
plt.show()
