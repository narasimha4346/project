import pandas as pd

# Load the CSV files
customer = pd.read_csv("N:\\zeotap\\customer.csv.csv")
product = pd.read_csv("N:\\zeotap\\product.csv.csv")
transaction = pd.read_csv("N:\\zeotap\\transaction.csv.csv")

# Step 1: Merge customer and transaction data based on CustomerID
customer_transaction = pd.merge(transaction, customer, on='CustomerID', how='inner')

# Step 2: Merge the result with the product data based on ProductID
final_data = pd.merge(customer_transaction, product, on='ProductID', how='inner')

# Step 3: Preview the final merged dataset
print("Merged Data Preview:")
print(final_data.head())

# Step 4: Save the merged dataset to a new CSV file
final_data.to_csv("N:\\zeotap\\merged_data.csv", index=False)

print("Merged data saved to 'merged_data.csv'.")
