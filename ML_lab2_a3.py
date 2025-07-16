import pandas as pd
import statistics
import matplotlib.pyplot as plt

# Load the Excel file (NOT CSV)
file_path = r"C:\Users\bramj\OneDrive\Pictures\Documents\irctc.xlsx"
df = pd.read_excel(file_path)

# Clean column names if needed (optional)
df.columns = df.columns.str.strip()

# ─── Task 1: Calculate Mean and Variance of the 'Price' column (assumed as column D) ───
price_data = df.iloc[:, 3]  # Column D is index 3
mean_price = statistics.mean(price_data)
variance_price = statistics.variance(price_data)

print(f"Mean of Price (Population): {mean_price}")
print(f"Variance of Price: {variance_price}")

# ─── Task 2: Wednesday Price Sample Mean ───
# First, convert date to datetime if not already
df['Date'] = pd.to_datetime(df['Date'])
df['Day'] = df['Date'].dt.day_name()

wednesday_prices = price_data[df['Day'] == 'Wednesday']
mean_wednesday = statistics.mean(wednesday_prices)

print(f"Sample Mean of Wednesday Prices: {mean_wednesday}")
print(f"Difference from Population Mean: {mean_wednesday - mean_price}")

# ─── Task 3: April Prices Sample Mean ───
april_prices = price_data[df['Date'].dt.month == 4]
mean_april = statistics.mean(april_prices)

print(f"Sample Mean of April Prices: {mean_april}")
print(f"Difference from Population Mean: {mean_april - mean_price}")

# ─── Task 4: Probability of Making a Loss (Chg% < 0) ───
chg_data = df.iloc[:, 8]  # Column I is index 8
chg_data = pd.to_numeric(chg_data, errors='coerce')  # Convert to numeric, handle non-numbers

loss_count = sum(map(lambda x: x < 0, chg_data.dropna()))
total_count = chg_data.dropna().count()
prob_loss = loss_count / total_count

print(f"Probability of making a loss (Chg% < 0): {prob_loss:.4f}")

# ─── Task 5: Probability of Making Profit on Wednesday ───
wednesday_chg = chg_data[df['Day'] == 'Wednesday']
profit_wednesday = sum(wednesday_chg > 0)
total_wednesday = wednesday_chg.count()
prob_profit_wed = profit_wednesday / total_wednesday

print(f"Probability of profit on Wednesday: {prob_profit_wed:.4f}")

# ─── Task 6: Conditional Probability P(Profit | Wednesday) ───
# Already computed as Task 5 — no change needed
print(f"Conditional Probability P(Profit | Wednesday): {prob_profit_wed:.4f}")

# ─── Task 7: Scatter Plot of Chg% vs Day of Week ───
plt.figure(figsize=(10, 5))
plt.scatter(df['Day'], chg_data, alpha=0.7, color='green')
plt.title('Chg% vs Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Change Percentage (Chg%)')
plt.grid(True)
plt.show()