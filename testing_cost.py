import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n_msoas = 10
n_periods = 20
mean_price = 3000
std_price = 100
mean_buildable_area = 10000  # hectares
std_buildable_area = 2000
base_cost = 3000
initial_base_cost = base_cost  # Store initial value
gamma = 0.005
delta = 0.1  # 10% profit margin required
price_elasticity = 1.8  # Price elasticity of supply

# Density constraints
initial_density_threshold = 15
max_density_threshold = 200
max_density_increase = 5

# Initialize MSOA data
msoas = pd.DataFrame({
    'msoa_id': range(1, n_msoas + 1),
    'price': lognorm.rvs(s=0.3, scale=mean_price, size=n_msoas),
    'buildable_area': np.maximum(np.random.normal(mean_buildable_area, std_buildable_area, n_msoas), 10),
    'initial_density': np.concatenate([
        np.random.uniform(1, 5, n_msoas // 3),      # rural
        np.random.uniform(5, 15, n_msoas // 3),     # suburban
        np.random.uniform(15, 50, n_msoas // 3)     # urban
    ])[np.random.choice([0, 1, 2], size=n_msoas)],
})

# Calculate existing homes after initializing msoas
msoas['existing_homes'] = msoas['initial_density'] * msoas['buildable_area']
msoas['current_homes'] = msoas['existing_homes']
msoas['cost'] = msoas['current_homes'] * base_cost  # Track cost

# Function to calculate density multiplier
def calculate_density_multiplier(current_density, initial_density, max_density):
    """
    Calculate a density multiplier that's bounded between 1 and 2
    based on how close current density is to maximum density
    """
    # Normalize density to be between 0 and 1
    normalized_density = (current_density - initial_density) / (max_density - initial_density)
    # Bound between 0 and 1
    normalized_density = np.clip(normalized_density, 0, 3)
    # Scale to 1-2 range
    return 1 + normalized_density

# Function to calculate optimal homes considering bounded density effects
def calculate_optimal_homes(row, total_homes_national, previous_total_new_homes):
    """
    Calculate optimal new homes considering bounded density effects
    """
    # Calculate current density
    current_density = row['current_homes'] / row['buildable_area']
    
    # Calculate density multiplier
    density_multiplier = calculate_density_multiplier(
        current_density,
        row['initial_density'],
        max_density_threshold
    )
    
    # Check if the price is sufficient to build new homes
    if row['price'] <= base_cost * (1 + delta):
        return 0  # No new homes if price is too low

    # Calculate optimal new homes based on price and base cost
    optimal_homes = (row['price'] - base_cost) / (gamma * base_cost * density_multiplier * 2)

    # Apply density constraints
    max_allowed_homes = min(
        row['buildable_area'] * max_density_threshold - row['current_homes'],
        row['existing_homes'] * max_density_increase - (row['current_homes'] - row['existing_homes'])
    )

    # Ensure optimal_homes is non-negative and does not exceed max_allowed_homes
    optimal_homes = max(0, min(optimal_homes, max_allowed_homes))

    return optimal_homes

# Placeholder for update_prices function
def update_prices(total_new_homes, total_homes):
    # Implement your price adjustment logic here
    return 1 - (total_new_homes / total_homes) * price_elasticity if total_homes > 0 else 1

# Track development over time
history = pd.DataFrame(columns=['period', 'total_homes', 'avg_price', 'homes_built', 'n_developments', 'total_cost', 'avg_density', 'avg_cost', 'std_cost', 'std_price'])

# Store initial prices and costs
initial_prices = msoas['price'].copy()
initial_costs = msoas['cost'].copy()

# Define average rate of building (1% of initial homes per year)
initial_total_homes = msoas['existing_homes'].sum()
average_building_rate = initial_total_homes * 0.01   # 1% of initial homes

# Initialize previous period's total new homes
previous_total_new_homes = 0

# Initialize a list to track base cost over time
base_cost_history = []

# Initialize a list to track costs over time
cost_history = []

# Run simulation
for period in range(n_periods):
    total_homes = msoas['current_homes'].sum()
    
    # Calculate new homes for the current period
    msoas['new_homes'] = msoas.apply(lambda x: calculate_optimal_homes(x, total_homes, previous_total_new_homes), axis=1)
    
    # Update home counts
    msoas['current_homes'] += msoas['new_homes']
    
    # Calculate current density
    msoas['current_density'] = msoas['current_homes'] / msoas['buildable_area']
    
    # Calculate the current rate of building
    current_building_rate = msoas['new_homes'].sum()
    
    # Calculate current building rate as percentage of initial homes
    current_rate = current_building_rate / total_homes
    
    # Calculate how much current rate exceeds 2%
    percentage_change = max(0, current_rate - 0.005)

    # Adjust base cost from initial value based on percentage change
    base_cost = initial_base_cost * (1 + 0.5 * percentage_change)  # Increase base cost by 0.5% for every 1% above 2%

    # Calculate costs for each MSOA based on the provided formula
    msoas['cost'] = base_cost * (1 + gamma * msoas['current_density'] * msoas['new_homes']) 

    # Calculate average density, average cost, and average price
    avg_density = msoas['current_density'].mean()
    avg_cost = msoas['cost'].mean()  # Calculate average cost
    avg_price = msoas['price'].mean()  # Calculate average price

    # Calculate standard deviation of costs and prices
    std_cost = msoas['cost'].std()  # Standard deviation of costs
    std_price = msoas['price'].std()  # Standard deviation of prices

    # Update prices based on new supply
    total_new_homes = msoas['new_homes'].sum()
    if total_new_homes > 0:
        price_adjustment = update_prices(total_new_homes, total_homes)
        msoas['price'] *= price_adjustment
    
    # Record history
    history.loc[period] = {
        'period': period,
        'total_homes': msoas['current_homes'].sum(),
        'avg_price': avg_price,
        'homes_built': total_new_homes,
        'n_developments': (msoas['new_homes'] > 0).sum(),
        'total_cost': msoas['cost'].sum(),  # Track total cost
        'avg_density': avg_density,  # Track average density
        'avg_cost': avg_cost,  # Track average cost
        'std_cost': std_cost,  # Track standard deviation of costs
        'std_price': std_price  # Track standard deviation of prices
    }
    
    # Stop if no more development
    if total_new_homes == 0:
        print(f"Development stopped after {period} periods")
        break
    
    # Update previous total new homes for the next iteration
    previous_total_new_homes = total_new_homes

    # Track the base cost
    base_cost_history.append(base_cost)  # Store the current base cost

    # Track costs
    cost_history.append(msoas['cost'].sum())  # Sum of costs for all MS

    # Inside your simulation loop, after calculating all the values for the period:
    print(f"\nPeriod {period}:")
    print(f"New homes per MSOA: {(total_new_homes / len(msoas)):,.1f}")
    print(f"Current building rate: {(current_building_rate / initial_total_homes * 100):,.1f}% of initial homes")
    print(f"Average homes per MSOA: {msoas['current_homes'].mean():,.1f}")
    print(f"Base cost: £{base_cost:,.2f}")
    print(f"Average cost per MSOA: £{avg_cost:,.2f}")
    print(f"Average price per MSOA: £{avg_price:,.2f}")
    print(f"Average density per MSOA: {avg_density:.2f} homes/hectare")
    print("-" * 50)  # Separator line for readability

# Create visualizations
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 18))

# Total homes over time
ax1.plot(history['period'], history['total_homes'])
ax1.set_title('Total Homes Over Time')
ax1.set_xlabel('Period')
ax1.set_ylabel('Total Homes')

# Average price over time
ax2.plot(history['period'], history['avg_price'])
ax2.set_title('Average Price Over Time')
ax2.set_xlabel('Period')
ax2.set_ylabel('Price per sqm (£)')

# New homes built per period
ax3.bar(history['period'], history['homes_built'])
ax3.set_title('New Homes Built per Period')
ax3.set_xlabel('Period')
ax3.set_ylabel('New Homes')

# Number of active developments per period
ax4.bar(history['period'], history['n_developments'])
ax4.set_title('Number of Active Developments')
ax4.set_xlabel('Period')
ax4.set_ylabel('Number of MSOAs with development')

# Average density over time
ax6.plot(history['period'], history['avg_density'])
ax6.set_title('Average Density Over Time')
ax6.set_xlabel('Period')
ax6.set_ylabel('Average Density (homes/hectare)')

# Average cost over time
ax5.plot(history['period'], history['avg_cost'])
ax5.set_title('Average Cost Over Time')
ax5.set_xlabel('Period')
ax5.set_ylabel('Average Cost per Home (£)')

plt.tight_layout()
plt.show()

# Final Summary Statistics
print("\nFinal Summary Statistics:")
print(f"Total periods simulated: {len(cost_history)}")
print(f"Total new homes built: {history['homes_built'].sum():,.0f}")
print(f"Initial total homes: {history['total_homes'].iloc[0]:,.0f}")
print(f"Final total homes: {history['total_homes'].iloc[-1]:,.0f}")
print(f"Initial average price: £{history['avg_price'].iloc[0]:,.0f}")
print(f"Final average price: £{history['avg_price'].iloc[-1]:,.0f}")
print(f"Price change: {((history['avg_price'].iloc[-1] / history['avg_price'].iloc[0]) - 1):,.1%}")
print(f"Initial average cost: £{history['avg_cost'].iloc[0]:,.0f}")
print(f"Final average cost: £{history['avg_cost'].iloc[-1]:,.0f}")
print(f"Cost change: {((history['avg_cost'].iloc[-1] / history['avg_cost'].iloc[0]) - 1):,.1%}")
print(f"Housing stock increase: {((history['total_homes'].iloc[-1] / history['total_homes'].iloc[0]) - 1):,.1%}")

# Save results
history.to_csv('development_history.csv', index=False)
msoas.to_csv('final_msoa_state.csv', index=False)

# Plotting the base cost over time
plt.figure(figsize=(10, 6))
plt.plot(base_cost_history, marker='o', linestyle='-', color='b')
plt.title('Base Cost Over Time')
plt.xlabel('Period')
plt.ylabel('Base Cost')
plt.grid()
plt.xticks(range(len(base_cost_history)))  # Set x-ticks to match the number of periods
plt.show()