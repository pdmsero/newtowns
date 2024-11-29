import numpy as np
import pandas as pd
from scipy.stats import lognorm

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n_msoas = 7264
n_periods = 10
mean_price = 10000
mean_buildable_area = 250
std_buildable_area = 250/20
base_cost = 3000
kappa = 4
price_elasticity = 1.8

# Generate density values for each type of area
section_size = n_msoas // 3
rural = np.random.uniform(1, 5, section_size)
suburban = np.random.uniform(5, 15, section_size)
urban = np.random.uniform(15, 50, section_size)
concatenated = np.concatenate([rural, suburban, urban])
choices = np.random.choice([0, 1, 2], size=n_msoas) * section_size
choices = choices + np.random.randint(0, section_size, size=n_msoas)

# Initialize MSOA data
msoas = pd.DataFrame({
    'msoa_id': range(1, n_msoas + 1),
    'price': (np.random.pareto(a=2.5, size=n_msoas) + 1) * mean_price/4,  # Pareto distribution
    'buildable_area': np.maximum(np.random.normal(mean_buildable_area, std_buildable_area, n_msoas), 10),
    'initial_density': concatenated[choices]
})

# Calculate initial homes
msoas['initial_homes'] = msoas['initial_density'] * msoas['buildable_area']
msoas['current_homes'] = msoas['initial_homes']
msoas['investment'] = 0.01*msoas['initial_homes']

# Initialize 'new_homes' to zero before the simulation loop
msoas['new_homes'] = 0

# Initialize history with new column
history = pd.DataFrame(columns=['period', 'total_homes', 'avg_price', 'homes_built', 'base_cost'])
history.loc[0] = {
    'period': 0,
    'total_homes': msoas['current_homes'].sum(),
    'avg_price': msoas['price'].mean(),
    'homes_built': 0,
    'base_cost': base_cost
}

def update_prices(new_homes, current_homes):
    return 1 - (new_homes / current_homes) * price_elasticity if current_homes > 0 else 1

# Run simulation
print(f"\nPeriod 0 (Initial state):")
print(msoas)

for period in range(1, n_periods + 1):
    # Calculate density penalty
    msoas['density'] = msoas['current_homes'] / msoas['buildable_area']
    msoas['density_penalty'] = 1 - 1/msoas['density']
        
    # Optimal homes calculation with constraints
    msoas['new_homes'] = np.where(
        (msoas['density'] >= 15) | (msoas['density'] >= msoas['initial_density'] * 5),
        0,  # No new homes if constraints are violated
        msoas['investment']*(2/3+np.sqrt(1+6/kappa*(msoas['price']-base_cost)/(base_cost) ))
    )
    msoas['new_homes'] = np.maximum(msoas['new_homes'], 0)
    msoas['new_homes'] = np.nan_to_num(msoas['new_homes'], nan=0.0, posinf=0.0, neginf=0.0)
    msoas['new_homes'] = np.floor(msoas['new_homes']).astype(int)
    
    msoas['current_homes'] += msoas['new_homes']
    msoas['density'] = msoas['current_homes'] / msoas['buildable_area']
    msoas['buildable_area'] -= msoas['new_homes'] / msoas['density']
    msoas['cost'] = base_cost * (1 + kappa / 2 * (msoas['new_homes'] / msoas['investment'] -1)**2)
    
    msoas['price'] *= msoas.apply(lambda row: update_prices(row['new_homes'], row['current_homes']), axis=1)
    
    msoas['investment'] = msoas['new_homes']

    total_homes_built = msoas['new_homes'].sum()
    history.loc[period] = {
        'period': period,
        'total_homes': msoas['current_homes'].sum(),
        'avg_price': msoas['price'].mean(),
        'homes_built': total_homes_built,
        'base_cost': base_cost
    }
    
    print(f"\nPeriod {period}:")
    print(msoas)
    print(f"Total homes built this period: {total_homes_built}")
    print(f"Total homes: {msoas['current_homes'].sum():,.0f}")
    print(f"Current base cost: {base_cost:.2f}")

