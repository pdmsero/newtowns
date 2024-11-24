# Housing Development Algorithm: Economic Intuition and Process

## Core Economic Logic

The algorithm models how housing development might respond to price signals while respecting environmental and planning constraints. It operates on the principle that development will occur where:
1. Prices are sufficiently high to justify construction costs
2. Land is available and not environmentally protected
3. Current density is low enough to allow additional development

## Protected Land Calculation

The first step establishes where building cannot occur. The algorithm creates buffer zones around protected areas:

$$
A_{i,\text{protected}} = \bigcup_{j \in J} \text{buffer}(A_{i,j}, b_j)
$$

This reflects real-world planning constraints where development is restricted not just within protected areas but also in their immediate vicinity. Buffer sizes vary by protection type:
- 5km for National Parks (substantial buffer reflecting their national importance)
- 2.5km for Areas of Outstanding Natural Beauty
- 200m for Sites of Special Scientific Interest and Ancient Woodland (more localized protection)

## Development Process

### Price Signals
In each iteration, development occurs where prices exceed construction costs by a sufficient margin:

$$
P_{i,t} > c(1+\delta)
$$

Where:
- $P_{i,t}$ is the local price per square meter
- $c$ is the construction cost (£3,000)
- $\delta$ is the required return (10%)

This reflects the real-world requirement that developers need a profit margin to justify construction.

### Rate of Development
The amount of new housing in buildable areas is proportional to:

$$
\Delta H_{i,t} = H_{i,t} \cdot s \cdot \frac{P_{i,t} - c(1+\delta)}{\max_j(P_{j,t} - c(1+\delta))}
$$

This formula captures several economic insights:
1. Development is faster where price premiums are larger
2. Areas with more existing housing can accommodate more new housing
3. The step size (s = 0.01) prevents unrealistic sudden changes

### Price Response
As new homes are built, prices adjust:

$$
P_{i,t+1} = P_{i,t} \cdot \left(1 - \epsilon \cdot \frac{\sum_i \Delta H_{i,t}}{\sum_i H_{i,t}}\right)
$$

The price elasticity ($\epsilon = 1.8$) determines how much prices fall as supply increases. This captures the market's response to increased housing supply.

## Key Features and Economic Implications

### 1. Market-Led Development
The algorithm allows development to occur where demand (as signaled by prices) is highest, subject to constraints. This mimics how real estate markets function, with developers responding to price signals.

### 2. Progressive Price Moderation
As more homes are built in high-price areas, prices gradually moderate. This reflects how increasing supply can improve affordability over time.

### 3. Environmental Protection
The algorithm maintains absolute protection for key environmental assets while allowing some development in less sensitive areas (e.g., partial Green Belt development).

### 4. Density-Aware Growth
By limiting development in already dense areas, the algorithm promotes expansion where it's most physically feasible.

### 5. Self-Limiting Mechanism
Development naturally slows and stops when:
- Prices fall below the development threshold
- Density limits are reached
- Protected land constraints bind

## Limitations and Assumptions

The algorithm makes several simplifying assumptions:
1. Uniform construction costs across locations
2. No local infrastructure constraints
3. Homogeneous housing types
4. No explicit modeling of transport links
5. Static environmental designations

These could be relaxed in more sophisticated versions of the model.

## Applications

This algorithm can help:
1. Identify areas with highest development potential
2. Estimate maximum housing capacity under current constraints
3. Model price impacts of different development patterns
4. Evaluate trade-offs between environmental protection and housing supply

The results suggest where planning policy might be unnecessarily restrictive and where development might most effectively improve housing affordability while respecting environmental constraints.