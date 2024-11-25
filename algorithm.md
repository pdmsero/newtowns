---
title: Algorithm for housing development
numbersections: false
---

# Core Economic Logic

The algorithm models how housing development might respond to price signals while respecting environmental and planning constraints. It assumes that development will occur where:

1. Prices are sufficiently high to justify construction costs
2. Land is available and not environmentally protected
3. Current density is low enough to allow additional development

# Protected Land Calculation

The first step establishes where building cannot happen. The algorithm creates buffer zones around protected areas:

$$
A_{i,\text{protected}} = \bigcup_{j \in J} \text{buffer}(A_{i,j}, b_j)
$$

This reflects real-world planning constraints where development is restricted not just within protected areas but also in their immediate vicinity. Buffer sizes vary by protection type:

- 5km for National Parks (substantial buffer reflecting their national importance)
- 2.5km for Areas of Outstanding Natural Beauty
- 200m for Sites of Special Scientific Interest and Ancient Woodland (more localized protection)

# Development Process

## Price Signals

In each iteration, development occurs where prices exceed construction costs by a sufficient margin:   

$$
P_{i,t} > c(1+\delta)
$$

Where:

- $P_{i,t}$ is the local price per square meter
- $c$ is the construction cost (£3,000)
- $\delta$ is the required return (10%)

This reflects the real-world requirement that developers need a profit margin to justify construction.

## Rate of Development

The amount of new housing in buildable areas is proportional to:

$$
\Delta H_{i,t} = H_{i,t} \cdot s \cdot \frac{P_{i,t} - c(1+\delta)}{\max_j(P_{j,t} - c(1+\delta))}
$$

This formula captures several key economic insights:

1. Development is faster where price differentials are larger
2. Areas with more existing housing can accommodate more new housing
3. The step size (s = 0.01) prevents unrealistic sudden changes

## Price Response

As new homes are built, prices adjust:

$$
P_{i,t+1} = P_{i,t} \cdot \left(1 - \epsilon \cdot \frac{\sum_i \Delta H_{i,t}}{\sum_i H_{i,t}}\right)
$$

The price elasticity ($\epsilon = 1.8$) determines how much prices fall as supply increases. This captures the market's response to increased housing supply.

# Key Features and Economic Implications

## Market-Led Development

The algorithm allows development to occur where demand (as signaled by prices) is highest, subject to constraints. This mimics how real estate markets function, with developers responding to price signals.

## Progressive Price Moderation

As more homes are built in high-price areas, prices gradually moderate. This reflects how increasing supply can improve affordability over time.

## Environmental Protection

The algorithm maintains absolute protection for key environmental assets while allowing some development in less sensitive areas (e.g., partial Green Belt development).

## Density-Aware Growth

By limiting development in already dense areas, the algorithm promotes expansion where it's most physically feasible.

## Self-Limiting Mechanism

Development naturally slows and stops when:

- Prices fall below the development threshold
- Density limits are reached
- Protected land constraints bind

# Limitations and Assumptions

The algorithm makes several simplifying assumptions:

1. Uniform construction costs across locations
2. No local infrastructure constraints
3. Homogeneous housing types
4. No explicit modeling of transport links
5. Static environmental designations

These could be relaxed in more sophisticated versions of the model.

# Applications

This algorithm can help:

1. Identify areas with highest development potential
2. Estimate maximum housing capacity under current constraints
3. Model price impacts of different development patterns
4. Evaluate trade-offs between environmental protection and housing supply

The results suggest where planning policy might be unnecessarily restrictive and where development might most effectively improve housing affordability while respecting environmental constraints.

# Convex cost building rate

The algorithm in the previous section is a simple linear model, which might not adequately capture the cost implications of increased building across different areas. In effect, by not having costs increase with the rate of development, the model may introduce a bias towards building in areas with low prices that would be outbid by areas with higher price differentials.

We address this problem by introducing a convex cost building rate, where the unit cost of building increases with the number of units built. We also introduce a density scaling factor, to account for the fact that the cost of building in denser areas is higher.

## Developer Profit Function

In each iteration, developers choose how many homes to build by maximizing profit:

$$
\Pi_{i,t} = P_{i,t} I_{i,t} - c_0(1 + \gamma d_{i,t} I_{i,t}) I_{i,t}
$$

Where:

- $P_{i,t}$ is the local price per square meter
- $c_0$ is the base construction cost (£3,000)
- $d$ is current density (homes per hectare)
- $\gamma$ controls how quickly costs rise with density
- $I_{i,t}$ is the number of new homes

This captures how construction becomes more expensive in denser areas, while maintaining convexity in the choice variable I.

## Profit Maximization Process

The profit function expands to:

$$
\Pi_{i,t} = P_{i,t} I_{i,t} - c_0(1 + \gamma d_{i,t} I_{i,t}) I_{i,t}
$$

To find the profit-maximizing development size, we take the first derivative with respect to I:

$$
\frac{d\Pi_{i,t}}{dI_{i,t}} = P_{i,t} - c_0 - 2c_0\gamma d_{i,t} I_{i,t}
$$

Setting this equal to zero:

$$
P_{i,t} - c_0 - 2c_0\gamma d_{i,t} I_{i,t} = 0
$$

And solving for I gives the optimal development size:

$$
I_{i,t}^* = \frac{P_{i,t} - c_0}{2c_0\gamma d_{i,t}}
$$

This formula tells us that:

1. Development is larger where prices exceed costs by more
2. Development is smaller in areas with higher existing density
3. The $\gamma$ parameter determines how strongly density constrains development
4. No development occurs when prices are below base construction costs

The formula provides an optimal development size for each MSOA, subject to the additional density constraints in the model.

By adjusting the cost function to reflect local conditions, we relax the following assumptions:

1. **Uniform construction costs across locations**: The model now allows for construction costs to vary based on the number of units built.
2. **No local infrastructure constraints**: The model can now account for the impact of local infrastructure on construction costs.