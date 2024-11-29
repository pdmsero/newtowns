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

# Convex investment adjustment costs

The algorithm in the previous section is a simple linear model, which might not adequately capture the cost implications of increased building across different areas. In effect, by not having costs increase with the rate of development, the model may introduce a bias towards building in areas with low prices that would be outbid by areas with higher price differentials.

We address this problem by introducing a convex cost building rate, where the unit cost of building increases with the number of units built. We also introduce a density scaling factor, to account for the fact that the cost of building in denser areas is higher.

## Developer Profit Function

In each iteration, developers choose how many homes to build by maximizing profit:

$$
\Pi_{i,t} = P_{i,t} I_{i,t} - c_0 \left(1 + \Phi ( I_{i,t}, I_{i,t-1}) \right) I_{i,t}
$$ 

Where:

- $P_{i,t}$ is the local price per square meter
- $c_0$ is the base construction cost (£3,000)
- $\Phi$ is a function that controls how quickly costs rise with changing investment in new homes
- $\Phi=\frac{\kappa}{2}(\frac{I_{i,t}}{I_{i,t-1}}-1)^2$
- $\kappa$ is a parameter that controls the degree to which adjustment costs increase with changing investment in new homes
- $I_{i,t}$ is the number of new homes

This captures how construction becomes more expensive the more investment deviates from the previous period, while maintaining convexity in the choice variable I.

## Profit Maximization Process

The profit function expands to:

$$
\Pi_{i,t} = P_{i,t} I_{i,t} - c_0\left(1 + \frac{\kappa}{2}\left(\frac{I_{i,t}}{I_{i,t-1}}-1\right)^2\right) I_{i,t}
$$

To find the profit-maximizing development size, we take the first derivative with respect to I:

$$
\frac{d\Pi_{i,t}}{dI_{i,t}} = P_{i,t} - c_0 - c_0\left(1 + \frac{\kappa}{2}\left(\frac{I_{i,t}}{I_{i,t-1}}-1\right)^2\right)- c_0 \kappa \left(\frac{I_{i,t}}{I_{i,t-1}}-1\right) \frac{I_{i,t}}{I_{i,t-1}}
$$

Setting this equal to zero:

$$
P_{i,t} - c_0 - c_0\left(1 + \frac{\kappa}{2}\left(\frac{I_{i,t}}{I_{i,t-1}}-1\right)^2\right)- c_0 \kappa \left(\frac{I_{i,t}}{I_{i,t-1}}-1\right) \frac{I_{i,t}}{I_{i,t-1}} = 0
$$

And solving for I gives the optimal development size:

$$
I_{i,t}^{*} = I_{i,t-1} \left[\frac{2}{3}+ \sqrt{6 \frac{P_{i,t} - c_0}{c_0 \kappa}} \right]
$$

There are two solutions to the optimal level of investment $I_{i,t}^*$. The first of these solutions (+) satisfies the second order condition for a maximum, while the second (-) does not. There is therefore only one economically meaningful solution.

This formula tells us that:

1. Development is larger where prices exceed costs by more
2. It is costly to very rapidly change investment in new homes, resulting in rapidly increasing costs

By adjusting the cost functions, we relax the assumption that construction costs are uniform across locations and that there are no costs to rapidly changing investment in new homes.