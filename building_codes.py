import pandas as pd  # Importing pandas for data manipulation and analysis
import geopandas as gpd  # Importing geopandas for working with geospatial data
import matplotlib.pyplot as plt  # Importing matplotlib for plotting data
import matplotlib  # Importing matplotlib for additional plotting functionalities
import numpy as np  # Importing numpy for numerical operations

# Loading price per square meter data for MSOAs from a CSV file
msoa_prices_2023 = pd.read_csv("data/raw_data/price_paid/psqm_by_MSOA_2023_2023.csv")[["MSOA21CD", "priceper_median"]]

# Loading dwelling stock data for MSOAs from a CSV file
msoa_dwelling_stock  = pd.read_csv("data/raw_data/population/msoa_dwellings_2021.csv")[["MSOA21CD", "MSOA21NM", "all_dwellings"]]

# Loading MSOA boundaries from a GeoPackage file
msoa_boundaries = gpd.read_file("data/raw_data/boundaries/msoa_2021_bfc.gpkg")[["MSOA21CD", "geometry"]]

# Loading green belt data from a shapefile
green_belt = gpd.read_file("data/raw_data/protected_land/green_belt/England_Green_Belt_2021-22_WGS84.shp")

# Loading Areas of Outstanding Natural Beauty (AONB) data from a shapefile
aonb = gpd.read_file("data/raw_data/protected_land/AONB/Areas_of_Outstanding_Natural_Beauty_EnglandPolygon.shp")

# Loading Site of Special Scientific Interest (SSSI) data from a GeoJSON file
sssi = gpd.read_file("data/raw_data/protected_land/SSSI.geojson")

# Loading ancient woodland data from a GeoJSON file
ancient_woodland = gpd.read_file("data/raw_data/protected_land/AncientWoodland.geojson")

# Loading national park data from a GeoJSON file
national_park = gpd.read_file("data/raw_data/protected_land/national_parks.geojson")

# Loading designated open land data from a GeoPackage file
des_open = gpd.read_file("data/raw_data/protected_land/ldn_des_open.gpkg")

# Loading lookup table for mapping MSOAs to parliamentary constituencies
lookup = pd.read_csv("data/raw_data/lookups/MSOA_to_parliament_best_fit.csv")[["MSOA21CD", "PCON25NM", "PCON25CD"]]

# Loading parliamentary constituencies data from a GeoPackage file
parliamentary_constituencies = gpd.read_file("data/raw_data/ukparliament/2024_constituencies.gpkg")

# Defining buffer distances for protected land features
buffer_national_park = 5_000 
buffer_aonb = 2_500
buffer_sssi = 200
buffer_ancient_woodland = 200

# Filtering MSOAs to only include those in England
msoa_boundaries = msoa_boundaries.loc[msoa_boundaries.MSOA21CD.str.startswith("E")]

# Reprojecting designated open land data to EPSG:27700
des_open = des_open.to_crs(epsg=27700)

# Buffering protected land features
national_park["geometry"] = national_park.geometry.buffer(buffer_national_park)
aonb["geometry"] = aonb.geometry.buffer(buffer_aonb)
sssi["geometry"] = sssi.geometry.buffer(buffer_sssi)
ancient_woodland["geometry"] = ancient_woodland.geometry.buffer(buffer_ancient_woodland)

# ===========================
# Combining and Buffering Section
# ===========================

# Combining and buffering protected land features
# This step combines the geometries of various protected land types into a single shape.
# The union_all method is used to merge the geometries, creating a single representation of all protected areas.
protected_land_shape = pd.concat([
    aonb[["geometry"]],
    sssi[["geometry"]],
    ancient_woodland[["geometry"]],
    national_park[["geometry"]],
    des_open[["geometry"]]
]).union_all()  # Using union_all to combine geometries

# Combining and buffering green belt data
# This step creates a single shape for the green belt areas by merging their geometries.
# The union_all method ensures that overlapping areas are combined into one continuous shape.
green_belt_shape = green_belt.union_all()  # Using union_all to combine geometries

# Creating a shape of non-protected green belt areas
# This step calculates the areas of green belt that are not protected by other designations.
# The difference method subtracts the protected land shape from the green belt shape,
# resulting in a new geometry that represents only the green belt areas that are not overlapped by protected land.
non_protected_green_belt_shape = green_belt_shape.difference(protected_land_shape)

# ===========================
# Visualization Section
# ===========================

# Plotting the shapes of protected and non-protected areas
# This section visualizes the different land types using matplotlib.
# A figure and axis are created for plotting, and each type of land is plotted with a different color for clarity.
fig, ax = plt.subplots(figsize=(12, 6))  # Creating a figure and axis for plotting
gpd.GeoSeries(green_belt_shape).plot(ax=ax, color="blue", label="Green Belt")  # Plotting green belt areas in blue
gpd.GeoSeries(non_protected_green_belt_shape).plot(ax=ax, color="red", label="Non-Protected Green Belt")  # Plotting non-protected green belt areas in red
gpd.GeoSeries(protected_land_shape).plot(ax=ax, color="green", alpha=0.5, label="Protected Land")  # Plotting protected land areas in green with transparency
ax.set_axis_off()  # Hiding the axis for better visualization
plt.legend()  # Adding a legend to the plot to identify land types
plt.show()  # Displaying the plot

# ===========================
# Buildable Area Calculation Section
# ===========================

# Calculate buildable area for each MSOA by excluding SSSI, AONB, and half of the Green Belt
# This section computes the available land for building in each MSOA.
# The intersection method is used to find the area of protected land within each MSOA,
# and the area property retrieves the actual area of that geometry.
msoa_boundaries["protected_land"] = msoa_boundaries.intersection(protected_land_shape).area  # Area of protected land within each MSOA

# Similarly, this line calculates the area of non-protected green belt within each MSOA.
msoa_boundaries["non_protected_green_belt"] = msoa_boundaries.intersection(non_protected_green_belt_shape).area  # Area of non-protected green belt within each MSOA

# This line calculates the total area of each MSOA, which will be used in further calculations.
msoa_boundaries["total_area"] = msoa_boundaries.area  # Total area of each MSOA

# ===========================
# Parameters for Housing Calculation Section
# ===========================

# Defining parameters for housing calculations
# These parameters will be used in subsequent calculations to estimate housing potential.
pes_housing = 1.8  # Average number of persons per dwelling

build_cost = 3_000  # Cost to build per unit area (e.g., per square meter)
build_cost_delta = 1  # Incremental cost adjustment for building

# Defining density thresholds for housing calculations
initial_density_threshold = 15  # Initial density threshold for housing (minimum dwellings per area)
max_density_threshold = 200  # Maximum density threshold for housing (maximum dwellings per area)
max_density_increase = 5  # Maximum increase in density allowed in calculations

# Percentage of green belt that is retained for development
pc_green_belt_retained = 0.5  # 50% of the non-protected green belt is considered for development

# Step size for calculations, used in iterative processes
step = 0.01  # Step size for calculations

# ===========================
# Final Buildable Area Calculation
# ===========================

# Calculating the buildable area for each MSOA
# This calculation considers the total area minus protected land and a portion of the non-protected green belt.
# The np.maximum function ensures that the buildable area cannot be negative.
msoa_boundaries["buildable_area"] = np.maximum(
    msoa_boundaries.area - msoa_boundaries["protected_land"] - (msoa_boundaries["non_protected_green_belt"] * pc_green_belt_retained), 
    0
) * 0.0001  # Converting area to appropriate units (e.g., hectares) by multiplying by 0.0001
# Sorting MSOAs by buildable area in descending order to prioritize areas with the most potential for development
msoa_boundaries.sort_values(by="buildable_area", ascending=False)  # Sorting MSOAs by buildable area in descending order

# ===========================
# Housing Development Simulation Section
# ===========================

# Joining dataframes and filtering for England MSOAs
# This step combines the price, dwelling stock, and buildable area data into a single DataFrame for analysis.
# It filters the joined DataFrame to only include MSOAs that are located in England.
joined = msoa_prices_2023.set_index("MSOA21CD").join(
    msoa_dwelling_stock.set_index("MSOA21CD"), how="left"
).join(
    msoa_boundaries.set_index("MSOA21CD"), how="left"
).reset_index()
joined = joined.loc[joined.MSOA21CD.str.startswith("E")]

# Setting up working variables
# These variables will be used in the iterative process to simulate housing development.
joined["working_price"] = joined["priceper_median"].astype(float)  # Converting price to float for calculations
joined["added_homes"] = 0.0  # Initializing a column to track the number of homes added
joined["working_homes"] = joined["all_dwellings"].astype(float)  # Current number of homes in each MSOA
joined["initial_density"] = joined["all_dwellings"] / (joined["buildable_area"] + 1)  # Calculating initial density of homes per buildable area
joined = joined.drop_duplicates(subset=["MSOA21CD"])  # Removing duplicate entries based on MSOA code

# Saving the working DataFrame to a CSV file for further analysis
joined.drop(columns=["geometry"]).to_csv("data/intermediate/missing_homes_working_df.csv", index=False)

# Creating a copy of the joined DataFrame for iterative calculations
working_df = joined.set_index("MSOA21CD").copy()
i = 0  # Iteration counter

# Initializing a DataFrame to track metrics during the simulation
tracking_df = pd.DataFrame(columns=[
    'iteration', 'msoas_built_in', 'homes_added', 'pc_homes', 'pc_change',
    'total_working_homes', 'avg_working_price', 'max_working_price', 'buildable_max_working_price'
])

# ===========================
# Iterative Housing Development Process
# ===========================

while True:
    i += 1  # Incrementing the iteration counter

    # Identifying buildable MSOAs based on several criteria
    # This step filters the working DataFrame to find MSOAs where housing can be developed.
    buildable_msoas = working_df.loc[
        (working_df.initial_density < initial_density_threshold)  # Check if initial density is below the threshold
        & (working_df.working_homes / (working_df.buildable_area + 1) < (working_df.initial_density * max_density_increase))  # Check if working density growth is within limits
        & (working_df.working_homes / (working_df.buildable_area + 1) < max_density_threshold)  # Check if working density is below the maximum threshold
        & (working_df.working_price > build_cost * build_cost_delta)  # Check if the working price is above the adjusted build cost
    ]

    # If no more buildable MSOAs are found, exit the loop
    if len(buildable_msoas) == 0:
        print(f"Ending: no more homes can be built (Iteration {i})")
        # Reporting the number of MSOAs that could not meet the criteria
        A = working_df.loc[working_df.initial_density < initial_density_threshold]
        print(f"{len(A)} MSOAs with initial density < {initial_density_threshold}")
        B = A.loc[(A.working_homes / (A.buildable_area + 1) < (A.initial_density * max_density_increase))]
        print(f"{len(B)} MSOAs with working density growth less than {max_density_increase}x")
        C = B.loc[(B.working_homes / (B.buildable_area + 1) < max_density_threshold)]
        print(f"{len(C)} MSOAs with working density < {max_density_threshold}")
        D = C.loc[C.working_price > build_cost * build_cost_delta]
        print(f"{len(D)} MSOAs with working price > {build_cost * build_cost_delta}")
        break  # Exit the loop if no more homes can be built

    # Determine the maximum working price among the buildable MSOAs
    max_price = buildable_msoas.working_price.max()
    max_delta = np.maximum(max_price - (build_cost * build_cost_delta), 1)  # Calculate the maximum price delta for scaling

    # Calculate the number of homes to add based on the working price and density
    homes_to_add = np.maximum(buildable_msoas.working_homes * step * (buildable_msoas.working_price - (build_cost * build_cost_delta)) / max_delta, 0)
    pc_homes = homes_to_add.sum() / working_df.working_homes.sum()  # Calculate the percentage of homes added
    pc_change = pc_homes * pes_housing  # Calculate the change in population based on housing added

    # Update the working DataFrame with the new homes added
    working_df.loc[buildable_msoas.index, "added_homes"] += homes_to_add  # Increment the added homes
    working_df.loc[buildable_msoas.index, "working_homes"] += homes_to_add  # Update the working homes count

    # Adjust the working price based on the percentage change in homes
    working_df["working_price"] = working_df["working_price"] * (1 - pc_change)

    # Track metrics for this iteration
    tracking_df.loc[i] = [
        i,
        buildable_msoas.MSOA21NM.shape[0],  # Number of MSOAs built in this iteration
        homes_to_add.sum(),  # Total homes added in this iteration
        pc_homes,  # Percentage of homes added
        pc_change,  # Change in population
        working_df.working_homes.sum(),  # Total working homes after this iteration
        working_df.working_price.median(),  # Median working price after this iteration
        working_df.working_price.max(),  # Maximum working price after this iteration
        buildable_msoas.working_price.max()  # Maximum working price among buildable MSOAs
    ]

    # Print progress every 50 iterations
    if i % 50 == 0:
        print(f"({i}) {homes_to_add.sum():.0f} homes built in {buildable_msoas.MSOA21NM.shape[0]} MSOAs. New national average price: {working_df['working_price'].median():.0f}. New national max price: {working_df['working_price'].max():.0f}. New buildable max price: {buildable_msoas.working_price.max():.0f}. {working_df.added_homes.sum():.0f} total homes built ({working_df.added_homes.sum() / working_df.all_dwellings.sum():.1%})")

# Final reporting of results
print(f"Final total homes built: {working_df.added_homes.sum():.0f} ({working_df.added_homes.sum() / working_df.all_dwellings.sum():.1%})")
print(f"Price drop: {working_df.added_homes.sum() / working_df.all_dwellings.sum() * pes_housing:.1%}")
print(f"Settings: {step=}, {initial_density_threshold=}, {max_density_threshold=}, {max_density_increase=}, {pc_green_belt_retained=}")
print(f"Buffers: {buffer_national_park=}, {buffer_aonb=}, {buffer_sssi=}, {buffer_ancient_woodland=}")

# ===========================
# London Area Analysis Section
# ===========================

# Defining the bounding box for the London area
LONDON_BBOX = [446170.3513, 95773.8205, 612601.8914, 270551.3753]
gdf = gpd.GeoDataFrame(working_df, geometry=working_df.geometry)  # Creating a GeoDataFrame for spatial analysis
london_area_msoas = gdf.cx[LONDON_BBOX[0]:LONDON_BBOX[2], LONDON_BBOX[1]:LONDON_BBOX[3]]  # Filtering for MSOAs within the London bounding box
print(f"London area homes built: {london_area_msoas.added_homes.sum():.0f} ({london_area_msoas.added_homes.sum() / working_df.added_homes.sum():.1%})")

# ===========================
# Parliamentary Constituency Analysis Section
# ===========================

# Joining the working DataFrame with the lookup table to associate MSOAs with parliamentary constituencies
output = working_df.join(lookup.set_index("MSOA21CD"), how="left")

# Grouping the data by parliamentary constituency to aggregate various metrics
# This step summarizes the total area, buildable area, added homes, working homes, and all dwellings for each constituency
output = output.groupby("PCON25CD").agg({
    "total_area": "sum",  # Summing total area for each constituency
    "buildable_area": "sum",  # Summing buildable area for each constituency
    "added_homes": "sum",  # Summing added homes for each constituency
    "working_homes": "sum",  # Summing working homes for each constituency
    "all_dwellings": "sum",  # Summing all dwellings for each constituency
    "PCON25NM": "first"  # Taking the first name of the constituency
})

# Joining the parliamentary constituencies data with the aggregated output
# This step ensures that the constituency boundaries are included in the analysis
parliamentary_constituencies = parliamentary_constituencies.set_index("PCON24CD")
parliamentary_constituencies = parliamentary_constituencies.join(output, how="left")

# Plotting the results of added homes by parliamentary constituency
# This visualization helps to understand the distribution of new homes across constituencies
parliamentary_constituencies.plot(column="added_homes", legend=True, figsize=(12, 6), cmap="coolwarm")
parliamentary_constituencies.sort_values(by="added_homes", ascending=False)  # Sorting constituencies by added homes
parliamentary_constituencies.added_homes.sum()  # Summing total added homes across constituencies

# ===========================
# Visualization of New Homes Built
# ===========================

# Create figure with two subplots to visualize new homes built nationally and in London
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Get common scale for both plots using logarithmic normalization
norm = matplotlib.colors.LogNorm(vmin=gdf.loc[gdf.added_homes > 0, 'added_homes'].min(), 
                                  vmax=gdf.loc[gdf.added_homes > 0, 'added_homes'].max())

# Plot national map of new homes built
gdf.loc[gdf.added_homes > 0].plot(column="added_homes", legend=True, norm=norm, ax=ax1, cmap="coolwarm")
ax1.set_title('New Homes Built - National')

# Plot London area map of new homes built
london_gdf = gdf.cx[LONDON_BBOX[0]:LONDON_BBOX[2], LONDON_BBOX[1]:LONDON_BBOX[3]]
london_gdf.loc[london_gdf.added_homes > 0].plot(column="added_homes", legend=True, norm=norm, ax=ax2, cmap="coolwarm")
ax2.set_title('New Homes Built - London Area')

plt.tight_layout()  # Adjust layout for better spacing

# ===========================
# Density Analysis Section
# ===========================

# Create figure with two subplots to visualize initial and final housing densities
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Calculate initial and final densities
initial_density = gdf.all_dwellings / (gdf.geometry.area / 10000)  # Dwellings per hectare
final_density = (gdf.all_dwellings + gdf.added_homes) / (gdf.geometry.area / 10000)  # Final density after adding homes

# Get common scale for both plots
vmin = min(initial_density.min(), final_density.min())
vmax = max(initial_density.max(), final_density.max())
norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)

# Plot initial density
gdf.plot(column=initial_density, ax=ax1, legend=True, norm=norm)
ax1.set_title('Initial Housing Density (dwellings/ha)')

# Plot final density 
gdf.plot(column=final_density, ax=ax2, legend=True, norm=norm)
ax2.set_title('Final Housing Density (dwellings/ha)')

plt.tight_layout()  # Adjust layout for better spacing

# ===========================
# Price Analysis Section
# ===========================

# Create figure with two subplots to visualize initial and final house prices
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Get min/max values across both columns for consistent scale
vmin = min(gdf['priceper_median'].min(), gdf['working_price'].min())
vmax = max(gdf['priceper_median'].max(), gdf['working_price'].max())
norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)

# Plot initial prices with log scale
gdf.plot(column='priceper_median', ax=ax1, legend=True, norm=norm)
ax1.set_title('Initial House Prices (Log Scale)')

# Plot final prices with log scale
gdf.plot(column='working_price', ax=ax2, legend=True, norm=norm)
ax2.set_title('Final House Prices (Log Scale)')

plt.tight_layout()  # Adjust layout for better spacing

# ===========================
# Summary of Results
# ===========================

# Displaying the top 10 MSOAs with the most added homes
working_df.sort_values(by="added_homes", ascending=False)[["MSOA21NM", "added_homes", "initial_density", "working_price", "all_dwellings"]].head(10)

# Displaying the top 10 MSOAs with the highest working prices
working_df.sort_values(by="working_price", ascending=False)[["MSOA21NM", "working_price", "added_homes", "initial_density", "all_dwellings"]].head(10)

# Reporting the total number of dwellings across all MSOAs
working_df.all_dwellings.sum()  # Total number of dwellings in the dataset

# Final output of the total number of homes built
np.float64(24927575.0)  # Example output, replace with actual calculation if needed