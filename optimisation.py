import pulp as lp
import pandas as pd
import numpy as np

# ----------------------------------------
# 1. Data Initialization
# ----------------------------------------

districts = ['Belconnen', 'Nth CBR & City', 'CBR East', 'South CBR', 'Tugg', 'Gungahlin', 'Woden Valley']

# Population Data
current_pop = np.array([104255, 59437, 1495, 31234, 87941, 82118, 77536])
projected_pop = np.array([140297, 90088, 1326, 36196, 88922, 86905, 138739])

# Power Consumption Data (Summer and Winter)
current_power_consumption_summer = np.array([109, 134, 17, 103, 133, 68, 77])  # MW
current_power_consumption_winter = np.array([121, 124, 18, 101, 149, 73, 80])  # MW

projected_power_consumption_summer = np.array([169, 169, 21, 103, 151, 73, 124])  # MW
projected_power_consumption_winter = np.array([192, 181, 21, 108, 153, 71, 156])  # MW

# Power Limits Data (Summer and Winter)
current_power_limits_summer = np.array([171, 209, 54, 142, 219, 76, 95]) * 0.8  # MW
current_power_limits_winter = np.array([190, 257, 54, 142, 252, 84, 114]) * 0.8  # MW

future_power_limits_summer = np.array([226, 264, 54, 114, 219, 131, 150])  # MW
future_power_limits_winter = np.array([245, 312, 54, 114, 252, 139, 169])  # MW

# Minimum Power Limits Data (Summer and Winter)
min_power_limits_summer = -np.array([48, 32.9, 54.4, 18.2, 18.4, 20.9, 32.7])
min_power_limits_winter = -np.array([50.8, 28.7, 56.5, 21.6, 37.7, 15.4, 38.4])

# EV Charger Data
ev_charger_power = 0.35  # MW per charger / charging station
max_chargers_per_iteration = 60  # Maximum chargers that can be added per year
max_chargers_per_season = max_chargers_per_iteration / 2  # Chargers per season
years = 17
seasons = years * 2  # Total number of seasons
safety_buffer = 0.90  # 90% of grid capacity

# Rooftop Solar Data
average_solar_per_person = 0.0044 * 2.5 # 4.4kWh over course of a day (averaged) per 1kW system per installation. Conservative 2.5kW system per household
initial_solar_percentage = 15 / 5  # percent at Year 1 (15% of population / 5 for the people in a household)
final_solar_percentage = 40 / 5  # percent at Year 15 (40% of population / 5 for the people in a household)

# Function to perform linear interpolation over seasons
def interpolate_over_seasons(start, end, total_seasons):
    return [start + (end - start) * season / (total_seasons - 1) for season in range(total_seasons)]

# Create a list to alternate between 'summer' and 'winter'
season_types = ['summer' if i % 2 == 0 else 'winter' for i in range(seasons)]

# Initialize Results Storage
yearly_results = []
total_chargers_placed = {district: 0 for district in districts}

# Running Total of Chargers Placed Across All Districts
previous_total_chargers_placed = 0

# ----------------------------------------
# 2. Optimization Loop for Each Season
# ----------------------------------------

for season in range(seasons):
    # Determine if it's summer or winter
    season_type = season_types[season]

    # Interpolate Population for the Current Season
    population_season = interpolate_over_seasons(current_pop, projected_pop, seasons)[season]
    total_population = sum(population_season)

    # Interpolate baseline power consumption for the current season and the opposite season
    if season_type == 'summer':
        power_season_start = current_power_consumption_summer
        power_season_end = projected_power_consumption_summer
        power_limits_start = current_power_limits_summer
        power_limits_end = future_power_limits_summer
        min_power_limits = min_power_limits_summer

        # Get the opposite season's baseline power consumption
        opposite_power_season_start = current_power_consumption_winter
        opposite_power_season_end = projected_power_consumption_winter
    else:
        power_season_start = current_power_consumption_winter
        power_season_end = projected_power_consumption_winter
        power_limits_start = current_power_limits_winter
        power_limits_end = future_power_limits_winter
        min_power_limits = min_power_limits_winter

        # Get the opposite season's baseline power consumption
        opposite_power_season_start = current_power_consumption_summer
        opposite_power_season_end = projected_power_consumption_summer

    # Interpolate power consumption and power limits for the current season
    power_season = interpolate_over_seasons(power_season_start, power_season_end, seasons)[season]
    max_power_limits = interpolate_over_seasons(power_limits_start, power_limits_end, seasons)[season]

    # Interpolate the opposite season's power consumption for the same season index
    opposite_power_season = interpolate_over_seasons(opposite_power_season_start, opposite_power_season_end, seasons)[season]

    # Calculate the maximum baseline consumption between current and opposite seasons
    max_baseline_power = np.maximum(power_season, opposite_power_season)

    # Total grid capacity
    total_grid_capacity = sum(max_power_limits)  # MW

    # Total maximum baseline power consumption (used for cumulative constraint)
    total_max_baseline_power = sum(max_baseline_power)  # MW

    # Calculate Percentage of People with Rooftop Solar for the Current Season
    percentage_solar = interpolate_over_seasons(initial_solar_percentage, final_solar_percentage, seasons)[season] / 100.0  # Fraction

    # Calculate Solar Power Generation and Adjust Max Power Limits
    solar_generation_district = []
    adjusted_max_limits = []
    for i, district in enumerate(districts):
        # Number of People with Rooftop Solar
        people_with_solar = population_season[i] * percentage_solar
        # Total Solar Generation in MW for the District
        solar_generation = people_with_solar * average_solar_per_person  # MW
        solar_generation_district.append(solar_generation)

        # Adjust Max Power Limit by Adding Solar Generation
        adjusted_max_limit = max_power_limits[i] + solar_generation
        adjusted_max_limits.append(adjusted_max_limit)

    # Adjusted Total Grid Capacity Including Solar
    total_adjusted_grid_capacity = sum(adjusted_max_limits)

    # Define the Linear Programming Problem
    prob = lp.LpProblem(f"EV_Charger_Placement_Season_{season+1}", lp.LpMaximize)

    # Decision Variables: Number of Chargers Added in Each District
    chargers_added = lp.LpVariable.dicts(f"ChargersAdded_{season+1}", districts, lowBound=0, cat='Integer')

    # Objective Function: Maximize Total Chargers Added This Season (Up to max_chargers_per_season)
    total_chargers_added = lp.lpSum([chargers_added[district] for district in districts])
    prob += total_chargers_added, "MaximizeTotalChargersAdded"

    # Constraints:

    # 1. Total Chargers Added Per Season Should Not Exceed max_chargers_per_season
    prob += total_chargers_added <= max_chargers_per_season, f"MaxChargersPerSeason_Season_{season+1}"

    # 2. Cumulative Power Usage Should Not Exceed 90% of Total Adjusted Grid Capacity
    # Total Max Power Consumption = Max Baseline Power + Power from Chargers (Previous and Current)
    cumulative_ev_charger_power = ev_charger_power * (previous_total_chargers_placed + total_chargers_added)
    cumulative_total_max_power_consumption = total_max_baseline_power + cumulative_ev_charger_power  # MW

    prob += cumulative_total_max_power_consumption <= safety_buffer * total_adjusted_grid_capacity, f"CumulativePowerConstraint_Season_{season+1}"

    # 3. Population-Based Proportionality Constraint for Charger Distribution
    for i, district in enumerate(districts):
        chargers_this_season = lp.lpSum([chargers_added[district] for district in districts])
        min_chargers = (population_season[i] / total_population) * chargers_this_season * 0.5  # 50% of proportional share
        prob += chargers_added[district] >= min_chargers, f"MinChargers_{district}_Season_{season+1}"

    # 4. No District can receive more than 35% of the total chargers added
    for i, district in enumerate(districts):
        prob += chargers_added[district] <= 0.35 * total_chargers_added, f"MaxChargersPerDistrict_{district}_Season_{season+1}"

    # 5. No District can go below its minimum power limit
    for i, district in enumerate(districts):
        district_consumption = power_season[i] + ev_charger_power * (total_chargers_placed[district] + chargers_added[district])
        prob += district_consumption >= min_power_limits[i], f"MinPowerLimit_{district}_Season_{season+1}"

    # 6. 'CBR East' receives at most 5% of the total chargers added
    prob += chargers_added['CBR East'] <= 0.05 * total_chargers_added, f"CBR_East_Max_5_Percent_Season_{season+1}"

    # Solve the Linear Programming Problem
    prob.solve()

    # Check if the Solution is Optimal
    if lp.LpStatus[prob.status] != 'Optimal':
        print(f"\nLinear program no longer optimal in Season {season+1}. Setting chargers added to zero for this and subsequent seasons.")
        # From this season onwards, set chargers_added_season to zeros
        chargers_added_season = {district: 0 for district in districts}
    else:
        # Extract the Results for the Current Season
        chargers_added_season = {district: int(lp.value(chargers_added[district])) for district in districts}
        # Update the Running Tally of Total Chargers Placed in Each District
        for district in districts:
            total_chargers_placed[district] += chargers_added_season[district]
        # Update the Running Total of Chargers Placed Across All Districts
        previous_total_chargers_placed += sum(chargers_added_season.values())

    # Calculate Power Consumption for EV Chargers Placed Up to the Current Season
    ev_charger_consumption = {
        district: total_chargers_placed[district] * ev_charger_power for district in districts
    }

    # Calculate Total Power Consumption per District
    total_consumption_district = {
        district: power_season[i] + ev_charger_consumption[district] for i, district in enumerate(districts)
    }

    # Calculate Spare Capacity per District
    spare_capacity_district = {
        district: adjusted_max_limits[i] - total_consumption_district[district] for i, district in enumerate(districts)
    }

    # Calculate Cumulative Total Max Power Consumption
    cumulative_total_max_power_consumption = total_max_baseline_power + sum(ev_charger_consumption.values())  # MW

    # Calculate Buffer Percentage
    buffer_percentage = cumulative_total_max_power_consumption / (safety_buffer * total_adjusted_grid_capacity)

    # Calculate Overall Spare Capacity
    overall_spare_capacity = (safety_buffer * total_adjusted_grid_capacity) - cumulative_total_max_power_consumption  # MW

    # Calculate the Year (Season Number Divided by 2)
    current_year = (season // 2) + 1

    # Store Results in the yearly_results List
    yearly_results.append({
        'Season': season + 1,
        'Year': current_year,
        'Season Type': season_type.capitalize(),
        'District': districts,
        'Solar Generation (MW)': solar_generation_district,
        'Adjusted Max Power Limit (MW)': adjusted_max_limits,
        'Power Consumption from Chargers (MW)': [ev_charger_consumption[district] for district in districts],
        'Total Power Consumption (MW)': [total_consumption_district[district] for district in districts],
        'Chargers Added This Season': [chargers_added_season[district] for district in districts],
        'Total Chargers Placed': [total_chargers_placed[district] for district in districts],
        'Spare Capacity (MW)': [spare_capacity_district[district] for district in districts],
        'Cumulative Total Max Power Consumption (MW)': cumulative_total_max_power_consumption,
        'Buffer Percentage': buffer_percentage,
        'Overall Spare Capacity (MW)': overall_spare_capacity
    })

# ----------------------------------------
# 3. Create a DataFrame and Save Results
# ----------------------------------------

# Create a DataFrame for All Results
all_results_df = pd.DataFrame()

for result in yearly_results:
    season_df = pd.DataFrame({
        'Season': [result['Season']] * len(districts),
        'Year': [result['Year']] * len(districts),
        'Season Type': [result['Season Type']] * len(districts),
        'District': result['District'],
        'Solar Generation (MW)': result['Solar Generation (MW)'],
        'Adjusted Max Power Limit (MW)': result['Adjusted Max Power Limit (MW)'],
        'Power Consumption from Chargers (MW)': result['Power Consumption from Chargers (MW)'],
        'Total Power Consumption (MW)': result['Total Power Consumption (MW)'],
        'Chargers Added This Season': result['Chargers Added This Season'],
        'Total Chargers Placed': result['Total Chargers Placed'],
        'Spare Capacity (MW)': result['Spare Capacity (MW)'],
        'Cumulative Total Max Power Consumption (MW)': [result['Cumulative Total Max Power Consumption (MW)']] * len(districts),
        'Buffer Percentage': [result['Buffer Percentage']] * len(districts),
        'Overall Spare Capacity (MW)': [result['Overall Spare Capacity (MW)']] * len(districts)
    })
    all_results_df = pd.concat([all_results_df, season_df], ignore_index=True)

# Save Results to CSV
csv_output_path = 'ev_charger_optimization_results.csv'  # Adjust the path as needed
all_results_df.to_csv(csv_output_path, index=False)

print(f"Results saved to {csv_output_path}")
