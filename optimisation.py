import pulp as lp
import pandas as pd
import numpy as np

# Data Initialization
districts = ['Belconnen', 'Nth CBR & City', 'CBR East', 'South CBR', 'Tugg', 'Gungahlin', 'Woden Valley']
current_pop = [104255, 59437, 1495, 31234, 87941, 82118, 77536]
projected_pop = [140297, 90088, 1326, 36196, 88922, 86905, 138739]
current_power_consumption = [
    max(109,121), # Belconnen
    max(134,124), # Nth CBR & City
    max(17,18), # CBR East
    max(103,101), # South CBR
    max(133,149), # Tugg
    max(68,73), # Gungahlin
    max(77,80) # Woden Valley
]  # MW
projected_power_consumption = [
    max(169,192),  # Belconnen
    max(169,181), # Nth CBR & City
    max(21,21), # CBR East
    max(103,108), # South CBR
    max(151,153), # Tugg
    max(73,71), # Gungahlin
    max(124,156) # Woden Valley
]  # MW
projected_power_consumption = [
    max(169,192),  # Belconnen
    max(169,181), # Nth CBR & City
    max(21,21), # CBR East
    max(103,108), # South CBR
    max(151,153), # Tugg
    max(73,71), # Gungahlin
    max(124,156) # Woden Valley
]  # MW

# Assuming that the Actual Power Limit is 80% of the 2 hour maximum
current_power_limits = [
    min(171,190) * 0.8, # Belconnen
    min(209,257) * 0.8, # Nth CBR & City
    min(54,54) * 0.8,  # CBR East
    min(142,142) * 0.8, # South CBR
    min(219,252) * 0.8, # Tugg
    min(76,84) * 0.8, # Gungahlin
    min(95,114) * 0.8, # Woden Valley
] # MW

future_power_limits = [
    min(226,245), # Belconnen
    min(264,312), # Nth CBR & City
    min(54,54),  # CBR East
    min(114,114), # South CBR
    min(219,252), # Tugg
    min(131,139), # Gungahlin
    min(150,169)  # Woden Valley
]  # MW
ev_charger_power = 0.35  # MW per charger / charging station
max_chargers_per_iteration = 60  # Maximum chargers that can be added per year
years = 17
safety_buffer = 0.90  # 90% of grid capacity

min_power_limits = [
    -min(48,50.8),   # Belconnen
    -min(32.9,28.7), # Nth CBR & City
    -min(54.4,56.5), # CBR East
    -min(18.2,21.6), # South CBR
    -min(18.4,37.7), # Tugg
    -min(20.9,15.4), # Gungahlin
    -min(32.7,38.4), # Woden Valley
]

# Rooftop Solar Data
average_solar_per_person = 0.0044  # MW per person
initial_solar_percentage = 15 / 5  # percent at Year 1 (15% of population / 5 for the people in a household)
final_solar_percentage = 40 / 5 # percent at Year 15 (50% of population / 5 for the people in a household)

# Linear Interpolation Function
def interpolate(start, end, years):
    return [start + (end - start) * year / (years - 1) for year in range(years)]

# Initialize Results Storage
yearly_results = []
total_chargers_placed = {district: 0 for district in districts}

# Running Total of Chargers Placed Across All Districts
previous_total_chargers_placed = 0

# Optimization Loop for Each Year
for year in range(years):
    # Interpolate Population and Baseline Power Consumption
    power_year = interpolate(np.array(current_power_consumption), np.array(projected_power_consumption), years)[year]
    population_year = interpolate(np.array(current_pop), np.array(projected_pop), years)[year]

    # Interpolate for available power capacity (without EV chargers)
    max_power_limits = interpolate(np.array(current_power_limits), np.array(future_power_limits), years)[year]

    # Total Population and Baseline Power Consumption for the Current Year
    total_population = sum(population_year)
    total_baseline_power = sum(power_year)  # MW

    # Total Grid Capacity (without solar)
    total_grid_capacity = sum(max_power_limits)  # MW

    # Calculate Percentage of People with Rooftop Solar for the Current Year
    percentage_solar = interpolate(initial_solar_percentage, final_solar_percentage, years)[year] / 100.0  # Fraction

    # Calculate Solar Power Generation and Adjust Max Power Limits
    solar_generation_district = []
    adjusted_max_limits = []
    for i, district in enumerate(districts):
        # Number of People with Rooftop Solar
        people_with_solar = population_year[i] * percentage_solar
        # Total Solar Generation in MW for the District
        solar_generation = people_with_solar * average_solar_per_person  # MW
        solar_generation_district.append(solar_generation)

        # Adjust Max Power Limit by Adding Solar Generation
        adjusted_max_limit = max_power_limits[i] + solar_generation
        adjusted_max_limits.append(adjusted_max_limit)

    # Adjusted Total Grid Capacity Including Solar
    total_adjusted_grid_capacity = sum(adjusted_max_limits)

    # Define the Linear Programming Problem
    prob = lp.LpProblem(f"EV_Charger_Placement_Year_{year+1}", lp.LpMaximize)

    # Decision Variables: Number of Chargers Added in Each District
    chargers_added = lp.LpVariable.dicts(f"ChargersAdded_{year+1}", districts, lowBound=0, cat='Integer')

    # Objective Function: Maximize Total Chargers Added This Year (Up to max_chargers_per_iteration)
    total_chargers_added = lp.lpSum([chargers_added[district] for district in districts])
    prob += total_chargers_added, "MaximizeTotalChargersAdded"

    # Constraints:

    # 1. Total Chargers Added Per Year Should Not Exceed max_chargers_per_iteration
    prob += total_chargers_added <= max_chargers_per_iteration, f"MaxChargersPerIteration_Year_{year+1}"

    # 2. Cumulative Power Usage Should Not Exceed 90% of Total Adjusted Grid Capacity
    # Total Power Consumption = Baseline Power + Power from Chargers (Previous and Current)
    cumulative_ev_charger_power = ev_charger_power * (previous_total_chargers_placed + total_chargers_added)
    cumulative_total_power_consumption = total_baseline_power + cumulative_ev_charger_power  # MW

    prob += cumulative_total_power_consumption <= safety_buffer * total_adjusted_grid_capacity, f"CumulativePowerConstraint_Year_{year+1}"

    # 3. Population-Based Proportionality Constraint for Charger Distribution
    for i, district in enumerate(districts):
        min_chargers = (population_year[i] / total_population) * max_chargers_per_iteration * 0.5  # 50% of proportional share
        min_chargers_int = int(np.ceil(min_chargers))
        prob += chargers_added[district] >= min_chargers_int, f"MinChargers_{district}_Year_{year+1}"

    # 4. District Power Consumption Should Not Exceed Adjusted Max Power Limits
    for i, district in enumerate(districts):
        # Total Power Consumption in District = Baseline Power + Chargers Added * EV Charger Power
        district_consumption = power_year[i] + ev_charger_power * chargers_added[district]
        prob += district_consumption <= adjusted_max_limits[i], f"DistrictConsumptionConstraint_{district}_Year_{year+1}"

    # 5. No District can recieve more than 25% of the total chargers added
    for i, district in enumerate(districts):
        prob += chargers_added[district] <= 0.25 * total_chargers_added, f"MaxChargersPerDistrict_{district}_Year_{year+1}"

    # 6. No District can go below its minimum power limit (how much power can be
    #    transferred from another district)
    for i, district in enumerate(districts):
        prob += district_consumption >= min_power_limits[i], f"MinPowerLimit_{district}_Year_{year+1}"

    # Needed to get the model to solve correctly at the end
    prob += chargers_added['CBR East'] <= 0.05 * total_chargers_added, f"CBR_East_Max_5_Percent_Year_{year+1}"

    # Solve the Linear Programming Problem
    prob.solve()

    # Check if the Solution is Optimal
    if lp.LpStatus[prob.status] != 'Optimal':
        print(f"\nLinear program no longer optimal in Year {year+1}. Setting chargers added to zero for this and subsequent years.")
        # From this year onwards, set chargers_added_year to zeros
        chargers_added_year = {district: 0 for district in districts}
    else:
        # Extract the Results for the Current Year
        chargers_added_year = {district: int(lp.value(chargers_added[district])) for district in districts}
        # Update the Running Tally of Total Chargers Placed in Each District
        for district in districts:
            total_chargers_placed[district] += chargers_added_year[district]
        # Update the Running Total of Chargers Placed Across All Districts
        previous_total_chargers_placed += sum(chargers_added_year.values())

    # Calculate Power Consumption for EV Chargers Placed Up to the Current Year
    ev_charger_consumption = {
        district: total_chargers_placed[district] * ev_charger_power for district in districts
    }

    # Calculate Total Power Consumption per District
    total_consumption_district = {
        district: power_year[i] + ev_charger_consumption[district] for i, district in enumerate(districts)
    }

    # Calculate Spare Capacity per District
    spare_capacity_district = {
        district: adjusted_max_limits[i] - total_consumption_district[district] for i, district in enumerate(districts)
    }

    # Calculate Cumulative Total Power Consumption
    cumulative_total_consumption = total_baseline_power + sum(ev_charger_consumption.values())  # MW

    # Calculate Buffer Percentage (How Close the Total Power is to the 90% Limit)
    buffer_percentage = cumulative_total_consumption / (safety_buffer * total_adjusted_grid_capacity)

    # Calculate Overall Spare Capacity
    overall_spare_capacity = (safety_buffer * total_adjusted_grid_capacity) - cumulative_total_consumption  # MW

    # Store Results in the yearly_results List
    yearly_results.append({
        'Year': year + 1,
        'District': districts,
        'Solar Generation (MW)': solar_generation_district,
        'Adjusted Max Power Limit (MW)': adjusted_max_limits,
        'Power Consumption from Chargers (MW)': [ev_charger_consumption[district] for district in districts],
        'Total Power Consumption (MW)': [total_consumption_district[district] for district in districts],
        'Chargers Added This Year': [chargers_added_year[district] for district in districts],
        'Total Chargers Placed': [total_chargers_placed[district] for district in districts],
        'Spare Capacity (MW)': [spare_capacity_district[district] for district in districts],
        'Cumulative Total Power Consumption (MW)': cumulative_total_consumption,
        'Buffer Percentage': buffer_percentage,
        'Overall Spare Capacity (MW)': overall_spare_capacity
    })

# Create a DataFrame for All Results
all_results_df = pd.DataFrame()

for result in yearly_results:
    year_df = pd.DataFrame({
        'Year': [result['Year']] * len(districts),
        'District': result['District'],
        'Solar Generation (MW)': result['Solar Generation (MW)'],
        'Adjusted Max Power Limit (MW)': result['Adjusted Max Power Limit (MW)'],
        'Power Consumption from Chargers (MW)': result['Power Consumption from Chargers (MW)'],
        'Total Power Consumption (MW)': result['Total Power Consumption (MW)'],
        'Chargers Added This Year': result['Chargers Added This Year'],
        'Total Chargers Placed': result['Total Chargers Placed'],
        'Spare Capacity (MW)': result['Spare Capacity (MW)'],
        'Cumulative Total Power Consumption (MW)': [result['Cumulative Total Power Consumption (MW)']] * len(districts),
        'Buffer Percentage': [result['Buffer Percentage']] * len(districts),
        'Overall Spare Capacity (MW)': [result['Overall Spare Capacity (MW)']] * len(districts)
    })
    all_results_df = pd.concat([all_results_df, year_df], ignore_index=True)

# Save Results to CSV
csv_output_path = 'ev_charger_optimization_results.csv'  # Adjust the path as needed
all_results_df.to_csv(csv_output_path, index=False)

print(f"Results saved to {csv_output_path}")
