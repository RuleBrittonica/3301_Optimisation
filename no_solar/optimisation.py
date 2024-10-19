import pulp as lp
import pandas as pd
import numpy as np

# Data
districts = ['Belconnen', 'Nth CBR & City', 'CBR East', 'South CBR', 'Tugg', 'Gungahlin', 'Woden Valley']
current_pop = [104000, 59500, 1495, 31234, 87941, 82118, 77536]
projected_pop = [140297, 90088, 1326, 36196, 88922, 86905, 138739]
current_power_consumption = [109, 134, 17, 103, 133, 68, 77]  # MW
projected_power_consumption = [169, 169, 21, 103, 151, 73, 124]  # MW
max_power_limits = [226, 264, 54, 114, 219, 131, 150]  # MW
ev_charger_power = 0.30  # MW per charger / charging station
max_chargers_per_iteration = 50  # We can add up to 75 chargers per iteration
years = 15
safety_buffer = 0.90  # We need to ensure usage is not more than 90% of grid capacity

# Total grid capacity (since suburbs can share power)
total_grid_capacity = sum(max_power_limits)  # MW

# Function to perform linear interpolation for power consumption over years
def interpolate(start, end, years):
    return [start + (end - start) * year / (years - 1) for year in range(years)]

# Initialize storage for results and running tally of chargers placed in each district
yearly_results = []
total_chargers_placed = {district: 0 for district in districts}

# Running total of chargers placed across all districts
previous_total_chargers_placed = 0

# Loop through each year and perform optimization
for year in range(years):
    # Interpolate population-based power consumption (without EV chargers)
    power_year = interpolate(np.array(current_power_consumption), np.array(projected_power_consumption), years)[year]
    population_year = interpolate(np.array(current_pop), np.array(projected_pop), years)[year]

    # Total population and baseline power consumption for the current year
    total_population = sum(population_year)
    total_baseline_power = sum(power_year)  # MW

    # Create a LP problem instance for the current year
    prob = lp.LpProblem(f"EV_Charger_Placement_Year_{year+1}", lp.LpMaximize)

    # Decision variables: Number of chargers added in each district (integer variables)
    chargers_added = lp.LpVariable.dicts(f"ChargersAdded_{year+1}", districts, lowBound=0, cat='Integer')

    # Objective function: Maximize the total number of chargers added this iteration (up to 75)
    total_chargers_added = lp.lpSum([chargers_added[district] for district in districts])
    prob += total_chargers_added, "MaximizeTotalChargersAdded"

    # Constraints:

    # 1. Total chargers added per iteration should not exceed 75
    prob += total_chargers_added <= max_chargers_per_iteration, f"MaxChargersPerIteration_Year_{year+1}"

    # 2. Cumulative power usage should not exceed 90% of the total grid capacity
    # Total power consumption = baseline power + power from chargers (previous and current)
    cumulative_ev_charger_power = ev_charger_power * (previous_total_chargers_placed + total_chargers_added)
    cumulative_total_power_consumption = total_baseline_power + cumulative_ev_charger_power  # MW

    prob += cumulative_total_power_consumption <= safety_buffer * total_grid_capacity, f"CumulativePowerConstraint_Year_{year+1}"

    # 3. Population-based proportionality constraint for charger distribution
    # To avoid non-linear constraints, set a minimum number of chargers per district based on their population proportion and a fraction of the maximum chargers per iteration
    for i, district in enumerate(districts):
        min_chargers = (population_year[i] / total_population) * max_chargers_per_iteration * 0.5  # 50% of proportional share
        # Since chargers_added[district] is an integer, we need to set the minimum as an integer
        # Use ceiling to ensure at least the minimum number is met
        min_chargers_int = int(np.ceil(min_chargers))
        prob += chargers_added[district] >= min_chargers_int, f"MinChargers_{district}_Year_{year+1}"

    # Solve the problem
    prob.solve()

    # Check if the solution is still optimal
    if lp.LpStatus[prob.status] != 'Optimal':
        print(f"\nLinear program no longer optimal in Year {year+1}. Setting chargers added to zero for this and subsequent years.")
        # From this year onwards, set chargers_added_year to zeros
        chargers_added_year = {district: 0 for district in districts}
        # Chargers placed remain the same
    else:
        # Extract the results for the current year
        chargers_added_year = {district: int(chargers_added[district].varValue) for district in districts}
        # Update the running tally of total chargers placed in each district and overall
        for district in districts:
            total_chargers_placed[district] += chargers_added_year[district]
        previous_total_chargers_placed += sum(chargers_added_year.values())

    # Calculate power consumption for EV chargers placed up to the current year
    ev_charger_power_consumption = {
        district: (total_chargers_placed[district]) * ev_charger_power for district in districts
    }

    # Calculate total power consumption (baseline power + power from EV chargers)
    total_power_consumption_district = {
        district: power_year[i] + ev_charger_power_consumption[district] for i, district in enumerate(districts)
    }

    # Calculate spare capacity per district
    spare_capacity_district = {
        district: max_power_limits[i] - total_power_consumption_district[district] for i, district in enumerate(districts)
    }

    # Calculate cumulative total power consumption
    cumulative_ev_charger_power = ev_charger_power * previous_total_chargers_placed
    cumulative_total_power_consumption = total_baseline_power + cumulative_ev_charger_power  # MW

    # Calculate buffer percentage (how close the total power is to the 90% limit)
    buffer_percentage = cumulative_total_power_consumption / (safety_buffer * total_grid_capacity)

    # Calculate Overall Spare Capacity
    overall_spare_capacity = (total_grid_capacity) - cumulative_total_power_consumption  # MW

    # Store results in the yearly_results list
    yearly_results.append({
        'Year': year + 1,
        'District': districts,
        'Baseline Power Consumption (MW)': list(power_year),
        'Power Consumption from Chargers (MW)': [ev_charger_power_consumption[district] for district in districts],
        'Total Power Consumption (MW)': [total_power_consumption_district[district] for district in districts],
        'Chargers Added This Year': [chargers_added_year[district] for district in districts],
        'Total Chargers Placed': [total_chargers_placed[district] for district in districts],
        'Spare Capacity (MW)': [spare_capacity_district[district] for district in districts],
        'Cumulative Total Power Consumption (MW)': cumulative_total_power_consumption,
        'Buffer Percentage': buffer_percentage,
        'Overall Spare Capacity (MW)': overall_spare_capacity  # Added Overall Spare Capacity
    })

# Create a DataFrame for all results
all_results_df = pd.DataFrame()

for result in yearly_results:
    year_df = pd.DataFrame({
        'Year': result['Year'],
        'District': result['District'],
        'Baseline Power Consumption (MW)': result['Baseline Power Consumption (MW)'],
        'Power Consumption from Chargers (MW)': result['Power Consumption from Chargers (MW)'],
        'Total Power Consumption (MW)': result['Total Power Consumption (MW)'],
        'Chargers Added This Year': result['Chargers Added This Year'],
        'Total Chargers Placed': result['Total Chargers Placed'],
        'Spare Capacity (MW)': result['Spare Capacity (MW)'],
        'Cumulative Total Power Consumption (MW)': result['Cumulative Total Power Consumption (MW)'],
        'Buffer Percentage': result['Buffer Percentage'],
        'Overall Spare Capacity (MW)': result['Overall Spare Capacity (MW)']  # Include Overall Spare Capacity
    })
    all_results_df = pd.concat([all_results_df, year_df], ignore_index=True)

# Save to CSV
csv_output_path = 'ev_charger_optimization_results.csv'  # Adjust the path as needed
all_results_df.to_csv(csv_output_path, index=False)

print(f"Results saved to {csv_output_path}")
