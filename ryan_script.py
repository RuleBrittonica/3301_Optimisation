import pulp as lp
import pandas as pd
import numpy as np

districts = ['Belconnen', 'Nth CBR & City', 'CBR East', 'South CBR', 'Tugg', 'Gungahlin', 'Woden Valley']
current_pop = [104000, 59500, 1495, 31234, 87941, 82118, 77536]
projected_pop = [140297, 90088, 1326, 36196, 88922, 86905, 138739]
current_power_consumption = [109, 134, 17, 103, 133, 68, 77]  # MW
projected_power_consumption = [169, 169, 21, 103, 151, 73, 124]  # MW
max_power_limits = [226, 264, 54, 114, 219, 131, 150]  # MW
ev_charger_power = 0.250  # MW per charger
chargers_added_per_year = 50  # We want to add exactly 50 chargers per year
years = 15
safety_buffer = 0.85

# Function to perform linear interpolation for power consumption over years
def interpolate(start, end, years):
    return [start + (end - start) * year / (years - 1) for year in range(years)]

# Initialize storage for results and running tally of chargers placed in each district
yearly_results = []
total_chargers_placed = {district: 0 for district in districts}

# Loop through each year and perform optimization
for year in range(years):
    # Interpolate population-based power consumption (without EV chargers)
    power_year = interpolate(np.array(current_power_consumption), np.array(projected_power_consumption), years)[year]
    population_year = interpolate(np.array(current_pop), np.array(projected_pop), years)[year]

    # Total population for the current year
    total_population = sum(population_year)

    # Create a LP problem instance for the current year
    prob = lp.LpProblem(f"EV_Charger_Placement_Year_{year+1}", lp.LpMinimize)

    # Decision variables: Power allocations for EV chargers in each district
    power_allocations = lp.LpVariable.dicts(f"PowerAllocation_{year+1}", districts, lowBound=0)

    # Objective function: minimize the total power allocated to EV chargers
    prob += lp.lpSum([power_allocations[district] for district in districts])

    # Constraints: Power allocations should not exceed substation limits minus current projected power usage
    for i, district in enumerate(districts):
        prob += power_allocations[district] <= max_power_limits[i] - power_year[i], f"MaxPowerLimit_{district}_Year_{year+1}"

    # Population-based proportionality constraint for charger distribution
    for i, district in enumerate(districts):
        prob += power_allocations[district] * (1 / ev_charger_power) >= chargers_added_per_year * (population_year[i] / total_population), f"PopulationConstraint_{district}_Year_{year+1}"

    # Total number of chargers across all districts should still be exactly 50
    prob += lp.lpSum([power_allocations[district] * (1 / ev_charger_power) for district in districts]) == chargers_added_per_year, f"ChargerConstraint_Year_{year+1}"

    # Solve the problem
    prob.solve()

    # Check if the solution is still optimal
    if lp.LpStatus[prob.status] != 'Optimal':
        print(f"\nLinear program no longer optimal in Year {year+1}. Stopping further optimization.")
        break

    # Extract the results for the current year
    optimized_power_allocations = {district: power_allocations[district].varValue for district in districts}
    chargers_added = {district: optimized_power_allocations[district] // ev_charger_power for district in districts}

    # Update the running tally of total chargers placed in each district
    for district in districts:
        total_chargers_placed[district] += chargers_added[district]

    # Calculate power consumption for EV chargers placed in the current and previous years
    ev_charger_power_consumption = {
        district: total_chargers_placed[district] * ev_charger_power for district in districts
    }

    # Calculate total power consumption (baseline power + power from EV chargers)
    total_power_consumption = {
        district: power_year[i] + ev_charger_power_consumption[district] for i, district in enumerate(districts)
    }

    # Calculate spare capacity in the grid
    spare_capacity = {
        district: max_power_limits[i] - total_power_consumption[district] for i, district in enumerate(districts)
    }

    # Calculate buffer percentage (how close the total power is to 85% of the maximum power)
    buffer_percentage = [
        total_power_consumption[district] / max_power_limits[i] for i, district in enumerate(districts)
    ]

    # Store results in the yearly_results list
    yearly_results.append({
        'Year': year + 1,
        'District': districts,
        'Baseline Power Consumption (kW)': list(power_year),
        'Power Consumption from Chargers (kW)': [ev_charger_power_consumption[district] for district in districts],
        'Total Power Consumption (kW)': [total_power_consumption[district] for district in districts],
        'Chargers Added This Year': [chargers_added[district] for district in districts],
        'Total Chargers Placed': [total_chargers_placed[district] for district in districts],
        'Buffer': buffer_percentage,
        'Spare Capacity (kW)': [spare_capacity[district] for district in districts]  # Add spare capacity
    })

# Create a DataFrame for all results
all_results_df = pd.DataFrame()

for result in yearly_results:
    year_df = pd.DataFrame({
        'Year': result['Year'],
        'District': result['District'],
        'Baseline Power Consumption (kW)': result['Baseline Power Consumption (kW)'],
        'Power Consumption from Chargers (kW)': result['Power Consumption from Chargers (kW)'],
        'Total Power Consumption (kW)': result['Total Power Consumption (kW)'],
        'Chargers Added This Year': result['Chargers Added This Year'],
        'Total Chargers Placed': result['Total Chargers Placed'],
        'Buffer': result['Buffer'],
        'Spare Capacity (kW)': result['Spare Capacity (kW)']  # Include spare capacity
    })
    all_results_df = pd.concat([all_results_df, year_df], ignore_index=True)

# Save to CSV
csv_output_path = 'ev_charger_optimization_results.csv'  # Adjust the path as needed
all_results_df.to_csv(csv_output_path, index=False)

print(f"Results saved to {csv_output_path}")
