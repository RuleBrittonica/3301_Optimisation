import pulp as lp
import pandas as pd
import numpy as np

#Data Initialization
districts = ['Belconnen', 'Nth CBR & City', 'CBR East', 'South CBR', 'Tugg', 'Gungahlin', 'Woden Valley']

#Population Data
current_pop = np.array([104255, 59437, 1495, 31234, 87941, 82118, 77536])
projected_pop = np.array([140297, 90088, 1326, 36196, 88922, 86905, 138739])

#Power Consumption Data (Summer and Winter)
current_power_consumption_summer = np.array([109, 134, 17, 103, 133, 68, 77])  #MW
current_power_consumption_winter = np.array([121, 124, 18, 101, 149, 73, 80])  #MW

projected_power_consumption_summer = np.array([169, 169, 21, 103, 151, 73, 124])  #MW
projected_power_consumption_winter = np.array([192, 181, 21, 108, 153, 71, 156])  #MW

#Power Limits Data (Summer and Winter)
current_power_limits_summer = np.array([171, 209, 54, 142, 219, 76, 95]) * 0.8  #MW
current_power_limits_winter = np.array([190, 257, 54, 142, 252, 84, 114]) * 0.8  #MW

future_power_limits_summer = np.array([226, 264, 54, 114, 219, 131, 150])  #MW
future_power_limits_winter = np.array([245, 312, 54, 114, 252, 139, 169])  #MW

#Minimum Power Limits Data (Summer and Winter)
min_power_limits_summer = -np.array([48, 32.9, 54.4, 18.2, 18.4, 20.9, 32.7])
min_power_limits_winter = -np.array([50.8, 28.7, 56.5, 21.6, 37.7, 15.4, 38.4])

#EV Charger Data
ev_charger_power = 0.35  #MW per charger / charging station
max_chargers_per_iteration = 50  #Maximum chargers that can be added per year
max_chargers_per_season = max_chargers_per_iteration / 2  #Chargers per season
years = 17
seasons = years * 2  #Total number of seasons
safety_buffer = 0.90  #90% of grid capacity


def interpolate_over_seasons(start, end, total_seasons):
    return [start + (end - start) * season / (total_seasons - 1) for season in range(total_seasons)]


season_types = ['summer' if i % 2 == 0 else 'winter' for i in range(seasons)]


yearly_results = []
total_chargers_placed = {district: 0 for district in districts}


previous_total_chargers_placed = 0

for season in range(seasons):

    season_type = season_types[season]


    population_season = interpolate_over_seasons(current_pop, projected_pop, seasons)[season]
    total_population = sum(population_season)


    if season_type == 'summer':
        power_season_start = current_power_consumption_summer
        power_season_end = projected_power_consumption_summer
        power_limits_start = current_power_limits_summer
        power_limits_end = future_power_limits_summer
        min_power_limits = min_power_limits_summer


        opposite_power_season_start = current_power_consumption_winter
        opposite_power_season_end = projected_power_consumption_winter
    else:
        power_season_start = current_power_consumption_winter
        power_season_end = projected_power_consumption_winter
        power_limits_start = current_power_limits_winter
        power_limits_end = future_power_limits_winter
        min_power_limits = min_power_limits_winter


        opposite_power_season_start = current_power_consumption_summer
        opposite_power_season_end = projected_power_consumption_summer


    power_season = interpolate_over_seasons(power_season_start, power_season_end, seasons)[season]
    max_power_limits = interpolate_over_seasons(power_limits_start, power_limits_end, seasons)[season]


    opposite_power_season = interpolate_over_seasons(opposite_power_season_start, opposite_power_season_end, seasons)[season]


    max_baseline_power = np.maximum(power_season, opposite_power_season)

    #Total grid capacity
    total_grid_capacity = sum(max_power_limits)  #MW

    #Total maximum baseline power consumption (used for cumulative constraint)
    total_max_baseline_power = sum(max_baseline_power)  #MW

    #Define the Problem
    prob = lp.LpProblem(f"EV_Charger_Placement_Season_{season+1}", lp.LpMaximize)

    #Decision Variables: Number of Chargers Added in Each District
    chargers_added = lp.LpVariable.dicts(f"ChargersAdded_{season+1}", districts, lowBound=0, cat='Integer')

    #Objective Function: Maximize Total Chargers Added This Season (Up to max_chargers_per_season)
    total_chargers_added = lp.lpSum([chargers_added[district] for district in districts])
    prob += total_chargers_added, "MaximizeTotalChargersAdded"


    prob += total_chargers_added <= max_chargers_per_season, f"MaxChargersPerSeason_Season_{season+1}"


    cumulative_ev_charger_power = ev_charger_power * (previous_total_chargers_placed + total_chargers_added)
    cumulative_total_max_power_consumption = total_max_baseline_power + cumulative_ev_charger_power  #MW

    prob += cumulative_total_max_power_consumption <= safety_buffer * total_grid_capacity, f"CumulativePowerConstraint_Season_{season+1}"


    for i, district in enumerate(districts):
        chargers_this_season = lp.lpSum([chargers_added[district] for district in districts])
        min_chargers = (population_season[i] / total_population) * chargers_this_season * 0.5  #50% of proportional share
        prob += chargers_added[district] >= min_chargers, f"MinChargers_{district}_Season_{season+1}"


    for i, district in enumerate(districts):
        prob += chargers_added[district] <= 0.35 * total_chargers_added, f"MaxChargersPerDistrict_{district}_Season_{season+1}"


    for i, district in enumerate(districts):
        district_consumption = power_season[i] + ev_charger_power * (total_chargers_placed[district] + chargers_added[district])
        prob += district_consumption >= min_power_limits[i], f"MinPowerLimit_{district}_Season_{season+1}"


    prob += chargers_added['CBR East'] <= 0.05 * total_chargers_added, f"CBR_East_Max_5_Percent_Season_{season+1}"

    #Solve the Linear Programming Problem
    prob.solve()

    #Check if the Solution is Optimal
    if lp.LpStatus[prob.status] != 'Optimal':
        print(f"\nLinear program no longer optimal in Season {season+1}-Setting chargers added to zero for this and subsequent seasons.")

        chargers_added_season = {district: 0 for district in districts}
    else:

        chargers_added_season = {district: int(lp.value(chargers_added[district])) for district in districts}

        for district in districts:
            total_chargers_placed[district] += chargers_added_season[district]

        previous_total_chargers_placed += sum(chargers_added_season.values())


    ev_charger_consumption = {
        district: total_chargers_placed[district] * ev_charger_power for district in districts
    }


    total_consumption_district = {
        district: power_season[i] + ev_charger_consumption[district] for i, district in enumerate(districts)
    }


    spare_capacity_district = {
        district: max_power_limits[i] - total_consumption_district[district] for i, district in enumerate(districts)
    }


    cumulative_total_max_power_consumption = total_max_baseline_power + sum(ev_charger_consumption.values())  #MW


    buffer_percentage = cumulative_total_max_power_consumption / (safety_buffer * total_grid_capacity)


    overall_spare_capacity = (safety_buffer * total_grid_capacity) - cumulative_total_max_power_consumption  #MW


    current_year = (season // 2) + 1

    #Store Results
    yearly_results.append({
        'Season': season + 1,
        'Year': current_year,
        'Season Type': season_type.capitalize(),
        'District': districts,
        'Baseline Power Consumption (MW)': list(power_season),
        'Max Baseline Power Consumption (MW)': list(max_baseline_power),
        'Power Consumption from Chargers (MW)': [ev_charger_consumption[district] for district in districts],
        'Total Power Consumption (MW)': [total_consumption_district[district] for district in districts],
        'Chargers Added This Season': [chargers_added_season[district] for district in districts],
        'Total Chargers Placed': [total_chargers_placed[district] for district in districts],
        'Spare Capacity (MW)': [spare_capacity_district[district] for district in districts],
        'Cumulative Total Power Consumption (MW)': cumulative_total_max_power_consumption,
        'Buffer Percentage': buffer_percentage,
        'Overall Spare Capacity (MW)': overall_spare_capacity
    })

#Create a DataFrame for All Results
all_results_df = pd.DataFrame()

for result in yearly_results:
    season_df = pd.DataFrame({
        'Season': [result['Season']] * len(districts),
        'Year': [result['Year']] * len(districts),
        'Season Type': [result['Season Type']] * len(districts),
        'District': result['District'],
        'Baseline Power Consumption (MW)': result['Baseline Power Consumption (MW)'],
        'Power Consumption from Chargers (MW)': result['Power Consumption from Chargers (MW)'],
        'Total Power Consumption (MW)': result['Total Power Consumption (MW)'],
        'Chargers Added This Season': result['Chargers Added This Season'],
        'Total Chargers Placed': result['Total Chargers Placed'],
        'Spare Capacity (MW)': result['Spare Capacity (MW)'],
        'Cumulative Total Power Consumption (MW)': [result['Cumulative Total Power Consumption (MW)']] * len(districts),
        'Buffer Percentage': [result['Buffer Percentage']] * len(districts),
        'Overall Spare Capacity (MW)': [result['Overall Spare Capacity (MW)']] * len(districts)
    })
    all_results_df = pd.concat([all_results_df, season_df], ignore_index=True)

#Save to CSV
csv_output_path = 'ev_charger_optimization_results.csv'
all_results_df.to_csv(csv_output_path, index=False)

print(f"Results saved to {csv_output_path}")
