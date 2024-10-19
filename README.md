# 3301_Optimisation

A codebase containing an optimisation study and relevant plotting for group ERPG02. The repository will be public until 3 weeks after the assignment is due. 

The codebase can be broken up into 3 groups

## Image Processing:
The original image is available from this link: https://yoursayconversations.act.gov.au/act-planning-review/draft-district-strategies
From here, it was processed with the `map_processing.py` script into `images/suburbs_processed.png`. This image is the image that all of the gifs are overlaid on. 

## Optimisation without Solar:
Inside the `no_solar` directory, there is a script called `matt_script.py`. This file contains our first attempt at the optimisation, without considering solar energy. It outputs its data as a .csv file, which the following scripts work off of. `ryan_script.py` is the first draft of the optimisation, but had numerous flaws. 

- `0_charger_consumption.py` - Plots the power consumption by the chargers in the district
- `1_spare_capacity.py` - Plots the spare capacity in the grid (i.e. grid capacity - power consumption - charger power)
- `2_total_chargers.py` - Plots the total number of chargers added over time

## Optimisation with Solar: 

This is the scripts in the main folder of this repository outline the optimisation considering rooftop solar power production. This power production is simply estimated based on the population and expected percentage of households with rooftop solar, and added onto the available power. 

The following files are very similar to their `no_solar` counterparts
- `0_charger_consumption.py` - Plots the power consumption by the chargers in the district
- `1_spare_capacity.py` - Plots the spare capacity in the grid (i.e. grid capacity - power consumption - charger power)
- `2_total_chargers.py` - Plots the total number of chargers added over time

With
- `3_solar_generation.py` Plotting the expected solar generation per district. 

