#!/bin/bash

source ".venv/Scripts/activate"

python -u "matt_script.py"

python -u "0_charger_consumption.py" &
python -u "1_spare_capacity.py" &
python -u "2_total_chargers.py" &
python -u "3_solar_generation.py" &

# Copy the gifs to the output folder
# Make sure the python scripts have finished running before copying the gifs
wait $!
wait $!

rm -rf "output"
mkdir "output"


cp "chargers_growth.gif" "output/chargers_growth.gif"
cp "spare_capacity_growth.gif" "output/spare_capacity_growth.gif"
cp "power_consumption_growth.gif" "output/power_consumption_growth.gif"
cp "solar_generation_growth.gif" "output/solar_generation_growth.gif"

wait $!

echo "All scripts have finished"