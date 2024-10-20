#!/bin/bash

source ".venv/Scripts/activate"

./run.sh

echo "Solar Optimisation finished"

cd no_solar

./run.sh

echo "No Solar Optimisation finished"
