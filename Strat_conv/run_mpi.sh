#!/bin/sh
echo "**STARTING**"
mpiexec -n 8 python3 strat_conv.py
mpiexec -n 8 python3 merge.py temp_salinity_8x12pm/ --cleanup
echo "**DONE**"
