#!/bin/sh
echo "**STARTING**"
mpiexec -n 6 python3 strat_conv.py
mpiexec -n 6 python3 merge.py ugm_24/ --cleanup
mpiexec -n 6 python3 strat_conv_30.py
mpiexec -n 6 python3 merge.py ugm_30/ --cleanup
mpiexec -n 6 python3 strat_conv_40.py
mpiexec -n 6 python3 merge.py ugm_40/ --cleanup
echo "**DONE**"
