#!/bin/sh
echo "**STARTING**"
mpiexec -n 6 python3 strat_conv.py
mpiexec -n 6 python3 merge.py ugm_28_1hr/ --cleanup
echo "**DONE**"
