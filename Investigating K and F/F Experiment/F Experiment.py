"""
Analysis of the behaviour of PRDE when the differential weight, F, is being altered.
"""

# Importing relevant packages to run the experiment
import matplotlib.pyplot as plt
import numpy as np
import csv
import random
from BSE import market_session
import time
startTime = time.time()


"""Defining function from week 8 BSE workshop to display results"""


def n_runs(n, trial_id, start_time, end_time, traders_spec, order_sched, F_value):

    for i in range(1, n):
        trialId = trial_id + '_' + "F=" + str(F_value) + "_" + str(i)
        tdump = open(trialId + '_avg_balance.csv', 'w')

        market_session(trialId, start_time, end_time, traders_spec, order_sched, tdump, dump_all, verbose)

        tdump.close()


"""End of function from week 8 BSE workshop"""

# Defining length of experiment to be 30 (simulated) days ~2.6e6 seconds
start_time = 0
end_time = 30 * 24 * 60 * 60  # Converting 30 days to seconds.
wait_time = 7200  # each strategy gets evaluated for 7200 seconds (2hrs) each -> whole set K takes 8hrs before mutation

# Defining the supply and demand schedule as symmetric with arbitrary range
sup_range = (50, 150)
dem_range = sup_range
stepmode = "fixed"
supply_schedule = [{'from': start_time, 'to': end_time, 'ranges':
                    [sup_range], 'stepmode': stepmode}]
demand_schedule = [{'from': start_time, 'to': end_time, 'ranges':
                    [dem_range], 'stepmode': stepmode}]

F_vals = [1.2, 1.6, 2.0]
for F in F_vals:
    # Introducing the traders to the market
    trader_params = {"k": 4, "F": F, "s_min": -1.0, "s_max": +1.0, "wait_time": wait_time}
    sellers_spec = [("PRDE", 30, trader_params)]
    seller_num = 30
    buyers_spec = sellers_spec
    buyer_num = seller_num
    traders_spec = {"sellers": sellers_spec, "buyers": buyers_spec}

    # Defining the order schedule, interval 10 -> ~2.6e5 orders given to each trader over the 30 days
    # drip-poisson time mode is used to more closely simulate a real market
    order_interval = 10
    order_sched = {"sup": supply_schedule, "dem": demand_schedule,
                   "interval": order_interval, "timemode": "drip-poisson"}

    # Don't want lots of output into the terminal, not useful to me at this stage
    verbose = False
    # dump all = False as we only want the total profits from each trader for the entire run for our stats analysis
    dump_all = False
    # Naming our experiment so CSVs written are appropriately named
    trial_id = "PRDE_F_Experiment"

    # Number of independent and identically distributed (iid) runs per experiment
    # n >= 30 is advised for parametric tests (like the t-test)
    n = 10

    # Producing empty arrays for time and price values to be added to later
    x = np.empty(0)
    y = np.empty(0)

    # Runs the interval n times from start to finish to plot results and supply/demand chart
    # plot_sup_dem(seller_num, [sup_range], buyer_num, [dem_range], stepmode)  # Same supply/demand as baseline
    n_runs(n, trial_id, start_time, end_time, traders_spec, order_sched, F)

executionTime = (time.time() - startTime)
print("Execution time in seconds: " + str(executionTime))
