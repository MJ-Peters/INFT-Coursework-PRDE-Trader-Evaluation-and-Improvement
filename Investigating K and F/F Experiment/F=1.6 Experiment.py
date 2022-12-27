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


"""Defining functions from week 8 BSE workshop to display results"""


# Small alteration made to save each market session plot as its own png
def n_runs_plot_trades(n, trial_id, start_time, end_time, traders_spec, order_sched, F_value):

    for i in range(1, n):
        trialId = trial_id + '_' + "F=" + str(F_value) + "_" + str(i)
        tdump = open(trialId + '_avg_balance.csv', 'w')

        market_session(trialId, start_time, end_time, traders_spec, order_sched, tdump, dump_all, verbose)

        tdump.close()

        with open(trialId + '_tape.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)
            x = np.empty(0)
            y = np.empty(0)

            for row in reader:
                time = float(row[1])
                price = float(row[2])
                x = np.append(x, time)
                y = np.append(y, price)
            plt.plot(x, y, 'x', color='black');
            plt.savefig("output" + "_" + str(i) + ".png")
            plt.show()


# def get_order_price(i, sched, n, mode):
#     pmin = min(sched[0][0], sched[0][1])
#     pmax = max(sched[0][0], sched[0][1])
#     prange = pmax - pmin
#     stepsize = prange / (n - 1)
#     halfstep = round(stepsize / 2.0)
#
#     if mode == 'fixed':
#         orderprice = pmin + int(i * stepsize)
#     elif mode == 'jittered':
#         orderprice = pmin + int(i * stepsize) + random.randint(-halfstep, halfstep)
#     elif mode == 'random':
#         if len(sched) > 1:
#             # more than one schedule: choose one equiprobably
#             s = random.randint(0, len(sched) - 1)
#             pmin = min(sched[s][0], sched[s][1])
#             pmax = max(sched[s][0], sched[s][1])
#         orderprice = random.randint(pmin, pmax)
#     return orderprice
#
#
# # Small alteration made to save plot as a png to computer
# def make_supply_demand_plot(bids, asks):
#     # total volume up to current order
#     volS = 0
#     volB = 0
#
#     fig, ax = plt.subplots()
#     plt.ylabel('Price')
#     plt.xlabel('Quantity')
#
#     pr = 0
#     for b in bids:
#         if pr != 0:
#             # vertical line
#             ax.plot([volB, volB], [pr, b], 'r-')
#         # horizontal lines
#         line, = ax.plot([volB, volB + 1], [b, b], 'r-')
#         volB += 1
#         pr = b
#     if bids:
#         line.set_label('Demand')
#
#     pr = 0
#     for s in asks:
#         if pr != 0:
#             # vertical line
#             ax.plot([volS, volS], [pr, s], 'b-')
#         # horizontal lines
#         line, = ax.plot([volS, volS + 1], [s, s], 'b-')
#         volS += 1
#         pr = s
#     if asks:
#         line.set_label('Supply')
#
#     if bids or asks:
#         plt.legend()
#     plt.savefig("sup_dem.png")
#     plt.show()
#
#
# def plot_sup_dem(seller_num, sup_ranges, buyer_num, dem_ranges, stepmode):
#     asks = []
#     for s in range(seller_num):
#         asks.append(get_order_price(s, sup_ranges, seller_num, stepmode))
#     asks.sort()
#     bids = []
#     for b in range(buyer_num):
#         bids.append(get_order_price(b, dem_ranges, buyer_num, stepmode))
#     bids.sort()
#     bids.reverse()
#
#     make_supply_demand_plot(bids, asks)


"""End of functions from week 8 BSE workshop"""

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

# Introducing the traders to the market
trader_params = {"k": 4, "F": 1.6, "s_min": -1.0, "s_max": +1.0, "wait_time": wait_time}
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
trial_id = "PRDE_Baseline"

# Number of independent and identically distributed (iid) runs per experiment
# n >= 30 is advised for parametric tests (like the t-test)
n = 10

# Producing empty arrays for time and price values to be added to later
x = np.empty(0)
y = np.empty(0)

# Runs the interval n times from start to finish to plot results and supply/demand chart
# plot_sup_dem(seller_num, [sup_range], buyer_num, [dem_range], stepmode)  # Same supply/demand as baseline
n_runs_plot_trades(n, trial_id, start_time, end_time, traders_spec, order_sched, 2.0)

executionTime = (time.time() - startTime)
print("Execution time in seconds: " + str(executionTime))
