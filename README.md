# Abstract
This paper presents an investigation into parameterized response differential evolution (PRDE) trader agents in the Bristol Stock
Exchange (BSE), a minimal limit-order-book-based financial exchange written in Python. The PRDE agent is an extension of two
previous trader agents, PRZI and PRSH, and uses the differential evolution (DE) genetic algorithm to optimize the stochastic
hill-climbing process. This paper describes the use of the PRDE agent in homogeneous experiments on BSE and the application of
statistical testing to assess its performance based on two parameters, the population size "k" and the differential weight "F". These
statistical tests showed that "F" has a significant impact on performance while "k" does not. Subsequently, PRDE was modified to create
the PRADE trader agent that uses a rudimentary adaptive differential evolution (ADE) to actively adjusts its mutation strategy using the
scaling parameter "f". After adaptation, PRADE and PRDE were placed in a balanced group test on BSE producing results that
statistically proved PRADE was significantly more profitable than PRDE. Two further experiments were run with different market
conditions, a market shock and a perfectly elastic market, to test the robustness of the PRADE algorithm. These experiments found
similar results; PRADE demonstrated its adaptability by significantly outperforming PRDE again. Following this, PRADE was altered
further to try and improve its performance: the rand/1 DE algorithm was swapped to the current-to-rand/1 algorithm. This resulted in
a much worse trader agent than the original rand/1 PRADE by a significant margin.

GitHub page for BSE by Dave Cliff, the simulated financial exchange upon which these experiments were run: https://github.com/davecliff/BristolStockExchange

The research paper written by Dave Cliff about BSE can be found using the citation below:

Cliff, D. (2018). BSE: A Minimal Simulation of a Limit-Order-Book Stock Exchange. In M. Affenzeller, et al. (Eds.), Proceedings 30th European Modeling and Simulation Symposium (EMSS 2018), pp. 194-203. DIME University of Genoa.
