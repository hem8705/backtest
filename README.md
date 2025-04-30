# Cont & Kukanov Smart Order Router Back-testing

This project implements a back-testing framework for a Smart Order Router (SOR) based on the static cost model introduced by Cont & Kukanov in "Optimal Order Placement in Limit Order Markets". The SOR splits a 5,000-share buy order across multiple venues to minimize execution costs.

## Approach

The implementation follows the algorithm described in the `allocator_pseudocode.txt` file, which provides a static allocation strategy across N venues for a single snapshot. The key components of the implementation are:

1. **Data Processing**: The system loads and preprocesses market data from `l1_day.csv`, extracting venue-specific ask prices and sizes.

2. **Allocator Implementation**: The core allocator algorithm as described in the pseudocode is implemented, computing the optimal order allocation across venues based on cost minimization.

3. **Back-testing Engine**: A simulation engine that replays market data and executes orders according to the allocation strategy, tracking execution statistics.

4. **Parameter Tuning**: A grid search over the three risk parameters (lambda_over, lambda_under, theta_queue) to find the combination that minimizes execution costs.

5. **Baseline Strategies**: Implementation of three baseline strategies for comparison:
   - Best Ask: Always takes the best available ask across venues
   - TWAP: Time-Weighted Average Price strategy with 60-second buckets
   - VWAP: Volume-Weighted Average Price strategy weighted by displayed size

6. **Performance Evaluation**: Comparison of the tuned SOR performance against the baseline strategies, reporting savings in basis points.

## Parameter Ranges

The parameter search is conducted over the following ranges:

- **lambda_over**: [0.001, 0.005, 0.01, 0.05, 0.1]
   - This parameter controls the penalty for exceeding the target quantity. Lower values allow more aggressive trading, while higher values discourage overtrading.

- **lambda_under**: [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
   - This parameter controls the penalty for not filling the target quantity. Higher values prioritize complete executions, potentially at the expense of higher costs.

- **theta_queue**: [0.0001, 0.0005, 0.001, 0.005, 0.01]
   - This parameter represents the queue-risk penalty. Higher values make the model more conservative about queue position, while lower values allow more aggressive limit order placement.

These ranges were chosen to cover a reasonable spectrum of risk preferences, from very aggressive to very conservative execution strategies.

## Suggested Improvement: Queue Position Model

One significant area for improving fill realism is implementing a more sophisticated queue position model. The current implementation assumes that orders are filled strictly in order of arrival (FIFO), and all displayed size at a given price level is equally accessible.

A more realistic approach would be to incorporate a probabilistic model for queue position that accounts for:

1. **Queue Dynamics**: In real markets, limit order queues exhibit complex dynamics, with orders being added and removed at various positions. A probabilistic model could account for the likelihood of orders being filled based on their position in the queue.

2. **Adverse Selection**: Orders at the front of the queue are more likely to be filled when prices are about to move unfavorably (due to informed trading). This could be modeled as a price-dependent fill probability.

3. **Fill Rates**: Different venues have different fill rates for similarly positioned orders due to variations in order flow and market participant behavior. Historical fill rates could be incorporated into the model.

Implementation would involve:
- Estimating a venue-specific fill probability function based on queue position
- Adjusting the expected execution quantity in the cost function to account for this probability
- Parameterizing the model based on historical fill data

This enhancement would make the SOR more robust in real-world scenarios where queue position significantly impacts execution outcomes, potentially leading to further cost savings.