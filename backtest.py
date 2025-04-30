#!/usr/bin/env python3
"""
Cont & Kukanov Smart Order Router Back-testing

This script implements a back-test for the static cost model introduced by Cont & Kukanov
for optimal order placement in limit order markets. It simulates the execution of a 5,000-share
buy order across multiple venues using the provided market data.

The script tunes the three risk parameters (lambda_over, lambda_under, theta_queue) to minimize
total execution cost, and compares the performance against three baseline strategies.
"""

import pandas as pd
import numpy as np
import json
from typing import List, Dict, Tuple, Any
import time


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# Global constants
TARGET_ORDER_SIZE = 5000  # Target size in shares
STEP_SIZE = 100           # Step size for allocator (as per pseudocode)


def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess the market data.
    
    Args:
        file_path: Path to the L1 market data CSV file
    
    Returns:
        DataFrame with preprocessed market data
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Convert timestamps to datetime
    df['ts_event'] = pd.to_datetime(df['ts_event'])
    
    # Keep only the first message per publisher_id (venue) per timestamp
    df = df.sort_values(['ts_event', 'publisher_id']).drop_duplicates(['ts_event', 'publisher_id'])
    
    # Sort by timestamp
    df = df.sort_values('ts_event')
    
    # Extract venue information: publisher_id, ask price, and ask size
    venue_data = df[['ts_event', 'publisher_id', 'ask_px_00', 'ask_sz_00']].copy()
    
    # Filter out rows with NaN in ask price or size
    venue_data = venue_data.dropna(subset=['ask_px_00', 'ask_sz_00'])
    
    # Convert numeric columns to appropriate types
    venue_data['ask_px_00'] = venue_data['ask_px_00'].astype(float)
    venue_data['ask_sz_00'] = venue_data['ask_sz_00'].astype(int)
    
    print(f"Processed {len(venue_data)} market data records.")
    return venue_data


class VenueState:
    """
    Represents the state of a single venue, including ask price, size, and execution fees/rebates.
    """
    def __init__(self, venue_id: int, ask: float, ask_size: int, fee: float = 0.0, rebate: float = 0.0):
        self.venue_id = venue_id
        self.ask = ask
        self.ask_size = ask_size
        self.fee = fee
        self.rebate = rebate
    
    def __repr__(self) -> str:
        return f"Venue({self.venue_id}, ask={self.ask}, size={self.ask_size}, fee={self.fee}, rebate={self.rebate})"


def compute_cost(split: List[int], venues: List[VenueState], order_size: int, 
                lambda_over: float, lambda_under: float, theta_queue: float) -> float:
    """
    Compute the expected cost of a specific order allocation across venues.
    
    This function implements the compute_cost function from the Cont-Kukanov pseudocode.
    
    Args:
        split: List of shares allocated to each venue
        venues: List of venue state objects
        order_size: Target order size
        lambda_over: Cost penalty per extra share bought
        lambda_under: Cost penalty per unfilled share
        theta_queue: Queue-risk penalty
        
    Returns:
        Total expected cost of the allocation
    """
    executed = 0
    cash_spent = 0.0
    
    for i in range(len(venues)):
        # Calculate executed quantity at this venue
        exe = min(split[i], venues[i].ask_size)
        executed += exe
        
        # Calculate cash spent at this venue
        cash_spent += exe * (venues[i].ask + venues[i].fee)
        
        # Calculate maker rebate if applicable
        maker_rebate = max(split[i] - exe, 0) * venues[i].rebate
        cash_spent -= maker_rebate
    
    # Calculate underfill and overfill
    underfill = max(order_size - executed, 0)
    overfill = max(executed - order_size, 0)
    
    # Calculate penalties
    risk_pen = theta_queue * (underfill + overfill)
    cost_pen = lambda_under * underfill + lambda_over * overfill
    
    return cash_spent + risk_pen + cost_pen


def allocate(order_size: int, venues: List[VenueState], 
            lambda_over: float, lambda_under: float, theta_queue: float) -> Tuple[List[int], float]:
    """
    Allocate shares across venues to minimize expected cost.
    
    This function implements the allocate function from the Cont-Kukanov pseudocode.
    Optimized to reduce unnecessary calculations and improve performance.
    
    Args:
        order_size: Target order size
        venues: List of venue state objects
        lambda_over: Cost penalty per extra share bought
        lambda_under: Cost penalty per unfilled share
        theta_queue: Queue-risk penalty
        
    Returns:
        Tuple of (best allocation, expected cost)
    """
    # Optimization: Special case for small number of venues
    if len(venues) == 0:
        return [], 0.0
    
    if len(venues) == 1:
        # With one venue, the only valid allocation is to send all shares there
        allocation = [min(order_size, venues[0].ask_size)]
        cost = compute_cost(allocation, venues, order_size, lambda_over, lambda_under, theta_queue)
        return allocation, cost
    
    step = STEP_SIZE  # Search in 100-share chunks
    
    # Optimization: For small orders, try simple allocations first
    if order_size <= 500:
        # Try sending all to the venue with the best ask price
        best_venue_index = min(range(len(venues)), key=lambda i: venues[i].ask)
        simple_allocation = [0] * len(venues)
        simple_allocation[best_venue_index] = order_size
        simple_cost = compute_cost(simple_allocation, venues, order_size, lambda_over, lambda_under, theta_queue)
        
        # Start with this as the best so far
        best_cost = simple_cost
        best_split = simple_allocation
    else:
        best_cost = float('inf')
        best_split = []
    
    # Generate all possible allocation combinations, but with optimizations
    splits = [[]]  # Start with an empty allocation list
    
    # Optimization: Filter venues by ask price and size
    # Sort venues by ask price to try the cheapest ones first
    venue_indices = sorted(range(len(venues)), key=lambda i: venues[i].ask)
    
    # Generate all possible allocation combinations
    for v_idx in range(len(venues)):
        v = venue_indices[v_idx]  # Use sorted indices
        new_splits = []
        
        for alloc in splits:
            used = sum(alloc)
            remaining = order_size - used
            
            # Skip if we've already allocated all shares
            if remaining <= 0:
                new_splits.append(alloc + [0])
                continue
                
            max_v = min(remaining, venues[v].ask_size)
            
            # Optimization: If this is the last venue, only consider the allocation that 
            # uses exactly the remaining shares
            if v_idx == len(venues) - 1:
                # Round to nearest step
                q = (max_v // step) * step
                new_alloc = alloc + [q]
                
                # If this allocation is valid, evaluate it
                if sum(new_alloc) == order_size:
                    cost = compute_cost(new_alloc, venues, order_size, lambda_over, lambda_under, theta_queue)
                    if cost < best_cost:
                        best_cost = cost
                        best_split = new_alloc
                
                # Also try allocating the remaining shares rounded to the step
                remainder = order_size - sum(new_alloc)
                if remainder > 0 and remainder <= venues[v].ask_size:
                    new_alloc[-1] += remainder
                    cost = compute_cost(new_alloc, venues, order_size, lambda_over, lambda_under, theta_queue)
                    if cost < best_cost:
                        best_cost = cost
                        best_split = new_alloc.copy()
                
                continue
            
            # For other venues, try different allocations
            for q in range(0, max_v + 1, step):
                new_alloc = alloc + [q]
                new_splits.append(new_alloc)
                
                # If we've allocated exactly the order size, evaluate this allocation immediately
                if sum(new_alloc) + (len(venues) - len(new_alloc)) * 0 == order_size:
                    # Add zeros for remaining venues
                    complete_alloc = new_alloc + [0] * (len(venues) - len(new_alloc))
                    cost = compute_cost(complete_alloc, venues, order_size, lambda_over, lambda_under, theta_queue)
                    if cost < best_cost:
                        best_cost = cost
                        best_split = complete_alloc
        
        splits = new_splits
    
    # Ensure we have a valid allocation
    if not best_split:
        # Fallback: Send all shares to the venue with best ask price
        best_venue_index = min(range(len(venues)), key=lambda i: venues[i].ask)
        best_split = [0] * len(venues)
        best_split[best_venue_index] = min(order_size, venues[best_venue_index].ask_size)
        best_cost = compute_cost(best_split, venues, order_size, lambda_over, lambda_under, theta_queue)
    
    return best_split, best_cost


def execute_strategy(data: pd.DataFrame, venues_config: Dict[int, Dict], 
                     lambda_over: float, lambda_under: float, theta_queue: float) -> Dict:
    """
    Run a back-test using the Cont-Kukanov allocator to execute orders as market data unfolds.
    Optimized version that processes data in chunks for better performance.
    
    Args:
        data: Preprocessed market data
        venues_config: Dictionary mapping venue IDs to their fee/rebate configuration
        lambda_over: Cost penalty parameter for over-execution
        lambda_under: Cost penalty parameter for under-execution
        theta_queue: Queue risk penalty parameter
        
    Returns:
        Dictionary with execution results
    """
    remaining_shares = TARGET_ORDER_SIZE
    cash_spent = 0.0
    filled_shares = 0
    weighted_price_sum = 0.0  # For calculating weighted average price
    
    # Precompute venue configurations to avoid repeated dictionary lookups
    venue_configs = {}
    for venue_id in data['publisher_id'].unique():
        config = venues_config.get(venue_id, {'fee': 0.0, 'rebate': 0.0})
        venue_configs[venue_id] = (config.get('fee', 0.0), config.get('rebate', 0.0))
    
    # Group data by timestamp to get snapshots
    # Use numpy operations for performance where possible
    timestamps = data['ts_event'].unique()
    
    for ts in timestamps:
        if remaining_shares <= 0:
            break
        
        # Get snapshot for this timestamp
        snapshot = data[data['ts_event'] == ts]
        
        # Quickly create venue states
        venues = []
        for _, row in snapshot.iterrows():
            venue_id = row['publisher_id']
            ask = row['ask_px_00']
            ask_size = row['ask_sz_00']
            
            # Get venue configuration (fee and rebate)
            fee, rebate = venue_configs[venue_id]
            
            venues.append(VenueState(venue_id, ask, ask_size, fee, rebate))
        
        # Skip if no valid venues
        if not venues:
            continue
        
        # Calculate optimal allocation
        allocation, _ = allocate(
            min(remaining_shares, TARGET_ORDER_SIZE), 
            venues,
            lambda_over,
            lambda_under,
            theta_queue
        )
        
        # Execute orders according to allocation
        for i, shares in enumerate(allocation):
            venue = venues[i]
            executed = min(shares, venue.ask_size)
            
            if executed > 0:
                # Record execution details - calculate weighted average on the fly
                this_execution_cost = executed * venue.ask
                cash_spent += this_execution_cost
                filled_shares += executed
                weighted_price_sum += this_execution_cost
                remaining_shares -= executed
        
        if remaining_shares <= 0:
            break
    
    # Calculate average execution price (weighted)
    avg_price = weighted_price_sum / filled_shares if filled_shares > 0 else 0.0
    
    return {
        'shares_filled': filled_shares,
        'cash_spent': cash_spent,
        'avg_price': avg_price,
        'remaining': remaining_shares
    }


def execute_best_ask_strategy(data: pd.DataFrame, venues_config: Dict[int, Dict]) -> Dict:
    """
    Run a "take the best ask" baseline strategy.
    
    Args:
        data: Preprocessed market data
        venues_config: Dictionary mapping venue IDs to their fee/rebate configuration
        
    Returns:
        Dictionary with execution results
    """
    remaining_shares = TARGET_ORDER_SIZE
    cash_spent = 0.0
    filled_shares = 0
    execution_prices = []
    
    # Group data by timestamp to get snapshots
    grouped = data.groupby('ts_event')
    
    for ts, snapshot in grouped:
        if remaining_shares <= 0:
            break
            
        # Find the best (lowest) ask across venues
        if len(snapshot) > 0:
            best_row = snapshot.loc[snapshot['ask_px_00'].idxmin()]
            venue_id = best_row['publisher_id']
            ask = best_row['ask_px_00']
            ask_size = best_row['ask_sz_00']
            
            # Execute order at the best ask
            executed = min(remaining_shares, ask_size)
            
            # Record execution details
            cash_spent += executed * ask
            filled_shares += executed
            execution_prices.extend([ask] * executed)
            remaining_shares -= executed
        
        if remaining_shares <= 0:
            break
    
    # Calculate average execution price
    avg_price = np.mean(execution_prices) if execution_prices else 0
    
    return {
        'shares_filled': filled_shares,
        'cash_spent': cash_spent,
        'avg_price': avg_price,
        'remaining': remaining_shares
    }


def execute_twap_strategy(data: pd.DataFrame, venues_config: Dict[int, Dict], bucket_seconds: int = 60) -> Dict:
    """
    Run a Time-Weighted Average Price (TWAP) baseline strategy with specified bucket size.
    
    Args:
        data: Preprocessed market data
        venues_config: Dictionary mapping venue IDs to their fee/rebate configuration
        bucket_seconds: Size of time buckets in seconds
        
    Returns:
        Dictionary with execution results
    """
    remaining_shares = TARGET_ORDER_SIZE
    cash_spent = 0.0
    filled_shares = 0
    execution_prices = []
    
    # Calculate the total time range
    start_time = data['ts_event'].min()
    end_time = data['ts_event'].max()
    total_seconds = (end_time - start_time).total_seconds()
    
    # Calculate number of buckets
    num_buckets = max(1, int(total_seconds / bucket_seconds))
    
    # Calculate shares per bucket
    shares_per_bucket = TARGET_ORDER_SIZE / num_buckets
    
    # Create time buckets
    buckets = [start_time + pd.Timedelta(seconds=i*bucket_seconds) for i in range(num_buckets + 1)]
    
    # Execute TWAP strategy
    for i in range(len(buckets) - 1):
        if remaining_shares <= 0:
            break
            
        # Get data for this time bucket
        bucket_data = data[(data['ts_event'] >= buckets[i]) & (data['ts_event'] < buckets[i+1])]
        
        # Calculate shares to execute in this bucket
        bucket_shares = min(remaining_shares, int(shares_per_bucket))
        
        if bucket_shares <= 0 or len(bucket_data) == 0:
            continue
        
        # Sort venues by ask price
        bucket_data = bucket_data.sort_values('ask_px_00')
        
        # Execute orders across venues
        for _, row in bucket_data.iterrows():
            venue_id = row['publisher_id']
            ask = row['ask_px_00']
            ask_size = row['ask_sz_00']
            
            executed = min(bucket_shares, ask_size)
            
            if executed > 0:
                # Record execution details
                cash_spent += executed * ask
                filled_shares += executed
                execution_prices.extend([ask] * executed)
                bucket_shares -= executed
                remaining_shares -= executed
            
            if bucket_shares <= 0:
                break
    
    # Calculate average execution price
    avg_price = np.mean(execution_prices) if execution_prices else 0
    
    return {
        'shares_filled': filled_shares,
        'cash_spent': cash_spent,
        'avg_price': avg_price,
        'remaining': remaining_shares
    }


def execute_vwap_strategy(data: pd.DataFrame, venues_config: Dict[int, Dict]) -> Dict:
    """
    Run a Volume-Weighted Average Price (VWAP) baseline strategy.
    
    Args:
        data: Preprocessed market data
        venues_config: Dictionary mapping venue IDs to their fee/rebate configuration
        
    Returns:
        Dictionary with execution results
    """
    remaining_shares = TARGET_ORDER_SIZE
    cash_spent = 0.0
    filled_shares = 0
    execution_prices = []
    
    # Group data by timestamp to get snapshots
    grouped = data.groupby('ts_event')
    
    for ts, snapshot in grouped:
        if remaining_shares <= 0:
            break
            
        # Calculate the total displayed size across all venues
        total_size = snapshot['ask_sz_00'].sum()
        if total_size <= 0:
            continue
        
        # Calculate the proportion of shares to execute at each venue
        for _, row in snapshot.iterrows():
            venue_id = row['publisher_id']
            ask = row['ask_px_00']
            ask_size = row['ask_sz_00']
            
            # Weight by displayed size
            proportion = ask_size / total_size
            shares_to_execute = min(remaining_shares, int(proportion * remaining_shares))
            executed = min(shares_to_execute, ask_size)
            
            if executed > 0:
                # Record execution details
                cash_spent += executed * ask
                filled_shares += executed
                execution_prices.extend([ask] * executed)
                remaining_shares -= executed
        
        if remaining_shares <= 0:
            break
    
    # Calculate average execution price
    avg_price = np.mean(execution_prices) if execution_prices else 0
    
    return {
        'shares_filled': filled_shares,
        'cash_spent': cash_spent,
        'avg_price': avg_price,
        'remaining': remaining_shares
    }


def parameter_search(data: pd.DataFrame, venues_config: Dict[int, Dict]) -> Dict[str, Any]:
    """
    Run a smart search to find optimal parameter values.
    
    Args:
        data: Preprocessed market data
        venues_config: Dictionary mapping venue IDs to their fee/rebate configuration
        
    Returns:
        Dictionary with optimal parameters and results
    """
    print("Starting parameter search...")
    start_time = time.time()
    
    # Define parameter grids - using a coarse grid first, then refining
    # Initial coarse grid
    lambda_over_values = [0.001, 0.01, 0.1]
    lambda_under_values = [0.001, 0.01, 0.1]
    theta_queue_values = [0.0001, 0.001, 0.01]
    
    best_result = None
    best_params = None
    best_avg_price = float('inf')
    
    # First pass: coarse grid search
    total_combinations = len(lambda_over_values) * len(lambda_under_values) * len(theta_queue_values)
    progress_count = 0
    elapsed_time = 0
    
    for lambda_over in lambda_over_values:
        for lambda_under in lambda_under_values:
            for theta_queue in theta_queue_values:
                # Execute strategy with these parameters
                result = execute_strategy(
                    data, 
                    venues_config, 
                    lambda_over, 
                    lambda_under, 
                    theta_queue
                )
                
                # Check if this is better than our best so far
                if result['shares_filled'] > 0 and result['avg_price'] < best_avg_price:
                    best_result = result
                    best_params = {
                        'lambda_over': lambda_over,
                        'lambda_under': lambda_under,
                        'theta_queue': theta_queue
                    }
                    best_avg_price = result['avg_price']
                
                # Print progress
                progress_count += 1
                if progress_count % 5 == 0:
                    elapsed_time = time.time() - start_time
                    print(f"Progress: {progress_count}/{total_combinations} combinations tested. Time: {elapsed_time:.2f}s")
    
    # Check if we have enough time for refinement
    time_per_combination = elapsed_time / total_combinations
    
    # If we have time, refine around the best parameters
    if elapsed_time < 60:  # If first pass took less than 60 seconds
        print("Refining parameter search around best values...")
        
        # Define refinement ranges around best parameters
        lambda_over_best = best_params['lambda_over']
        lambda_under_best = best_params['lambda_under']
        theta_queue_best = best_params['theta_queue']
        
        # Create refined parameter grids
        lambda_over_refined = [max(0.0001, lambda_over_best * 0.5), lambda_over_best, min(1.0, lambda_over_best * 2.0)]
        lambda_under_refined = [max(0.0001, lambda_under_best * 0.5), lambda_under_best, min(1.0, lambda_under_best * 2.0)]
        theta_queue_refined = [max(0.00001, theta_queue_best * 0.5), theta_queue_best, min(0.1, theta_queue_best * 2.0)]
        
        # Remove duplicates and sort
        lambda_over_refined = sorted(list(set(lambda_over_refined)))
        lambda_under_refined = sorted(list(set(lambda_under_refined)))
        theta_queue_refined = sorted(list(set(theta_queue_refined)))
        
        # Second pass: refined grid search
        refined_combinations = len(lambda_over_refined) * len(lambda_under_refined) * len(theta_queue_refined)
        
        # Estimate remaining time
        estimated_time = time_per_combination * refined_combinations
        print(f"Estimated time for refinement: {estimated_time:.2f}s")
        
        if estimated_time < 50:  # Only proceed if estimated time is reasonable
            for lambda_over in lambda_over_refined:
                for lambda_under in lambda_under_refined:
                    for theta_queue in theta_queue_refined:
                        # Skip if this is the exact same combination as the best from first pass
                        if (lambda_over == lambda_over_best and 
                            lambda_under == lambda_under_best and 
                            theta_queue == theta_queue_best):
                            continue
                        
                        # Execute strategy with these parameters
                        result = execute_strategy(
                            data, 
                            venues_config, 
                            lambda_over, 
                            lambda_under, 
                            theta_queue
                        )
                        
                        # Check if this is better than our best so far
                        if result['shares_filled'] > 0 and result['avg_price'] < best_avg_price:
                            best_result = result
                            best_params = {
                                'lambda_over': lambda_over,
                                'lambda_under': lambda_under,
                                'theta_queue': theta_queue
                            }
                            best_avg_price = result['avg_price']
    
    # Calculate execution time
    execution_time = time.time() - start_time
    print(f"Parameter search completed in {execution_time:.2f} seconds.")
    
    return {
        'best_params': best_params,
        'best_result': best_result,
        'execution_time': execution_time
    }


def calculate_savings_bps(tuned_avg_price: float, baseline_avg_price: float) -> float:
    """
    Calculate savings in basis points.
    
    Args:
        tuned_avg_price: Average price from tuned strategy
        baseline_avg_price: Average price from baseline strategy
        
    Returns:
        Savings in basis points
    """
    return (baseline_avg_price - tuned_avg_price) / baseline_avg_price * 10000


def main():
    """
    Main function to run the back-test.
    """
    # Load and preprocess data
    data = load_and_preprocess_data('l1_day.csv')
    
    # Configure venues with fees and rebates
    # In a real-world scenario, these values would be loaded from a configuration file
    venues_config = {
        1: {'fee': 0.0003, 'rebate': 0.0002},  # Example venue 1
        2: {'fee': 0.0002, 'rebate': 0.0003},  # Example venue 2
        3: {'fee': 0.0004, 'rebate': 0.0001},  # Example venue 3
        # Add more venues as needed
    }
    
    # Ensure all venues in the data have a configuration
    for venue_id in data['publisher_id'].unique():
        if venue_id not in venues_config:
            venues_config[venue_id] = {'fee': 0.0003, 'rebate': 0.0002}
    
    # Find optimal parameters
    search_results = parameter_search(data, venues_config)
    best_params = search_results['best_params']
    best_result = search_results['best_result']
    
    print(f"Best parameters found: {best_params}")
    
    # Run baseline strategies for comparison
    print("Running baseline strategies...")
    best_ask_result = execute_best_ask_strategy(data, venues_config)
    twap_result = execute_twap_strategy(data, venues_config)
    vwap_result = execute_vwap_strategy(data, venues_config)
    
    # Calculate savings in basis points
    savings_vs_best_ask = calculate_savings_bps(best_result['avg_price'], best_ask_result['avg_price'])
    savings_vs_twap = calculate_savings_bps(best_result['avg_price'], twap_result['avg_price'])
    savings_vs_vwap = calculate_savings_bps(best_result['avg_price'], vwap_result['avg_price'])
    
    # Prepare output JSON
    output = {
        'best_parameters': {
            'lambda_over': best_params['lambda_over'],
            'lambda_under': best_params['lambda_under'],
            'theta_queue': best_params['theta_queue']
        },
        'tuned_strategy': {
            'total_cash_spent': best_result['cash_spent'],
            'avg_fill_price': best_result['avg_price'],
            'shares_filled': best_result['shares_filled']
        },
        'baselines': {
            'best_ask': {
                'total_cash_spent': best_ask_result['cash_spent'],
                'avg_fill_price': best_ask_result['avg_price'],
                'shares_filled': best_ask_result['shares_filled']
            },
            'twap': {
                'total_cash_spent': twap_result['cash_spent'],
                'avg_fill_price': twap_result['avg_price'],
                'shares_filled': twap_result['shares_filled']
            },
            'vwap': {
                'total_cash_spent': vwap_result['cash_spent'],
                'avg_fill_price': vwap_result['avg_price'],
                'shares_filled': vwap_result['shares_filled']
            }
        },
        'savings_bps': {
            'vs_best_ask': savings_vs_best_ask,
            'vs_twap': savings_vs_twap,
            'vs_vwap': savings_vs_vwap
        }
    }
    
    # Output the JSON to stdout
    print(json.dumps(output, indent=2, cls=NumpyEncoder))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        # Output error as JSON for consistent parsing
        error_output = {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        print(json.dumps(error_output))