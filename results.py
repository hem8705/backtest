#!/usr/bin/env python3
"""
Results Visualization for Cont & Kukanov Smart Order Router Back-testing

This script generates a cumulative cost plot comparing the tuned smart order router
against the baseline strategies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
import json
import sys
from typing import Dict, List, Any, Tuple


def load_results() -> Dict:
    """Load results from stdin or a file."""
    try:
        # Try to read from stdin
        results_json = input()
        return json.loads(results_json)
    except json.JSONDecodeError:
        print("Error: Could not parse JSON input. Please pipe the output of backtest.py to this script.")
        sys.exit(1)


def simulate_cumulative_costs(data: pd.DataFrame, 
                             tuned_params: Dict[str, float],
                             venues_config: Dict[int, Dict]) -> Dict[str, pd.DataFrame]:
    """
    Simulate the cumulative costs of each strategy over time.
    
    Args:
        data: Preprocessed market data
        tuned_params: Parameters for the tuned strategy
        venues_config: Venue configuration with fees and rebates
        
    Returns:
        Dictionary mapping strategy names to DataFrames with cumulative costs
    """
    from backtest import execute_strategy, execute_best_ask_strategy, execute_twap_strategy, execute_vwap_strategy
    
    # Group data by timestamp to get snapshots
    grouped = data.groupby('ts_event')
    timestamps = sorted(data['ts_event'].unique())
    
    # Initialize result dictionaries for each strategy
    strategies = {
        'Tuned SOR': {'params': tuned_params},
        'Best Ask': {},
        'TWAP': {},
        'VWAP': {}
    }
    
    # Initialize cumulative statistics for each strategy
    for strategy in strategies:
        strategies[strategy]['cumulative_costs'] = []
        strategies[strategy]['cumulative_shares'] = []
        strategies[strategy]['timestamps'] = []
    
    # Simulate each strategy, accumulating costs over time
    for ts in timestamps:
        snapshot = data[data['ts_event'] == ts]
        
        # Get current state for each strategy
        for strategy_name, strategy_data in strategies.items():
            # Get the current slice of data up to this timestamp
            current_data = data[data['ts_event'] <= ts]
            
            if strategy_name == 'Tuned SOR':
                # Execute tuned strategy
                result = execute_strategy(
                    current_data, 
                    venues_config, 
                    tuned_params['lambda_over'],
                    tuned_params['lambda_under'],
                    tuned_params['theta_queue']
                )
            elif strategy_name == 'Best Ask':
                # Execute best ask strategy
                result = execute_best_ask_strategy(current_data, venues_config)
            elif strategy_name == 'TWAP':
                # Execute TWAP strategy
                result = execute_twap_strategy(current_data, venues_config)
            elif strategy_name == 'VWAP':
                # Execute VWAP strategy
                result = execute_vwap_strategy(current_data, venues_config)
            
            # Store cumulative statistics
            strategy_data['cumulative_costs'].append(result['cash_spent'])
            strategy_data['cumulative_shares'].append(result['shares_filled'])
            strategy_data['timestamps'].append(ts)
    
    # Convert to DataFrame
    result_dfs = {}
    for strategy_name, strategy_data in strategies.items():
        df = pd.DataFrame({
            'timestamp': strategy_data['timestamps'],
            'cumulative_cost': strategy_data['cumulative_costs'],
            'cumulative_shares': strategy_data['cumulative_shares']
        })
        # Calculate average price
        df['avg_price'] = np.where(
            df['cumulative_shares'] > 0,
            df['cumulative_cost'] / df['cumulative_shares'],
            0
        )
        result_dfs[strategy_name] = df
    
    return result_dfs


def generate_cumulative_cost_plot(strategy_dfs: Dict[str, pd.DataFrame], output_path: str = 'results.pdf'):
    """
    Generate a cumulative cost plot comparing the strategies.
    
    Args:
        strategy_dfs: Dictionary mapping strategy names to DataFrames with cumulative costs
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Set up styles for each strategy
    styles = {
        'Tuned SOR': {'color': 'blue', 'linestyle': '-', 'linewidth': 2},
        'Best Ask': {'color': 'red', 'linestyle': '--', 'linewidth': 1.5},
        'TWAP': {'color': 'green', 'linestyle': '-.', 'linewidth': 1.5},
        'VWAP': {'color': 'purple', 'linestyle': ':', 'linewidth': 1.5}
    }
    
    # Plot each strategy
    for strategy_name, df in strategy_dfs.items():
        if df.empty or df['cumulative_shares'].max() == 0:
            continue
            
        plt.plot(
            df['timestamp'], 
            df['avg_price'],
            label=f"{strategy_name}",
            **styles.get(strategy_name, {})
        )
    
    # Calculate the price range for the y-axis
    all_prices = []
    for df in strategy_dfs.values():
        if not df.empty:
            valid_prices = df[df['avg_price'] > 0]['avg_price']
            if not valid_prices.empty:
                all_prices.extend(valid_prices.tolist())
    
    if all_prices:
        min_price = min(all_prices) * 0.9995  # Slight padding
        max_price = max(all_prices) * 1.0005
        plt.ylim(min_price, max_price)
    
    # Format the plot
    plt.title('Cumulative Execution Cost Comparison', fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Average Execution Price', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Format the date on the x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.xticks(rotation=45)
    
    # Add savings annotation
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    
    # Optionally show the plot
    plt.close()


def main():
    """Main execution function"""
    # Load results
    results = load_results()
    
    # Load the original data for timeline simulation
    data = pd.read_csv('l1_day.csv')
    data['ts_event'] = pd.to_datetime(data['ts_event'])
    data = data.sort_values(['ts_event', 'publisher_id']).drop_duplicates(['ts_event', 'publisher_id'])
    data = data.sort_values('ts_event')
    venue_data = data[['ts_event', 'publisher_id', 'ask_px_00', 'ask_sz_00']].copy()
    venue_data = venue_data.dropna(subset=['ask_px_00', 'ask_sz_00'])
    venue_data['ask_px_00'] = venue_data['ask_px_00'].astype(float)
    venue_data['ask_sz_00'] = venue_data['ask_sz_00'].astype(int)
    
    # Configure venues with fees and rebates
    # In a real-world scenario, these values would be loaded from a configuration file
    venues_config = {
        1: {'fee': 0.0003, 'rebate': 0.0002},  # Example venue 1
        2: {'fee': 0.0002, 'rebate': 0.0003},  # Example venue 2
        3: {'fee': 0.0004, 'rebate': 0.0001},  # Example venue 3
        # Add more venues as needed
    }
    
    # Ensure all venues in the data have a configuration
    for venue_id in venue_data['publisher_id'].unique():
        if venue_id not in venues_config:
            venues_config[venue_id] = {'fee': 0.0003, 'rebate': 0.0002}
    
    # Get best parameters from results
    tuned_params = results['best_parameters']
    
    # Simulate cumulative costs
    strategy_dfs = simulate_cumulative_costs(venue_data, tuned_params, venues_config)
    
    # Generate cumulative cost plot
    generate_cumulative_cost_plot(strategy_dfs)
    
    # Print summary
    print("\nPerformance Summary:")
    print(f"Tuned SOR avg price: {results['tuned_strategy']['avg_fill_price']:.6f}")
    print(f"Best Ask avg price: {results['baselines']['best_ask']['avg_fill_price']:.6f}")
    print(f"TWAP avg price: {results['baselines']['twap']['avg_fill_price']:.6f}")
    print(f"VWAP avg price: {results['baselines']['vwap']['avg_fill_price']:.6f}")
    print("\nSavings (bps):")
    print(f"vs Best Ask: {results['savings_bps']['vs_best_ask']:.2f}")
    print(f"vs TWAP: {results['savings_bps']['vs_twap']:.2f}")
    print(f"vs VWAP: {results['savings_bps']['vs_vwap']:.2f}")


if __name__ == "__main__":
    main()