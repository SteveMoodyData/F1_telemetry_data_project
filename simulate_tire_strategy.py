#!/usr/bin/env python3
"""
F1 Tire Strategy Simulator

Simulate different tire strategies for a race and compare outcomes.
Uses historical data to predict lap times based on compound and tire age.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.ingestion.telemetry_loader import TelemetryLoader

sns.set_style('darkgrid')


class TireStrategySimulator:
    """Simulates race strategies based on tire performance data."""
    
    def __init__(self, race='Monaco', year=2024):
        """Initialize simulator with race data."""
        self.race = race
        self.year = year
        self.loader = TelemetryLoader()
        self.tire_models = {}
        self.load_race_data()
    
    def load_race_data(self):
        """Load and process race data for tire modeling."""
        print(f"Loading {self.year} {self.race} GP data...")
        
        session = self.loader.load_session(self.year, self.race, 'R')
        laps = self.loader.extract_lap_data(session)
        
        # Clean data
        self.data = laps[
            (laps['LapTime'].notna()) &
            (laps['IsAccurate'] == True) &
            (laps['Compound'].notna()) &
            (laps['TyreLife'].notna())
        ].copy()
        
        # Convert Timedelta to seconds if needed
        if pd.api.types.is_timedelta64_dtype(self.data['LapTime']):
            self.data['LapTime'] = self.data['LapTime'].dt.total_seconds()
        
        print(f"✓ Loaded {len(self.data)} valid laps")
        
        # Build tire performance models
        self._build_tire_models()
    
    def _build_tire_models(self):
        """Build degradation models for each compound."""
        print("\nBuilding tire performance models...")
        
        for compound in self.data['Compound'].unique():
            compound_data = self.data[self.data['Compound'] == compound]
            
            # Group by tire age
            perf_by_age = compound_data.groupby('TyreLife')['LapTime'].agg([
                'median', 'mean', 'std', 'count'
            ])
            
            # Only use ages with sufficient data
            perf_by_age = perf_by_age[perf_by_age['count'] >= 3]
            
            if len(perf_by_age) < 3:
                print(f"  ✗ {compound}: Insufficient data")
                continue
            
            self.tire_models[compound] = {
                'base_time': perf_by_age['median'].iloc[0],  # Lap 1 time
                'degradation': perf_by_age['median'].to_dict(),
                'max_age': perf_by_age.index.max()
            }
            
            print(f"  ✓ {compound}: Base {self.tire_models[compound]['base_time']:.3f}s, "
                  f"Max age {self.tire_models[compound]['max_age']:.0f} laps")
    
    def predict_lap_time(self, compound, tire_age):
        """
        Predict lap time for given compound and tire age.
        
        Args:
            compound: Tire compound
            tire_age: Age of tire in laps
        
        Returns:
            Predicted lap time in seconds
        """
        if compound not in self.tire_models:
            return None
        
        model = self.tire_models[compound]
        
        # Use actual data if available
        if tire_age in model['degradation']:
            return model['degradation'][tire_age]
        
        # Extrapolate if beyond data
        if tire_age > model['max_age']:
            # Use last known + additional degradation
            last_time = model['degradation'][model['max_age']]
            extra_laps = tire_age - model['max_age']
            # Assume degradation continues (conservative estimate)
            deg_rate = 0.05  # 0.05s per lap beyond max_age
            return last_time + (extra_laps * deg_rate)
        
        # Interpolate
        ages = sorted(model['degradation'].keys())
        return np.interp(tire_age, ages, [model['degradation'][a] for a in ages])
    
    def simulate_strategy(self, strategy):
        """
        Simulate a race strategy.
        
        Args:
            strategy: List of (compound, stint_length) tuples
                     Example: [('MEDIUM', 20), ('HARD', 30)]
        
        Returns:
            Dictionary with race simulation results
        """
        lap_times = []
        compounds_used = []
        tire_ages = []
        current_lap = 0
        
        for compound, stint_length in strategy:
            for lap_in_stint in range(1, stint_length + 1):
                lap_time = self.predict_lap_time(compound, lap_in_stint)
                
                if lap_time is None:
                    print(f"✗ Cannot simulate {compound} - no data")
                    return None
                
                lap_times.append(lap_time)
                compounds_used.append(compound)
                tire_ages.append(lap_in_stint)
                current_lap += 1
        
        # Add pit stop time (assume 25 seconds per stop)
        num_stops = len(strategy) - 1
        pit_time = num_stops * 25.0
        
        total_time = sum(lap_times) + pit_time
        avg_lap = np.mean(lap_times)
        
        return {
            'strategy': strategy,
            'total_laps': len(lap_times),
            'lap_times': lap_times,
            'compounds': compounds_used,
            'tire_ages': tire_ages,
            'race_time': total_time,
            'avg_lap_time': avg_lap,
            'num_stops': num_stops,
            'pit_time': pit_time
        }
    
    def compare_strategies(self, strategies):
        """
        Compare multiple strategies.
        
        Args:
            strategies: Dictionary of {name: strategy} pairs
        
        Returns:
            DataFrame with comparison results
        """
        print(f"\n{'='*70}")
        print("STRATEGY COMPARISON")
        print('='*70)
        
        results = []
        
        for name, strategy in strategies.items():
            print(f"\nSimulating: {name}")
            print(f"  Strategy: {strategy}")
            
            sim = self.simulate_strategy(strategy)
            
            if sim is None:
                continue
            
            results.append({
                'Strategy': name,
                'RaceTime': sim['race_time'],
                'AvgLapTime': sim['avg_lap_time'],
                'NumStops': sim['num_stops'],
                'TotalLaps': sim['total_laps']
            })
            
            print(f"  Race time: {sim['race_time']:.1f}s "
                  f"({sim['race_time']/60:.1f} min)")
            print(f"  Avg lap: {sim['avg_lap_time']:.3f}s")
        
        if not results:
            print("✗ No strategies could be simulated")
            return None
        
        df = pd.DataFrame(results).sort_values('RaceTime')
        
        print(f"\n{'='*70}")
        print("RESULTS (sorted by race time)")
        print('='*70)
        print(df.to_string(index=False))
        
        # Time differences
        fastest = df.iloc[0]['RaceTime']
        df['TimeDelta'] = df['RaceTime'] - fastest
        
        print(f"\nTime Differences from Fastest:")
        for _, row in df.iterrows():
            if row['TimeDelta'] == 0:
                print(f"  {row['Strategy']:30s}: Fastest ✓")
            else:
                print(f"  {row['Strategy']:30s}: +{row['TimeDelta']:.1f}s")
        
        return df
    
    def plot_strategy_comparison(self, strategies):
        """Visualize strategy comparison."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))
        
        # Plot lap times over race
        ax1 = axes[0]
        for (name, strategy), color in zip(strategies.items(), colors):
            sim = self.simulate_strategy(strategy)
            if sim:
                ax1.plot(range(1, len(sim['lap_times']) + 1), 
                        sim['lap_times'], 
                        label=name, color=color, linewidth=2)
        
        ax1.set_xlabel('Lap Number')
        ax1.set_ylabel('Lap Time (seconds)')
        ax1.set_title(f'{self.year} {self.race} GP - Strategy Comparison')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot cumulative time
        ax2 = axes[1]
        for (name, strategy), color in zip(strategies.items(), colors):
            sim = self.simulate_strategy(strategy)
            if sim:
                cumulative = np.cumsum(sim['lap_times'])
                # Add pit stops
                for i in range(sim['num_stops']):
                    stop_lap = sum([s[1] for s in strategy[:i+1]])
                    if stop_lap < len(cumulative):
                        cumulative[stop_lap:] += 25.0
                
                ax2.plot(range(1, len(cumulative) + 1), 
                        cumulative, 
                        label=name, color=color, linewidth=2)
        
        ax2.set_xlabel('Lap Number')
        ax2.set_ylabel('Cumulative Time (seconds)')
        ax2.set_title('Cumulative Race Time (including pit stops)')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        output_file = f'./data/output/strategy_comparison_{self.race.lower()}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved: {output_file}")


# Example usage
if __name__ == "__main__":
    # Initialize simulator
    sim = TireStrategySimulator(race='Monaco', year=2024)
    
    # Define strategies to compare (adjust based on actual race length)
    strategies = {
        'Conservative (1-stop)': [
            ('HARD', 78)
        ],
        'Standard (1-stop)': [
            ('MEDIUM', 40),
            ('HARD', 38)
        ],
        'Aggressive (2-stop)': [
            ('SOFT', 25),
            ('MEDIUM', 28),
            ('MEDIUM', 25)
        ],
        'Alternative (2-stop)': [
            ('MEDIUM', 30),
            ('SOFT', 20),
            ('MEDIUM', 28)
        ]
    }
    
    # Compare strategies
    results = sim.compare_strategies(strategies)
    
    if results is not None:
        # Visualize
        sim.plot_strategy_comparison(strategies)
        
        print("\n✓ Strategy simulation complete!")
        print(f"  Best strategy: {results.iloc[0]['Strategy']}")
