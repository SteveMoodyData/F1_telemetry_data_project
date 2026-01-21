#!/usr/bin/env python3
"""
F1 2024 Season Tire Analysis

Comprehensive tire performance analysis across the 2024 season:
- Tire degradation patterns by compound
- Optimal stint lengths
- Compound performance by circuit type
- Temperature effects on tire life
- Strategy comparison across teams
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

from src.ingestion.telemetry_loader import TelemetryLoader

# Set plotting style
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (14, 8)


class TireAnalyzer2024:
    """Analyzes tire performance data from 2024 F1 season."""
    
    def __init__(self, cache_dir='./data/cache'):
        """Initialize analyzer with FastF1 loader."""
        self.loader = TelemetryLoader(cache_dir=cache_dir)
        self.season_data = {}
        self.tire_data = None
        
    def load_season_races(self, races=None, session_type='R'):
        """
        Load multiple races from 2024 season.
        
        Args:
            races: List of GP names, or None for all available
            session_type: 'R' for Race, 'Q' for Qualifying
        
        Returns:
            DataFrame with all lap data
        """
        # 2024 Season races (first half)
        if races is None:
            races = [
                'Bahrain', 'Saudi Arabia', 'Australia', 'Japan',
                'China', 'Miami', 'Emilia Romagna', 'Monaco',
                'Canada', 'Spain', 'Austria', 'Great Britain',
                'Hungary', 'Belgium', 'Netherlands', 'Italy'
            ]
        
        print(f"Loading {len(races)} races from 2024 season...")
        all_laps = []
        
        for race in races:
            try:
                print(f"  Loading {race}...", end=' ')
                session = self.loader.load_session(2024, race, session_type)
                laps = self.loader.extract_lap_data(session)
                all_laps.append(laps)
                print(f"✓ {len(laps)} laps")
            except Exception as e:
                print(f"✗ Failed: {e}")
                continue
        
        if all_laps:
            self.tire_data = pd.concat(all_laps, ignore_index=True)
            print(f"\n✓ Loaded {len(self.tire_data)} total laps from {len(all_laps)} races")
            return self.tire_data
        else:
            print("✗ No data loaded")
            return None
    
    def analyze_tire_degradation(self):
        """
        Analyze tire degradation over stint life.
        
        Returns:
            DataFrame with degradation metrics by compound
        """
        print("\n" + "="*70)
        print("TIRE DEGRADATION ANALYSIS")
        print("="*70)
        
        # Filter valid laps
        valid = self.tire_data[
            (self.tire_data['LapTime'].notna()) &
            (self.tire_data['IsAccurate'] == True) &
            (self.tire_data['Compound'].notna()) &
            (self.tire_data['TyreLife'].notna())
        ].copy()
        
        # Convert Timedelta to seconds if needed
        if pd.api.types.is_timedelta64_dtype(valid['LapTime']):
            valid['LapTime'] = valid['LapTime'].dt.total_seconds()
        
        # Calculate degradation by compound
        degradation = []
        
        for compound in ['SOFT', 'MEDIUM', 'HARD']:
            compound_data = valid[valid['Compound'] == compound]
            
            if len(compound_data) < 50:  # Need sufficient data
                continue
            
            # Group by tire age
            age_groups = compound_data.groupby('TyreLife')['LapTime'].agg([
                'mean', 'median', 'count', 'std'
            ])
            
            # Calculate degradation rate (seconds per lap)
            if len(age_groups) > 5:
                # Linear regression of lap time vs tire age
                x = age_groups.index.values
                y = age_groups['median'].values
                
                # Only use first 30 laps to avoid outliers
                mask = x <= 30
                if sum(mask) > 5:
                    x_fit = x[mask]
                    y_fit = y[mask]
                    
                    # Simple linear fit
                    deg_rate = np.polyfit(x_fit, y_fit, 1)[0]
                else:
                    deg_rate = np.nan
            else:
                deg_rate = np.nan
            
            degradation.append({
                'Compound': compound,
                'TotalLaps': len(compound_data),
                'AvgStintLength': compound_data['TyreLife'].mean(),
                'MaxStintLength': compound_data['TyreLife'].max(),
                'DegradationRate': deg_rate,
                'AvgLapTime': compound_data['LapTime'].mean(),
                'MedianLapTime': compound_data['LapTime'].median()
            })
        
        deg_df = pd.DataFrame(degradation)
        
        print("\nDegradation Summary:")
        print(deg_df.to_string(index=False))
        
        return deg_df
    
    def plot_tire_degradation(self):
        """Create visualization of tire degradation patterns."""
        valid = self.tire_data[
            (self.tire_data['LapTime'].notna()) &
            (self.tire_data['IsAccurate'] == True) &
            (self.tire_data['Compound'].notna()) &
            (self.tire_data['TyreLife'].notna()) &
            (self.tire_data['TyreLife'] <= 40)  # Focus on realistic stint lengths
        ].copy()
        
        # Convert Timedelta to seconds if needed
        if pd.api.types.is_timedelta64_dtype(valid['LapTime']):
            valid['LapTime'] = valid['LapTime'].dt.total_seconds()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Define compound colors
        compound_colors = {
            'SOFT': '#FF1E1E',
            'MEDIUM': '#FFF500', 
            'HARD': '#EBEBEB'
        }
        
        # Plot 1: Lap time vs Tire Age
        ax1 = axes[0, 0]
        for compound in ['SOFT', 'MEDIUM', 'HARD']:
            data = valid[valid['Compound'] == compound]
            if len(data) > 0:
                grouped = data.groupby('TyreLife')['LapTime'].median()
                ax1.plot(grouped.index, grouped.values, 
                        marker='o', label=compound, 
                        color=compound_colors[compound], linewidth=2)
        
        ax1.set_xlabel('Tire Age (laps)')
        ax1.set_ylabel('Median Lap Time (seconds)')
        ax1.set_title('Tire Degradation by Compound - 2024 Season')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot 2: Stint Length Distribution
        ax2 = axes[0, 1]
        stint_lengths = []
        labels = []
        colors = []
        
        for compound in ['SOFT', 'MEDIUM', 'HARD']:
            data = valid[valid['Compound'] == compound]
            if len(data) > 0:
                max_ages = data.groupby(['GrandPrix', 'Driver', 'Stint'])['TyreLife'].max()
                stint_lengths.append(max_ages.values)
                labels.append(compound)
                colors.append(compound_colors[compound])
        
        if stint_lengths:
            ax2.boxplot(stint_lengths, labels=labels, patch_artist=True)
            for patch, color in zip(ax2.artists, colors):
                patch.set_facecolor(color)
        
        ax2.set_ylabel('Stint Length (laps)')
        ax2.set_title('Stint Length Distribution by Compound')
        ax2.grid(axis='y', alpha=0.3)
        
        # Plot 3: Lap Time Variance by Tire Age
        ax3 = axes[1, 0]
        for compound in ['SOFT', 'MEDIUM', 'HARD']:
            data = valid[valid['Compound'] == compound]
            if len(data) > 0:
                variance = data.groupby('TyreLife')['LapTime'].std()
                ax3.plot(variance.index, variance.values,
                        marker='s', label=compound,
                        color=compound_colors[compound], linewidth=2)
        
        ax3.set_xlabel('Tire Age (laps)')
        ax3.set_ylabel('Lap Time Std Dev (seconds)')
        ax3.set_title('Tire Consistency vs Age (Lower = More Consistent)')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # Plot 4: Compound Usage by Race
        ax4 = axes[1, 1]
        compound_usage = valid.groupby(['GrandPrix', 'Compound']).size().unstack(fill_value=0)
        
        if len(compound_usage) > 0:
            compound_usage.plot(kind='bar', ax=ax4, 
                              color=[compound_colors.get(c, 'gray') for c in compound_usage.columns],
                              stacked=True)
            ax4.set_xlabel('Grand Prix')
            ax4.set_ylabel('Number of Laps')
            ax4.set_title('Compound Usage by Race')
            ax4.legend(title='Compound')
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save figure
        output_dir = Path('./data/output')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / 'tire_degradation_2024.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved: {output_file}")
        
        return fig
    
    def analyze_optimal_stint_length(self):
        """
        Determine optimal stint length for each compound.
        
        Returns:
            DataFrame with optimal stint recommendations
        """
        print("\n" + "="*70)
        print("OPTIMAL STINT LENGTH ANALYSIS")
        print("="*70)
        
        valid = self.tire_data[
            (self.tire_data['LapTime'].notna()) &
            (self.tire_data['IsAccurate'] == True) &
            (self.tire_data['Compound'].notna()) &
            (self.tire_data['TyreLife'].notna())
        ].copy()
        
        # Convert Timedelta to seconds if needed
        if pd.api.types.is_timedelta64_dtype(valid['LapTime']):
            valid['LapTime'] = valid['LapTime'].dt.total_seconds()
        
        results = []
        
        for compound in ['SOFT', 'MEDIUM', 'HARD']:
            data = valid[valid['Compound'] == compound]
            
            if len(data) < 100:
                continue
            
            # Calculate average pace degradation by stint length
            pace_by_age = data.groupby('TyreLife')['LapTime'].agg(['median', 'count'])
            pace_by_age = pace_by_age[pace_by_age['count'] >= 10]  # Min sample size
            
            if len(pace_by_age) < 5:
                continue
            
            # Find point where degradation accelerates
            # Using rolling difference to find inflection point
            pace_change = pace_by_age['median'].diff().rolling(3).mean()
            
            # Optimal = before significant degradation spike
            threshold = pace_change.quantile(0.75)  # Top 25% degradation
            optimal_age = pace_change[pace_change > threshold].index.min()
            
            if pd.isna(optimal_age):
                optimal_age = pace_by_age.index.max()
            
            # Get stint statistics
            stint_data = data.groupby(['GrandPrix', 'Driver', 'Stint']).agg({
                'TyreLife': 'max',
                'LapTime': 'mean'
            })
            
            avg_stint = stint_data['TyreLife'].mean()
            common_stint = stint_data['TyreLife'].mode()[0] if len(stint_data) > 0 else np.nan
            
            results.append({
                'Compound': compound,
                'OptimalMaxAge': optimal_age,
                'AverageStintLength': avg_stint,
                'MostCommonStint': common_stint,
                'LongestStint': data['TyreLife'].max(),
                'TotalStints': len(stint_data)
            })
        
        results_df = pd.DataFrame(results)
        
        print("\nOptimal Stint Lengths:")
        print(results_df.to_string(index=False))
        print("\nRecommendations:")
        for _, row in results_df.iterrows():
            print(f"  {row['Compound']}: {row['OptimalMaxAge']:.0f} laps "
                  f"(teams average {row['AverageStintLength']:.1f})")
        
        return results_df
    
    def analyze_compound_performance_by_circuit(self):
        """
        Analyze which compounds perform best at different circuit types.
        
        Returns:
            DataFrame with compound performance by circuit
        """
        print("\n" + "="*70)
        print("COMPOUND PERFORMANCE BY CIRCUIT")
        print("="*70)
        
        valid = self.tire_data[
            (self.tire_data['LapTime'].notna()) &
            (self.tire_data['IsAccurate'] == True) &
            (self.tire_data['Compound'].notna())
        ].copy()
        
        # Convert Timedelta to seconds if needed
        if pd.api.types.is_timedelta64_dtype(valid['LapTime']):
            valid['LapTime'] = valid['LapTime'].dt.total_seconds()
        
        # Group by circuit and compound
        circuit_perf = valid.groupby(['GrandPrix', 'Compound']).agg({
            'LapTime': ['median', 'mean', 'count']
        }).reset_index()
        
        circuit_perf.columns = ['GrandPrix', 'Compound', 'MedianLapTime', 'MeanLapTime', 'LapCount']
        
        # Find fastest compound per circuit
        fastest = circuit_perf.loc[circuit_perf.groupby('GrandPrix')['MedianLapTime'].idxmin()]
        
        print("\nFastest Compound by Circuit:")
        for _, row in fastest.iterrows():
            print(f"  {row['GrandPrix']:20s}: {row['Compound']:6s} "
                  f"({row['MedianLapTime']:.2f}s, {row['LapCount']:.0f} laps)")
        
        # Compound win count
        print("\nCompound Performance Summary:")
        compound_wins = fastest['Compound'].value_counts()
        for compound, count in compound_wins.items():
            print(f"  {compound}: Fastest at {count} circuit(s)")
        
        return circuit_perf
    
    def export_tire_data(self, filename='tire_analysis_2024.csv'):
        """Export processed tire data to CSV."""
        output_dir = Path('./data/output')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / filename
        
        # Select relevant columns
        export_cols = [
            'GrandPrix', 'Circuit', 'Session', 'Driver', 'LapNumber',
            'LapTime', 'Compound', 'TyreLife', 'Stint', 'FreshTyre',
            'Sector1Time', 'Sector2Time', 'Sector3Time'
        ]
        
        available_cols = [col for col in export_cols if col in self.tire_data.columns]
        export_data = self.tire_data[available_cols]
        
        export_data.to_csv(output_file, index=False)
        print(f"\n✓ Tire data exported: {output_file}")
        print(f"  {len(export_data)} laps, {export_data['GrandPrix'].nunique()} races")
        
        return output_file


def main():
    """Run comprehensive 2024 tire analysis."""
    print("="*70)
    print("F1 2024 SEASON - TIRE ANALYSIS")
    print("="*70)
    
    # Initialize analyzer
    analyzer = TireAnalyzer2024()
    
    # Load data (first 8 races for demo - adjust as needed)
    races = [
        'Bahrain', 'Saudi Arabia', 'Australia', 'Japan',
        'China', 'Miami', 'Emilia Romagna', 'Monaco'
    ]
    
    print(f"\nLoading {len(races)} races from 2024...")
    print("(This will take 5-10 minutes on first run, then uses cache)\n")
    
    tire_data = analyzer.load_season_races(races=races)
    
    if tire_data is None or len(tire_data) == 0:
        print("✗ No data loaded. Check internet connection.")
        return
    
    # Run analyses
    analyzer.analyze_tire_degradation()
    analyzer.analyze_optimal_stint_length()
    analyzer.analyze_compound_performance_by_circuit()
    
    # Create visualizations
    print("\nGenerating visualizations...")
    analyzer.plot_tire_degradation()
    
    # Export data
    analyzer.export_tire_data()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nOutputs saved to ./data/output/:")
    print("  - tire_degradation_2024.png")
    print("  - tire_analysis_2024.csv")
    print("\nKey Findings:")
    print("  - Check degradation rates to optimize pit strategy")
    print("  - Review optimal stint lengths vs actual team strategies")
    print("  - Identify which compounds work best at each circuit")


if __name__ == "__main__":
    main()
