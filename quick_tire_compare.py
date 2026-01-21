#!/usr/bin/env python3
"""
Quick Tire Comparison Script

Compare tire performance between two races or compounds.
Useful for quick analysis and strategy insights.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.ingestion.telemetry_loader import TelemetryLoader

sns.set_style('darkgrid')


def compare_compounds_single_race(race='Monaco', year=2024):
    """
    Compare all tire compounds at a single race.
    
    Args:
        race: Grand Prix name
        year: Season year
    """
    print(f"Analyzing tire compounds at {year} {race} GP...")
    
    loader = TelemetryLoader()
    session = loader.load_session(year, race, 'R')
    laps = loader.extract_lap_data(session)
    
    # Clean data
    valid = laps[
        (laps['LapTime'].notna()) &
        (laps['IsAccurate'] == True) &
        (laps['Compound'].notna())
    ].copy()
    
    # Convert Timedelta to seconds if needed
    if pd.api.types.is_timedelta64_dtype(valid['LapTime']):
        valid['LapTime'] = valid['LapTime'].dt.total_seconds()
    
    print(f"\nCompound Usage:")
    compound_counts = valid['Compound'].value_counts()
    for compound, count in compound_counts.items():
        avg_time = valid[valid['Compound'] == compound]['LapTime'].median()
        print(f"  {compound:12s}: {count:4d} laps, Median: {avg_time:.3f}s")
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Box plot
    compound_colors = {'SOFT': '#FF1E1E', 'MEDIUM': '#FFF500', 'HARD': '#EBEBEB'}
    
    compounds = sorted(valid['Compound'].unique())
    data_to_plot = [valid[valid['Compound'] == c]['LapTime'].values for c in compounds]
    
    bp = axes[0].boxplot(data_to_plot, labels=compounds, patch_artist=True)
    for patch, compound in zip(bp['boxes'], compounds):
        patch.set_facecolor(compound_colors.get(compound, 'gray'))
    
    axes[0].set_ylabel('Lap Time (seconds)')
    axes[0].set_title(f'{year} {race} GP - Compound Performance')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Degradation plot
    for compound in compounds:
        data = valid[valid['Compound'] == compound]
        if len(data) > 10:
            deg = data.groupby('TyreLife')['LapTime'].median()
            axes[1].plot(deg.index, deg.values, marker='o', 
                        label=compound, color=compound_colors.get(compound, 'gray'),
                        linewidth=2)
    
    axes[1].set_xlabel('Tire Age (laps)')
    axes[1].set_ylabel('Median Lap Time (seconds)')
    axes[1].set_title('Tire Degradation')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_file = f'./data/output/tire_comparison_{race.lower().replace(" ", "_")}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    
    return valid


def compare_two_races(race1='Monaco', race2='Monza', year=2024, compound='SOFT'):
    """
    Compare same compound performance at two different circuits.
    
    Args:
        race1: First Grand Prix
        race2: Second Grand Prix  
        year: Season year
        compound: Tire compound to compare
    """
    print(f"Comparing {compound} tire at {race1} vs {race2}...")
    
    loader = TelemetryLoader()
    
    # Load both races
    session1 = loader.load_session(year, race1, 'R')
    laps1 = loader.extract_lap_data(session1)
    
    session2 = loader.load_session(year, race2, 'R')
    laps2 = loader.extract_lap_data(session2)
    
    # Filter for specific compound
    def filter_compound(df, comp):
        filtered = df[
            (df['LapTime'].notna()) &
            (df['IsAccurate'] == True) &
            (df['Compound'] == comp)
        ].copy()
        
        # Convert Timedelta to seconds if needed
        if len(filtered) > 0 and pd.api.types.is_timedelta64_dtype(filtered['LapTime']):
            filtered['LapTime'] = filtered['LapTime'].dt.total_seconds()
        
        return filtered
    
    race1_data = filter_compound(laps1, compound)
    race2_data = filter_compound(laps2, compound)
    
    if len(race1_data) == 0 or len(race2_data) == 0:
        print(f"✗ Insufficient {compound} tire data for comparison")
        return
    
    print(f"\n{race1}: {len(race1_data)} {compound} laps")
    print(f"{race2}: {len(race2_data)} {compound} laps")
    
    # Statistics
    print(f"\nPerformance Summary ({compound}):")
    print(f"  {race1:15s}: Median {race1_data['LapTime'].median():.3f}s, "
          f"Avg Stint {race1_data['TyreLife'].mean():.1f} laps")
    print(f"  {race2:15s}: Median {race2_data['LapTime'].median():.3f}s, "
          f"Avg Stint {race2_data['TyreLife'].mean():.1f} laps")
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Lap time distribution
    axes[0].hist(race1_data['LapTime'], bins=30, alpha=0.6, label=race1, color='blue')
    axes[0].hist(race2_data['LapTime'], bins=30, alpha=0.6, label=race2, color='red')
    axes[0].set_xlabel('Lap Time (seconds)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'{compound} Tire - Lap Time Distribution')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Degradation comparison
    deg1 = race1_data.groupby('TyreLife')['LapTime'].median()
    deg2 = race2_data.groupby('TyreLife')['LapTime'].median()
    
    axes[1].plot(deg1.index, deg1.values, marker='o', label=race1, color='blue', linewidth=2)
    axes[1].plot(deg2.index, deg2.values, marker='s', label=race2, color='red', linewidth=2)
    axes[1].set_xlabel('Tire Age (laps)')
    axes[1].set_ylabel('Median Lap Time (seconds)')
    axes[1].set_title(f'{compound} Tire Degradation')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_file = f'./data/output/tire_comparison_{race1.lower()}_{race2.lower()}_{compound.lower()}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")


def quick_tire_summary(race='Bahrain', year=2024):
    """
    Generate quick tire summary for a race.
    
    Args:
        race: Grand Prix name
        year: Season year
    """
    print(f"\n{'='*70}")
    print(f"{year} {race.upper()} GP - TIRE SUMMARY")
    print('='*70)
    
    loader = TelemetryLoader()
    session = loader.load_session(year, race, 'R')
    laps = loader.extract_lap_data(session)
    
    valid = laps[
        (laps['LapTime'].notna()) &
        (laps['IsAccurate'] == True) &
        (laps['Compound'].notna())
    ].copy()
    
    # Convert Timedelta to seconds if needed
    if pd.api.types.is_timedelta64_dtype(valid['LapTime']):
        valid['LapTime'] = valid['LapTime'].dt.total_seconds()
    
    # Convert sector times if they exist and are Timedelta
    for sector_col in ['Sector1Time', 'Sector2Time', 'Sector3Time']:
        if sector_col in valid.columns and pd.api.types.is_timedelta64_dtype(valid[sector_col]):
            valid[sector_col] = valid[sector_col].dt.total_seconds()
    
    # Overall statistics
    print(f"\nOverall: {len(valid)} valid laps")
    print(f"Drivers: {valid['Driver'].nunique()}")
    print(f"Compounds used: {', '.join(valid['Compound'].unique())}")
    
    # Per compound analysis
    print(f"\n{'Compound':<12} {'Laps':<8} {'Median Time':<15} {'Avg Stint':<12} {'Max Stint'}")
    print('-'*70)
    
    for compound in sorted(valid['Compound'].unique()):
        data = valid[valid['Compound'] == compound]
        
        # Get stint information
        stints = data.groupby(['Driver', 'Stint'])['TyreLife'].max()
        
        print(f"{compound:<12} {len(data):<8} "
              f"{data['LapTime'].median():>6.3f}s        "
              f"{stints.mean():>6.1f} laps   "
              f"{stints.max():>3.0f} laps")
    
    # Top 5 fastest laps by compound
    print(f"\nFastest Laps by Compound:")
    for compound in sorted(valid['Compound'].unique()):
        data = valid[valid['Compound'] == compound]
        fastest = data.nsmallest(1, 'LapTime').iloc[0]
        print(f"  {compound:12s}: {fastest['LapTime']:.3f}s "
              f"(Lap {fastest['LapNumber']:.0f}, {fastest['Driver']})")
    
    return valid


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick F1 Tire Analysis')
    parser.add_argument('--race', default='Bahrain', help='Grand Prix name')
    parser.add_argument('--year', type=int, default=2024, help='Season year')
    parser.add_argument('--compare-race', help='Second race to compare')
    parser.add_argument('--compound', default='SOFT', help='Compound for comparison')
    
    args = parser.parse_args()
    
    if args.compare_race:
        # Compare two races
        compare_two_races(args.race, args.compare_race, args.year, args.compound)
    else:
        # Single race analysis
        quick_tire_summary(args.race, args.year)
        compare_compounds_single_race(args.race, args.year)
    
    print("\n✓ Analysis complete!")
