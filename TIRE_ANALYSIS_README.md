# F1 2024 Tire Analysis Scripts

Three specialized scripts for analyzing tire performance in the 2024 F1 season.

## Scripts Overview

### 1. `analyze_tires_2024.py` - Full Season Analysis
Comprehensive tire analysis across multiple races.

**What it does:**
- Loads multiple races from 2024 season
- Analyzes tire degradation by compound
- Finds optimal stint lengths
- Compares compound performance by circuit type
- Exports data and visualizations

**Usage:**
```bash
python analyze_tires_2024.py
```

**Outputs:**
- `tire_degradation_2024.png` - 4-panel visualization
- `tire_analysis_2024.csv` - Complete lap data

**Example insights:**
- SOFT tires degrade at 0.08s/lap
- Optimal MEDIUM stint: 28 laps
- HARD tires fastest at Bahrain

---

### 2. `quick_tire_compare.py` - Race Comparison
Quick comparison of compounds at specific races.

**What it does:**
- Compare all compounds at single race
- Compare same compound at different circuits
- Generate quick summaries

**Usage:**

Single race analysis:
```bash
python quick_tire_compare.py --race Monaco
```

Compare two races:
```bash
python quick_tire_compare.py --race Monaco --compare-race Monza --compound SOFT
```

**Arguments:**
- `--race`: Grand Prix name (default: Bahrain)
- `--compare-race`: Second GP to compare
- `--compound`: Tire compound (default: SOFT)
- `--year`: Season year (default: 2024)

**Outputs:**
- Race-specific visualization
- Terminal summary with statistics

**Example commands:**
```bash
# Analyze Monaco tire usage
python quick_tire_compare.py --race Monaco

# Compare SOFT tires at Monaco vs Spa
python quick_tire_compare.py --race Monaco --compare-race "Belgium" --compound SOFT

# Check Bahrain compounds
python quick_tire_compare.py --race Bahrain
```

---

### 3. `simulate_tire_strategy.py` - Strategy Simulator
Simulate and compare different tire strategies.

**What it does:**
- Builds tire degradation models from race data
- Simulates different pit strategies
- Compares race times including pit stops
- Visualizes strategy evolution

**Usage:**
```bash
python simulate_tire_strategy.py
```

Edit the script to customize strategies:
```python
strategies = {
    'Conservative (1-stop)': [
        ('HARD', 78)
    ],
    'Aggressive (2-stop)': [
        ('SOFT', 25),
        ('MEDIUM', 28),
        ('MEDIUM', 25)
    ],
}
```

**Strategy format:**
Each strategy is a list of `(compound, stint_length)` tuples.

**Outputs:**
- `strategy_comparison_[race].png` - Lap time & cumulative time plots
- Terminal comparison with time deltas

**Example output:**
```
Strategy Comparison:
  Conservative (1-stop):    6250.3s
  Standard (1-stop):        6245.8s ← Fastest
  Aggressive (2-stop):      6251.2s (+5.4s)
```

---

## Installation

Make sure you have the required packages:
```bash
pip install fastf1 pandas numpy matplotlib seaborn
```

## Data Caching

First run downloads race data (~500MB per race) and caches it locally in `./data/cache/`. Subsequent runs use the cache and are much faster (~2 seconds).

## Available 2024 Races

```python
# First half of 2024 season
races = [
    'Bahrain', 'Saudi Arabia', 'Australia', 'Japan',
    'China', 'Miami', 'Emilia Romagna', 'Monaco',
    'Canada', 'Spain', 'Austria', 'Great Britain',
    'Hungary', 'Belgium', 'Netherlands', 'Italy'
]
```

## Example Workflows

### Workflow 1: Circuit Preparation
Preparing for Monaco GP:
```bash
# 1. Check historical tire data
python quick_tire_compare.py --race Monaco

# 2. Simulate strategies
python simulate_tire_strategy.py  # (edit for Monaco)

# 3. Compare to similar street circuit
python quick_tire_compare.py --race Monaco --compare-race "Saudi Arabia" --compound SOFT
```

### Workflow 2: Season Overview
Understanding tire trends:
```bash
# 1. Load full season data
python analyze_tires_2024.py

# 2. Review tire_analysis_2024.csv for patterns

# 3. Deep dive specific races
python quick_tire_compare.py --race Bahrain
python quick_tire_compare.py --race Monaco
```

### Workflow 3: Strategy Analysis
Finding optimal strategy:
```bash
# 1. Analyze recent race
python quick_tire_compare.py --race "Great Britain"

# 2. Simulate different approaches
python simulate_tire_strategy.py  # edit strategies

# 3. Compare to actual team strategies
# (review race reports)
```

## Tire Compounds

**Color codes:**
- 🔴 SOFT (Red) - Fastest, degrades quickly
- 🟡 MEDIUM (Yellow) - Balanced performance
- ⚪ HARD (White) - Durable, slower

**Pirelli 2024 Compounds:**
- C1 (Hardest) → HARD
- C2 → HARD or MEDIUM
- C3 → MEDIUM
- C4 → SOFT or MEDIUM  
- C5 (Softest) → SOFT

Actual compound allocation varies by race.

## Interpreting Results

### Degradation Rate
- **< 0.05s/lap**: Minimal degradation
- **0.05-0.10s/lap**: Normal degradation
- **> 0.10s/lap**: High degradation

### Optimal Stint Length
Recommended maximum before significant pace loss.
Teams often push beyond for strategy reasons.

### Strategy Simulation
- Account for pit stop time (~25s including in/out laps)
- Track position can outweigh pure pace
- Safety cars/red flags change everything

## Tips & Tricks

**1. Race-specific Analysis:**
```python
# In analyze_tires_2024.py, modify:
races = ['Monaco', 'Singapore', 'Baku']  # Street circuits only
```

**2. Driver Comparison:**
```python
# Filter for specific drivers
verstappen_laps = valid[valid['Driver'] == 'VER']
hamilton_laps = valid[valid['Driver'] == 'HAM']
```

**3. Custom Strategy:**
```python
# In simulate_tire_strategy.py
my_strategy = [
    ('MEDIUM', 15),  # Opening stint
    ('SOFT', 10),    # Qualifying sim
    ('HARD', 53)     # Long run to finish
]
```

## Common Issues

**"Race not found"**
- Check spelling (case-sensitive)
- Use full name: "Great Britain" not "Silverstone"
- Some races: "Emilia Romagna", "Saudi Arabia"

**"Insufficient data"**
- Race might not have that compound
- Try different compound with `--compound`

**Memory errors**
- Load fewer races in `analyze_tires_2024.py`
- Reduce to 3-5 races for initial testing

## Output Files

All outputs saved to `./data/output/`:
- `tire_degradation_2024.png`
- `tire_analysis_2024.csv`
- `tire_comparison_[race].png`
- `strategy_comparison_[race].png`

## Next Steps

1. **Run `analyze_tires_2024.py`** to get season overview
2. **Pick a race with `quick_tire_compare.py`** for deep dive
3. **Simulate strategies** to find optimal approach
4. **Export data** and build custom analyses

## Advanced Customization

All scripts can be imported as modules:

```python
from analyze_tires_2024 import TireAnalyzer2024

analyzer = TireAnalyzer2024()
analyzer.load_season_races(races=['Monaco', 'Singapore'])
analyzer.analyze_tire_degradation()
```

Happy analyzing! 🏎️💨
