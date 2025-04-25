# Event Study Analysis: Spiess and Affleck-Graves (1995)

## Overview
This project performs an event study and long-run performance analysis of equity issues using data from Spiess and Affleck-Graves (1995, JFE).

## Data Sources
- Equity issues data from Spiess and Affleck-Graves (1995, JFE) via course website
- Market returns and Fama-French factors from Kenneth French's data library

## Analysis
1. Event study with {-1, +1} window around announcement dates
2. Calendar time analysis using Mitchell and Stafford (2000) methodology
3. Comparison between CAPM and Fama-French three-factor model

## Replication Instructions
1. Place the equity issues data in `data/raw/equity_issues.csv`
2. Download Fama-French factors to `data/external/ff_factors.csv`
3. Run `code/data_prep/01_prepare_data.py`
4. Run `code/analysis/02_event_study.py`
5. Run `code/analysis/03_calendar_time.py`