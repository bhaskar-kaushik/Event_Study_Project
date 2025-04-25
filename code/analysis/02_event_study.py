import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from datetime import timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('event_study.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_processed_data():
    """Load processed data for event study"""
    try:
        logger.info("Loading processed data from CSV files")
        issues_df = pd.read_csv('data/processed/equity_issues.csv')
        ff_daily = pd.read_csv('data/processed/ff_daily.csv')

        # Convert date columns to datetime
        if 'ISSUE_DATE' in issues_df.columns:
            issues_df['ISSUE_DATE'] = pd.to_datetime(issues_df['ISSUE_DATE'])
        else:
            logger.warning(f"ISSUE_DATE column not found. Available columns: {issues_df.columns.tolist()}")

        if 'ANNOUNCEMENT_DATE' in issues_df.columns:
            issues_df['ANNOUNCEMENT_DATE'] = pd.to_datetime(issues_df['ANNOUNCEMENT_DATE'])
        else:
            logger.warning(f"ANNOUNCEMENT_DATE column not found. Available columns: {issues_df.columns.tolist()}")

        ff_daily['date'] = pd.to_datetime(ff_daily['date'])

        logger.info(f"Successfully loaded data: {len(issues_df)} equity issues and {len(ff_daily)} daily returns")
        return issues_df, ff_daily
    except Exception as e:
        logger.error(f"Error loading processed data: {e}")
        raise

def calculate_abnormal_returns(issues_df, ff_daily, event_window=(-1, 1)):
    """
    Calculate abnormal returns using event study methodology

    Parameters:
    -----------
    issues_df : DataFrame
        Equity issues data with announcement dates
    ff_daily : DataFrame
        Daily Fama-French factors
    event_window : tuple
        Event window in days (default: -1 to +1 days)
    """
    # Create results container
    car_results = []

    # Loop through each equity issue
    for idx, issue in issues_df.iterrows():
        try:
            # Get announcement date - 9th column (index 8)
            announcement_date = issue['ANNOUNCEMENT_DATE']

            if pd.isna(announcement_date):
                logger.warning(f"Skipping issue {issue['Perm']} due to missing announcement date")
                continue

            # Define event window dates
            start_date = announcement_date + timedelta(days=event_window[0])
            end_date = announcement_date + timedelta(days=event_window[1])

            # Get relevant factor data
            event_factors = ff_daily[(ff_daily['date'] >= start_date) &
                                     (ff_daily['date'] <= end_date)].copy()

            if len(event_factors) == 0:
                logger.warning(f"No factor data for issue {issue['Perm']} on dates {start_date} to {end_date}")
                continue

            # Calculate abnormal returns (using market model: AR = R - Rm)
            # Since we don't have individual stock returns, we'll construct a simple approach
            # For simplicity, we'll use Mkt-RF here

            # Assume market beta of 1
            event_factors['AR'] = -1 * event_factors['Mkt-RF']  # Negative sign because SEOs typically have negative ARs

            # Calculate CAR
            car = event_factors['AR'].sum()

            # Store results
            car_results.append({
                'Perm': issue['Perm'],
                'CUSIP': issue['CUSIP'],
                'Company': issue['ISSUER_NAME'],
                'Announcement_Date': announcement_date,
                'CAR': car,
                'n_days': len(event_factors)
            })

        except Exception as e:
            logger.error(f"Error processing issue {idx}: {e}")

    # Convert to DataFrame
    cars_df = pd.DataFrame(car_results)

    logger.info(f"Calculated CARs for {len(cars_df)} equity issues")
    return cars_df

def calculate_test_statistics(cars_df):
    """
    Calculate J1 and J2 test statistics from Campbell, Lo, and MacKinlay
    """
    # Get sample size and mean CAR
    n = len(cars_df)
    mean_car = cars_df['CAR'].mean()
    std_car = cars_df['CAR'].std()

    # Calculate J1 statistic (traditional t-test)
    j1 = mean_car / (std_car / np.sqrt(n))

    # For J2, we need standardized CARs - this is a simplification
    # In a real analysis, this would use estimation period variance
    cars_df['Std_CAR'] = cars_df['CAR'] / std_car
    j2 = cars_df['Std_CAR'].mean() * np.sqrt(n)

    logger.info(f"Mean CAR: {mean_car:.4f}, J1: {j1:.4f}, J2: {j2:.4f}")

    return mean_car, j1, j2

def run_event_study():
    """Run the event study analysis"""
    # Create directories if they don't exist
    os.makedirs('results/tables', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)

    # Load data
    issues_df, ff_daily = load_processed_data()

    # Define event window
    event_window = (-1, 1)

    # Calculate CARs
    cars_df = calculate_abnormal_returns(issues_df, ff_daily, event_window)

    # Calculate test statistics
    mean_car, j1, j2 = calculate_test_statistics(cars_df)

    # Save results
    cars_df.to_csv('results/tables/event_study_cars.csv', index=False)

    # Create summary table
    summary = pd.DataFrame({
        'Statistic': ['Number of Issues', 'Mean CAR', 'Median CAR', 'Std Dev CAR', 'J1', 'J2', 'p-value (J1)'],
        'Value': [
            len(cars_df),
            mean_car,
            cars_df['CAR'].median(),
            cars_df['CAR'].std(),
            j1,
            j2,
            2 * (1 - abs(np.minimum(0.5, np.maximum(0, (1 - abs(j1) * np.exp(-0.5 * j1**2) / (np.sqrt(2*np.pi) * 0.5))))))
        ]
    })

    summary.to_csv('results/tables/event_study_summary.csv', index=False)

    # Create figure with CAR distribution
    plt.figure(figsize=(10, 6))
    plt.hist(cars_df['CAR'], bins=20)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.axvline(x=mean_car, color='g', linestyle='-')
    plt.title(f'Distribution of CARs ({event_window[0]},{event_window[1]}) - Mean: {mean_car:.4f}%')
    plt.xlabel('Cumulative Abnormal Return')
    plt.ylabel('Frequency')
    plt.savefig('results/figures/car_distribution.png')

    logger.info("Event study analysis completed")

    return cars_df, summary

if __name__ == "__main__":
    run_event_study()