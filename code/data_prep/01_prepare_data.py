import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_prep.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_equity_issues():
    """Load equity issues data from Spiess and Affleck-Graves (1995)"""
    try:
        # Load the Excel file with skipping header rows
        logger.info("Loading equity issues data from Excel file")
        df = pd.read_excel('data/raw/seospiess.xls', skiprows=3)
        logger.info(f"Loaded equity issues data: {len(df)} observations")

        # Print column information for debugging
        logger.info(f"Original columns: {df.columns.tolist()}")

        # Column names based on your Excel file structure
        # Based on the image you shared, column 7 is ISSUE_DATE and column 8 is ANNOUNCEMENT_DATE
        # Adjust these indices based on your actual data
        new_columns = [
            'Perm', 'CUSIP', 'SIC', 'ISSUER_NAME', 'SHARES', 'PRICE',
            'OFFER_SIZE', 'ISSUE_DATE', 'ANNOUNCEMENT_DATE', 'PERIOD'
        ]

        # Rename columns if the number matches
        if len(df.columns) >= len(new_columns):
            df.columns = new_columns + list(df.columns[len(new_columns):])
            logger.info(f"Renamed columns to: {df.columns.tolist()}")
        else:
            logger.warning(f"Column count mismatch: expected at least {len(new_columns)}, got {len(df.columns)}")

        # Convert all problematic columns to string
        for col in df.columns:
            if col in ['Perm', 'CUSIP', 'SIC']:
                logger.info(f"Converting {col} to string type")
                df[col] = df[col].astype(str)

        # Ensure date columns are datetime
        if 'ISSUE_DATE' in df.columns:
            logger.info("Converting ISSUE_DATE to datetime")
            df['ISSUE_DATE'] = pd.to_datetime(df['ISSUE_DATE'], errors='coerce')

        if 'ANNOUNCEMENT_DATE' in df.columns:
            logger.info("Converting ANNOUNCEMENT_DATE to datetime")
            df['ANNOUNCEMENT_DATE'] = pd.to_datetime(df['ANNOUNCEMENT_DATE'], errors='coerce')

        return df

    except Exception as e:
        logger.error(f"Error loading equity issues data: {e}")
        raise

def load_ff_factors():
    """Load Fama-French factors"""
    try:
        # Load the daily factors - skip rows and set column names
        logger.info("Loading daily Fama-French factors")
        daily_ff = pd.read_csv('data/raw/F-F_Research_Data_Factors_daily.CSV',
                               skiprows=5,
                               header=None,
                               names=['date', 'Mkt-RF', 'SMB', 'HML', 'RF'])

        # Convert date format for daily data (YYYYMMDD)
        daily_ff['date'] = pd.to_datetime(daily_ff['date'], format='%Y%m%d', errors='coerce')

        # Convert factors from percent to decimal
        for col in ['Mkt-RF', 'SMB', 'HML', 'RF']:
            daily_ff[col] = pd.to_numeric(daily_ff[col], errors='coerce') / 100

        logger.info(f"Loaded daily Fama-French factors: {len(daily_ff)} observations")

        # Load monthly factors - skip rows and set column names
        logger.info("Loading monthly Fama-French factors")
        monthly_ff = pd.read_csv('data/raw/F-F_Research_Data_Factors.CSV',
                                 skiprows=4,
                                 header=None,
                                 names=['date', 'Mkt-RF', 'SMB', 'HML', 'RF'])

        # Convert date format for monthly data (YYYYMM)
        monthly_ff['date'] = pd.to_datetime(monthly_ff['date'].astype(str), format='%Y%m', errors='coerce')

        # Convert factors from percent to decimal
        for col in ['Mkt-RF', 'SMB', 'HML', 'RF']:
            monthly_ff[col] = pd.to_numeric(monthly_ff[col], errors='coerce') / 100

        logger.info(f"Loaded monthly Fama-French factors: {len(monthly_ff)} observations")

        return daily_ff, monthly_ff

    except Exception as e:
        logger.error(f"Error loading Fama-French factors: {e}")
        raise

def prepare_data():
    """Prepare data for event study analysis"""
    # Create directories if they don't exist
    os.makedirs('data/processed', exist_ok=True)

    # Load the data
    issues_df = load_equity_issues()
    daily_ff, monthly_ff = load_ff_factors()

    # Save directly to CSV (safer option)
    logger.info("Saving data to CSV files")
    issues_df.to_csv('data/processed/equity_issues.csv', index=False)
    daily_ff.to_csv('data/processed/ff_daily.csv', index=False)
    monthly_ff.to_csv('data/processed/ff_monthly.csv', index=False)

    logger.info("Data processed and saved successfully")

    return issues_df, daily_ff, monthly_ff

if __name__ == "__main__":
    prepare_data()