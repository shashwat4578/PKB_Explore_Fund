import requests
import pandas as pd
import io
import os
import time
from datetime import datetime, timedelta

def get_latest_nav_data(start_date_str, end_date_str, max_retries=3):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36"
        )
    }

    url = (
        "https://portal.amfiindia.com/DownloadNAVHistoryReport_Po.aspx"
        f"?tp=1&frmdt={start_date_str}&todt={end_date_str}"
    )

    print(f"Fetching data from {start_date_str} to {end_date_str}...")

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=120)
            response.raise_for_status()

            lines = response.text.splitlines()

            if not any("Scheme Code" in line for line in lines):
                print("Warning: AMFI response does not contain valid headers.")
                if attempt < max_retries:
                    time.sleep(3)
                    continue
                return None

            header = None
            cleaned_data = []
            current_category = "Unknown"

            for line in lines:
                line = line.strip().lstrip("\ufeff")

                if not line:
                    continue

                parts = [part.strip() for part in line.split(";")]

                # Identify the column header.
                if parts[0].strip() == "Scheme Code":
                    header = parts
                    continue

                # Ignore rows before the column header.
                if header is None:
                    if "(" in line and ")" in line:
                        current_category = line
                    continue

                # Identify scheme data rows by numeric scheme code.
                try:
                    scheme_code = float(parts[0])
                except (ValueError, IndexError):
                    if "(" in line and ")" in line:
                        current_category = line
                    continue

                if len(parts) < len(header):
                    parts.extend([""] * (len(header) - len(parts)))
                elif len(parts) > len(header):
                    parts = parts[:len(header)]

                parts[0] = str(int(scheme_code))
                cleaned_data.append(parts + [current_category])

            if not cleaned_data or header is None:
                print("No valid NAV records found in the response.")
                return None

            df = pd.DataFrame(
                cleaned_data,
                columns=header + ["Category"]
            )

            # Standardize column names and fill alternate scheme names.
            df.columns = df.columns.str.strip()

            if "NAV Name" in df.columns:
                if "Scheme Name" in df.columns:
                    df["Scheme Name"] = df["Scheme Name"].replace(
                        "", pd.NA
                    ).fillna(df["NAV Name"])
                else:
                    df["Scheme Name"] = df["NAV Name"]

            # Convert numeric fields safely.
            df["Scheme Code"] = pd.to_numeric(
                df["Scheme Code"], errors="coerce"
            ).astype("Int64")

            numeric_cols = [
                "Net Asset Value",
                "Repurchase Price",
                "Sale Price",
            ]

            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Remove rows without a valid scheme code.
            df = df.dropna(subset=["Scheme Code"])

            print(f"Retrieved {len(df)} NAV records.")
            return df

        except requests.RequestException as e:
            print(f"Attempt {attempt}/{max_retries} failed: {e}")
        except Exception as e:
            print(f"Error processing NAV data: {e}")
            return None

        if attempt < max_retries:
            time.sleep(3)

    print("Failed to fetch NAV data after all retries.")
    return None
    
def update_year_parquet(year):
    output_dir = f"nav_history/year={year}"
    file_path = f"{output_dir}/data.parquet"
    os.makedirs(output_dir, exist_ok=True)
    
    today = datetime.now()
    if year < today.year:
        # For past years, we should have the data already, but let's check
        end_date_dt = datetime(year, 12, 31)
    else:
        # For current year
        end_date_dt = today
    
    end_date_str = end_date_dt.strftime("%d-%b-%Y")
    
    if os.path.exists(file_path):
        try:
            existing_df = pd.read_parquet(file_path)
            if 'Date' in existing_df.columns:
                existing_df['Date_dt'] = pd.to_datetime(existing_df['Date'], format='%d-%b-%Y', errors='coerce')
                last_date = existing_df['Date_dt'].max()
                if pd.notnull(last_date):
                    start_date_dt = last_date + timedelta(days=1)
                    if start_date_dt > end_date_dt:
                        print(f"Data for {year} is already up to date ({last_date.strftime('%Y-%m-%d')}).")
                        return
                    start_date_str = start_date_dt.strftime("%d-%b-%Y")
                else:
                    start_date_str = f"01-Jan-{year}"
            else:
                start_date_str = f"01-Jan-{year}"
        except Exception as e:
            print(f"Error reading existing parquet: {e}. Re-fetching.")
            start_date_str = f"01-Jan-{year}"
    else:
        start_date_str = f"01-Jan-{year}"

    # If the range is too large (more than 45 days), fetch in monthly chunks to avoid timeouts
    start_date_dt = datetime.strptime(start_date_str, "%d-%b-%Y")
    
    all_new_dfs = []
    current_start = start_date_dt
    while current_start < end_date_dt:
        # Calculate end of current month or end_date_dt
        next_month = (current_start.replace(day=1) + timedelta(days=32)).replace(day=1)
        current_end = min(next_month - timedelta(days=1), end_date_dt)
        
        chunk_df = get_latest_nav_data(current_start.strftime("%d-%b-%Y"), current_end.strftime("%d-%b-%Y"))
        if chunk_df is not None and not chunk_df.empty:
            all_new_dfs.append(chunk_df)
        
        current_start = current_end + timedelta(days=1)
        if current_start < end_date_dt:
            time.sleep(2) # Respect AMFI rate limit

    if all_new_dfs:
        new_df = pd.concat(all_new_dfs, ignore_index=True)
        if os.path.exists(file_path):
            existing_df = pd.read_parquet(file_path)
            if 'Date_dt' in existing_df.columns:
                existing_df = existing_df.drop(columns=['Date_dt'])
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates()
            combined_df.to_parquet(file_path, engine='pyarrow', compression='snappy')
            print(f"DONE: Updated {file_path}")
        else:
            new_df.to_parquet(file_path, engine='pyarrow', compression='snappy')
            print(f"DONE: Created {file_path}")
    else:
        print(f"No new data found for {year}")

if __name__ == "__main__":
    import sys
    
    current_year = datetime.now().year
    
    if "--all" in sys.argv:
        # Full historical backfill
        years = list(range(2021, current_year + 1))
    else:
        # Daily update: Sync current and previous year
        years = [current_year - 1, current_year]
        
    for year in years:
        update_year_parquet(year)
