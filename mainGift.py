

# Import pkgs
import pandas as pd
import glob
import locale
import numpy as np
import numpy as np


# Setting
locale.setlocale(locale.LC_ALL, 'fa_IR.UTF-8')



# Functions
def quantile_score(vec, score):
    scorevec = np.zeros(len(vec))
    qu = np.quantile(vec, np.linspace(0, 1, score + 1))
    scorevec[(vec <= qu[1]) & (vec >= qu[0])] = 1
    for i in range(1, score - 1):
        scorevec[(vec <= qu[i + 1]) & (vec > qu[i])] = i + 1
    scorevec[vec > qu[score]] = score
    return scorevec




# Read Data
# Folder containing Parquet files
folder_path = "Data/data/*.parquet"

# List all parquet files
parquet_files = glob.glob(folder_path)

# Read and concatenate all files
df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)


DimDate = pd.read_csv("Data/data/DimDate.csv")

# Functions

def quantile_score(vec, score):
    scorevec = np.zeros(len(vec))
    qu = np.quantile(vec, np.linspace(0, 1, score + 1))
    scorevec[(vec <= qu[1]) & (vec >= qu[0])] = 1
    for i in range(1, score - 1):
        scorevec[(vec <= qu[i + 1]) & (vec > qu[i])] = i + 1
    scorevec[vec > qu[score]] = score
    return scorevec

def generate_dates(years, months, days30, days31):
    dates = []
    for month in months:
        if month in Months:
            dates.append(f"{years}{month}{days31}")
        else:
            dates.append(f"{years}{month}{days30}")
    return dates


# Parameters

start_date = 14030101  # Start train
target_date = 14030631  # End train

Years = ["1403"]
Months = ["{:02d}".format(i) for i in range(1, 13)]  # Generates "01" to "12"
Days30 = ["{:02d}".format(i) for i in range(1, 31)]  # Generates "01" to "30"
Days31 = ["{:02d}".format(i) for i in range(1, 32)]  # Generates "01" to "31"

print("Years:", Years)
print("Months:", Months)
print("Days30:", Days30)
print("Days31:", Days31)



main_dates = generate_dates(Years, Months, Days30, Days31)

# Remove specific dates
excluded_dates = {14021031, 14021131, 14021231, 14030731, 14030831, 14030931, 14031031, 14031131, 14031231}
main_dates = [date for date in main_dates if date not in excluded_dates]

# Data Preperation

# Convert date column to string
df["date_CHR"] = df["date"].astype(str)

# Create Miladi_Num by extracting and concatenating substrings
df["Miladi_Num"] = df["date_CHR"].str[:4] + df["date_CHR"].str[5:7] + df["date_CHR"].str[8:10]

# Convert Miladi column to string
DimDate["Miladi_CHR"] = DimDate["Miladi"].astype(str)

# Create Miladi_Num in DimDate
DimDate["Miladi_Num"] = DimDate["Miladi_CHR"].str[:4] + DimDate["Miladi_CHR"].str[5:7] + DimDate["Miladi_CHR"].str[8:10]

# Select specific columns
DimDateS = DimDate[["Jalali_1", "Miladi_Num"]]

# Left join on Miladi_Num
df = df.merge(DimDateS, on="Miladi_Num", how="left")

# Create Shamsi_Date and convert to numeric
df["Shamsi_Date"] = (df["Jalali_1"].str[:4] + df["Jalali_1"].str[5:7] + df["Jalali_1"].str[8:10]).astype(int)


# Filters
TrainRFM = df[df["module"] == "Giftcard"]
TrainRFM = TrainRFM[TrainRFM["payment_status"] == "payed"]


# R Calculation

# Group by user_id and calculate min and max dates
TrainRFM_RB = TrainRFM.groupby("user_id", as_index=False).agg(
    minDate=("Shamsi_Date_Num", "min"),
    maxDate=("Shamsi_Date_Num", "max")
)

# Initialize R_InitDF DataFrame
R_InitDF = pd.DataFrame(columns=["user_id", "R"])

# Calculate R
for i in range(len(TrainRFM_RB)):
    R_Init = (np.where(MainDates == EndDate)[0][0] - np.where(MainDates == TrainRFM_RB.loc[i, "maxDate"])[0][0]) + 1
    R_InitDF.loc[i] = [TrainRFM_RB.loc[i, "user_id"], R_Init]
    print(i)

# Merge R values back into TrainRFM_RB
TrainRFM_RB = TrainRFM_RB.merge(R_InitDF, on="user_id", how="left")

# Convert R column to numeric
TrainRFM_RB["R"] = pd.to_numeric(TrainRFM_RB["R"])

# Remove R_InitDF from memory
del R_InitDF

print(TrainRFM_RB.head)


