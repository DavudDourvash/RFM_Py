##### Start making dash

# import pkgs
import pandas as pd
import glob
import locale
import numpy as np
from datetime import date

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

def generate_dates(years, months, days30, days31):
    dates = []
    for month in months:
        if month in Months:
            dates.append(f"{years}{month}{days31}")
        else:
            dates.append(f"{years}{month}{days30}")
    return dates


# Parameters

start_date_jalali = 14030601
target_date_jalali = 14031130
start_date = date(2024, 8, 22)  # Start train
target_date = date(2025, 2, 18)  # End train
moduleSelected = "Onlineshopping"


# Read Data
# Folder containing Parquet files
folder_path = "Data/data/*.parquet"

# List all parquet files
parquet_files = glob.glob(folder_path)

# Read and concatenate all files
df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

DimDate = pd.read_csv("Data/data/DimDate.csv")


# Date Manipulations and Preperations
DimDateS = DimDate[["Miladi3", "Jalali_1"]]
# Convert Miladi column to string
DimDateS["Miladi3"] = DimDateS["Miladi3"].astype(str)
DimDateS.rename(columns={"Miladi3": "date_CHR"}, inplace=True)


df['date_CHR'] = df['date'].astype(str)

# Left join on Miladi_Num
df = df.merge(DimDateS, on="date_CHR", how="left")

# Create Shamsi_Date and convert to numeric
df["Shamsi_Date_Num"] = (df["Jalali_1"].str[:4] + df["Jalali_1"].str[5:7] + df["Jalali_1"].str[8:10]).astype(int)



# Filter & Select Trains
# Assuming df is already a pandas DataFrame
TrainRFM = df[df["module"] == moduleSelected]

TrainRFM = TrainRFM[TrainRFM["payment_status"] == "payed"]
TrainRFM = TrainRFM[TrainRFM["status"] == "finished"]

print(TrainRFM.columns)
print(TrainRFM.shape)

# select data for modeling
TrainRFM = TrainRFM[['user_id', '_id', 'initial_total', 'module','Shamsi_Date_Num', 'date']]

# print(TrainRFM.head(5))

# make R, F, M, L

# !pip install jdatetime
import jdatetime
import datetime
from datetime import date
# from datetime import datetime

# Group by user_id and calculate min and max dates
TrainRFM_RB = TrainRFM.groupby("user_id", as_index=False).agg(
    minDate=("Shamsi_Date_Num", "min"),
    maxDate=("Shamsi_Date_Num", "max")
)



def jalali_to_miladi(jalali_date):
    # Extract year, month, day from the integer Jalali date
    year = jalali_date // 10000
    month = (jalali_date % 10000) // 100
    day = jalali_date % 100

    # Convert to Gregorian using jdatetime
    gregorian_date = jdatetime.date(year, month, day).togregorian()

    # Return formatted Gregorian date
    return gregorian_date




TrainRFM_RB['minDate_Miladi'] = TrainRFM_RB['minDate'].apply(jalali_to_miladi)
TrainRFM_RB['maxDate_Miladi'] = TrainRFM_RB['maxDate'].apply(jalali_to_miladi)

# TrainRFM_RB['maxDate_Miladi'] = TrainRFM_RB['maxDate_Miladi'].apply(
#     lambda x: datetime.strptime(x, '%Y-%m-%d').date()
# )

TrainRFM_RB = TrainRFM_RB[
    (TrainRFM_RB['maxDate'] >= start_date_jalali) & (TrainRFM_RB['maxDate'] <= target_date_jalali)]

TrainRFM_RB['R'] = TrainRFM_RB['maxDate_Miladi'].apply(lambda d: (target_date - d).days)


# calculate F
TrainRFM_FB = TrainRFM.groupby('user_id').size().reset_index(name='F')

TrainRFM_RFB = TrainRFM_RB.merge(TrainRFM_FB, how='left', on='user_id')


# Calculate M

TrainRFM_MB = TrainRFM.groupby('user_id').agg(M=('initial_total', 'sum')).reset_index()

TrainRFMRFMB = TrainRFM_RFB.merge(TrainRFM_MB, on='user_id', how='left')

TrainRFMRFMB['M'] = TrainRFMRFMB['M'].astype(float).astype(int)


#Calculate L

TrainRFMRFMBL = TrainRFMRFMB

TrainRFMRFMBL['L'] = TrainRFMRFMBL.apply(lambda row: (row['maxDate_Miladi'] - row['minDate_Miladi']).days, axis=1) + 1


# Score R, F, M

# R
TrainRFMRFMBL['R_Norm'] = (TrainRFMRFMBL['R'] - TrainRFMRFMBL['R'].min()) / (TrainRFMRFMBL['R'].max() - TrainRFMRFMBL['R'].min())

TrainRFMRFMBL['RNormScore'] = quantile_score(TrainRFMRFMBL['R_Norm'], 5)

TrainRFMRFMBL['RNormScore'] = 6 - TrainRFMRFMBL['RNormScore']



import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import numpy as np

# فرض بر اینکه دیتافریم RFM آماده‌ست
rfm_df = TrainRFMRFMBL[['user_id', 'R', 'F', 'M']]
# انتخاب فقط ستون‌های عددی برای محاسبه میانگین
rfm_df_numeric = rfm_df.select_dtypes(include=[np.number])

# محاسبه میانگین برای ستون‌های عددی
rfm_avg = rfm_df_numeric.mean().reset_index()

# تغییر نام ستون‌ها برای نتیجه‌ی میانگین
rfm_avg.columns = ['Metric', 'Average']

rfm_avg = rfm_df[['R', 'F', 'M']].mean().reset_index()

rfm_avg.columns = ['Metric', 'Value']

app = dash.Dash(__name__)

# لیست کردن تمامی user_id ها برای استفاده در کرکره
user_ids = TrainRFMRFMBL['user_id'].unique()

# تعریف لایه داشبورد
app.layout = html.Div([
    html.H1("نمایش RFM کاربران", style={'textAlign': 'center'}),

    # باکس انتخابی (Dropdown) برای انتخاب user_id
    dcc.Dropdown(
        id='user_id_dropdown',
        options=[{'label': user_id, 'value': user_id} for user_id in user_ids],
        value=user_ids[0],  # مقدار پیش‌فرض
        style={'width': '50%', 'margin': 'auto'}
    ),

    # جایی که مقادیر R, F, M نمایش داده خواهند شد
    html.Div(id='rfm_values', style={'textAlign': 'center', 'marginTop': '20px'}),

    # نمودار میله‌ای RFM
    dcc.Graph(id='rfm_bar_chart')
])


# کال‌بک برای نمایش مقادیر RFM بر اساس انتخاب user_id
@app.callback(
    [Output('rfm_values', 'children'),
     Output('rfm_bar_chart', 'figure')],
    [Input('user_id_dropdown', 'value')]
)
def update_rfm(user_id):
    # فیلتر کردن دیتافریم بر اساس user_id انتخاب شده
    user_rfm = TrainRFMRFMBL[TrainRFMRFMBL['user_id'] == user_id]

    # استخراج مقادیر R, F, M
    R_value = user_rfm['R'].iloc[0]
    F_value = user_rfm['F'].iloc[0]
    M_value = user_rfm['M'].iloc[0]

    # نمایش مقادیر RFM
    rfm_display = html.Div([
        html.P(f"R: {R_value}"),
        html.P(f"F: {F_value}"),
        html.P(f"M: {M_value}")
    ])

    # ایجاد نمودار میله‌ای از مقادیر RFM
    rfm_fig = {
        'data': [
            {'x': ['R', 'F', 'M'], 'y': [R_value, F_value, M_value], 'type': 'bar', 'name': user_id}
        ],
        'layout': {
            'title': f"RFM for user {user_id}",
            'xaxis': {'title': 'Metric'},
            'yaxis': {'title': 'Value'}
        }
    }

    return rfm_display, rfm_fig


# اجرای اپلیکیشن
if __name__ == '__main__':
    app.run_server(debug=True)