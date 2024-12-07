import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress
import matplotlib.image as image
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
import os
import warnings

# warnings.simplefilter(action='ignore', category=FutureWarning)
###################################
############# Feeding #############
###################################

parser = argparse.ArgumentParser()

parser.add_argument("-m", "--mode", help="Choose running modality ['prof_vert_archivio', 'prof_vert']", default="data/")
parser.add_argument("-d", "--date", help="Use bash command 'date +%Y%m%d_%H%M' ")
parser.add_argument("-i", "--input", required=False, help="the data-input folder", default="data/")
parser.add_argument("-o", "--output", required=False, help="the output folder", default="maps/")
# parser.print_help()
args = parser.parse_args()

running_modality = ['prof_vert_archivio', 'prof_vert']
if args.mode not in running_modality:
    print(f"{args.mode} is not present in the running modality")
    exit()

##########
# creazione directory
if not os.path.isdir(f'{args.output}/{args.mode}/{args.date[:-5]}/'):
    os.makedirs(f'{args.output}/{args.mode}/{args.date[:-5]}/')

###################################
###################################
###################################

# Carica il file CSV

file_path = f'{args.input}{args.date[:-5]}/dati_{args.date}.csv' 
try:
    data = pd.read_csv(file_path,na_values="NA")
except:
    print("Something went wrong, can't find data file")
    exit()

data.columns = data.columns.str.strip()
data['datetime'] = pd.to_datetime(data['datetime'])
filtered_data = data[
    ~data['stazione'].str.contains('Camporella|Civago|Villa Minozzo', na=False)
][['datetime', 'quota', 'temp_val', 'dewpoint', 'temp_wet_bulb']].dropna()
sorted_data = filtered_data[['datetime', 'quota', 'temp_val', 'dewpoint', 'temp_wet_bulb']].sort_values(by='quota', ascending=True)


# Function to analyze and regress data
def analyze_and_regress(data, value_column, target_value, elevation_column):
    """
    Analyze the dataset to find values close to the target and perform regression.
    """
    # Drop rows with NaN in the necessary columns
    valid_data = data.dropna(subset=[value_column, elevation_column])
    value_exist = True

    # Check if there are any values below the target
    below_target = valid_data[valid_data[value_column] < target_value]
    if below_target.empty:
        print(f"{value_column}: No values below {target_value}. Regression not performed.")
        value_exist = False
        return None

    # Check if any value matches the target exactly
    if (valid_data[value_column] == target_value).any():
        print(f"{value_column}: Value {target_value} exists in the dataset. No regression needed.")
        # value_exist = 0 
        # print(value_exist)
        return None

    # Identify rows closest to the target value
    valid_data['diff'] = np.abs(valid_data[value_column] - target_value)
    closest_two = valid_data.nsmallest(2, 'diff')

    if len(closest_two) < 2:
        print(f"{value_column}: Not enough data points for regression.")
        value_exist = False
        print(value_exist)
        # return value_exist

    # Perform linear regression
    x = closest_two[value_column].values
    y = closest_two[elevation_column].values
    slope, intercept, _, _, _ = linregress(x, y)

    # Calculate elevation at target value
    elevation_at_target = slope * target_value + intercept
    print(f"{value_column}: Elevation at {target_value} is {elevation_at_target:.2f} m.")
    return elevation_at_target, value_exist

zero_termico, zero_flag = analyze_and_regress(sorted_data, 'temp_val', 0, 'quota')

# Perform analysis for wet bulb temperature ('temp_wet_bulb') at 1.5°C
quota_neve, snow_flag = analyze_and_regress(sorted_data, 'temp_wet_bulb', 1.5, 'quota')

fig, ax = plt.subplots(figsize=(12, 12))

if not sorted_data['temp_val'].isna().all():
    ax.plot(
        sorted_data['temp_val'].dropna(),
        sorted_data['quota'][sorted_data['temp_val'].notna()],
        label='Temperatura (°C)',
        linewidth=3,
        color='red',
    )

if not sorted_data['dewpoint'].isna().all():
    ax.plot(
        sorted_data['dewpoint'].dropna(),
        sorted_data['quota'][sorted_data['dewpoint'].notna()],
        label='Punto di rugiada (°C)',
        color='green',
    )

if not sorted_data['temp_wet_bulb'].isna().all():
    ax.plot(
        sorted_data['temp_wet_bulb'].dropna(),
        sorted_data['quota'][sorted_data['temp_wet_bulb'].notna()],
        label='Temperatura bulbo umido (°C)',
        color='blue',
    )

try:
    # Load the logo
    logo = image.imread('/volume1/web/images/reggioemiliameteo-logo-mappe.jpg')
    # Create an OffsetImage
    image_box = OffsetImage(logo, zoom=0.5)  # Adjust zoom as necessary
    # Position the logo
    ab = AnnotationBbox(image_box, (0.05, 0), frameon=False,
                         xycoords="axes fraction", box_alignment=(0, 0))
    plt.gca().add_artist(ab)  # Add the image to the plot
except Exception as e:
    print("Error adding logo:", e)
    

plt.ylim([0, 2100])
plt.xlabel('Gradi (°C)')
plt.ylabel('Quota (m)')
plt.title(f'Temperatura, Punto di rugiada e Temperatura di bulbo umido vs Quota \n{sorted_data["datetime"].dt.strftime("%Y-%m-%d %H:%M").iloc[0]}')
plt.legend(loc='upper right')

# if (zero_flag==True) & (quota_neve<2100):
#     plt.axhline(y=zero_termico , color='k', linestyle='--', )#label=f'Zero termico')
#     plt.text(sorted_data['temp_val'].iloc[1], zero_termico, f'Zero termico {int(np.round(zero_termico/10)*10)} m', color='black', ha='right', va='bottom', fontsize=8)
# if  (snow_flag==True) & (quota_neve<2000) :
#     plt.axhline(y=quota_neve, color='teal', linestyle='--',)# label='Quota neve'
#     plt.text(sorted_data['temp_val'].iloc[1], quota_neve, f'Stima quota neve {int(np.round(quota_neve/10)*10)} m', color='teal', ha='right', va='bottom', fontsize=8)

plt.grid(True)

try:
    fig.savefig(
            f"{args.output}/{args.mode}/{args.date[:-5]}/prof_vert_{args.date}.png",
            bbox_inches="tight",
            transparent=False,
        )
except:
    print("Can't save the figure, is the output path correct?")
    exit()
