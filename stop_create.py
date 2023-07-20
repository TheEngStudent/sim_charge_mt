################################### READ ME #######################################
"""
This program used to copy over the re-simulated vehicle stop data, but has now been
repurposed to instead perform the same algorithm that ev-fleet-sim does. This used to 
save the files to the same directory as the folder_name.py

The program now calculates if the vehicle is stopped based on two parameters:
    -   is the vehicles velocity lower than 10 km/h - this can be considered as 
        stopping due to gps inaccuracy
    -   is the vehicle within a 20m radius of this stop location in the next instance

"""
import os
import pandas as pd
from haversine import haversine
import sys
import time


source_folder = "D:\Masters\Simulations\Simulation_1\Inputs"


# Display progress bar for simulation
def progress_bar(current, total, start_time):
    bar_length = 40
    filled_length = int(bar_length * current / total)
    percentage = current / total * 100
    elapsed_time = int((time.time() - start_time))  # Calculate elapsed time in minutes
    bar = '=' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write(f'\r[{bar}] {percentage:.2f}% Elapsed Time: {elapsed_time} seconds')
    sys.stdout.flush()



# Iterate through the subfolders in the source folder
for root, dirs, files in os.walk(source_folder):
    for folder in dirs:
        # Construct the full path to the current subfolder
        subfolder_path = os.path.join(root, folder)

        # Iterate through the files within the current subfolder
        for file_name in os.listdir(subfolder_path):

            # Check if the file name matches the desired name
            if file_name == "fcd.out.csv":
                # Construct the full path to the current file
                file_path = os.path.join(subfolder_path, file_name)

                vehicle = pd.read_csv(file_path)
                vehicle_lat = vehicle['vehicle_x']
                vehicle_long = vehicle['vehicle_y']
                vehicle_speed = vehicle['vehicle_speed']

                vehicle_stopped = False

                save_stop = pd.DataFrame(columns = ['Stop'])

                start_time = time.time()

                print(f'{subfolder_path}')

                for i in range(0, len(vehicle_lat)):


                    # Vehicle has stopped for the first time
                    if vehicle_speed[i] < 1 and vehicle_stopped == False:
                        stop_location = (vehicle_lat[i], vehicle_long[i])
                        vehicle_stopped = True
                        save_stop.loc[i] = vehicle_stopped
                    elif vehicle_stopped == False:
                        save_stop.loc[i] = vehicle_stopped

                    # Vehicle is stopped, but need to see if it is continued to stop
                    if vehicle_stopped == True:
                        current_location = (vehicle_lat[i], vehicle_long[i])
                        distance_drifted = haversine(current_location, stop_location, unit = 'm')
                        if vehicle_speed[i] >= 10 or distance_drifted >= 25:
                            vehicle_stopped = False
                            save_stop.loc[i] = vehicle_stopped
                        else:
                            save_stop.loc[i] = vehicle_stopped

                    progress_bar(i, len(vehicle_lat), start_time)

                print('\n')    
            
                
                file_path = os.path.join(subfolder_path, 'stop_time.csv')

                save_stop.to_csv(file_path, index=False)


                










"""

import pandas as pd
from datetime import datetime, timedelta
import os

### Change directory for each new simulation run required
source_folder = 'D:/Masters/Chris Data Model/GoMetro_Stop/Temporal_Clusters/Stop_Labels/'
destination_folder = 'D:/Masters/Simulations/Simulation_1/Inputs/'
file_prefix = 'stop_labels_Vehicle_'
file_suffix = '.csv'
file_common = 'Vehicle_'


### Functions
def count_files_in_folder(folder_path):
    file_count = 0
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            file_count += 1
    return file_count


new_file_name = 'stop_data' + file_suffix

num_files = count_files_in_folder(source_folder)

print(f"Number of files in the folder: {num_files}")

for i in range(1, num_files + 1):

    # Create file path to read
    full_path = source_folder + file_prefix + str(i) + file_suffix
    # Read file
    df = pd.read_csv(full_path)
    #Change date format to split
    df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S')

    # Split the data by days
    grouped = df.groupby(df['Time'].dt.date)

    new_common = file_common + str(i)

    print(f'Vehicle: {i}')

    for date, data in grouped:

        # Get the day value
        day_str = date.strftime('%d')
        # Create folder to save in based on day value
        full_path = destination_folder + new_common + '/' + new_common + '_' + day_str + '/' # //Vehicle_i/Vehicle_i_day/
        save_path = full_path + new_file_name

        # Save file to location
        data.to_csv(save_path, index=False)

        

"""