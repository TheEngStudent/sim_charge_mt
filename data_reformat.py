################################################## READ ME ##################################################
"""
    This programme ultimately transforms the data to be used by sim_charge.py

    This program looks at the the various number of folders and subfolders within a specified location. For any
    scenario, it will count the number of vehicles present. It will then iterate through each vehicle accordingly. 
    It then counts the number of days present within a specified vehicle. For each day in a vehicle, the nescessary
    data for sim_charge.py is created. This data has the following format for final_sec:

    Time_of_Day | Energy_Consumption    | Latitude  | Longitude | Battery_Capacity  | SOC       | Stop      
    --------------------------------------------------------------------------------------------------------
    (string)    | (float)               | (float)   | (float)   | (float)           | (float)   | (boolean) 
    [YYYY/MM/DD | [Wh/s]                | [degres]  | [degrees] | [kWh]             | [%]       |           

    This is done to test if indeed the vehicle has driven or if there is bad data that exists. This is is then
    transformed to the following vehicle_day format to be used by sim_charge.

    Time_of_Day | Energy_Consumption    | Latitude  | Longitude | Stop      | 20_Min_Stop   | Hub_Location  | Available_Charging    | HC_Location   | Home_Charging     
    -------------------------------------------------------------------------------------------------------------------------------------------------------------------
    (string)    | (float)               | (float)   | (float)   | (boolean) | (boolean)     | (boolean)     | (boolean)             | (boolean)     | (boolean)           
    [YYYY/MM/DD | [Wh/s]                | [degres]  | [degrees] |           |               |               |                       |               |                   
      HH:MM:SS]


    It is important to note the battery capacity given here is for 70kWh, and the SOC is calculated for no charging 
    scenario. That is why sim_charge.py is used for the entire algorithm. The only nescessary columns for sim_charge.py
    is:
        Time_of_Day
        Energy_Consumption
        Available_Charging
        Home_Charging
    The other columns are merely there for ease of use and visualisations to ensure that the code is running smoothly.
    The various days are plotted indivually and all the days for a vehicle are also plotted on one graph. It is important
    to note that if this code is used for a new scenario, the directory folders specified later on need to be changed
    accordingly for the scenario.

    The HC_Location and Home_Charging is created in the second iteration of going through everything. This portion of code
    also re-samples the day to start from 04:00:00 to 03:59:59 of the next day. It is able to get the stop location for home
    charging by finding the most occuring location between 20:00:00 and 03:59:59. This is set as the stop oint for the vehicle
    and if the vehicle is within a 25m radius of this point, then it is available to charge at home.

    TO_NOTE -- This is setup for data that is for a month, would need to change for data that is more than a month

"""

import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from haversine import haversine

### Constants
battery_capacity = 70 #kWh
file_name_1 = 'battery.out.csv'
file_name_2 = 'fcd.out.csv'
file_name_3 = 'stop_time.csv'
csv_save_name_1 = 'final_sec.csv'
csv_save_name_2 = 'final_min.csv'
csv_save_name_3 = 'vehicle_day_sec.csv'
png_file_name = 'Time_vs_SOC.png'
png_total_name = 'Daily_SOC.png'

### Change directories for each time running
source_folder = "D:/Masters/Simulations/Simulation_1/Inputs/"
destination_folder = "D:/Masters/Simulations/Simulation_1/Usable_Data/"
folder_prefix = "Vehicle_" 

### Box coordinates - Stellenbosch Taxi Rank
stop_location = [
    (-33.932359, 18.857750),  
    (-33.932359, 18.859046),       
    (-33.933172, 18.859046),      
    (-33.933172, 18.857750)       
]

### Constants
specified_length = 86400 # number of seconds in a day

### Folder Functions
def count_folders_with_prefix(directory_path, prefix):
    folder_count = 0
    for folder_name in os.listdir(directory_path):
        if os.path.isdir(os.path.join(directory_path, folder_name)) and folder_name.startswith(prefix):
            folder_count += 1
    return folder_count

def copy_folder_structure(source_folder, destination_folder):
    for root, dirs, _ in os.walk(source_folder):
        relative_path = os.path.relpath(root, source_folder)
        destination_path = os.path.join(destination_folder, relative_path)
        os.makedirs(destination_path, exist_ok=True)
        print(f"Created folder: {destination_path}")

def get_last_two_values_as_strings(directory_path):
    folder_names = os.listdir(directory_path)
    last_two_values_array = []
    for folder_name in folder_names:
        if os.path.isdir(os.path.join(directory_path, folder_name)):
            last_two_values = folder_name[-2:]
            last_two_values_array.append(str(last_two_values))
    
    return last_two_values_array

### General Functions
def is_point_in_stop(point):
    lat, lon = point
    latitudes = [coord[0] for coord in stop_location]
    longitudes = [coord[1] for coord in stop_location]
    min_lat, max_lat = min(latitudes), max(latitudes)
    min_lon, max_lon = min(longitudes), max(longitudes)
    
    if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
        return True
    else:
        return False
    
def count_20_minutes(column_stop):
    empty_stop = np.empty_like(column_stop)
    first_true_flag = 1
    counter_true = 0

    ### Check if vehicle has stopped for more than 20 minutes
    for j in range(0, len(column_stop) - 1):

        ### Counts the number of true values starting at the specific index 
        if column_stop[j] == True:
            if first_true_flag == 1:
                current_index = j
                first_true_flag = 0
            elif first_true_flag == 0:
                counter_true = counter_true + 1

        ### If false, 20_min_stop is automatically false
        if column_stop[j] == False:
            #merged_data.loc[j, '20_Min_Stop'] = False
            empty_stop[j] = False
            counter_true = 0

        if column_stop[j] != column_stop[j + 1]:
            first_true_flag = 1
            if counter_true >= 1200:
                for m in range(current_index, current_index + counter_true + 1):
                    #merged_data.loc[m, '20_Min_Stop'] = True
                    empty_stop[m] = True
                counter_true = 0
            elif counter_true < 1200 and column_stop[j + 1] == False:
                for g in range(current_index, current_index + counter_true + 1):
                    #merged_data.loc[g, '20_Min_Stop'] = Fals
                    empty_stop[g] = False
                counter_true = 0            
    else:
        if counter_true >= 1200:
            for m in range(current_index, current_index + counter_true + 2):
                #merged_data.loc[m, '20_Min_Stop'] = True
                empty_stop[m] = True

    return empty_stop


def is_point_at_home(row, most_common):
    target_latitude = most_common[0]
    target_longitude = most_common[1]
    point_latitude = row['Latitude']
    point_longitude = row['Longitude']
    distance = haversine((target_latitude, target_longitude), (point_latitude, point_longitude), unit = 'm')
    return distance <= 150  


### Create output folders to save everything to
os.makedirs(destination_folder, exist_ok=True)  # Create the destination folder if it doesn't exist
copy_folder_structure(source_folder, destination_folder)

### Get the number of folders within the directory
### Change based on simulation
num_folders = count_folders_with_prefix(source_folder, folder_prefix)

### Iterate through each vehicle in directory
for i in range(1, num_folders + 1):

    ### Get the number of days within that folder
    ### No need to change for different smulation
    new_folder = source_folder + folder_prefix + str(i) + '/'
    new_folder_prefix = folder_prefix + str(i) + '_' # Vehicle_i_
    num_days = count_folders_with_prefix(new_folder, new_folder_prefix)

    day_num_array = get_last_two_values_as_strings(new_folder)

    data_list = [] # Initialise for each new iteration after daily data has been plotted 

    ### Iterate through each day in vehicle
    for k in range(1, num_days + 1):

        ### Read the file from Inputs
        file_folder = new_folder + new_folder_prefix + day_num_array[k - 1] + '/' # Vehicle_i_k
        full_path_1 = file_folder + file_name_1 # battery data
        full_path_2 = file_folder + file_name_2 # fcd data
        full_path_3 = file_folder + file_name_3 # stop data
        battery_data = pd.read_csv(full_path_1)
        fcd_data = pd.read_csv(full_path_2)
        stop_data = pd.read_csv(full_path_3)       

        ### Read file columns
        # Time data
        column_time = battery_data['timestep_time']
        column_time = column_time.astype(int)
        column_time = pd.to_datetime(column_time, origin='unix', unit='s')
        column_time = pd.to_datetime(column_time).dt.time
        column_time= [time.strftime('%H:%M:%S') for time in column_time] # getting the data to right format
        date = pd.to_datetime('1900-01-01')
        column_time = pd.to_datetime(date.strftime('%Y-%m-%d') + ' ' + pd.Series(column_time).apply(str))

        # Energy data
        column_energy = battery_data['vehicle_energyConsumed']

        # Location data
        column_lat = fcd_data['vehicle_y']
        column_long = fcd_data['vehicle_x']

        # Stop data - Read from data
        column_stop = stop_data['Stop']

        ### Create new data frame with read data
        data = pd.DataFrame({'Time_of_Day': column_time, 'Energy_Consumption': column_energy, 
                             'Latitude': column_lat, 'Longitude': column_long,
                             'Stopped': column_stop})

        ### Fill in the missing data
        # 00:00:00 to (data[start_time] - 1)
        start_time_1 = datetime.strptime('00:00:00', '%H:%M:%S')
        end_time_1 = datetime.combine(datetime(1900, 1, 1), column_time.iloc[0].time()).strftime('%Y-%m-%d %H:%M:%S')
        end_time_1 = datetime.strptime(end_time_1, '%Y-%m-%d %H:%M:%S')
        end_time_1 = end_time_1 - timedelta(seconds = 1)  # Subtract one second from end_time_1
        end_time_1 = end_time_1.strftime('%Y-%m-%d %H:%M:%S')
        # (data[end_time] + 1) to 23:59:59
        start_time_2 = datetime.combine(datetime(1900, 1, 1), column_time.iloc[-1].time()).strftime('%Y-%m-%d %H:%M:%S')
        start_time_2 = datetime.strptime(start_time_2, '%Y-%m-%d %H:%M:%S')
        start_time_2 = start_time_2 + timedelta(seconds = 1)  # Subtract one second from end_time_1
        start_time_2 = start_time_2.strftime('%Y-%m-%d %H:%M:%S')
        end_time_2 = datetime.strptime('23:59:59', '%H:%M:%S')
        # Create time data
        time_range_1 = pd.date_range(start = start_time_1, end = end_time_1, freq='S')
        time_range_2 = pd.date_range(start = start_time_2, end = end_time_2, freq='S')
        # Bottom data
        new_data_1 = pd.DataFrame({'Time_of_Day': time_range_1})
        new_data_1['Energy_Consumption'] = 0
        new_data_1['Latitude'] = column_lat.iloc[0]
        new_data_1['Longitude'] = column_long.iloc[0]
        new_data_1['Stopped'] = True # Same reason as below
        # Top data
        new_data_2 = pd.DataFrame({'Time_of_Day': time_range_2})
        new_data_2['Energy_Consumption'] = 0
        new_data_2['Latitude'] = column_lat.iloc[-1]
        new_data_2['Longitude'] = column_long.iloc[-1]
        new_data_2['Stopped'] = True # This changed due to the fact that there is no data for vehicle past point and so is assumed to have stopped
        # Merge bottom, original and top data
        merged_data = pd.concat([new_data_1, data, new_data_2])
        merged_data.reset_index(drop=True, inplace=True)
        
        # If data goes above one day, then trim it to one day length (86400 seconds)
        if len(merged_data) > specified_length:
            merged_data = merged_data[:specified_length]

        
        ### Create a minute version of data
        merged_data_minute = merged_data.copy()
        #merged_data_minute = merged_data_minute.drop('Stopped', axis=1)
        merged_data_minute = merged_data_minute.set_index('Time_of_Day')
        merged_data_minute = merged_data_minute.resample('1Min').agg({
            'Energy_Consumption': 'sum', 
            'Latitude': 'last', 
            'Longitude': 'last',
            })
        merged_data_minute.reset_index(inplace=True)

        ### Calaculate battery capacity and SOC
        # Set battery capacity - second then minute data
        merged_data.loc[0, 'Battery_Capacity'] = battery_capacity
        merged_data_minute.loc[0, 'Battery_Capacity'] = battery_capacity

        # Calculate SOC
        negative_energy = -merged_data['Energy_Consumption']/1000
        negative_energy_min = -merged_data_minute['Energy_Consumption']/1000

        merged_data['Battery_Capacity'] = negative_energy.cumsum() + merged_data['Battery_Capacity'].iloc[0]
        merged_data_minute['Battery_Capacity'] = negative_energy_min.cumsum() + merged_data_minute['Battery_Capacity'].iloc[0]

        merged_data['SOC'] = (merged_data['Battery_Capacity']/battery_capacity)*100
        merged_data_minute['SOC'] = (merged_data_minute['Battery_Capacity']/battery_capacity)*100

        ### check if vehicle actually drove that day, if not, don't save the graphs and delete folder
        if(merged_data['SOC'].iloc[-1] < 90):
            ### Plot the data and save as PNG in later code - per day only
            plt.figure()
            plt.plot(merged_data_minute['Time_of_Day'], merged_data_minute['SOC'])
            plt.xlabel('Time of Day')
            plt.ylabel('SOC')
            plt.title(f'Vehicle {i} - Day {k}')
            plt.xticks(rotation=45, ha='right')

            ### Save the data to Outputs location folder
            # Create the file path
            folder_save_path = destination_folder + folder_prefix + str(i) + '/' + new_folder_prefix + day_num_array[k - 1] + '/'
            # Save secondly and minuely data
            full_save_path_1 = folder_save_path + csv_save_name_1
            full_save_path_2 = folder_save_path + csv_save_name_2
            merged_data.to_csv(full_save_path_1, index=False)
            merged_data_minute.to_csv(full_save_path_2, index=False)
            # Save PNG
            png_full_path = folder_save_path + png_file_name
            plt.savefig(png_full_path, dpi=300, bbox_inches='tight')
            print(f'Vehicle {i} - Day {k} = Done')
            plt.close()

            ### Save data to be plotted
            data_list.append(merged_data_minute)
        else:
            print(f'Vehicle {i} - Day {k} = Did not drive')
            folder_save_path = destination_folder + folder_prefix + str(i) + '/' + new_folder_prefix + day_num_array[k - 1] + '/'
            shutil.rmtree(folder_save_path)
    
    ### Plot each days SOC per vehicle
    plt.figure()
    for j, data in enumerate(data_list):
        plt.plot(data['Time_of_Day'], data['SOC'], label = f'Day {j + 1}')
    plt.xlabel('Time of Day')
    plt.ylabel('SOC')
    plt.title(f'SOC Vehicle {i}')
    plt.legend()
    plt.xticks(rotation=45, ha='right')

    png_save_path = destination_folder + folder_prefix + str(i) + '/'
    png_total_path = png_save_path + png_total_name
    plt.savefig(png_total_path, dpi=300, bbox_inches='tight')
    plt.close()


#################################################################################################
################ Reframe each vehicle to start from 04:00:00 an end at 03:59:59 #################
#################################################################################################


print('Reframing data points')

        
destination_folder = "D:/Masters/Simulations/Simulation_1/Usable_Data/"
days = [str(num).zfill(2) for num in range(1, 32)]  # Days in the month


total_shift = 4 * 60 * 60

### Get the number of folders within the directory
### Change based on simulation
num_folders = count_folders_with_prefix(destination_folder, folder_prefix)

### Iterate through each vehicle in directory
for i in range(1, num_folders + 1):

    ### Get the number of days within that folder
    ### No need to change for different smulation
    new_folder = destination_folder + folder_prefix + str(i) + '/'
    new_folder_prefix = folder_prefix + str(i) + '_' # Vehicle_i_
    num_days = count_folders_with_prefix(new_folder, new_folder_prefix)

    day_num_array = get_last_two_values_as_strings(new_folder)

    ### Iterate through each day in vehicle
    for k in range(0, len(days)):

        ### Read the file from Inputs
        file_folder_1 = new_folder + new_folder_prefix + days[k] + '/' # Vehicle_i_k

        # If that day exists
        if os.path.exists(file_folder_1):
            print(f'{file_folder_1} exists: Reframing')

            # Create path to read file
            full_path = file_folder_1 + csv_save_name_1
            # File path exists, read the file
            day_1 = pd.read_csv(full_path)

            file_folder_2 = new_folder + new_folder_prefix + days[k + 1] + '/' # Vehicle_i_k+1

            # If day following starting day exists, use that dataset
            if os.path.exists(file_folder_2):
                # Create path to read file
                full_path = file_folder_2 + csv_save_name_1
                # File path exists, read the file
                day_2 = pd.read_csv(full_path)

                day_1['Time_of_Day'] = pd.to_datetime(day_1['Time_of_Day'])
                day_2['Time_of_Day'] = pd.to_datetime(day_2['Time_of_Day'])

                day_1_filtered = day_1[day_1['Time_of_Day'].dt.time >= pd.to_datetime('04:00:00').time()]
                day_2_filtered = day_2[day_2['Time_of_Day'].dt.time <= pd.to_datetime('03:59:59').time()]

            # just use ending points
            else:
                day_1['Time_of_Day'] = pd.to_datetime(day_1['Time_of_Day'])

                day_1_filtered = day_1[day_1['Time_of_Day'].dt.time >= pd.to_datetime('04:00:00').time()]

                last_values = day_1_filtered.iloc[-1]
                day_2_filtered = pd.DataFrame([last_values] * total_shift)

                # Add the time column to the filled DataFrame starting at the specified time
                day_2_filtered['Time_of_Day'] = pd.date_range(start = '00:00:00', periods = total_shift, freq = 'S')
            
            ### Create the new vehiclee day
            vehicle_day = pd.concat([day_1_filtered, day_2_filtered])

            vehicle_day['Time_of_Day'] = pd.to_datetime(vehicle_day['Time_of_Day'])
            home_charging_location = vehicle_day[(vehicle_day['Time_of_Day'].dt.time >= pd.to_datetime('20:00:00').time()) |
                 (vehicle_day['Time_of_Day'].dt.time <= pd.to_datetime('03:59:59').time())]

            most_common_combination = home_charging_location.groupby(['Latitude', 'Longitude']).size().idxmax()

            vehicle_day = vehicle_day.drop(['Battery_Capacity', 'SOC'], axis = 1)
            vehicle_day = vehicle_day.reset_index(drop=True)

            x = vehicle_day['Stopped']
            ### Determine if vehicle is avaliable to charge
            # Determine 20 minute stop classification
            vehicle_day['20_Min_Stop'] = count_20_minutes(vehicle_day['Stopped'])

            # Check to see if vehicle has stopped in specified location
            vehicle_day['Hub_Location'] = vehicle_day[['Latitude', 'Longitude']].apply(lambda x: is_point_in_stop(x), axis=1)

            # Create Available_Charging column - is the vehicle able to charge
            vehicle_day['Available_Charging'] = vehicle_day['Hub_Location'] & vehicle_day['20_Min_Stop']

            vehicle_day['HC_Location'] = vehicle_day[['Latitude', 'Longitude']].apply(lambda x: is_point_at_home(x, most_common_combination), axis=1)

            vehicle_day['Home_Charging'] = vehicle_day['HC_Location'] & vehicle_day['20_Min_Stop']

            save_path = file_folder_1 + csv_save_name_3

            vehicle_day.to_csv(save_path, index=False)
        else:
            print(f'{file_folder_1} does not exist')


