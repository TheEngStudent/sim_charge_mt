################################################## READ ME ##################################################
"""
    This programme simulates the vehicle driving and charging with limited number of chargers and different battery
    sizes and charging rates. Furthermore, charging takes place at the specified location, in this case the
    Stellenbosch Taxi Rank. The option is also there to include home charging into the simulation. The results are
    then categorised as Percentage_Day_Completion (how many vehicles completed there trip in that day) and 
    Vehicle_Completion (of the trips the vehicle needs to take, how many has it succesfully completed)
    
    TODO Add the home charging scenario as well

    When the program rearranges the data, it reads it from the individual folders of the vehicle per day,
    and combines the vehicles active on a specific day to the structure as follows:

    Day_k

    Vehicle_1   |   Vehicle_2   |   Vehicle_5   |   --------    |   Vehicle_n
    ----------------------------------------------------------------------------
    [ec/soc/    |   [ec/soc/    |   [ec/soc/    |   --------    |   [ec/soc/  
        cf/ac]  |       cf/ac]  |       cf/ac]  |   --------    |       cf/ac]

    Where:
        ec == energy consumed
        soc == state of charge
        cf == charging flag
        ac == available charging

    Each of these for variables have their own save files which the simulation then reads from. The simulation
    takes one row at a time and does the logic based on the values within the rows. It is important to note that
    the data is given as secondly data. Furthermore, other important values required for the vehicle are also
    created as dataframes for that day.

    The charging is classified as CP/CV (Constant Power followed by Constant Voltage). As a result, the battery 
    voltages and currents for each vehicle are thus also modelled and saved as graphs and files. TODO simulate 
    the other voltages

    TO_NOTE this programme has been written with a month of data only, and so should be changed if more data is
    given.
     
"""

########## This only works because data is for a month, would need to be updated for multiple months ###########
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import os
import pandas as pd
import sys
import time
import math
import numpy as np

### Change directory for each new simulation
source_folder = 'D:/Masters/Simulations/Simulation_1/Usable_Data/'
destination_folder = 'D:/Masters/Simulations/Simulation_1/Outputs/'
file_common = 'Vehicle_'
file_name = 'vehicle_day_sec.csv'
save_common = 'Day_'
file_suffix = '.csv'

### Subfolders to save files in
original_folder = 'Original_Data/'
energy_consumption_folder = 'Original_EC/'
charging_comsumption_folder = 'Charging_EC/'
soc_folder = 'Daily_SOC/'
charging_flag_folder = 'Charging_Flag/'
available_charging_folder = 'Available_Charging/'
grid_power_folder = 'Grid_Power/'
home_charging_folder = 'Home_Charging/'

file_name_ac = 'Available_Charging'
file_name_ec = 'Energy_Consumption'
file_name_cf = 'Charging_Variable'
file_name_soc = 'SOC'
file_name_hc = 'Home_Charging'

save_name_ac = 'available_charging.csv'
save_name_ec = 'energy_consumption.csv'
save_name_soc = 'soc.csv'
save_name_cf = 'charging_variable.csv'
save_name_gp = 'grid_power.csv'
save_name_charger = 'charger_association.csv'
save_name_V_b = 'battery_voltage.csv'
save_name_I_b = 'battery_current.csv'

### Constants for simulation
# Battery Model Parameters
battery_parameters = {
    'V_nom': 3.7, # [V]
    'V_max': 4.15, # [V]
    'R_cell': 0.148, # [Ohm]
    'Q_nom': 2.2, # [Ah]
    'E_nom': 8.14, # [Wh]
    'M_p': 78, # Number of batteries in parallel
    'M_s': 110, # Number of batteries in series
    'a_v': 0.06792, # [V/Wh]
    'b_v': 3.592 # [V] 
}

battery_capacity = battery_parameters['E_nom']*battery_parameters['M_p']*battery_parameters['M_s'] # Wh
R_eq = (battery_parameters['M_s'] * battery_parameters['R_cell']) / battery_parameters['M_p'] # Ohm

# Grid Model Parameters
grid_parameters = {
    'num_chargers': 6,
    'P_max': 22, # [kW]
    'efficiency': 0.88,
    'soc_upper_limit': 80,
    'soc_lower_limit': 0,
    'home_charge': True, # Set for each sim you wish to desire
    'home_power': 7.2 # [kW]
}


days = [str(num).zfill(2) for num in range(1, 32)]  # Days in the month
num_vehicles = 17 # Total number of vehicles used in the sim

# length of lists
length_days = len(days)

integer_list = list(range(0, 86400))
total_items = len(integer_list)

colour_list = [ '#d9ff00',
                '#00ffd5',
                '#00ff15',
                '#f2ff00',
                '#0019ff',
                '#ffea00',
                '#ff2f00',
                '#00ff88',
                '#ff9d00',
                '#ef65db',
                '#653a2a',
                '#ffa200',
                '#bfff00',
                '#a481af',
                '#e7596a',
                '#d65fb2',
                '#9f5d54',
                '#a67311' ]

color_palette = {'Vehicle_' + str(i): colour_list[i - 1] for i in range(1, num_vehicles + 1)}

### Dictionary for vehicle valid driving day
vehicle_valid_drive = {'Vehicle_' + str(i): True for i in range(1, num_vehicles + 1)}

day_exists = {save_common + day: True for day in days}

vehicle_total_trips = {'Vehicle_' + str(i): 0 for i in range(1, num_vehicles + 1)}
vehicle_end_soc = {'Vehicle_' + str(i): 0 for i in range(1, num_vehicles + 1)}
vehicle_completed_trips = {'Vehicle_' + str(i): 0 for i in range(1, num_vehicles + 1)}

day_total_trips = {save_common + day: 0 for day in days}
day_end_soc = {save_common + day: 0 for day in days}
day_completed_trips = {save_common + day: 0 for day in days}


### Colour dictionary for vehicle tracking through each graph

another_colour = colour_list[17]


### Functions
# Count the number of folders
def count_folders(directory):
    folder_count = 0
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_dir():
                folder_count += 1
    return folder_count

# Create folder if it does not exist
def create_folder(directory):
    # Create folder if not exist
    if not os.path.exists(directory):
        os.makedirs(directory)

# Display progress bar for simulation
def progress_bar(current, total, start_time):
    bar_length = 40
    filled_length = int(bar_length * current / total)
    percentage = current / total * 100
    elapsed_time = int((time.time() - start_time) / 60)  # Calculate elapsed time in minutes
    bar = '=' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write(f'\r[{bar}] {percentage:.2f}% Elapsed Time: {elapsed_time} minutes')
    sys.stdout.flush()

# Main algorithm for calculating
def simulate_charge(og_ec, og_ac, og_soc, og_cf, og_hc, grid_power, charger, priority_vehicles, start_time, battery_capacity, 
                    V_t, V_b, I_t, I_b, V_oc, V_oc_eq, CP_flag, battery_parameters, grid_parameters, vehicle_valid_drive):
    
    # Iterate over each row of og_ec
    for index in range(1, len(og_ec)):
        
        row = og_ac.loc[index]
       
        # If SOC is below 100, the vehicle needs to be charged
        # only calculates new SOC later on, so has to be previous soc
        charging_mask_ac = og_ac.loc[index] & (og_soc.loc[index - 1] < 100)

        # Is home charging available
        if grid_parameters['home_charge'] == True:
            charging_mask_hc = og_hc.loc[index] & (og_soc.loc[index - 1] < 100)
            og_cf.loc[index, charging_mask_hc] = -1
        
        ### Charger distribution algorithm
        true_columns = og_ac.columns[charging_mask_ac]  # Get column labels with true values
        row_headings = []

        matrix = [[value1 == value2 for value2 in charger.loc[index - 1]] for value1 in true_columns]
        # Print matrix
        for vehicle_name, row in zip(true_columns, matrix):
            row_headings.append(vehicle_name)
            # print(vehicle_name, row)
        
        missing_vehicles = set(priority_vehicles).difference(row_headings)
        # Remove vehicle if it has lost its position to charge
        for vehicle in missing_vehicles:
            if vehicle in priority_vehicles:
                priority_vehicles.remove(vehicle)
        
        # Iterate over rows and row headings
        for vehicle_name, row in zip(true_columns, matrix):
            if any(row): # If vehicle has been assigned, keep vehicle in charger
                column_index = row.index(True)
                column_heading = charger.columns[column_index]

                charger.loc[index, column_heading] = vehicle_name
            else: # If vehicle has not been assigned charger, add to list of vehicles needing to be charged
                if vehicle_name not in priority_vehicles:
                    priority_vehicles.append(vehicle_name)

        # Reorganise pirioty_vehicles to have the lowest SOC at the top
        priority_vehicles = sorted(priority_vehicles, key = lambda x: og_soc.loc[index - 1, x])

        for k in range(0, len(priority_vehicles)):
            if any(charger.loc[index] == ''): # If available charger, add vehicle to it
                next_col = charger.loc[index].eq('').idxmax()
                charger.loc[index, next_col] = priority_vehicles[0]
                priority_vehicles.pop(0)
            else:
                # Check if there is a vehicle with an SOC higher than 80% that is on charge
                # If there is, remove vehicle from charger and add it to prirority_vehicles
                # Top value from priority_vehicles gets added to charger
                highest_soc_remove = sorted(charger.loc[index], key = lambda x: og_soc.loc[index - 1, x], reverse = True)

                for w in range(0, len(highest_soc_remove)):

                    if og_soc.loc[index - 1, priority_vehicles[0]] < grid_parameters['soc_upper_limit']: # It should only swap vehicles if the soc in priority vehicles is less than 80

                        column_name = highest_soc_remove[w] # Assign the highest column name with the highest soc to remove first
                        if og_soc.loc[index - 1, column_name] > grid_parameters['soc_upper_limit']:
                            column_to_replace = charger.loc[index] == column_name # find the location where that value is
                            charger.loc[index, column_to_replace.idxmax()] = priority_vehicles[0] # make it equal to the highest priority vehicle
                            priority_vehicles.pop(0)
                            if vehicle_name not in priority_vehicles:
                                priority_vehicles.append(column_name) # add the vehicle that was on charge to priority list

                    else:    
                        break # If no available slot, break for loop to save processing power

        ### Update og_cf based on charger distibution
        for charger_col in charger.columns:
            assigned_vehicle = charger.loc[index, charger_col]
            if assigned_vehicle:
                og_cf.loc[index, assigned_vehicle] = 1
        
        ### Calculate battery characteristics
        for col_name in og_soc.columns:

            # Check if vehicle has gone below 20% for the day
            if og_soc.loc[index - 1, col_name] <= grid_parameters['soc_lower_limit']:
                vehicle_valid_drive[col_name] = False

            # Calculate open circuit voltage
            V_oc.loc[index, col_name] = battery_parameters['a_v']*( (og_soc.loc[index - 1, col_name]/100)*battery_parameters['E_nom'] ) + battery_parameters['b_v']
            # Calculate battery pack open circuit voltage
            V_oc_eq.loc[index, col_name] = battery_parameters['M_s']*V_oc.loc[index, col_name]

            ### Check to see if the vehicle is on charge
            # Update the necessary power and battery characteristics
            if og_cf.loc[index, col_name] == 1:

                # Vehicle is charging at constant power (CP)
                if CP_flag[col_name] == 1:
                    grid_power.loc[index, col_name] = grid_parameters['P_max']*1000

                    V_b.loc[index, col_name] = V_oc_eq.loc[index, col_name]/2 + math.sqrt( grid_parameters['efficiency']*grid_power.loc[index, col_name]*R_eq 
                                                    + 0.25*(V_oc_eq.loc[index, col_name] ** 2) )
                    
                    V_t.loc[index, col_name] = V_b.loc[index, col_name]/battery_parameters['M_s']

                    if V_t.loc[index, col_name] < battery_parameters['V_max']:
                        # Calculate battery charging current
                        I_b.loc[index, col_name] = ( V_b.loc[index, col_name] - V_oc_eq.loc[index, col_name] ) / R_eq
                    else:
                        # Vehcile is no longer in constant power charging, but now constant voltage charging
                        CP_flag[col_name] = 0    

                # Vehicle is charging at constant voltage (CV)
                if CP_flag[col_name] == 0:

                    V_t.loc[index, col_name] = battery_parameters['V_max']
                    V_b.loc[index, col_name] = V_t.loc[index, col_name]*battery_parameters['M_s']

                    I_b.loc[index, col_name] = ( battery_parameters['M_s']*battery_parameters['V_max'] - V_oc_eq.loc[index, col_name] ) / R_eq

                    grid_power.loc[index, col_name] = ( battery_parameters['M_s']*battery_parameters['V_max']*I_b.loc[index, col_name] ) / grid_parameters['efficiency']

                # Calculate cell current for next charging opportunity
                I_t.loc[index, col_name] = I_b.loc[index, col_name] / battery_parameters['M_p']

                # Update SOC for charging
                og_soc.loc[index, col_name] = og_soc.loc[index - 1, col_name] + (((grid_power.loc[index, col_name])/3600)/battery_capacity)*100

                if og_soc.loc[index, col_name] > 100:
                    og_soc.loc[index, col_name] = 100
            
            # If vehicle is not on charge, simply update the battery characteristics
            elif og_cf.loc[index, col_name] == 0:

                grid_power.loc[index, col_name] = 0

                """
                # Calculate driving battery characterisitics
                roots = np.roots([ 1,
                                -V_oc_eq.loc[index, col_name],
                                (og_ec.loc[index, col_name]*3600)*R_eq])

                V_b.loc[index, col_name] = np.max(roots[roots > 0])
                V_t.loc[index, col_name] = V_b.loc[index, col_name] / battery_parameters['M_s']

                if V_t.loc[index, col_name] > battery_parameters['V_max']:
                    V_t.loc[index, col_name] = battery_parameters['V_max']
                    V_b.loc[index, col_name] = V_t.loc[index, col_name]*battery_parameters['M_s']

                I_b.loc[index, col_name] = ( V_b.loc[index, col_name] - V_oc_eq.loc[index, col_name] ) / R_eq
                I_t.loc[index, col_name] = I_b.loc[index, col_name] / battery_parameters['M_p']
                """

                # Update SOC for driving
                og_soc.loc[index, col_name] = og_soc.loc[index - 1, col_name] - (og_ec.loc[index, col_name]/battery_capacity)*100
                

                if CP_flag[col_name] == 0:
                    CP_flag[col_name] = 1
            
            # If vehicle is charging at home cf == -1
            else: 
                # Vehicle is charging at constant power (CP)
                if CP_flag[col_name] == 1:
                    grid_power.loc[index, col_name] = grid_parameters['home_power']*1000

                    V_b.loc[index, col_name] = V_oc_eq.loc[index, col_name]/2 + math.sqrt( grid_parameters['efficiency']*grid_power.loc[index, col_name]*R_eq 
                                                    + 0.25*(V_oc_eq.loc[index, col_name] ** 2) )
                    
                    V_t.loc[index, col_name] = V_b.loc[index, col_name]/battery_parameters['M_s']

                    if V_t.loc[index, col_name] < battery_parameters['V_max']:
                        # Calculate battery charging current
                        I_b.loc[index, col_name] = ( V_b.loc[index, col_name] - V_oc_eq.loc[index, col_name] ) / R_eq
                    else:
                        # Vehcile is no longer in constant power charging, but now constant voltage charging
                        CP_flag[col_name] = 0    

                # Vehicle is charging at constant voltage (CV)
                if CP_flag[col_name] == 0:

                    V_t.loc[index, col_name] = battery_parameters['V_max']
                    V_b.loc[index, col_name] = V_t.loc[index, col_name]*battery_parameters['M_s']

                    I_b.loc[index, col_name] = ( battery_parameters['M_s']*battery_parameters['V_max'] - V_oc_eq.loc[index, col_name] ) / R_eq

                    grid_power.loc[index, col_name] = ( battery_parameters['M_s']*battery_parameters['V_max']*I_b.loc[index, col_name] ) / grid_parameters['efficiency']

                # Calculate cell current for next charging opportunity
                I_t.loc[index, col_name] = I_b.loc[index, col_name] / battery_parameters['M_p']

                # Update SOC for charging
                og_soc.loc[index, col_name] = og_soc.loc[index - 1, col_name] + (((grid_power.loc[index, col_name])/3600)/battery_capacity)*100

                if og_soc.loc[index, col_name] > 100:
                    og_soc.loc[index, col_name] = 100
            

        if index % 100 == 0:  # Update the progress and elapsed time every 100 iterations
            progress_bar(index, total_items, start_time)


# Saving graphs functions
def save_individual_graphs(og_soc, V_b, save_folder, day, timedelta_index):

    for column in og_soc.columns:
 
        # Create a new figure and axis for each column
        fig, ax1 = plt.subplots(figsize = (12, 9))

        # Plot the first graph (V_b) on the first axis        
        ax1.set_xlabel('Time of Day')
        ax1.set_ylabel('Battery Pack Voltage [V]', color = another_colour)
        ax1.set_ylim(0, 500)  # Set the desired y-axis limits for V_b
        ax1.tick_params(axis = 'y', colors = another_colour)
        ax1.set_xticks(ax1.get_xticks())
        ax1.set_xticklabels(ax1.get_xticks(), rotation = 45)

        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())

        ax1.plot(timedelta_index, V_b[column], color = another_colour)

        ax2 = ax1.twinx()

        # Plot the second graph (og_soc) on the second axis
        
        ax2.set_ylabel('SOC [%]', color = color_palette[column])
        ax2.set_ylim(-20, 105)  # Set the desired y-axis limits for og_soc
        ax2.tick_params(axis = 'y', colors = color_palette[column])
        ax2.set_xticks(ax2.get_xticks())
        ax2.set_xticklabels(ax2.get_xticks(), rotation = 45)

        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())

        ax2.plot(timedelta_index, og_soc[column], color = color_palette[column])

        ax1.set_title(column)

        plt.rcParams['figure.dpi'] = 600
                        
        # Save the plot to a specific location as a png
        save_path = save_folder + column + '_' + day + '.png'
        plt.savefig(save_path)
        # Save the plot to a specific location as a svg
        save_path = save_folder + column + '_' + day + '.svg'
        plt.savefig(save_path, format = 'svg')

                        
        # Close the figure to free up memory
        plt.close(fig)  


def save_complete_graphs(og_soc, grid_power, day, save_folder, timedelta_index):

    ### Plot and save all vehicles graph
    plt.figure()
    for column in og_soc.columns:
        plt.plot(timedelta_index, og_soc[column], color = color_palette[column], label = column)
    plt.xlabel('Time of Day')
    plt.ylabel('SOC [%]')
    plt.title('Day_' + day + ' SOC')
    plt.ylim(-20, 140)
    plt.tight_layout()
    plt.xticks(rotation=45)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.legend(loc = 'upper center', ncol = 4)
    plt.subplots_adjust(bottom = 0.2)

    save_path = save_folder + 'Day_' + day + '_SOC.png'
    plt.savefig(save_path)
    # Save the plot to a specific location as a svg
    save_path = save_folder + 'Day_' + day + '_SOC.svg'
    plt.savefig(save_path, format = 'svg')
    plt.close()

    ### Plot grid power usage
    grid_sums = grid_power.sum(axis = 1)
    grid_sums = grid_sums/1000

    plt.figure()
    plt.plot(timedelta_index, grid_sums)
    plt.xlabel('Time of Day')
    plt.ylabel('Power [kW]')
    plt.title('Grid Power for Day_' + day)
    plt.ylim(0, 170)
    plt.xticks(rotation=45)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.subplots_adjust(bottom = 0.2)

    save_path = save_folder + 'Grid_Power_Day_' + day + '.png'
    plt.savefig(save_path)
    # Save the plot to a specific location as a svg
    save_path = save_folder + 'Grid_Power_Day_' + day + '.svg'
    plt.savefig(save_path, format = 'svg')
    plt.close()



########################################################################################################################
################### Create nescessary original files that the simulation runs off of ###################################
########################################################################################################################

num_folders = count_folders(source_folder)

for k in range(0, length_days):  # Cycle through for each day

    data_list_ec = []  # create empty list to combine data at the end
    data_list_ac = []
    data_list_hc = []

    name_list = []  # create empty list for valid vehicles

    for i in range(1, num_folders + 1):  # Check in each vehicle if there is a valid driving day

        # Create file path to read
        sub_path = source_folder + file_common + str(i) + '/'  # //Vehicle_i/

        sub_sub_path = sub_path + file_common + str(i) + '_' + days[k] + '/'  # //Vehicle_i_k/

        if os.path.exists(sub_sub_path):
            # Create path to read file
            full_path = sub_sub_path + file_name

            # File path exists, read the file
            df = pd.read_csv(full_path)
            
            data_list_ec.append(df['Energy_Consumption'])
            data_list_ac.append(df['Available_Charging'])
            data_list_hc.append(df['Home_Charging'])
            name_list.append(str(i))
        else:
            # File path does not exist, skip
            print(f"File path '{sub_sub_path}' does not exist. Skipping...")

    if data_list_ec:
        ### For Energy_Consumption data
        # Perform functions when the list is not empty
        combined_data_ec = pd.concat(data_list_ec, axis=1)
        vehicle_columns = [file_common + name for name in name_list]
        combined_data_ec.columns = vehicle_columns
        # Save the combined DataFrame to a CSV file
        save_name = save_common + days[k] + '_' + file_name_ec + file_suffix
        save_folder = destination_folder + original_folder + energy_consumption_folder
        save_path = save_folder + save_name
        create_folder(save_folder)
        combined_data_ec.to_csv(save_path, index=False)

        ### For Available_Charging data
        # Perform functions when the list is not empty
        combined_data_ac = pd.concat(data_list_ac, axis=1)
        vehicle_columns = [file_common + name for name in name_list]
        combined_data_ac.columns = vehicle_columns
        # Save the combined DataFrame to a CSV file
        save_name = save_common + days[k] + '_' + file_name_ac + file_suffix
        save_folder = destination_folder + original_folder + available_charging_folder
        save_path = save_folder + save_name
        create_folder(save_folder)
        combined_data_ac.to_csv(save_path, index=False)

        ### For Home_Charging data
        # Perform functions when the list is not empty
        combined_data_hc = pd.concat(data_list_hc, axis=1)
        vehicle_columns = [file_common + name for name in name_list]
        combined_data_hc.columns = vehicle_columns
        # Save the combined DataFrame to a CSV file
        save_name = save_common + days[k] + '_' + file_name_hc + file_suffix
        save_folder = destination_folder + original_folder + home_charging_folder
        save_path = save_folder + save_name
        create_folder(save_folder)
        combined_data_hc.to_csv(save_path, index=False)

        ### For starting off SOC data
        combined_data_soc = combined_data_ec.copy()
        combined_data_soc[:] = 100
        # Save the combined DataFrame to a CSV file
        save_name = save_common + days[k] + '_' + file_name_soc + file_suffix
        save_folder = destination_folder + original_folder + soc_folder
        save_path = save_folder + save_name
        create_folder(save_folder)
        combined_data_soc.to_csv(save_path, index=False)

        ### For starting off Charging_Variable data
        combined_data_cf = combined_data_ec.copy()
        combined_data_cf[:] = 0
        # Save the combined DataFrame to a CSV file
        save_name = save_common + days[k] + '_' + file_name_cf + file_suffix
        save_folder = destination_folder + original_folder + charging_flag_folder
        save_path = save_folder + save_name
        create_folder(save_folder)
        combined_data_cf.to_csv(save_path, index=False)
        
    else:
        # Skip over when the list is empty
        print("Day does not exist. Skipping...")




#######################################################################################################################
############################################ Main simulating code #####################################################
#######################################################################################################################

### Initialise scenario
scenario_folder = 'SCE_' + str(grid_parameters['P_max']) + 'kW_N' + str(grid_parameters['num_chargers']) + '_B' + str(round(battery_capacity/1000)) + '_HC_' + str(grid_parameters['home_charge']) + '/'
print(f'Scenario {scenario_folder}')
save_folder = destination_folder + scenario_folder
create_folder(save_folder)


### Iterate over each day in the month
for m in range(0, length_days): 
    
    # Create file paths to read nescessary data
    read_name_ec = save_common + days[m] + '_' + file_name_ec + file_suffix # Day_i_Data.csv
    read_name_ac = save_common + days[m] + '_' + file_name_ac + file_suffix
    read_name_soc = save_common + days[m] + '_' + file_name_soc + file_suffix
    read_name_cf = save_common + days[m] + '_' + file_name_cf + file_suffix
    read_name_hc = save_common + days[m] + '_' + file_name_hc + file_suffix

    read_path_ec = destination_folder + original_folder + energy_consumption_folder + read_name_ec
    read_path_ac = destination_folder + original_folder + available_charging_folder + read_name_ac
    read_path_soc = destination_folder + original_folder + soc_folder + read_name_soc
    read_path_cf = destination_folder + original_folder + charging_flag_folder + read_name_cf
    read_path_hc = destination_folder + original_folder + home_charging_folder + read_name_hc

    # If file exists for one day then that day exists
    if os.path.exists(read_path_ec):

        # Create save folder only if that day exists
        day_folder = save_common + days[m] + '/' # Day_i/
        save_folder = destination_folder + scenario_folder + day_folder
        create_folder(save_folder)
        
        # Read the nescessary files
        og_ec = pd.read_csv(read_path_ec)
        og_ac = pd.read_csv(read_path_ac)
        og_soc = pd.read_csv(read_path_soc)
        og_cf = pd.read_csv(read_path_cf)
        og_hc = pd.read_csv(read_path_hc)

        # Get the number of total trips for the day
        day_total_trips[save_common + days[m]] = len(og_soc.columns)

        # Add to the total number of trips Vehicle x is supossed to complete
        for vehicle_trip in og_soc.columns:

            vehicle_total_trips[vehicle_trip] = vehicle_total_trips[vehicle_trip] + 1

            ### Check to see if vehicle is valid for driving day and drop the columns if not
            # if vehicle_valid_drive[vehicle_trip] == False:
            #     og_ec = og_ec.drop(vehicle_trip, axis=1)
            #     og_ac = og_ac.drop(vehicle_trip, axis=1)
            #     og_soc = og_soc.drop(vehicle_trip, axis=1)
            #     og_cf = og_cf.drop(vehicle_trip, axis=1)
            #     og_hc = og_hc.drop(vehicle_trip, axis=1) 

    
        # If there is vehicles to simulate, then simulate
        if not og_ac.empty:

            # Create nescessary other data frames
            grid_power = pd.DataFrame(0, index = range(total_items), columns = og_ec.columns)

            charger = pd.DataFrame('', index = range(len(og_ec)), columns = [f'Charger_{w}' for w in range(1, grid_parameters['num_chargers'] + 1)])

            # Battery characteriistic dataframes
            V_t = pd.DataFrame(0, index = range(total_items), columns = og_ec.columns)
            V_b = pd.DataFrame(0, index = range(total_items), columns = og_ec.columns)
            I_t = pd.DataFrame(0, index = range(total_items), columns = og_ec.columns)
            I_b = pd.DataFrame(0, index = range(total_items), columns = og_ec.columns)
            V_oc = pd.DataFrame(0, index = range(total_items), columns = og_ec.columns)
            V_oc_eq = pd.DataFrame(0, index = range(total_items), columns = og_ec.columns)

            ### Re-initialise each vehicle to constant power charging
            CP_flag = {'Vehicle_' + str(i): 1 for i in range(1, num_vehicles + 1)}

            priority_vehicles = []
            
            print(f'Day {days[m]} Simulating')

            start_time = time.time()

            ### Perform different functions depending on if home_charging or not
            ### Simulate actual data
            simulate_charge(og_ec, og_ac, og_soc, og_cf, og_hc, grid_power, charger, priority_vehicles, start_time, battery_capacity,
                            V_t, V_b, I_t, I_b, V_oc, V_oc_eq, CP_flag, battery_parameters, grid_parameters, vehicle_valid_drive) # Does the actual simulating of vehicles

            ### See how many vehicles have completed their trips
            for vehicle_name in og_soc.columns:
                # Did the vehicle not cross the 0% boundary
                if vehicle_valid_drive[vehicle_name] == True:
                    vehicle_completed_trips[vehicle_name] = vehicle_completed_trips[vehicle_name] + 1
                    day_completed_trips[save_common + days[m]] = day_completed_trips[save_common + days[m]] + 1

                if og_soc.iloc[-1][vehicle_name] >= 90:
                    vehicle_end_soc[vehicle_name] = vehicle_end_soc[vehicle_name] + 1
                    day_end_soc[save_common + days[m]] = day_end_soc[save_common + days[m]] + 1
                
                    
            ### All vehicles become valid again to drive
            vehicle_valid_drive = {'Vehicle_' + str(i): True for i in range(1, num_vehicles + 1)}

            ### Prepare for plotting
            timedelta_index = pd.to_timedelta(integer_list, unit='s')
            base_date = pd.to_datetime('04:00:00')
            timedelta_index = base_date + timedelta_index

            ### Plot and save individual vehicle graphs
            print('\nSaving graphs')
            save_individual_graphs(og_soc, V_b, save_folder, days[m], timedelta_index)
            save_complete_graphs(og_soc, grid_power, days[m], save_folder, timedelta_index)
                
            ### Save dataframes
            print('Saving files')
            save_path = save_folder + save_name_ec
            og_ec.to_csv(save_path, index=False)

            save_path = save_folder + save_name_cf
            og_cf.to_csv(save_path, index=False)

            save_path = save_folder + save_name_ac
            og_ac.to_csv(save_path, index=False)

            save_path = save_folder + save_name_soc
            og_soc.to_csv(save_path, index=False)

            save_path = save_folder + save_name_gp
            grid_power.to_csv(save_path, index=False)

            save_path = save_folder + save_name_charger
            charger.to_csv(save_path, index=False)

            save_path = save_folder + save_name_V_b
            V_b.to_csv(save_path, index=False)

            save_path = save_folder + save_name_I_b
            I_b.to_csv(save_path, index=False)

        else:
            print(f'Day {days[m]} Simulating')
            print('No vehicles to simulate')


    # If day does not exist
    else:

        day_total_trips[save_common + days[m]] = 1

        day_exists[save_common + days[m]] = False

        print(f'Day {days[m]} does not exist')
    

#################################################################################################################
########## Save and calculate the percentage completions for the vehicles as well as the days####################
#################################################################################################################



### Vehicle Succesful Trips for day - was it able to stay above 0%
# Calculate completion and uncompletion percentages
completion_percentages = [(vehicle_completed_trips[vehicle] / vehicle_total_trips[vehicle]) * 100 for vehicle in vehicle_total_trips]
uncompletion_percentages = [100 - percentage for percentage in completion_percentages]

# Create the figure and axis objects
fig, ax = plt.subplots()
x = np.arange(len(vehicle_total_trips)) * 1.7
bar_width = 1
bar1 = ax.bar(x, completion_percentages, bar_width, label = 'Completed Trips', color = '#FFA500')
bar2 = ax.bar(x, uncompletion_percentages, bar_width, bottom=completion_percentages, label = 'Uncompleted Trips', color = '#ADD8E6')

for rect, completion_percentage in zip(bar1 + bar2, completion_percentages):
    height = rect.get_height()
    if completion_percentage > 0:
            if completion_percentage < 10:
                ax.text(rect.get_x() + rect.get_width() / 2, height + 2, f'{completion_percentage:.1f}%', ha = 'center', va = 'bottom', fontsize = 8, rotation = 90)
            else:
                ax.text(rect.get_x() + rect.get_width() / 2, height / 2, f'{completion_percentage:.1f}%', ha = 'center', va = 'center', fontsize = 8, rotation = 90)

ax.set_xticks(x)
ax.set_xticklabels(range(1, num_vehicles + 1), fontsize = 6)

ax.set_ylabel('Percentage [%]')
ax.set_ylim(0, 115)

ax.set_title('Vehicle_Day Completion Rate')
ax.set_xlabel('Vehicle')
plt.legend(loc = 'upper center', ncol = 2)

plt.tight_layout()

save_path = destination_folder + scenario_folder + 'Vehicle_Day_Trip_Completion.png'
plt.savefig(save_path)
# Save the plot to a specific location as a svg
save_path = destination_folder + scenario_folder + 'Vehicle_Day_Trip_Completion.svg'
plt.savefig(save_path, format = 'svg')



### Succesful Day Trips - did all the vehicles of that day stay above 0%
# Calculate completion and uncompletion percentages
completion_percentages = [(day_completed_trips[vehicle] / day_total_trips[vehicle]) * 100 for vehicle in day_total_trips]
uncompletion_percentages = [100 - percentage for percentage in completion_percentages]

# Create the figure and axis objects
fig, ax = plt.subplots()
x = np.arange(len(day_total_trips)) * 3
bar_width = 2
bar1 = ax.bar(x, completion_percentages, bar_width, label = 'Completed Trips', color = '#FFA500')
bar2 = ax.bar(x, uncompletion_percentages, bar_width, bottom=completion_percentages, label = 'Uncompleted Trips', color = '#ADD8E6')

for rect, completion_percentage, vehicle_name in zip(bar1 + bar2, completion_percentages, day_total_trips.keys()):
    if day_exists[vehicle_name]:
        height = rect.get_height()
        if completion_percentage > 0:
            if completion_percentage < 10:
                ax.text(rect.get_x() + rect.get_width() / 2, height + 2, f'{completion_percentage:.1f}%', ha = 'center', va = 'bottom', fontsize = 8, rotation = 90)
            else:
                ax.text(rect.get_x() + rect.get_width() / 2, height / 2, f'{completion_percentage:.1f}%', ha = 'center', va = 'center', fontsize = 8, rotation = 90)


for i, exists in enumerate(day_exists.values()):
    if not exists:
        bar1[i].set_height(0)
        bar2[i].set_height(0)

ax.set_xticks(x)
ax.set_xticklabels(range(1, len(days) + 1), fontsize = 6)

ax.set_ylabel('Percentage [%]')
ax.set_xlabel('Day')
ax.set_ylim(0, 115)

ax.set_title('Daily Completion Rate')
plt.legend(loc = 'upper center', ncol = 2)

plt.tight_layout()

save_path = destination_folder + scenario_folder + 'Daily_Valid_Trip_Completion.png'
plt.savefig(save_path)
# Save the plot to a specific location as a svg
save_path = destination_folder + scenario_folder + 'Daily_Valid_Trip_Completion.svg'
plt.savefig(save_path, format = 'svg')
            


### Vehicle Valid Trips for next day - was it able to get back to 0%
# Calculate completion and uncompletion percentages
completion_percentages = [(vehicle_end_soc[vehicle] / vehicle_total_trips[vehicle]) * 100 for vehicle in vehicle_total_trips]
uncompletion_percentages = [100 - percentage for percentage in completion_percentages]

# Create the figure and axis objects
fig, ax = plt.subplots()
x = np.arange(len(vehicle_total_trips)) * 1.7
bar_width = 1
bar1 = ax.bar(x, completion_percentages, bar_width, label = 'Valid', color = '#FFA500')
bar2 = ax.bar(x, uncompletion_percentages, bar_width, bottom=completion_percentages, label = 'Invalid', color = '#ADD8E6')

for rect, completion_percentage in zip(bar1 + bar2, completion_percentages):
    height = rect.get_height()
    if completion_percentage > 0:
            if completion_percentage < 10:
                ax.text(rect.get_x() + rect.get_width() / 2, height + 2, f'{completion_percentage:.1f}%', ha = 'center', va = 'bottom', fontsize = 8, rotation = 90)
            else:
                ax.text(rect.get_x() + rect.get_width() / 2, height / 2, f'{completion_percentage:.1f}%', ha = 'center', va = 'center', fontsize = 8, rotation = 90)

ax.set_xticks(x)
ax.set_xticklabels(range(1, num_vehicles + 1), fontsize = 6)

ax.set_ylabel('Percentage [%]')
ax.set_ylim(0, 115)

ax.set_title('Vehicle_Day Valid Completion Rate')
ax.set_xlabel('Vehicle')
plt.legend(loc = 'upper center', ncol = 2)

plt.tight_layout()

save_path = destination_folder + scenario_folder + 'Vehicle_Day_Valid_Completion.png'
plt.savefig(save_path)
# Save the plot to a specific location as a svg
save_path = destination_folder + scenario_folder + 'Vehicle_Day_Valid_Completion.svg'
plt.savefig(save_path, format = 'svg')



### Valid for next Day Trips - did all the vehicles manage to get to 100% SOC at the end of the day
# Calculate completion and uncompletion percentages
completion_percentages = [(day_end_soc[vehicle] / day_total_trips[vehicle]) * 100 for vehicle in day_total_trips]
uncompletion_percentages = [100 - percentage for percentage in completion_percentages]

# Create the figure and axis objects
fig, ax = plt.subplots()
x = np.arange(len(day_total_trips)) * 3
bar_width = 2
bar1 = ax.bar(x, completion_percentages, bar_width, label = 'Validity', color = '#FFA500')
bar2 = ax.bar(x, uncompletion_percentages, bar_width, bottom=completion_percentages, label = 'Invalidity', color = '#ADD8E6')

for rect, completion_percentage, vehicle_name in zip(bar1 + bar2, completion_percentages, day_total_trips.keys()):
    if day_exists[vehicle_name]:
        height = rect.get_height()
        if completion_percentage > 0:
            if completion_percentage < 10:
                ax.text(rect.get_x() + rect.get_width() / 2, height + 2, f'{completion_percentage:.1f}%', ha = 'center', va = 'bottom', fontsize = 8, rotation = 90)
            else:
                ax.text(rect.get_x() + rect.get_width() / 2, height / 2, f'{completion_percentage:.1f}%', ha = 'center', va = 'center', fontsize = 8, rotation = 90)


for i, exists in enumerate(day_exists.values()):
    if not exists:
        bar1[i].set_height(0)
        bar2[i].set_height(0)

ax.set_xticks(x)
ax.set_xticklabels(range(1, len(days) + 1), fontsize = 6)

ax.set_ylabel('Percentage [%]')
ax.set_xlabel('Day')
ax.set_ylim(0, 115)

ax.set_title('Daily Validity Rate')
plt.legend(loc = 'upper center', ncol = 2)

plt.tight_layout()

save_path = destination_folder + scenario_folder + 'Daily_Valid_Next_Trip.png'
plt.savefig(save_path)
# Save the plot to a specific location as a svg
save_path = destination_folder + scenario_folder + 'Daily_Valid_Next_Trip.svg'
plt.savefig(save_path, format = 'svg')
        