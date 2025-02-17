import glob
import os
from tqdm import tqdm
import pathlib
import guitarpro as pygp


def write_tokens_from_file(instrument_list, current_text, new_text):
    """Write the lines of the current_text that concern the instruments in instrument_list to new_text.

    Args:
        instrument_list (list): List of instruments to keep.
        current_text (str): Text to read from.
        new_text (file): File to write to.
    """
    
    in_measure = False
    check_nfx = False
    for line in current_text.split("\n"):
        # write everything until we reach the measures
        
        if "measure" in line:
            new_text.write(line + "\n")
            in_measure=True
        
        elif not in_measure:
            new_text.write(line + "\n")
        
        if check_nfx:
            if "nfx" in line or "bfx" in line:
                new_text.write(line + "\n")
            else:
                check_nfx = False
        
        if in_measure:
            # print only the lines that contain the instrument and the next if they are nfx or bfx
            if (any(instr in line for instr in instrument_list)) or ('wait' in line):
                new_text.write(line + "\n")
                check_nfx = True
                
def merge_consecutive_waits(file_path):
    """Merge consecutive wait tokens in a file.

    Args:
        file_path (str): Path to the file to merge consecutive wait tokens in.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    merged_lines = []
    wait_sum = 0

    for line in lines:
        line = line.strip()
        if line.startswith("wait:"):
            # Extract the numeric value and add to the current sum
            wait_value = int(line.split(":")[1])
            wait_sum += wait_value
        else:
            # If we encounter a non-wait line, finalize the current wait sum
            if wait_sum > 0:
                merged_lines.append(f"wait:{wait_sum}")
                wait_sum = 0
            merged_lines.append(line)
    
    # Add any remaining wait sum at the end
    if wait_sum > 0:
        merged_lines.append(f"wait:{wait_sum}")
    
    # Write the merged lines back to the same file
    with open(file_path, 'w') as file:
        file.write("\n".join(merged_lines))


def get_tokens_inst_iter_files(instrument_dict, path_to_read_folder, path_to_write_folder):
    """Get the tokens of the instruments in instrument_dict from the files in path_to_read_folder and write them to path_to_write_folder.

    Args:
        instrument_dict (dict): Dictionary with instruments as keys and boolean values indicating whether to keep them or not.
        path_to_read_folder (str): Path to the folder containing the files to read.
        path_to_write_folder (str): Path to the folder to write the files to.
    """
    
    # Generate write folder if it does not exist
    if not os.path.exists(path_to_write_folder):
        os.makedirs(path_to_write_folder)
    
    # Generate a list of the instruments to keep
    instrument_list = [key for key, value in instrument_dict.items() if value]
    n_skipped = 0
    
    # Read the txt files in the folder
    for current_file in glob.glob(f"{path_to_read_folder}/*.txt"):
        with open(current_file) as f_1:
            current_text = f_1.read()
            
            instruments_in_file ={
                "distorted0": False,
                "distorted1": False,
                "distorted2": False,
                "clean0": False,
                "clean1": False,
                "bass": False,
                "leads": False,    
                "pads": False,
                "drums": False,
            }
            
            for instrument in instruments_in_file.keys():
                if instrument in current_text:
                    instruments_in_file[instrument] = True
            
            if not any(instruments_in_file[instrument] for instrument in instrument_list):
                n_skipped += 1
                continue # Skip the file if it does not contain any of the instruments we want
            
            # Generate file name by appending the instruments to the original file name
            file_name = current_file.split("\\")[-1].split(".")[0] # Remove the .txt extension
            file_name = "_".join([file_name, "_".join(instrument_list)]) # Append the instruments
            file_path = f"{path_to_write_folder}/{file_name}.txt" # Add the .text and the path
            
            # Generate a txt file only containing the lines refering to the instrument
            with open(file_path, "w") as new_text:
                
                write_tokens_from_file(instrument_list, current_text, new_text)
                
            # Merge consecutive wait commands
            merge_consecutive_waits(file_path)
    
    return n_skipped
            

def get_tokens_inst_iter_folders(instrument_to_keep, path_to_general_read_folder, path_to_general_write_folder):
    """Iterate over the DadaGP dataset and apply get_tokens_inst to the files in the dataset.
    Write back the files to the write folder BGTG.

    Args:
        instrument_to_keep (dict): Dictionary with instruments as keys and boolean values indicating whether to keep them or not.
        path_to_general_read_folder (str): Path to the folder containing the DadaGP dataset.
        path_to_general_write_folder (str): Path to the folder BGTG where we write the files.
    """
    
    n_skipped = 0
    # Iterate twice over the folders in the general folder (alphabetical and group)
    with tqdm(total=4871, desc="Scores computed") as pbar:
            
        for read_alphabetical_folder in glob.glob(f"{path_to_general_read_folder}/*"):
            # Create a folder with the same name in the write folder
            
            # Retrieve the alphabetical folder name (last part of the path)
            alphabetical_order = read_alphabetical_folder.split("\\")[-1]
            write_alphabetical_folder = f"{path_to_general_write_folder}/{alphabetical_order}"
            
            if not os.path.exists(write_alphabetical_folder):
                os.makedirs(write_alphabetical_folder)
                    
            for read_group_folder in glob.glob(f"{read_alphabetical_folder}/*"):
                # Create a folder with the same name in the write folder
                
                group_order = read_group_folder.split('\\')[-1]

                write_group_folder = f"{write_alphabetical_folder}/{group_order}"
                
                if not os.path.exists(write_group_folder):
                    os.makedirs(write_group_folder)
                                    
                #  Apply get_tokens_inst to the files in the group folder
                n_skipped += get_tokens_inst_iter_files(instrument_to_keep, read_group_folder, write_group_folder)
                pbar.update(1)
    
    print(f"Skipped {n_skipped} files, because they did not contain any of the instruments we want.")


def get_tokens_inst_iter_df_rg(df_rg, path_to_general_read_folder, path_to_general_write_folder):
    """Iterate over the DadaGP dataset and apply get_tokens_inst to the files in the dataset.
    Write back the files to the write folder BGTG.

    Args:
        df_rg (pd.DataFrame): DataFrame with the DadaGP dataset with the rythmic information.
        path_to_general_read_folder (str): Path to the folder containing the DadaGP dataset.
        path_to_general_write_folder (str): Path to the folder BGTG where we write the files.
    """
    
    # Iterate twice over the folders in the general folder (alphabetical and group)
    for str_path in tqdm(df_rg['Dadagp_Path'].unique()):
        
        # Replace extension with txt
        txt_file_path = str_path.replace('.pygp.gp5', '.tokens.txt')
        
        df_track = df_rg[df_rg['Dadagp_Path'] == str_path]
        
        instruments_to_keep = {
            "distorted0": False,
            "distorted1": False,
            "distorted2": False,
            "distorted3": False,
            "distorted4": False,
            "distorted5": False,
            "clean0": False,
            "clean1": False,
            "clean2": False,
            "clean3": False,
            "clean4": False,
            "bass": False,
            "leads": False,    
            "pads": False,
            "drums": False,
        }
        
        # Create a folder with the same name in the write folder
        
        write_folder = str_path.replace(path_to_general_read_folder, path_to_general_write_folder)
        write_folder = write_folder.replace('.pygp.gp5', '')
        write_folder = pathlib.Path(write_folder).with_suffix('')
        # Remove the last part of the path
        write_folder = write_folder.parent 
        write_folder.mkdir(parents=True, exist_ok=True)
            
        rythmic_instruments = df_track[df_track['is_track_rythmic']]['Dadagp_Name'].unique()
        
        for instrument in rythmic_instruments:
            instruments_to_keep[instrument] = True
            instruments_to_keep['bass'] = False
        
        # Generate file name by appending the instruments to the original file name
        file_name = df_track['File_Name'].iloc[0]
        instrument_list = [key for key, value in instruments_to_keep.items() if value]
        file_name = "_".join([file_name, "rythmic"]) # Append 'rythmic'
        file_path = f"{write_folder}/{file_name}.txt" # Add the .text and the path

        with open(txt_file_path) as f_1:
            current_text = f_1.read()
        # Generate a txt file only containing the lines refering to the instrument
            with open(file_path, "w") as new_text:
                
                write_tokens_from_file(instrument_list, current_text, new_text)
                
            # Merge consecutive wait commands
            merge_consecutive_waits(file_path)



# def ultimate_get_filename(row, list_filenames, list_paths):
#     name = row['Fichier']
#     name = name.split('/')[-1]
#     name = name.split('.')[0]
#     splits = name.split('-')
#     current_name = splits[0]
#     i = 1
#     while current_name not in list_filenames:
#         # print(current_name)
#         try:
#             current_name += '-' + splits[i]
#             i += 1
#         except:
#             row['Dadagp_Path'] = 'error'
#             row['File_Name'] = 'error'
#             return row
    
#     path_index = list_filenames.index(current_name)
#     path = list_paths[path_index]
#     row['Dadagp_Path'] = path
#     row['File_Name'] = current_name
#     return row

   
def ultimate_get_filename_2(row, series_filenames, series_paths):
    """Get the filename of the file in the series_filenames that matches the row.

    Args:
        row (pandas.Series): Row of the DataFrame.
        series_filenames (pandas.Series): Series with the filenames.
        series_paths (pandas.Series): Series with the paths.

    Returns:
        pandas.Series: Row with the 'Dadagp_Path' and 'File_Name' filled.
    """
    name = row['Fichier']
    name = name.split('/')[-1]
    name = name.split('.')[0]
    splits = name.split('-')
    current_name = splits[0]
    i = 1
    while current_name not in series_filenames.values:
        # print(current_name)
        try:
            current_name += '-' + splits[i]
            i += 1
        except:
            row['Dadagp_Path'] = 'error'
            row['File_Name'] = 'error'
            return row
    
    row['File_Name'] = current_name
    indexes = series_filenames[series_filenames == current_name].index
    paths = series_paths[indexes].values
    for path in paths:
        row_part = row['Partie']
        gp_file = pygp.parse(str(path))
        file_parts = [track.name for track in gp_file.tracks]
        if row_part in file_parts:
            row['Dadagp_Path'] = path
            break

    return row