import glob
import os
from tqdm import tqdm

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
                    
                write_group_folder = f"{write_alphabetical_folder}/{group_order}"
                
                #  Apply get_tokens_inst to the files in the group folder
                n_skipped += get_tokens_inst_iter_files(instrument_to_keep, read_group_folder, write_group_folder)
                pbar.update(1)
    
    print(f"Skipped {n_skipped} files, because they did not contain any of the instruments we want.")
                    
               