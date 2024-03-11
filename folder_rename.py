import os

def remove_suffix(folder_name, suffix):
    """
    Removes the specified suffix from a folder name.
    :param folder_name: The original folder name.
    :param suffix: The suffix to remove.
    :return: The modified folder name.
    """
    if folder_name.endswith(suffix):
        return folder_name[:-len(suffix)]
    return folder_name

# Specify the directory containing your folders
directory_path = "/mnt/myhdd/Abdominal_1k/zerohataune/"

# Iterate through each folder
for folder_name in os.listdir(directory_path):
    full_path = os.path.join(directory_path, folder_name)
    if os.path.isdir(full_path):
        new_folder_name = remove_suffix(folder_name, "_0000")
        new_full_path = os.path.join(directory_path, new_folder_name)
        os.rename(full_path, new_full_path)
        print(f"Renamed '{folder_name}' to '{new_folder_name}'")

print("Folder names updated successfully!")
