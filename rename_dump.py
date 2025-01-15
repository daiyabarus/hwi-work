import os

root_dir = r'D:\HW\Export\XML_LTE_NS_20250106'

for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename == 'CFGDATA.XML':
            upper_folder_name = os.path.basename(os.path.dirname(dirpath))
            new_filename = f"{upper_folder_name}.XML"
            old_file_path = os.path.join(dirpath, filename)
            new_file_path = os.path.join(dirpath, new_filename)
            os.rename(old_file_path, new_file_path)
            print(f"Renamed {old_file_path} to {new_file_path}")

# INFO: This script will rename all CFGDATA.XML files to the upper folder name.XML
# ├───E_TBT875_KodimTebing
# │   └───AUTOBAKDATA20250106033705
# |        └───CFGDATA.XML
# └───E_TBT876_BroholTebing
#     └───AUTOBAKDATA20250106033635
#         └───CFGDATA.XML


# ├───E_TBT875_KodimTebing
# │   └───AUTOBAKDATA20250106033705
# |        └───E_TBT875_KodimTebing.XML
# └───E_TBT876_BroholTebing
#     └───AUTOBAKDATA20250106033635
#         └───E_TBT876_BroholTebing.XML