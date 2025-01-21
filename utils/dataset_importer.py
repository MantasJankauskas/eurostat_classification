import os
import urllib.request
import zipfile
import shutil

def download_dataset():
    url = "http://madm.dfki.de/files/sentinel/EuroSAT.zip"
    zip_filename = "EuroSAT.zip"
    extract_dir = "dataset"

    if not os.path.exists(zip_filename):
        print("Parsisiunčiama EuroSAT duomenų rinkinys...")
        urllib.request.urlretrieve(url, zip_filename)
        print("Parsisiuntimas baigtas!")
    else:
        print("Failas jau egzistuoja:", zip_filename)

    if not os.path.exists(extract_dir):
        print("Išarchyvuojama...")
        with zipfile.ZipFile(zip_filename, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Išarchyvavimas baigtas!")

        os.remove(zip_filename)
        print(f"Ištrintas ZIP failas: {zip_filename}")

        extracted_folders = os.listdir(extract_dir)
        for folder_name in extracted_folders:
            if folder_name == "2750":
                source_folder = os.path.join(extract_dir, folder_name)
                destination_folder = extract_dir
                for item in os.listdir(source_folder):
                    shutil.move(os.path.join(source_folder, item), destination_folder)
                os.rmdir(source_folder)
                print(f"Perkelti failai iš {folder_name} į {extract_dir}")
                break
    else:
        print("Duomenys jau išarchyvuoti:", extract_dir)
