from zipfile import ZipFile

def extract_all(filepath):

    """
    Extract all the files in the current directory. 
    """

    with ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall('')


def zip_csv(file, name="ZipFile"):

    """
    Zip a csv file.
    """
    
    file.to_csv(f'{name}.zip', 
              index=False, 
              compression={'method': 'zip', 'archive_name': f'{name}.csv'})
