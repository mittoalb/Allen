import h5py
import numpy as np
import os
import click
from .log import setup_custom_logger, info, error, warning, debug

def data_scope(assigned_title):
    """
    Extracts relevant data scope from the assigned title.

    Parameters:
    assigned_title (str): Title assigned to a dataset.

    Returns:
    str: Comma-separated list of matching keywords in the title, or 0 if no matches found.
    """
    keywords = [b'dark', b'flat', b'projections']  # Use bytes-like objects for keywords
    matching_keywords = [keyword.decode('utf-8') for keyword in keywords if keyword in assigned_title.lower()]
    return ', '.join(matching_keywords) if matching_keywords else None

def ReadDatasets(group, datasets, param):
    """
    Recursively reads datasets within an HDF5 group, identifies 3D arrays, and extracts relevant information.

    Parameters:
    group (h5py.Group): HDF5 group to explore.
    datasets (list): List to store identified datasets along with their associated group and type.
    param (dict): Dictionary to store additional parameters like scan_range, half_acquisition, and npoints.
    """
    esrf_meta = {
        'scan_range': 'scan_range',
        'half_acquisition': 'half',
        'npoints': 'angles'
    }
    for name, item in group.items():
        if isinstance(item, h5py.Group):
            # Recursively search within groups
            ReadDatasets(item, datasets, param)
        elif isinstance(item, h5py.Dataset) and item.ndim == 3 and name == 'image':
            path = '/' + str(group).split('/')[1] + '/title'
            try:
                title_dataset = group.file[path]
                typeD = data_scope(title_dataset[()])
                if typeD:
                    datasets.append((item, group, typeD))
            except KeyError:
                error(f"Dataset '{path}' not found.")
        elif name in esrf_meta:
            param[name] = item[()]

def read_linked_files(dataset):
    """
    Reads linked files from a virtual dataset.

    Parameters:
    - dataset (h5py.VirtualDataset): Input virtual dataset.

    Returns:
    - list of str: List of linked file names.
    """
    virtual_sources = dataset.virtual_sources()
    linked_files = []
    for source in virtual_sources:
        linked_files.append(source.file_name)
    return linked_files

def assign_data(data_path, group_path, dataset, output_file, key):
    """
    Assigns virtual datasets to an output HDF5 file based on linked files.

    Parameters:
    - data_path (str): Path to the directory containing linked HDF5 files.
    - group_path (str): Path to the group within the linked HDF5 files.
    - dataset (numpy.ndarray): Input dataset to be linked.
    - output_file (h5py.File): HDF5 file where the virtual dataset will be created.
    - key (str): Key for the virtual dataset in the output file.
    """
    # Read linked files from the dataset
    linked_files = read_linked_files(dataset)
    # Define the group path within the HDF5 file
    gpath = '/entry_0000/' + group_path + '/data'
    # Create a virtual layout for the output dataset
    layout = h5py.VirtualLayout(dataset.shape, dtype=float)
    if key != '/exchange/data' and len(linked_files) > 1:
        # Removes averaged whites
        file_path = os.path.join(data_path, linked_files[0])
        # Create a virtual source for the current file
        vsource = h5py.VirtualSource(file_path, gpath, shape=(int(dataset.shape[0] / len(linked_files)), dataset.shape[1], dataset.shape[2]))
        layout[:int(dataset.shape[0] / len(linked_files)), :, :] = vsource
    else:
        # Iterate over each HDF5 file
        for n, hdf5_file in enumerate(linked_files):
            # Create the full path to the HDF5 file
            file_path = os.path.join(data_path, hdf5_file)
            # Create a virtual source for the current file
            vsource = h5py.VirtualSource(file_path, gpath, shape=(int(dataset.shape[0] / len(linked_files)), dataset.shape[1], dataset.shape[2]))
            # Assign the virtual source to the layout
            layout[n * int(dataset.shape[0] / len(linked_files)):(n + 1) * int(dataset.shape[0] / len(linked_files)), :, :] = vsource
    # Create a virtual dataset in the output file linked to the virtual layout
    output_file.create_virtual_dataset(key, layout)

def APS_format(input_file, new_file_path, reference):
    """
    Converts ESRF to APS IMG data structure.

    Parameters:
    input_file (str): Path to the existing HDF5 file (ESRF format).
    new_file_path (str): Path to the new HDF5 file (APS IMG format).
    """
    data_path = os.path.dirname(input_file)
    with h5py.File(input_file, 'r') as input_file, h5py.File(new_file_path, 'w') as outfile:
        # Create groups
        APS_base = '/exchange/'
        data_g = outfile.create_group(APS_base)
        defa = outfile.create_group('/default')
        meas = outfile.create_group('/measurement')
        proc = outfile.create_group('/process')
        # Use the previous function to identify 3D arrays and 'title' entries
        datasets_list = []
        param = {}
        ReadDatasets(input_file, datasets_list, param)
        data_map = {
            'projections': 'data',
            'flat': 'data_white',
            'dark': 'data_dark'
        }
        for dataset, group, measurement in datasets_list:
            target_path = dataset.name
            group_path = '/'.join(group.name.split('/')[2:])
            key = data_map.get(measurement, None)
            if key:
                try:
                    key = f'{key}F' if key == 'data_white' and 'data_white' in data_g else key
                    assign_data(data_path, group_path, dataset, outfile, APS_base + key)
                except ValueError:
                    pass
        # Create angle array
        data_g.create_dataset(f'theta', data=np.linspace(0, int(param['scan_range']), int(param['npoints'])))
        if reference is not None:
            # Get new data path
            data_path = os.path.dirname(reference)
            with h5py.File(reference, 'r') as ref_file:
                # Use the previous function to identify 3D arrays and 'title' entries
                datasets_list = []
                param = {}
                ReadDatasets(ref_file, datasets_list, param)
                data_map = {
                    'flat': 'data_white',
                    'dark': 'data_dark'
                }
                for dataset, group, measurement in datasets_list:
                    target_path = dataset.name
                    group_path = '/'.join(group.name.split('/')[2:])
                    key = data_map.get(measurement, None)
                    if key:
                        try:
                            key = f'{key}F' if key == 'data_white' and 'data_white' in data_g else key
                            assign_data(data_path, group_path, dataset, outfile, APS_base + key)
                        except ValueError:
                            pass

@click.command()
@click.option('-i', '--fname', required=True, type=click.Path(exists=True), help='Path to the existing HDF5 file.')
@click.option('-o', '--outname', required=True, type=click.Path(), help='Path to the new HDF5 file.')
@click.option('-r', '--reference', default=None, type=click.Path(), help='Path to the reference HDF5 file.')
def main(fname, outname, reference):
    setup_custom_logger(stream_to_console=True)
    APS_format(fname, outname, reference)

if __name__ == "__main__":
    main()

