import os
import h5py
import click
import fabio
import numpy as np
from .log import setup_custom_logger, info, error, warning, debug

setup_custom_logger()

def read_edf_files(input_path, numberofdark, data_radix):
    """
    Reads EDF files from the specified input directory, categorizing them into data, dark, and white images.

    Args:
        input_path (str): Path to the directory containing EDF files.
        numberofdark (int): The number to divide the dark data values by.
        data_radix (str): Radix name to identify the data files.

    Returns:
        tuple: Containing lists of data images, dark images, white images, and angles (empty for now).
    """
    data = []
    data_dark = []
    data_white = []
    angles = []
    for filename in sorted(os.listdir(input_path)):
        if filename.endswith(".edf"):
            filepath = os.path.join(input_path, filename)
            edf_file = fabio.open(filepath)
            image_data = edf_file.data

            if "darkHST" in filename:
                data_dark.append(image_data / numberofdark)
            elif "refHST" in filename:
                data_white.append(image_data)
            elif data_radix in filename:
                data.append(image_data)
    
    info(f"Read {len(data)} data files, {len(data_dark)} dark files, and {len(data_white)} white files.")
    return data, data_dark, data_white, angles

def generate_theta(num_data_files, arange):
    """
    Generates an array of angles for the given number of data files and angular range.

    Args:
        num_data_files (int): Number of data files (projections).
        arange (int): Angular range (either 180 or 360).

    Returns:
        numpy.ndarray: Array of angles in radians.
    """
    angular_step = arange / num_data_files
    info(f"Angular step found: {angular_step}")
    return np.array(np.linspace(0, arange, num=num_data_files, endpoint=False))# * (np.pi / 180.0))

def get_data_radix(input_path):
    """
    Retrieves the radix name from the input path.

    Args:
        input_path (str): Path to the input directory.

    Returns:
        str: Radix name.
    """
    data_radix = os.path.basename(os.path.normpath(input_path))
    debug(f"Data radix: {data_radix}")
    return data_radix

def saveh5(output_path, data, data_dark, data_white, angles):
    """
    Saves the data, dark images, white images, and angles to an HDF5 file.

    Args:
        output_path (str): Path to the output HDF5 file.
        data (list): List of data images.
        data_dark (list): List of dark images.
        data_white (list): List of white images.
        angles (list): List of angles.
    """
    with h5py.File(output_path, 'w') as hdf_file:
        exchange_group = hdf_file.create_group('/exchange')
        exchange_group.create_dataset('data', data=data)
        exchange_group.create_dataset('data_dark', data=data_dark)
        exchange_group.create_dataset('data_white', data=data_white)
        exchange_group.create_dataset('theta', data=angles)
    info(f"Data saved to {output_path}")

@click.command()
@click.option('--input_path', required=True, type=click.Path(exists=True), help="Path to the input directory containing EDF files.")
@click.option('--output_path', required=True, type=click.Path(), help="Path to the output HDF5 file.")
@click.option('--arange', required=True, type=int, help="Angular range (either 180 or 360).")
@click.option('--numberofdark', default=20, type=int, help="Number to divide the dark data values by. Default is 20.")
def convert_edf_to_hdf5(input_path, output_path, arange, numberofdark):
    """
    Main function to convert EDF files to an HDF5 file. Reads the EDF files, generates angles, and saves to HDF5.

    Args:
        input_path (str): Path to the input directory containing EDF files.
        output_path (str): Path to the output HDF5 file.
        arange (int): Angular range (either 180 or 360).
        numberofdark (int): Number to divide the dark data values by. Default is 20.
    """
    data_radix = get_data_radix(input_path)
    info(f"Data Radix Name: {data_radix}")

    data, data_dark, data_white, angles = read_edf_files(input_path, numberofdark, data_radix)
    
    num_data_files = len(data)
    info(f"Number of projections found: {num_data_files}")
    angles = generate_theta(num_data_files, arange)
    info(f"Number of angles generated: {len(angles)}")

    saveh5(output_path, data, data_dark, data_white, angles)
    
def main():
    convert_edf_to_hdf5()

if __name__ == '__main__':
    main()

