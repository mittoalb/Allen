import click
import h5py
import numpy as np
import os

def print_attrs_and_datasets(name, obj, output_file):
    """
    Print attributes and datasets of the given HDF5 object to the output file.

    Parameters:
    - name: str, the name of the HDF5 object.
    - obj: h5py object, the HDF5 group or dataset object.
    - output_file: file object, the file to which attributes and datasets are written.
    """
    # Iterate over the attributes of the object
    for key, value in obj.attrs.items():
        try:
            # Decode bytes to string if necessary
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            # Write the attribute name and value to the output file
            output_file.write(f"{name}: {key} -> {value}\n")
        except Exception as e:
            # Handle any errors that occur during attribute processing
            output_file.write(f"Error processing attribute {key} in {name}: {e}\n")

    # Check if the object is a dataset and write its values to the output file
    if isinstance(obj, h5py.Dataset):
        try:
            data = np.array(obj)
            output_file.write(f"{name} dataset values: {data}\n")
        except Exception as e:
            # Handle any errors that occur during dataset processing
            output_file.write(f"Error processing dataset {name}: {e}\n")

@click.command()
@click.argument('file_path', type=click.Path(exists=True))
def process_hdf5(file_path):
    """
    Process the given HDF5 file, extracting attributes and dataset values, 
    and saving them to a text file with the same base name as the input file.

    Parameters:
    - file_path: str, the path to the input HDF5 file.
    """
    # Determine the output file path by changing the extension to .txt
    output_path = os.path.splitext(file_path)[0] + ".txt"
    with open(output_path, 'w', encoding='utf-8') as output_file:
        # Open the HDF5 file in read mode
        with h5py.File(file_path, 'r') as f:
            # Visit all items in the HDF5 file and apply the print_attrs_and_datasets function
            f.visititems(lambda name, obj: print_attrs_and_datasets(name, obj, output_file))

    # Print a message indicating the output file has been saved
    print(f"Output saved to {output_path}")

def main():
    process_hdf5()
if __name__ == '__main__':
    main()

