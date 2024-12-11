import os
import shutil
from glob import glob

import tifffile as tf
import numpy as np
from multiprocessing import Pool
import argparse
   
#Generate a copy of the volume   
def GenerateRealVol(source_dir, dest_dir, start_num, end_num,k):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for i in range(start_num, end_num + 1):
        src_file = os.path.join(source_dir, f"recon_{i:05d}.tiff")
        dest_file = os.path.join(dest_dir, f"recon_{k:05d}.tiff")
        shutil.copyfile(src_file, dest_file)
        k += 1   
        
#Create a virtual copy of the volume
def GenerateVirtualVol(source_dir, dest_dir, start_num, end_num, k):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    print(source_dir)
    for i in range(start_num, end_num + 1):
        src_file = os.path.join(source_dir, f"recon_{i:05d}.tiff")
        dest_file = os.path.join(dest_dir, f"recon_{k:05d}.tiff")
        os.symlink(src_file, dest_file)
        k += 1  
    return k


def process_image(max_val, min_val, image_path):
    image = tf.imread(image_path)
    return max_val.append(np.max(image)), min_val.append(np.min(image))


def bin_convert(binning,source_directory, start_vol, end_vol,formatf):
    for i in binning:
            bin_dir = source_directory + "VOL_" + str(i) + '_' + str(formatf)
            if not os.path.exists(bin_dir):
                os.makedirs(bin_dir)
    
    image_files = [os.path.join(source_directory + "VOL/", file) for file in os.listdir(source_directory + "VOL/") if file.endswith(".tiff")]
    c=0
    max_val = []
    min_val = []
    
    for file in image_files:
    	if c > 1500 and c < 1600:
    		process_image(max_val, min_val, file)
    	c = c + 1

    overall_max, overall_min = np.max(max_val), np.min(min_val)
    n = 0 #out file counter
    
    for file in image_files[start_vol:end_vol]:
        image = tf.imread(file)
        
        for i in binning:
                bin_dir = source_directory + "VOL_" + str(i) + '_' + str(formatf)
                if formatf == 8:
                      binned_image = image.reshape(image.shape[0] // i, i, image.shape[1] // i, i).mean(axis=(1, 3))
                      image_8bit = ((binned_image - overall_min) / (overall_max - overall_min) * 255).astype(np.uint8)
                      output_file = os.path.join(bin_dir, f"recon_{n:05d}.tiff")
                      tf.imwrite(output_file, image_8bit)
                if formatf == 16:
                      binned_image = image.reshape(image.shape[0] // i, i, image.shape[1] // i, i).mean(axis=(1, 3))
                      image_16bit = ((binned_image - overall_min) / (overall_max - overall_min) * 65535).astype(np.uint16)
                      output_file = os.path.join(bin_dir, f"recon_{n:05d}.tiff")
                      tf.imwrite(output_file, image_16bit)
        n += 1


########################################################


def main():
    parser = argparse.ArgumentParser(description='Assemble data stack and save binned copies.')
    parser.add_argument('--source_path', type=str, required=True, help='The source path containing the data. Provide the full path.')
    parser.add_argument('--starting_line', type=int, required=True, help='The starting line number in each frame for overlap.')
    parser.add_argument('--ending_line', type=int, required=True, help='The ending line number in each frame for overlap.')
    parser.add_argument('--start_vol', type=int, required=True, help='The starting line in the full volume.')
    parser.add_argument('--end_vol', type=int, required=True, help='The ending line in the full volume.')
    parser.add_argument('--binning', type=int, nargs='+', default=[1,2,4,8,16], help='List of binning values for conversion.')
    parser.add_argument('--format', type=int, default='16', help='8 bit or 16 bit')

    args = parser.parse_args()

    subdirectories = glob(args.source_path + "/**/", recursive=True)
    reco_dirs = [dir for dir in subdirectories if 'APS_rec' in dir]

    l = 0
    for dir in reco_dirs:
        if l == 0: 
            k = 0
        k = GenerateVirtualVol(dir, args.source_path + "VOL/", args.starting_line, args.ending_line, k)
        l += 1

    bin_convert(args.binning, args.source_path, args.start_vol, args.end_vol, args.format)

if __name__ == "__main__":
    main()


