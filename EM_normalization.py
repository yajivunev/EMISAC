#!/Users/vijay/miniforge3/envs/bs/bin/python

"""This code was originally developed by Samaneh Azadi in Prof. Pieter Abbeel's lab at UC Berkeley 
with the collaboration of Jeremy Maitin-Shepard. Updated for Python 3 compatibility."""

"""Copyright (C) {2014}  {Samaneh Azadi, Jeremy Maitin-Shepard, Pieter Abbeel}

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 3 of the License.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>."""

import os
import sys
import getopt
import numpy as np
import scipy as sp
import scipy.sparse as scsp
import scipy.ndimage as ndimage
from scipy.stats.mstats import mquantiles
import h5py
import time
from itertools import *
from skimage.morphology import rectangle
from skimage.filters import rank
from joblib import Parallel, delayed
from os.path import split as split_path, join as join_path
from fnmatch import filter as fnfilter
from PIL import Image
import multiprocessing
import subprocess
import threading
import ctypes

def usage():
    print("Help:")
    print("Usage : Normalization_command.py [options] <I> <w_x> <w_y> <par_x> <par_y> <n_cpu> <output-path>\n")
    print("<I>        STRING  The path to 3D input image on which you want to do normalization.")
    print("           (it should be an image file of .h5 or .tif formats)")
    print("<stp_x>    INT The sampling rate of the input image in the x coordinate. ")
    print("<stp_y>    INT The sampling rate of the input image in the y coordinate. ")
    print("           NOTE: w_x should be dividable by stp_x, and w_y should be dividable by stp_y.")
    print("<w_x>      INT     The minimum length of the artifacts you want to remove from the ")
    print("           image in X-Z direction (It is recommended to choose it around 32).")
    print("<w_y>      INT     The minimum length of the artifacts you want to remove from the")
    print("           image in Y-Z direction(It is recommended to choose it around 32).")
    print("           NOTE: Not to deface the image in the borders, you should use the")
    print("           values for w_x and w_y to which both the length of your input image  ")
    print("           and downsampled input image in x and y directions are dividable respectively!!")
    print("           EXAMPLE: if the size of your image is [1024,1020,100], ")
    print("           you can use stp_x=4, stp_y=2, w_x=32, w_y=30.")
    print("<par_x>    INT A paramter that shows you can divide your image in x-direction in ")
    print("           'par_x' partitions to speed up the algorithm.")
    print("           A higher value causes a faster process, but it should not be too large")
    print("<par_y>    INT     A paramter that shows you can divide your image in y-direction in ")
    print("           'par_y' partitions to speed up the algorithm.")
    print("           A higher value causes a faster process, but it should not be too large")
    print("           You should consider the following note...")
    print("           NOTE:(size_of_x_dim/w_x) & (size_of_downsampled_x_dim)should be dividable ")
    print("           to par_x (and the same for y-dim)...")
    print("<n_cpu>    INT     Defines the number of CPUs used for parallel computation on the partitions. ")
    print("           This param depends on the number of the CPUs of your computer.")
    print("           NOTE: max(num_parallel)= (# of CPUs). However, if your memory")
    print("           is too low, you should not use all of the CPUs")
    print("<output_path> STRING    Output directory where the generated normalized files will be placed.")
    print("options: ")
    print("--grp     STRING   The group name of your input file if it is .h5, and the group name of")
    print("           the output files. [Default: 'stack']")
    print("--up_fg    0/1     If you do not want to upsample your input image in the z-direction")
    print("           not to have a better resolution, you should put the 'up_fg=0'.[Default:1]")
    print("--up_z     INT     If you decide to upsample the input image, 'up_z' is the parameter")
    print("           that shows the size of the upsmapled image is how many times larger ")
    print("           than the original image in z-dir.[Default: 4]")           
    print("--fact    FLOAT    Determines the accuracy of the optimization process.[Default: 1e10]")
    print("           Typical values for fact are: 1e12 for low accuracy; 1e7 for moderate ")
    print("           accuracy; 10.0 for extremely high accuracy. The lower the 'fact' value,")
    print("           the higher the running time. The values in [1e8,1e10] produce good results.")
    print("--cns_fg   0/1     If you want to increase the contrast of the output image, set the ")
    print("           cns_fg=1. [Default=0]")
    print("--cns_low  FLOAT ")
    print("--cns_high FLOAT   If you have set cns_fg=1, you can choose which low and high quantiles")
    print("           be removed. [Default=(0.00001,0.99999)]")    
    print("--lm      FLOAT    Determines the ratio of the regularization term to the loss function.")
    print("-h           Help. Print Usage.")

def thread_proc(v_queue, finished_count, value_portion_arr, r_org, c_org, s_portion, z_division, beta, alpha, Dimg_org, window_size_x, window_size_y, selem, lower_val_beta, upper_val_beta, lower_val_alpha, upper_val_alpha, cnst, accuracy1, accuracy2):
    while True:
        try:
            bucket = v_queue.get_nowait()
        except:
            break
        

        beta_mat_block = beta[:, :, int(z_division * s_portion) + bucket]
        alpha_mat_block = alpha[:, :, int(z_division * s_portion) + bucket]

        correction_mat_pixel = scsp.diags(
            Dimg_org[int(int(z_division * s_portion) + bucket) * r_org * c_org:
                    int(int(z_division * s_portion) + bucket + 1) * r_org * c_org],
            0
        )
        correction_mat2_pixel = scsp.eye(r_org * c_org, r_org * c_org)
        correction_mat_pixel = scsp.hstack([correction_mat_pixel, correction_mat2_pixel])
        
        beta_mat_block = np.tile(beta_mat_block[:, np.newaxis], (1, window_size_x))
        beta_mat = np.tile(beta_mat_block[:, np.newaxis], (1, window_size_y))
        
        beta_mat = np.uint16((1.0 / accuracy1) * beta_mat)
        print(beta_mat_block.shape, beta_mat.shape, "beta_mat")
        beta_changed = rank.mean_bilateral(beta_mat, footprint=selem, s0=lower_val_beta, s1=upper_val_beta)
        beta_changed = np.float32(beta_changed) * accuracy1
        
        beta_vec = np.reshape(beta_changed, (r_org * c_org, 1), order='F')
        alpha_mat_block = np.tile(alpha_mat_block[:, np.newaxis], (1, window_size_x))
        alpha_mat = np.tile(alpha_mat_block[:, np.newaxis], (1, window_size_y))
        
        alpha_mat = alpha_mat + cnst
        alpha_mat = np.uint16((1.0 / accuracy2) * alpha_mat)
        alpha_changed = rank.mean_bilateral(alpha_mat, footprint=selem, s0=lower_val_alpha, s1=upper_val_alpha)
        alpha_changed = (np.float32(alpha_changed) * accuracy2) - cnst
        
        alpha_vec = np.reshape(alpha_changed, (r_org * c_org, 1), order='F')
        param_vec = np.vstack([beta_vec, alpha_vec])
        corrected_data_smoothed = correction_mat_pixel @ param_vec
        corrected_data_smoothed = np.reshape(corrected_data_smoothed, (r_org, c_org), order='F')

        with finished_count.get_lock():
            finished_count.value += 1
        
        with value_portion_arr.get_lock():
            value_portion_arr[..., bucket] = corrected_data_smoothed


def read_image_stack(fn):
    """Read a 3D volume of images in .tif or .h5 formats into a numpy.ndarray.
    This function attempts to automatically determine input file types and
    wraps specific image-reading functions.
    Adapted from gala.imio (https://github.com/janelia-flyem/gala)
    """
    if os.path.isdir(fn):
        fn += '/'
    d, fn = split_path(os.path.expanduser(fn))
    if len(d) == 0: 
        d = '.'
    fns = fnfilter(os.listdir(d), fn)
    if len(fns) == 1 and fns[0].endswith('.tif'):
        stack = read_multi_page_tif(join_path(d, fns[0]))
    elif fn.endswith('.h5'):
        with h5py.File(join_path(d, fn), 'r') as data:
            stack = data[group_name][()]
    return np.squeeze(stack)

def pil_to_numpy(img):
    """Convert a PIL Image object to a numpy array."""
    ar = np.squeeze(np.array(img.getdata()).reshape((img.size[1], img.size[0], -1)))
    return ar
       
def read_multi_page_tif(fn, crop=[None]*6):
    """Read a multi-page tif file into a numpy array.
    Currently, only grayscale images are supported.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = crop
    img = Image.open(fn)
    pages = []
    if zmin is not None and zmin > 0:
        img.seek(zmin)
    eof = False
    while not eof and img.tell() != zmax:
        pages.append(pil_to_numpy(img)[..., np.newaxis])
        try:
            img.seek(img.tell()+1)
        except EOFError:
            eof = True
    return np.concatenate(pages, axis=-1)

def make_shared_farray(shape):
    """Create a shared memory array for multiprocessing."""
    size = int(np.prod(shape))
    arr = multiprocessing.Array(ctypes.c_float, size)
    x = np.frombuffer(arr.get_obj(), dtype=np.float32)
    return (x.reshape(shape), arr)

def smoothing_func(lower_val_beta, upper_val_beta, lower_val_alpha, upper_val_alpha):
    """Eliminate blocking effects from previous stages."""
    print('Parameters found successfully and saved.')
    print('Please wait for smoothing over the parameters...')
    
    smooth_ngbh_x = 5
    smooth_ngbh_y = 5
    selem = rectangle(smooth_ngbh_x, smooth_ngbh_y)
    accuracy1 = 0.0025
    accuracy2 = 0.05
    r = r_org // num_partitions_x
    c = c_org // num_partitions_y
    max_shared_array_size = 33000000

    # def do_bilateral():
    #     num_threads = num_parallel
    #     volume_size = np.prod([r_org, c_org, s])
    #     volume_parts = np.ceil(volume_size / (100 * max_shared_array_size))
    #     value_total = np.zeros((r_org, c_org, s))
        
    #     for z_division in np.arange(volume_parts):
    #         s_portion = int(s / volume_parts)
    #         s_portion_prev = s_portion
    #         if z_division == volume_parts:
    #             s_portion = s - volume_parts * s_portion_prev
                
    #         value_portion, value_portion_arr = make_shared_farray([r_org, c_org, s_portion])
    #         v_queue = multiprocessing.Queue()
    #         finished_count = multiprocessing.Value(ctypes.c_long, 0)
            
    #         for bucket in range(s_portion):
    #             v_queue.put(bucket)

    #         def thread_proc():
    #             while True:
    #                 try:
    #                     bucket = v_queue.get_nowait()
    #                 except:
    #                     break
                    
    #                 beta_mat_block = beta[:, :, (z_division * s_portion) + bucket]
    #                 alpha_mat_block = alpha[:, :, (z_division * s_portion) + bucket]
        
    #                 correction_mat_pixel = scsp.diags(
    #                     Dimg_org[((z_division * s_portion) + bucket) * r_org * c_org:
    #                             ((z_division * s_portion) + bucket + 1) * r_org * c_org],
    #                     0
    #                 )
    #                 correction_mat2_pixel = scsp.eye(r_org * c_org, r_org * c_org)
    #                 correction_mat_pixel = scsp.hstack([correction_mat_pixel, correction_mat2_pixel])
                    
    #                 beta_mat_block = np.tile(beta_mat_block[:, np.newaxis], (1, window_size_x))
    #                 beta_mat = np.tile(beta_mat_block[:, np.newaxis], (1, window_size_y))
                    
    #                 beta_mat = np.uint16((1.0 / accuracy1) * beta_mat)
    #                 beta_changed = rank.mean_bilateral(beta_mat, footprint=selem, s0=lower_val_beta, s1=upper_val_beta)
    #                 beta_changed = np.float32(beta_changed) * accuracy1
                    
    #                 beta_vec = np.reshape(beta_changed, (r_org * c_org, 1), order='F')
    #                 alpha_mat_block = np.tile(alpha_mat_block[:, np.newaxis], (1, window_size_x))
    #                 alpha_mat = np.tile(alpha_mat_block[:, np.newaxis], (1, window_size_y))
                    
    #                 alpha_mat = alpha_mat + cnst
    #                 alpha_mat = np.uint16((1.0 / accuracy2) * alpha_mat)
    #                 alpha_changed = rank.mean_bilateral(alpha_mat, footprint=selem, s0=lower_val_alpha, s1=upper_val_alpha)
    #                 alpha_changed = (np.float32(alpha_changed) * accuracy2) - cnst
                    
    #                 alpha_vec = np.reshape(alpha_changed, (r_org * c_org, 1), order='F')
    #                 param_vec = np.vstack([beta_vec, alpha_vec])
    #                 corrected_data_smoothed = correction_mat_pixel @ param_vec
    #                 corrected_data_smoothed = np.reshape(corrected_data_smoothed, (r_org, c_org), order='F')
            
    #                 with finished_count.get_lock():
    #                     finished_count.value += 1
                    
    #                 with value_portion_arr.get_lock():
    #                     value_portion[..., bucket] = corrected_data_smoothed

    #         procs = []
    #         for _ in range(num_threads):
    #             p = multiprocessing.Process(target=thread_proc)
    #             p.start()
    #             procs.append(p)
    #         for p in procs:
    #             p.join()
                
    #         value_total[:, :, (z_division * s_portion_prev):(z_division * s_portion_prev + s_portion)] = value_portion
            
    #     return value_total

    def do_bilateral():
        num_threads = num_parallel
        volume_size = np.prod([r_org, c_org, s])
        volume_parts = np.ceil(volume_size / (100 * max_shared_array_size))
        value_total = np.zeros((r_org, c_org, s))
        
        for z_division in np.arange(volume_parts):
            s_portion = int(s / volume_parts)
            s_portion_prev = s_portion
            if z_division == volume_parts:
                s_portion = s - volume_parts * s_portion_prev
                
            value_portion, value_portion_arr = make_shared_farray([r_org, c_org, s_portion])
            v_queue = multiprocessing.Queue()
            finished_count = multiprocessing.Value(ctypes.c_long, 0)
            
            for bucket in range(s_portion):
                v_queue.put(bucket)

            procs = []
            for _ in range(num_threads):
                p = multiprocessing.Process(target=thread_proc, args=(v_queue, finished_count, value_portion_arr, r_org, c_org, s_portion, z_division, beta, alpha, Dimg_org, window_size_x, window_size_y, selem, lower_val_beta, upper_val_beta, lower_val_alpha, upper_val_alpha, cnst, accuracy1, accuracy2))
                p.start()
                procs.append(p)
            for p in procs:
                p.join()
                
            value_total[:, :, int(z_division * s_portion_prev):int(z_division * s_portion_prev + s_portion)] = value_portion
            
        return value_total


    alpha = np.zeros((r_org // window_size_x, c_org // window_size_y, s))
    beta = np.zeros((r_org // window_size_x, c_org // window_size_y, s))

    for partnum_y in range(num_partitions_y):
        for partnum_x in range(num_partitions_x):
            partnum = num_partitions_x * partnum_y + partnum_x
            with h5py.File(f'{output_path}/parameters_final_partition{step}_{partnum}.h5', 'r') as f:
                param = f[group_name][()]
                alpha_part = np.reshape(param[len(param)//2:], (r//window_size_x, c//window_size_y, s), order='F')
                beta_part = np.reshape(param[:len(param)//2], (r//window_size_x, c//window_size_y, s), order='F')
                alpha[(partnum_x*r//window_size_x):((partnum_x+1)*r//window_size_x),
                     (partnum_y*c//window_size_y):((partnum_y+1)*c//window_size_y), :] = alpha_part
                beta[(partnum_x*r//window_size_x):((partnum_x+1)*r//window_size_x),
                    (partnum_y*c//window_size_y):((partnum_y+1)*c//window_size_y), :] = beta_part

    Corrected_data_smoothed = do_bilateral()
    
    var = np.ptp(Corrected_data_smoothed)
    Corrected_data_smoothed = (Corrected_data_smoothed - np.min(Corrected_data_smoothed)) * 255.0 / var
    
    with h5py.File(f'{output_path}/normalized_data_final.h5', 'w') as f:
        f.create_dataset(group_name, data=Corrected_data_smoothed)
    
    return Corrected_data_smoothed

def AutoContrast(img, quantile_low, quantile_high):
    """Change the contrast of the input image by cutting the proper quantiles."""
    print("Please wait to change the contrast...")
    r, c, s = img.shape
    print(r, c, s)
    
    img = np.reshape(img, (r*c*s, 1), order='F')
    q = mquantiles(img, [quantile_low, quantile_high])
    img = (img - q[0]) * 255 / float(q[1] - q[0])
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8).astype(np.float32)
    img = np.reshape(img, (r, c, s), order='F')
    img = img.astype(np.uint8)
    
    print(img.shape)
    with h5py.File(f'{output_path}/higher_contrast_normalized_data.h5', 'w') as f:
        f.create_dataset(group_name, data=img)
    
    return img

def minimizer_downsampled(partnum, step):
    """Divide the image into several partitions and normalize each partition."""
    img = img_down[
        ((partnum % num_partitions_x) * r_down // num_partitions_x):
        ((partnum % num_partitions_x) * r_down // num_partitions_x + r_down // num_partitions_x),
        (partnum // num_partitions_x * c_down // num_partitions_y):
        (partnum // num_partitions_x * c_down // num_partitions_y + c_down // num_partitions_y),
        :]
    
    img_path = f'{output_path}/img_partition{partnum}.h5'
    with h5py.File(img_path, 'w') as da:
        da.create_dataset(group_name, data=img)
    
    print("WINDOWS??", window_size_x, window_size_y, step_x, step_y)

    # Call for the main normalization function
    subprocess.run(["python", "normalize_partition.py", "--grp", group_name, "--lm", str(lmbda), "--beta_bnd_l", str(lower_bnd_beta),
              "--beta_bnd_u", str(upper_bnd_beta), "--alpha_bnd_l", str(lower_bnd_alpha), "--alpha_bnd_u", str(upper_bnd_alpha),
              "--cnst", str(cnst), "--fact", str(fact), "--beta_val_l", str(lower_val_beta), "--beta_val_u", str(upper_val_beta),
              "--alpha_val_l", str(lower_val_alpha), "--alpha_val_u", str(upper_val_alpha), img_path, str(partnum),
              str(window_size_x//step_x), str(window_size_y//step_y), str(num_partitions_x), str(num_partitions_y),
              str(num_parallel), str(step), output_path])

def minimizer(partnum, step):
    """Divide the image into several partitions and normalize each partition."""
    img = img_org[
        ((partnum % num_partitions_x) * r_org // num_partitions_x):
        ((partnum % num_partitions_x) * r_org // num_partitions_x + r_org // num_partitions_x),
        (partnum // num_partitions_x * c_org // num_partitions_y):
        (partnum // num_partitions_x * c_org // num_partitions_y + c_org // num_partitions_y),
        :]
    
    img_path = f'{output_path}/img_partition{partnum}.h5'
    with h5py.File(img_path, 'w') as da:
        da.create_dataset(group_name, data=img)
    
    # Call for the main normalization function
    subprocess.run(["python", "normalize_partition.py", "--grp", group_name, "--lm", str(lmbda), "--beta_bnd_l", str(lower_bnd_beta),
              "--beta_bnd_u", str(upper_bnd_beta), "--alpha_bnd_l", str(lower_bnd_alpha), "--alpha_bnd_u", str(upper_bnd_alpha),
              "--cnst", str(cnst), "--fact", str(fact), "--beta_val_l", str(lower_val_beta), "--beta_val_u", str(upper_val_beta),
              "--alpha_val_l", str(lower_val_alpha), "--alpha_val_u", str(upper_val_alpha), img_path, str(partnum),
              str(window_size_x), str(window_size_y), str(num_partitions_x), str(num_partitions_y),
              str(num_parallel), str(step), output_path])
def main():
    global img_org, img_down, r_org, c_org, s, r_down, c_down, Dimg_org, Dimg_down
    global group_name, window_size_x, window_size_y, num_partitions_x, num_partitions_y
    global num_parallel, step, lmbda, fact, output_path, lower_val_beta, upper_val_beta
    global lower_val_alpha, upper_val_alpha, lower_bnd_beta, upper_bnd_beta
    global lower_bnd_alpha, upper_bnd_alpha, cnst, step_x, step_y

    opts, args = getopt.getopt(sys.argv[1:], "h", 
                              ["grp=", "up_fg=", "up_z=", "cnst=", "fact=", 
                               "cns_fg=", "cns_low=", "cns_high=", "lm="])

    # Setting the default parameters
    tt0 = time.time()
    group_name = 'stack'
    upsample_flag = 1
    upsample_z_param = 2
    lower_bnd_beta = 0.5
    upper_bnd_beta = 10000
    lower_bnd_alpha = -100
    upper_bnd_alpha = 10000
    fact = 1e10
    fact1 = fact
    lower_val_beta = 4000
    upper_val_beta = 4000
    lower_val_alpha = 4000
    upper_val_alpha = 4000
    cnst = abs(lower_bnd_alpha)
    contrast_flag = 0
    quantile_low = 0.00001
    quantile_high = 0.99999
    lm_cns = 100

    if len(args) != 9:
        print("Error: you have not entered all required inputs")
        usage()
        sys.exit(1)

    # Parse optional arguments
    for o, a in opts:
        if o == "--grp":
            group_name = str(a)
        elif o == "--up_fg":
            upsample_flag = int(a)
        elif o == "--up_z":
            upsample_z_param = int(a)
        elif o == "--fact":
            fact1 = float(a)
        elif o == "--cns_fg":
            contrast_flag = int(a)
        elif o == "--cns_low":
            quantile_low = float(a)
        elif o == "--cns_high":
            quantile_high = float(a)
        elif o == "--lm":
            lm_cns = float(a)
        elif o == "-h":
            usage()
            sys.exit(0)
        else:
            usage()
            sys.exit(1)

    # Read inputs
    print("Reading the Inputs...")
    img_org = read_image_stack(args[0])

    # Parse positional arguments
    step_x = int(args[1])
    step_y = int(args[2])
    window_size_x = int(args[3])
    window_size_y = int(args[4])
    num_partitions_x = int(args[5])
    num_partitions_y = int(args[6])
    num_parallel = int(args[7])
    output_path = args[8]
    lmbda = lm_cns / float((32/float(window_size_x))**2)

    # Initialize image processing
    img_org = img_org.astype(np.float32)
    r_org, c_org, s = img_org.shape
    
    for i in range(s):
        img_slice = img_org[:,:,i]
        img_org[:,:,i] = (img_slice - np.min(img_slice)) * 255 / np.ptp(img_slice)
    
    print(r_org, c_org, s)
    Dimg_org = np.reshape(img_org, (r_org*c_org*s,), order='F')

    # Upsampling
    if upsample_flag == 1:
        print("Upsampling the image...")
        resampled_img = np.zeros((r_org, c_org, s*upsample_z_param))
        img_org = img_org.astype(np.uint8)
        
        for i in range(c_org):
            resampled_img[:,i,:] = ndimage.zoom(img_org[:,i,:], 
                                              (1, upsample_z_param), 
                                              order=3)
        
        img_org = resampled_img.astype(np.float32)
        print(img_org.shape)
        r_org, c_org, s = img_org.shape
        
        with h5py.File(f'{output_path}/upsampled_train_input.h5', 'w') as g:
            g.create_dataset(group_name, data=img_org)
        
        Dimg_org = np.reshape(img_org, (r_org*c_org*s,), order='F')

    # Print parameters
    print(f"Parameters used in optimization process:")
    print(f"--grp {group_name} --up_fg {upsample_flag} --up_z {upsample_z_param} --lm {lmbda} "
          f"--fact {fact} --cns_fg {contrast_flag} --cns_low {quantile_low} --cns_high {quantile_high} "
          f"{args[0]} {window_size_x} {window_size_y} {num_partitions_x} {num_partitions_y} "
          f"{num_parallel} {output_path}")

    # Process downsampled version
    step = 1
    lmbda = lm_cns / float((step_x*step_y)*(32/float(window_size_x))**2)
    print(f'step {step}')
    img_down = img_org[::step_x, ::step_y, :]
    r_down, c_down, s = img_down.shape
    partnum_list = []
    Dimg_down = np.reshape(img_down, (r_down*c_down*s,), order='F')
    
    res = Parallel(n_jobs=num_parallel, verbose=100)(
        delayed(minimizer_downsampled)(partnum, step) 
        for partnum in range(num_partitions_x * num_partitions_y)
    )

    # Process full resolution
    step = 2
    lmbda = lm_cns / float((32/float(window_size_x))**2)
    print(f'step {step}')
    fact = fact1
    
    res = Parallel(n_jobs=num_parallel, verbose=100)(
        delayed(minimizer)(partnum, step) 
        for partnum in range(num_partitions_x * num_partitions_y)
    )

    # Smooth image
    del img_down
    #del img_org
    #img_org = smoothing_func(lower_val_beta, upper_val_beta, lower_val_alpha, upper_val_alpha)
    r_org, c_org, s = img_org.shape
    Dimg_org = np.reshape(img_org, (r_org*c_org*s,), order='F')

    # Final processing
    window_size_x = window_size_x // 2
    window_size_y = window_size_y // 2
    
    step = 3
    lmbda = lm_cns / float((32/float(window_size_x))**2)
    print(f'step {step}')
    
    res = Parallel(n_jobs=num_parallel, verbose=100)(
        delayed(minimizer)(partnum, step) 
        for partnum in range(num_partitions_x * num_partitions_y)
    )

    del img_org
    #img_org = smoothing_func(lower_val_beta, upper_val_beta, lower_val_alpha, upper_val_alpha)

    # Apply contrast enhancement if requested
    if contrast_flag == 1:
        data = AutoContrast(img_org, quantile_low, quantile_high)

    print(f"Total time: {time.time()-tt0}")
    print("Finished Successfully.")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()