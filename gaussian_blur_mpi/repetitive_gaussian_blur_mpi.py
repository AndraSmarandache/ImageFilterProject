import numpy as np
import cv2
from mpi4py import MPI
import time

def create_gaussian_kernel(size, sigma):
    kernel = np.zeros((size, size))
    center = size//2
    for y in range(size):
        for x in range(size):
            diff_x = x - center
            diff_y = y - center
            kernel[y,x] = np.exp(-(diff_x**2 + diff_y**2)/(2*sigma**2))
    return kernel/kernel.sum()

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    radius = 350
    sigma = radius/3.0
    kernel_size = 2*radius + 1
    input_path = "images/resized_image_32768x32768.jpg"
    output_path = "images/filtered_image_32768x32768.jpg"

    if rank == 0:
        image = cv2.imread(input_path)
        if image is None: comm.Abort(1)
        height, width, _ = image.shape
        kernel = create_gaussian_kernel(kernel_size, sigma)
    else:
        image, height, width = None, None, None
        kernel = np.empty((kernel_size, kernel_size))

    height = comm.bcast(height, root=0)
    width = comm.bcast(width, root=0)
    comm.Bcast(kernel, root=0)

    rows_per_process = height // size
    start_row = rank * rows_per_process
    end_row = height if rank == size-1 else start_row + rows_per_process
    extra_top = min(radius, start_row)
    extra_bottom = min(radius, height - end_row)

    local_data = np.empty((rows_per_process + extra_top + extra_bottom, width, 3), dtype=np.uint8)
    if rank == 0:
        for i in range(1, size):
            other_start = i * rows_per_process
            other_end = height if i == size-1 else other_start + rows_per_process
            other_top = min(radius, other_start)
            other_bottom = min(radius, height - other_end)
            comm.Send(image[other_start-other_top:other_end+other_bottom], dest=i, tag=i)
        local_data = image[start_row-extra_top:end_row+extra_bottom]
    else:
        comm.Recv(local_data, source=0, tag=rank)

    process_start = time.time() * 1000  
    local_result = np.empty((rows_per_process, width, 3), dtype=np.uint8)
    for c in range(3):
        for j in range(25):
         filtered = cv2.filter2D(local_data[:,:,c], -1, kernel, borderType=cv2.BORDER_REFLECT)
        local_result[:,:,c] = filtered[extra_top:extra_top+rows_per_process]
    
    if rank == 0:
        result = np.empty((height, width, 3), dtype=np.uint8)
        result[start_row:end_row] = local_result
        for i in range(1, size):
            other_start = i * rows_per_process
            other_end = height if i == size-1 else other_start + rows_per_process
            recv_buf = np.empty((other_end-other_start, width, 3), dtype=np.uint8)
            comm.Recv(recv_buf, source=i, tag=i+100)
            result[other_start:other_end] = recv_buf
        process_end = time.time() * 1000 
        
        cv2.imwrite(output_path, result)
        print(f"Parallel Processing Time: {process_end-process_start:.2f}")
    else:
        comm.Send(local_result, dest=0, tag=rank+100)

if __name__ == "__main__":
    main()
