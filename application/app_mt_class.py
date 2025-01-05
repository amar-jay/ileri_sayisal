from ctypes import *
from typing import List
import cv2
import numpy as np
import vart
import os
import pathlib
import xir
import threading
import time
import sys
import argparse

divider = '------------------------------------'


def load_npy(image_path):
    image = np.load(image_path)
    return image


def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]


def runDPU(id, start, dpu, img):
    '''get tensor'''
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)
    output_ndim = tuple(outputTensors[0].dims)
    # print(output_ndim)

    save_path = '/Xilinx/output/'
    batchSize = input_ndim[0]
    n_of_images = len(img)
    count = 0
    write_index = start
    ids = []
    ids_max = 50
    outputData = []
    for i in range(ids_max):
        outputData.append([np.empty(output_ndim, dtype=np.float32, order="C")])
    while count < n_of_images:
        if (count + batchSize <= n_of_images):
            runSize = batchSize
        else:
            runSize = n_of_images - count

        '''prepare batch input/output '''
        inputData = []
        inputData = [np.empty(input_ndim, dtype=np.float32, order="C")]

        '''init input image to input buffer '''
        for j in range(runSize):
            imageRun = inputData[0]
            imageRun[j, ...] = img[(count + j) % n_of_images].reshape(input_ndim[1:])
        '''run with batch '''
        job_id = dpu.execute_async(inputData, outputData[len(ids)])
        ids.append((job_id, runSize, start + count))
        count = count + runSize
        if count < n_of_images:
            if len(ids) < ids_max - 1:
                continue
        for index in range(len(ids)):
            dpu.wait(ids[index][0])
            write_index = ids[index][2]
            '''store output vectors '''
            for j in range(ids[index][1]):
                out_q[write_index] = np.argmax(outputData[index][0][j])
                save_name = '{}_{}.png'.format(index, j)
                imsave = outputData[index][0][j] * 255
                cv2.imwrite(os.path.join(save_path, save_name), imsave.astype('uint8'))
                write_index += 1
        ids = []


def app(image_dir, threads, model):
    listimage = os.listdir(image_dir)
    runTotal = len(listimage)
    print(runTotal)
    global out_q
    out_q = [None] * runTotal
    g = xir.Graph.deserialize(model)
    subgraphs = get_child_subgraph_dpu(g)
    all_dpu_runners = []
    for i in range(threads):
        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))

    # input scaling
    input_fixpos = all_dpu_runners[0].get_input_tensors()[0].get_attr("fix_point")

    ''' preprocess images '''
    print(divider)
    print('Pre-processing', runTotal, 'images...')
    img = []
    for i in range(runTotal):
        path = os.path.join(image_dir, listimage[i])
        img.append(load_npy(path))

    '''run threads '''
    print('Starting', threads, 'threads...')
    threadAll = []
    start = 0
    for i in range(threads):
        if (i == threads - 1):
            end = len(img)
        else:
            end = start + (len(img) // threads)
        in_q = img[start:end]
        t1 = threading.Thread(target=runDPU, args=(i, start, all_dpu_runners[i], in_q))
        threadAll.append(t1)
        start = end

    time1 = time.time()
    for x in threadAll:
        x.start()
    for x in threadAll:
        x.join()
    time2 = time.time()
    timetotal = time2 - time1

    fps = float(runTotal / timetotal)
    print(divider)
    print("Throughput=%.2f fps, total frames = %.0f, time=%.4f seconds" % (fps, runTotal, timetotal))

    return


# only used if script is run as 'main' from command line
def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--image_dir', type=str, default='images/', help='Path to folder of images. Default is images')
    ap.add_argument('-t', '--threads', type=int, default=1, help='Number of threads. Default is 1')
    ap.add_argument('-m', '--model', type=str, default='models/unet_compiled.xmodel',
                    help='Path of xmodel. Default is customcnn.xmodel')

    args = ap.parse_args()

    print('Command line options:')
    print(' --image_dir : ', args.image_dir)
    print(' --threads   : ', args.threads)
    print(' --model     : ', args.model)

    app(args.image_dir, args.threads, args.model)


if __name__ == '__main__':
    main()