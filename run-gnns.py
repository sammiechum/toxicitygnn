#!/usr/bin/env python3

import datetime
from os.path import isfile
import signal
from socket import gethostname
import sys
import time
import argparse

from som.common import Task

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, 
                    help="Dataset file name to process")
parser.add_argument('output', type=str,
                    help='Output subfolder')
#parser.add_argument('grid_search', type=int,
#                    help='Which grid search to run (1=GNN size, 2=GNN Conv, 3=Over splits)')
parser.add_argument('-d', '--depth', type=int, required=False, default=None,
                    help="Depth of GNN to use")
parser.add_argument('-w', '--width', type=int, required=False, default=None,
                    help="Width of GNN to use")
parser.add_argument('-c', '--convolution', type=str, required=False, default=None,
                    help="Convolution operation to use")
parser.add_argument('-p', '--pooling', type=str, required=False, default=None,
                    help="Pooling operation to use")
parser.add_argument('-e', '--epochs', type=int, required=False, default=None,
                    help="Number of epochs to run")
parser.add_argument('-s', '--splits', type=int, required=False, default=None,
                    help="Number of splits to use")

args = parser.parse_args()

data_path = args.dataset
output_folder_path = args.output
#which_gridSearch = args.grid_search

if args.depth is None: real_depths = [1,2,3,4]
else: real_depths = [args.depth]

if args.width is None: real_widths = [32,64,128,256,512]
else: real_widths = [args.width]

if args.convolution is None: real_convolutions = ['gat', 'resgat', 'transform', 'gine', 'crystal', 'gen', 'path', 'general']
else: real_convolutions = [args.convolution]

if args.pooling is None: real_poolings = ['mean']
else: real_poolings = [args.pooling]

if args.epochs is None: real_epochs = 200
else: real_epochs = args.epochs

if args.splits is None: real_splits = 3
else: real_splits = args.splits

options_dict = {
    'depth': real_depths,
    'width': real_widths,
    'convolution': real_convolutions,
    'pooling': real_poolings,
    'num_epochs': real_epochs,
    'num_splits': real_splits
}

queued = []

for split in range(options_dict['num_splits']):
    for options in [(f'--epochs={options_dict["num_epochs"]}',)]:
        for d in options_dict['depth']:
            for w in options_dict['width']:
                for conv in options_dict["convolution"]:
                    for pool in options_dict["pooling"]:
                        queued.append(Task('./gnn-node.py', f'--data={data_path}', f'--split={split}', f'--width={w}', f'--depth={d}', f'--conv={conv}', f'--pool={pool}', *options))
                        

'''
if which_gridSearch == 1: # Grid Search over GNN depths and widths with Convolution/Pooling fixed
    for split in [0]:
        for options in [('--epochs=200',)]:
            for d in [1,2,3]:
                for w in [32,64,128,256]:
                    queued.append(Task('./gnn-node.py', f'--data={data_path}', f'--split={split}', f'--width={w}', f'--depth={d}', f'--conv={options_dict["convolution"]}', f'--pool={options_dict["pooling"]}', *options))
                    
if which_gridSearch == 2: # Grid Search over GNN depths and widths with Convolution/Pooling fixed
    for split in [0]:
        for options in [('--epochs=200',)]:
            for d in [options_dict['depth']]:
                for w in [options_dict['width']]:
                    for conv in ['cheb1k', 'cheb', 'cheb10k', 'cheb15k']:
                        for pool in ['mean', 'max', 'add']:
                            queued.append(Task('./gnn-node.py', f'--data={data_path}', f'--split={split}', f'--width={w}', f'--depth={d}', f'--conv={conv}', f'--pool={pool}', *options))
                            
if which_gridSearch == 3: # Grid Search over dataset splits
    for split in range(10):
        for options in [('--epochs=200',)]:
            for d in [options_dict['depth']]:
                for w in [options_dict['width']]:
                    for conv in [options_dict["convolution"]]:
                        for pool in [options_dict["pooling"]]:
                            queued.append(Task('./gnn-node.py', f'--data={data_path}', f'--split={split}', f'--width={w}', f'--depth={d}', f'--conv={conv}', f'--pool={pool}', *options))
                            
'''
def status(message=''):
    print('GNNs     ', datetime.datetime.now(), message, flush=True)

try:
    import torch_geometric
except:
    status('Pytorch Geometric is not available in this environment')
    exit()


AVAILABLE_GPUS_FILENAME = 'vp-gpus.txt'

def getInstalledGpus():
    if gethostname() == 'ramallah':
        return set(range(6))
    elif gethostname() == 'hebron':
        return set(range(2))
    else:
        return set([0])

def loadAvailableGpus(currentGpus=None):
    installed = getInstalledGpus()
    if isfile(AVAILABLE_GPUS_FILENAME):
        try:
            gpus = set()
            with open(AVAILABLE_GPUS_FILENAME, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    else:
                        gpus.update(set(int(gpu.strip()) for gpu in line.split(',')))
            return gpus & installed
        except Exception as e:
            gpus = currentGpus if currentGpus is not None else installed
            try:
                with open(AVAILABLE_GPUS_FILENAME, 'w') as f:
                    f.write(','.join('%d' % gpu for gpu in sorted(gpus)))
                    f.write('\n# This is a comment describing the error that occurred previously:')
                    f.write('\n# %s %s\n' % (datetime.datetime.now(), str(e).replace('\n', '')))
            except:
                pass
            return gpus
    else:
        return installed
def saveAvailableGpus(gpus):
    with open('data/vp-gpus.txt', 'w') as f:
        f.write(','.join('%d' % gpu for gpu in sorted(gpus)))

gpus = loadAvailableGpus()
for arg in sys.argv:
    if arg.startswith('--gpus'):
        if arg == '--gpus=all':
            gpus = getInstalledGpus()
        else:
            gpus = set(int(gpu) for gpu in arg.split('=', 1)[1].split(','))
saveAvailableGpus(gpus)
status()
status('Started fresh run with GPUs ' + ', '.join('%d' % gpu for gpu in sorted(gpus)))
running = {gpu: None for gpu in gpus} # gpu index: Task


def shutdown(sig, frame):
    status('Shutting down...')
    for task in running.values():
        if task is not None and not task.done():
            task.terminate()
            status('Terminated task %s' % task)
signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)


while queued or any(task is not None and not task.done() for task in running.values()):
    runningGpus = set(running.keys())
    availableGpus = loadAvailableGpus(runningGpus)
    for removedGpu in (runningGpus - availableGpus):
        status('Removed GPU %d' % removedGpu)
        task = running[removedGpu]
        task.terminate()
        status('Terminated task %s' % task)
        del running[removedGpu]
        queued.append(task)
    for addedGpu in (availableGpus - runningGpus):
        status('Added GPU %d' % addedGpu)
        running[addedGpu] = None

    if queued:
        freeGpu = None
        for gpu, task in running.items():
            if task is None or task.done():
                freeGpu = gpu
                break
        if freeGpu is None:
            time.sleep(1)
            continue

        task = queued.pop(0)
        running[freeGpu] = task
        status('Started task %s on GPU %d' % (task, freeGpu))
        task.start(output_folder_path, f'--gpu={freeGpu}')

status('Done')
