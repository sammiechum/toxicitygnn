#!/usr/bin/env python3

import datetime
from os.path import isfile
import signal
from socket import gethostname
import sys
import time

from som.common import Task

queued = []
for split in range(1):
    for options in [('--cyp', '--epochs=200'), ('--noncyp', '--epochs=300'), ('--epochs=400',)]:
        for d in range(1):
            for w in [512]:
                queued.append(Task('./gnn-node.py', '--split=%d' % split, '--width=%d' % w, '--depth=%d' % d, '--conv=cheb1k', '--save', *options))
                queued.append(Task('./gnn-node.py', '--split=%d' % split, '--width=%d' % w, '--depth=%d' % d, '--conv=cheb', '--save', *options))
                queued.append(Task('./gnn-node.py', '--split=%d' % split, '--width=%d' % w, '--depth=%d' % d, '--conv=cheb10k', '--save', *options))
                queued.append(Task('./gnn-node.py', '--split=%d' % split, '--width=%d' % w, '--depth=%d' % d, '--conv=cheb15k', '--save', *options))
                queued.append(Task('./gnn-node.py', '--split=%d' % split, '--width=%d' % w, '--depth=%d' % d, '--conv=cheb1k', '--nokcf'))
                queued.append(Task('./gnn-node.py', '--split=%d' % split, '--width=%d' % w, '--depth=%d' % d, '--conv=cheb', '--nokcf'))
                queued.append(Task('./gnn-node.py', '--split=%d' % split, '--width=%d' % w, '--depth=%d' % d, '--conv=cheb10k', '--nokcf'))
                queued.append(Task('./gnn-node.py', '--split=%d' % split, '--width=%d' % w, '--depth=%d' % d, '--conv=cheb15k', '--nokcf'))
                queued.append(Task('./gnn-node.py', '--split=%d' % split,  '--width=%d' % w, '--depth=%d' % d, '--conv=mf0', *options))
                queued.append(Task('./gnn-node.py', '--split=%d' % split,  '--width=%d' % w, '--depth=%d' % d, '--conv=mf5', *options))
                queued.append(Task('./gnn-node.py', '--split=%d' % split,  '--width=%d' % w, '--depth=%d' % d, '--conv=mf10', *options))
                queued.append(Task('./gnn-node.py', '--split=%d' % split,  '--width=%d' % w, '--depth=%d' % d, '--conv=gin', *options))
                queued.append(Task('./gnn-node.py', '--split=%d' % split,  '--width=%d' % w, '--depth=%d' % d, '--conv=gcn', *options))

    # options = ('--epochs=400',)
    # queued.append(Task('./gnn-edge.py', '--split=%d' % split, '--gnn-width=64', '--gnn-depth=%d' % d, '--cls-width=128', '--cls-depth=1', *options))
    # queued.append(Task('./gnn-edge.py', '--split=%d' % split, '--gnn-width=64', '--gnn-depth=%d' % d, '--cls-width=256', '--cls-depth=1', *options))
    # queued.append(Task('./gnn-edge.py', '--split=%d' % split, '--gnn-width=64', '--gnn-depth=%d' % d, '--cls-width=128', '--cls-depth=2', *options))
    # queued.append(Task('./gnn-edge.py', '--split=%d' % split, '--gnn-width=64', '--gnn-depth=%d' % d, '--cls-width=256', '--cls-depth=2', *options))
    # queued.append(Task('./gnn-edge.py', '--split=%d' % split, '--gnn-width=128', '--gnn-depth=%d' % d, '--cls-width=256', '--cls-depth=1', *options))
    # queued.append(Task('./gnn-edge.py', '--split=%d' % split, '--gnn-width=256', '--gnn-depth=%d' % d, '--cls-width=256', '--cls-depth=1', *options))
    # queued.append(Task('./gnn-edge.py', '--split=%d' % split, '--gnn-width=256', '--gnn-depth=%d' % d, '--cls-width=256', '--cls-depth=1', '--nokcf', *options))
    # queued.append(Task('./gnn-edge.py', '--split=%d' % split, '--gnn-width=256', '--gnn-depth=%d' % d, '--cls-width=256', '--cls-depth=1', '--gnn-conv=cheb10k', *options))
    # queued.append(Task('./gnn-edge.py', '--split=%d' % split, '--gnn-width=512', '--gnn-depth=%d' % d, '--cls-width=256', '--cls-depth=1', '--gnn-conv=cheb10k', *options))
    # queued.append(Task('./gnn-edge.py', '--split=%d' % split, '--gnn-width=512', '--gnn-depth=%d' % d, '--cls-width=256', '--cls-depth=1', *options))


# for split in range(1):
#     options = ('--epochs=400',)
#     for d in [1]: # todo, put this in above, with d = 5
#         queued.append(Task('./gnn-edge.py', '--split=%d' % split, '--gnn-width=64', '--gnn-depth=%d' % d, '--cls-width=128', '--cls-depth=1', *options))
#         queued.append(Task('./gnn-edge.py', '--split=%d' % split, '--gnn-width=64', '--gnn-depth=%d' % d, '--cls-width=256', '--cls-depth=1', *options))
#         queued.append(Task('./gnn-edge.py', '--split=%d' % split, '--gnn-width=64', '--gnn-depth=%d' % d, '--cls-width=128', '--cls-depth=2', *options))
#         queued.append(Task('./gnn-edge.py', '--split=%d' % split, '--gnn-width=64', '--gnn-depth=%d' % d, '--cls-width=256', '--cls-depth=2', *options))
#         queued.append(Task('./gnn-edge.py', '--split=%d' % split, '--gnn-width=128', '--gnn-depth=%d' % d, '--cls-width=256', '--cls-depth=1', *options))
#         queued.append(Task('./gnn-edge.py', '--split=%d' % split, '--gnn-width=256', '--gnn-depth=%d' % d, '--cls-width=256', '--cls-depth=1', *options))
#         queued.append(Task('./gnn-edge.py', '--split=%d' % split, '--gnn-width=256', '--gnn-depth=%d' % d, '--cls-width=256', '--cls-depth=1', '--nokcf', *options))
#         queued.append(Task('./gnn-edge.py', '--split=%d' % split, '--gnn-width=256', '--gnn-depth=%d' % d, '--cls-width=256', '--cls-depth=1', '--gnn-conv=cheb10k', *options))
#         queued.append(Task('./gnn-edge.py', '--split=%d' % split, '--gnn-width=256', '--gnn-depth=%d' % d, '--cls-width=256', '--cls-depth=1', '--gnn-conv=cheb15k', *options))
#         queued.append(Task('./gnn-edge.py', '--split=%d' % split, '--gnn-width=512', '--gnn-depth=%d' % d, '--cls-width=256', '--cls-depth=1', '--gnn-conv=cheb10k', *options))
#         queued.append(Task('./gnn-edge.py', '--split=%d' % split, '--gnn-width=512', '--gnn-depth=%d' % d, '--cls-width=256', '--cls-depth=1', '--gnn-conv=cheb15k', *options))
#         queued.append(Task('./gnn-edge.py', '--split=%d' % split, '--gnn-width=512', '--gnn-depth=%d' % d, '--cls-width=256', '--cls-depth=1', *options))


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
    with open('/Users/sammiechum/Downloads/gnntox/data/vp-gpus.txt', 'w') as f:
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
        task.start('--gpu=%d' % freeGpu)

status('Done')
