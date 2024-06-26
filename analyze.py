#!/usr/bin/env python3

from os.path import isfile
import pickle
import sys

from sklearn.metrics import roc_auc_score

# Concordance is equivalent to AUROC assuming binary outcomes, see
# https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/1471-2288-12-82

try:
    filename = sys.argv[1]
except:
    print('Usage: %s results-filename [--dist] [--test] [--export] [--force-kcf] [--cyp/noncyp]' % sys.argv[0])
    exit()
reportDistribution = '--dist' in sys.argv
testMode = '--test' in sys.argv
considerCyp = '--cyp' in sys.argv
considerNonCyp = '--noncyp' in sys.argv

print(' '.join(sys.argv))

useDatasetWithKcfTypes = True
if '-nokcf' in filename and not '--force-kcf' in sys.argv:
    useDatasetWithKcfTypes = False

if useDatasetWithKcfTypes:
    with open('data/dataset2.pkl', 'rb') as f:
        dataset = pickle.load(f)
else:
    with open('data/dataset-nokcf.pkl', 'rb') as f:
        dataset = pickle.load(f)

with open(filename, 'rb') as f:
    results = pickle.load(f)

dataType = results['type']
assert dataType in ['node', 'edge']
split = results['split']

if testMode:
    molecules = [dataset['mols'][i] for i in dataset['splits'][split]['test']]
    predictions = results['test']
else:
    molecules = [dataset['mols'][i] for i in dataset['splits'][split]['valid']]
    predictions = results['valid']

modelConsidersCyp = '-cyp' in filename
modelConsidersNonCyp = '-noncyp' in filename
expectedMolIndices = []
if modelConsidersCyp != modelConsidersNonCyp:
    if modelConsidersCyp:
        expectedMolIndices = [i for i in range(len(molecules)) if molecules[i]['isCyp']]
    elif modelConsidersNonCyp:
        expectedMolIndices = [i for i in range(len(molecules)) if molecules[i]['isNonCyp']]
else:
    expectedMolIndices = range(len(molecules))
del results

if set(expectedMolIndices) != set(predictions.keys()):
    print('Molecules in predictions (%d) do not match the expected molecules (%d)' % (len(predictions), len(expectedMolIndices)))
    print(sorted(expectedMolIndices)[0:20])
    print(sorted(set(predictions.keys()))[0:20])
    exit()

def expectedRPrecision(relevant, total):
    retrieved = 0 # expected number of ones picked
    for i in range(relevant): # ...over R attempts
        retrieved += (relevant - i) / (total - i)
    return retrieved / relevant

def expectedTopTwoCorrectness(relevant, total):
    top = 2
    retrieved = 0 # expected number of ones picked
    for i in range(top): # ...over the top N attempts
        retrieved += (relevant - i) / (total - i)
    return retrieved / top

molRPrec = 0
molRPrecExpect = 0
molPrecAtK = [0, 0, 0] # indices represent (k - 1)
molPrecAtKCount = [0, 0, 0]
molAuroc = 0
molAurocCount = 0
# molAurocExpect = 0.5 https://stats.stackexchange.com/questions/346184/what-is-the-expected-value-of-aurocc-for-random-predictions, see "3.1. Random performance" in fawcett2006
molTopTwoCorrect = 0
molTopTwoCorrectExpect = 0

allTrues = []
allPreds = []

positiveScores = [] if reportDistribution else None
negativeScores = [] if reportDistribution else None

print('Processing molecules (%d total)...' % len(molecules))
numMolecules = 0
numAtoms = 0
numBonds = 0
for datasetIndex, pred in predictions.items():
    mol = molecules[datasetIndex]

    if considerCyp != considerNonCyp:
        if considerCyp and not mol['isCyp']:
            continue
        elif considerNonCyp and not mol['isNonCyp']:
            continue

    if dataType == 'edge':
        true = [0] * len(mol[dataType]['edges'])
        for i in mol[dataType]['posEdges']:
            true[i] = 1
    else:
        true = mol[dataType]['y']

    assert all(y == 0 or y == 1 for y in true)

    allTrues.extend(true)
    allPreds.extend(pred)

    predIndices = sorted(range(len(pred)), key=lambda i: pred[i], reverse=True)

    total = len(true)
    relevant = sum(true)
    retrieved = sum(true[i] for i in predIndices[0:relevant])
    molRPrec += retrieved / relevant
    molRPrecExpect += expectedRPrecision(relevant, total)

    retrieved = 0
    for k in range(len(molPrecAtK)):
        if k >= relevant:
            break
        if true[predIndices[k]]:
            retrieved += 1
        molPrecAtK[k] += retrieved / (k + 1)
        molPrecAtKCount[k] += 1

    try:
        molAuroc += roc_auc_score(true, pred)
        molAurocCount += 1
    except ValueError:
        pass

    if true[predIndices[0]] == 1 or (total >= 2 and true[predIndices[1]] == 1):
        molTopTwoCorrect += 1
    molTopTwoCorrectExpect += expectedTopTwoCorrectness(relevant, total)

    numMolecules += 1
    numAtoms += len(mol[dataType]['x'])
    numBonds += len(mol[dataType]['edges'])

    if positiveScores is not None:
        positiveScores.extend(pred[i] for i in range(len(pred)) if true[i])
        negativeScores.extend(pred[i] for i in range(len(pred)) if not true[i])

    if numMolecules != 0 and numMolecules % 1000 == 0:
        print('   %d done' % numMolecules)

molRPrec /= numMolecules
molRPrecExpect /= numMolecules
molAuroc /= molAurocCount
print(' * 01 Molecular R-Precision: %f' % molRPrec)
for k in range(len(molPrecAtK)):
    molPrecAtK[k] /= molPrecAtKCount[k]
    print(' * 02 Molecular Precision @ %d: %f' % (k + 1, molPrecAtK[k]))
print(' * 03 Molecular AUROC: %f' % molAuroc)
print(' * 04 Expected Molecular R-Precision: %f' % molRPrecExpect)
print(' * 05 Top-2 Correctness Rate: %f' % (molTopTwoCorrect / numMolecules))
print(' * 06 Expected Top-2 Correctness Rate: %f' % (molTopTwoCorrectExpect / numMolecules))

if not reportDistribution:
    predIndices = sorted(range(len(allPreds)), key=lambda i: allPreds[i], reverse=True)

    total = len(allTrues)
    relevant = sum(allTrues)
    retrieved = sum(allTrues[i] for i in predIndices[0:relevant])
    print(' * 11 Atomic R-Precision: %f' % (retrieved / relevant))
    atomicAuroc = roc_auc_score(allTrues, allPreds)
    print(' * 13 Atomic AUROC: %f' % atomicAuroc)
    print(' * 14 Expected Atomic R-Precision: %f' % expectedRPrecision(relevant, total))

    print(' * 91 Molecules: %d' % numMolecules)
    print(' * 92 Atoms: %d' % numAtoms)
    print(' * 93 Bonds: %d' % numBonds)
    print(' * 94 Candidates: %d' % total)
    print(' * 95 Positives: %d' % relevant)
    print()

if positiveScores is not None:
    print('Saving score distribution for true SOMs...')

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 1, sharex=True, tight_layout=True)

    setName = filename.rsplit('/', 1)[1].rsplit('.', 1)[0]
    fig.suptitle('Distribution of scores in ' + setName)

    axs[0].set_title('True SOMs')
    axs[0].hist(positiveScores, bins=20)

    axs[1].set_title('Non-SOMs')
    axs[1].hist(negativeScores, bins=20)

    axs[1].set_ylabel('Number of atoms')
    axs[1].set_xlabel('Score given by model')

    plt.savefig(setName + '.score-dist.png')
