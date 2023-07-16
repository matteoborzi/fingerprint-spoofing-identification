from itertools import chain, combinations
from mlpr import mcol
import numpy
import json
import os

test = ''
LOG = True


def get_full_path(filename: str) -> str:
    root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(root, filename)


IMAGE_PATH = get_full_path('images')


def powerset(iterable, start):
    s = list(iterable)
    return chain.from_iterable(combinations(s,r) for r in range(start, len(s)+1))


def log_test(s: str, end='', flush=True):
    if LOG:
        print(s, end=end, flush=flush)


def load(filename: str) -> tuple:
    """ This function loads a dataset with m features and n samples.
        Returns:
            * the samples array with shape (m,n)
            * labels 1-D integer array
    """
    samples = []
    labels = []

    filename = get_full_path(filename)
    with open(filename, 'r') as f:
        for line in f:
            # Protect the execution when the file contains an empty line at the end
            try:
                line = line.split(',')
                samples.append(mcol(numpy.array([float(i) for i in line[:-1]], dtype=numpy.float64)))
                labels.append(int(line[-1]))
            except:
                pass
    return numpy.hstack(samples), numpy.array(labels, dtype=numpy.int32)


def json_dump(data: dict, filename: str) -> None:
    with open(f'{get_full_path("data")}/{filename}.json', 'w') as f:
        json.dump(data, f, indent=4)


def json_load(filename: str) -> dict:
    with open(f'{get_full_path("data")}/{filename}.json','r') as f:
        data = json.load(f)
    return data


def save_table(table, filename: str) -> None:
    with open(f'{get_full_path("logs")}/{filename}.log','w') as f:
        f.write(str(table).replace('λ','l').replace('π','p'))


def clean_filename(filename: str) -> str:
    return filename.replace(" ", "_").replace("(","").replace(")","").replace("-","").replace("+","").replace("__","_")