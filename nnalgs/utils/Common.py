import os
import json
from pathlib import Path


def walk_dir(dirname, pattern):
    files = []
    for base, _, name in os.walk(dirname):
        if pattern in base:
            files += [os.path.join(base, name_i) for name_i in name]
    return sorted(files)


def read_json(filename):
    filename = Path(filename)
    with filename.open('rt') as handle:
        return json.load(handle)


def write_json(content, filename):
    filename = Path(filename)
    with filename.open('wt') as handle:
        json.dump(content, handle, indent=2)
