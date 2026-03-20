import os
import h5py
from contextlib import contextmanager


class HDF5File:
    def __init__(self, file_name):
        self.file_name = file_name
        self._file = None
        self._count = 0

        if not os.path.exists(self.file_name):
            with h5py.File(self.file_name, 'a'): pass

    def __enter__(self):
        if self._file is None: h5py.File(self.file_name, 'a')
        self._count += 1
        return self

    def __exit__(self, *args):
        self._count -= 1
        if self._file and self._count == 0:
            self._file.close()
            self._file = None

    @contextmanager
    def _get_file(self):
        if self._file:
            yield self._file
        else:
            file = h5py.File(self.file_name, 'a')
            try:
                yield file
            finally:
                file.close()

    def is_path(self, path):
        with self._get_file() as file:
            return path in file

    def create_group(self, path):
        with self._get_file() as file:
            if path in file: del file[path]
            file.create_group(path)

    def create_dataset(self, path, shape):
        with self._get_file()as file:
            if path in file: del file[path]
            file.create_dataset(path, shape=shape, maxshape=(None,) * len(shape), compression='gzip', dtype='f4')

    def resize(self, path, shape):
        with self._get_file() as file:
            file[path].resize(shape)

    def copy_metadata(self, source, target):
        with self._get_file() as file:
            for key, value in file[source].attrs.items():
                file[target].attrs[key] = value

    def save_metadata(self, path, **kwargs):
        with self._get_file() as file:
            for key, value in kwargs.items():
                file[path].attrs[key] = value

    def load_metadata(self, path, keys):
        if isinstance(keys, str): keys = (keys,)
        with self._get_file() as file:
            return file[path].attrs[keys[0]] if len(keys) == 1 else tuple(file[path].attrs[key] for key in keys)

    def save(self, path, index, data):
        with self._get_file() as file:
            file[path][index] = data

    def load(self, path, index):
        with self._get_file() as file:
            return file[path][index]
