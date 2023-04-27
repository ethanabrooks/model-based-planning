import collections
import os
import pickle

from rich.console import Console

console = Console()


class Config(collections.Mapping):
    def __init__(self, _class, verbose=True, savepath=None, **kwargs):
        self._class = _class
        self._dict = {}

        for key, val in kwargs.items():
            self._dict[key] = val

        if verbose:
            console.print(self)

        if savepath is not None:
            savepath = os.path.join(*savepath) if type(savepath) is tuple else savepath
            pickle.dump(self, open(savepath, "wb"))
            print(f"Saved config to: {savepath}\n")
        self.savepath = savepath

    def __rich_repr__(self):
        yield self._class
        for key in sorted(self._dict.keys()):
            val = self._dict[key]
            yield key, val

    def __iter__(self):
        return iter(self._dict)

    def __getitem__(self, item):
        return self._dict[item]

    def __len__(self):
        return len(self._dict)

    def __call__(self):
        return self.make()

    def __getattr__(self, attr):
        if attr == "_dict" and "_dict" not in vars(self):
            self._dict = {}
        try:
            return self._dict[attr]
        except KeyError:
            raise AttributeError(attr)

    def make(self):
        if "GPT" in str(self._class) or "Trainer" in str(self._class):
            ## GPT class expects the config as the sole input
            return self._class(self)
        else:
            return self._class(**self._dict)
