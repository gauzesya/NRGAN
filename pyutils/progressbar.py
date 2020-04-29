"""Main module."""
import time
import shutil


class progressbar(object):

    def __init__(self, obj, width_proportion=1, decostr=None):
        assert(hasattr(obj, '__iter__'))
        self._obj = list(obj)
        assert(width_proportion <= 1 and width_proportion > 0)
        self._width_proportion = width_proportion
        if decostr is not None:
            assert(type(decostr) is str)
        self._decostr = decostr
        self._i = 0
        self._len = len(self._obj)


    def __iter__(self):
        return self

    def __next__(self):
        if self._i == self._len:
            print()
            raise StopIteration()

        return_value = self._obj[self._i]
        self._i += 1

        # print progress bar
        if self._decostr is not None:
            bar_length = int((shutil.get_terminal_size()[0] - 9 - len(self._decostr)) * self._width_proportion)
        else:
            bar_length = int((shutil.get_terminal_size()[0] - 8) * self._width_proportion)

        per = self._i / self._len
        progress = int(per*bar_length)
        bar = "#" * progress + "." * (bar_length-progress)
        if self._decostr is not None:
            print('\r' + self._decostr + ' [{0}] {1:03d}%'.format(bar, int(per*100)), end="")
        else:
            print('\r[{0}] {1:03d}%'.format(bar, int(per*100)), end="")
        return return_value


# test
if __name__=='__main__':
    for i in progressbar(zip(range(100), range(100)), decostr='test'):
        time.sleep(0.1)
