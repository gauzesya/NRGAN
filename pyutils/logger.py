# -*- coding: utf-8 -*-

class logger:

    def __init__(self, log_filename):
        self._f = open(log_filename, 'w')

    def __call__(self, log, end='\n', only_file=False):
        if only_file is False:
            print(log, end=end)
        self._f.write(str(log)+end)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._f.close()


if __name__=='__main__':
    with logger('test.txt') as printl:
        printl('test')
        printl([1,2,3])
