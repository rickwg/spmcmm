import numpy as np


class SimpleDataImporter:
    """
    Simple class which import csv data with the :class:`numpy` function :func:`numpy.loadtxt`.
    """

    def __init__(self, i_file_name, i_delimiter=' '):
        self._data = np.array([])
        self.read_table(i_file_name, i_delimiter)

    # ---------------------------------------------------------------------------------------------#
    def read_table(self, i_file_name, i_delimiter=' '):
        """
        Read a file.

        :param i_file_name: A filename of the file.
        :type i_file_name: string
        """
        try:
            self._data = np.loadtxt(i_file_name, delimiter=i_delimiter)
            # reshape for k-means
            if 1 == len(self._data.shape):
                self._data = self._data.reshape(self._data.size, 1)

        except IOError as ioErr:
            print 'cannot open file: {}'.format(ioErr, i_file_name)

    def get_data(self):
        """
        Return the loaded csv data.

        :returns: Return a numpy array.
        :rtype: numpy.array()
        """
        return self._data
