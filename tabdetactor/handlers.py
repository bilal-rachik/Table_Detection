import os
import sys


from .core import TableList
from .parsers import Stream, Lattice



class PDFHandler(object):
    """Handles all operations like temp directory creation, splitting
    file into single page PDFs, parsing each PDF and then removing the
    temp directory.
    Parameters
    ----------
    filepath : str
        Filepath or URL of the PDF file.
    pages : str, optional (default: '1')
        Comma-separated page numbers.
        Example: '1,3,4' or '1,4-end' or 'all'.
    password : str, optional (default: None)
        Password for decryption.
    """

    def __init__(self, filepath):
        self.filepath = filepath
        #if not filepath.lower().endswith(".pd"):
        #    raise NotImplementedError("File format not supported")
    def parse(
        self, flavor="lattice", suppress_stdout=False, layout_kwargs={}, **kwargs
    ):
        """Extracts tables by calling parser.get_tables on all single
        page PDFs.
        Parameters
        ----------
        flavor : str (default: 'lattice')
            The parsing method to use ('lattice' or 'stream').
            Lattice is used by default.
        suppress_stdout : str (default: False)
            Suppress logs and warnings.
        layout_kwargs : dict, optional (default: {})
            A dict of `pdfminer.layout.LAParams <https://github.com/euske/pdfminer/blob/master/pdfminer/layout.py#L33>`_ kwargs.
        kwargs : dict
            See camelot.read_pdf kwargs.
        Returns
        -------
        tables : camelot.core.TableList
            List of tables found in PDF.
        """
        tables = []
        parser = Lattice(**kwargs) if flavor == "lattice" else Stream(**kwargs)
        t = parser.extract_tables(
                    self.filepath, suppress_stdout=suppress_stdout, layout_kwargs=layout_kwargs
                )
        tables.extend(t)
        return TableList(sorted(tables))