import os
import sys
from libpagexml3.libpagexml3 import PageXMLParser
from pprint import pprint
from time import time

DATA_FILE = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "test", "data", "tacicc_BV046303055_0061.xml",
    )
)


def test_load_example_xml():
    parser = PageXMLParser.from_file(DATA_FILE)
    page = parser.get_page()
    #assert False


if __name__=="__main__":
    test_load_example_xml()
