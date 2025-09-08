import sys
from libpagexml3.libpagexml3 import PageXMLParser
from pprint import pprint
from time import time
def test_load_example_xml():
    n = time()
    for i in range(100):
        
        parser = PageXMLParser.from_file("./test/data/tacicc_BV046303055_0061.xml")
        page = parser.get_page()
    print(time() - n)
    #assert False


if __name__=="__main__":
    test_load_example_xml()
