# Library to read and write PageXML files

## Basic Usage

**Loading a Document**
```
from libpagexml3.libpagexml3 import PageXMLParser, PageXMLParserMode, PageXMLPage

def load_pxml(filename: str) -> PageXMLPage:
    page = PageXMLParser.from_file(filename, strictness = PageXMLParserMode.EASY).get_page()
    return page
```

**Writing PageXML files**
```
from libpagexml3.libpagexml3 import PageXMLBuilder, PageXMLPage

def write_pxml(page: PageXMLPage) -> str:
    pstr = PageXMLBuilder.build_page(page=page, creator="DemoScript")
    return pstr
```
