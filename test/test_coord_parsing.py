valid_cases = [
    "1,2",
    "0,0",
    "3.5,4.7",
    "-1,2 -3,4",
    "10,-20 30.5,40.25 -5.5,-6.6",
    "123456,987654",
    "0.0001,0.9999 -9999.9999,8888.0001",
    "-0.0,0.0",
    "1,2 3.0,4.0 5,6.000 7.000,8",
    "42.42,24.24 -0.001,0.001 100,-100",
    "1,2   3,4"   ,       # multiple spaces (depends if allowed)
]

invalid_cases = [
    "1.2.3,4",            # too many decimals
    "1,2,3"   ,           # too many numbers per point
    "1;2 3;4"  ,          # wrong separator
    "(1,2) (3,4)" ,       # parentheses not allowed
    "1,2, 3,4"   ,        # extra comma spacing
    "1 2,3 4"   ,         # missing comma
    ",1,2"    ,           # leading comma
    "1,2 "    ,           # trailing space may be tricky
    "abc,1 2,3"   ,       # non-numeric text
    "1,,2 3,4"   ,        # double comma
    "1. ,2"      ,        # stray decimal point
    "-.5,2"      ,        # missing leading zero in float
    "1,2- 3,4"   ,        # weird hyphen placement
    "1.0.0,2"   ,         # malformed float
    ""          ,         # empty string
    " "         ,         # just a space
]

from libpagexml3.libpagexml3 import PageXMLParser
import numpy as np


def parser1(coords: str):
    points = []

    for match in PageXMLParser._coords_regex.finditer(coords):
        x, y = int(match.group(1)), int(match.group(3))

        points.append((x, y))
    
    return np.array(points)

def parser2(coords: str):
    points = []
    for m in coords.split():
        if not m: continue
        if "-" in m: continue
        x,y = m.split(",")
        points.append((int(float(x)), int(float(y))))
    # above alternative split implementation is faster, but will
    return np.array(points)

import pytest

def test_parsers():
    for x in valid_cases:
        assert np.all(parser1(x) == parser2(x)), f"{x} => {parser1(x)} <-> {parser2(x)}"

    for y in invalid_cases:
        parser1_raises = False
        parser2_raises = False
        try:
            parser1(y)
        except Exception:
            parser1_raises = True
        
        try:
            parser2(y)
        except Exception:
            parser2_raises = True
        
        assert parser1_raises == parser2_raises
        assert parser1_raises

