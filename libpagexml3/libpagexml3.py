import abc
import dataclasses
import enum
import functools
import itertools
import math
import re
import sys
import xml.etree.ElementTree as ET
import xml.sax.saxutils
from collections import namedtuple, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict
import datetime

import numpy as np
from loguru import logger


def rotate_points(points: np.ndarray, tx:float, ty:float, cos_phi:float, sin_phi:float, pbx: float ,pby: float):
    if len(points.shape) < 2:
        return None

    points = np.array(points, dtype=np.float)
    points[:,0] -= float(tx)
    points[:,1] -= float(ty)
    new_points = np.zeros_like(points,dtype=np.float)
    new_points[:,0] = points[:,0] * cos_phi - points[:,1] * sin_phi + pbx
    new_points[:,1] = points[:,0] * sin_phi + points[:,1] * cos_phi + pby
    return new_points


@enum.unique
class PageRegionTypes(enum.Enum):
    IMAGE_REGION = 'ImageRegion'
    TEXT_REGION = 'TextRegion'
    GRAPHIC_REGION = 'GraphicRegion'
    SEPARATOR_REGION = 'SeparatorRegion'
    TABLE_REGION = 'TableRegion'
    MUSIC_REGION = 'MusicRegion'
    NOISE_REGION = 'NoiseRegion'
    BORDER_REGION = 'Border'
    MATHS_REGION = 'MathsRegion'
    UNKNOWN_REGION = 'UnknownRegion'


    @staticmethod
    def ignored_regions():
        return {"PrintSpace", "Border"}

    @staticmethod
    def should_ignore(region_str):
        return region_str in PageRegionTypes.ignored_regions()

    def is_text(self):
        return self == PageRegionTypes.TEXT_REGION

    @classmethod
    def supported_regions(cls):
        return set(e.value for e in cls)


class PageXMLTextTypes(enum.Enum):
    PARAGRAPH = 'paragraph'
    HEADING = 'heading'
    HEADER = 'header'
    CATCH_WORD = 'catch_word'
    PAGE_NUMBER = 'page_number'
    SIGNATURE_MARK = 'signature_mark'
    MARGINALIA = 'marginalia'
    OTHER = 'other'
    DROP_CAPITAL = 'drop_capital'
    CAPITAL_WORD = 'capital_word'
    FOLD_COUNT = 'fold_count'
    FLOATING = 'floating'
    CAPTION = 'caption'
    CITATION = 'citation'
    LINE_NUMBER = 'line_number'
    ENDNOTE = 'endnote'
    IGNORE = 'ignore'
    TOCENTRY = 'toc_entry'
    FOOTNOTE = 'footnote'
    FOOTNOTE_CONTINUED = 'footnote_continued'
    FOOTER = 'footer'
    COLUMN = 'column'
    CREDIT = 'credit'
    EMPTY = ''

    def color(self):
        return {
            PageXMLTextTypes.PARAGRAPH: (255, 0, 0),
            PageXMLTextTypes.HEADING: (0, 0, 255),
            PageXMLTextTypes.HEADER: (0, 255, 255),
            PageXMLTextTypes.CATCH_WORD: (255, 255, 0),
            PageXMLTextTypes.PAGE_NUMBER: (255, 0, 255),
            PageXMLTextTypes.SIGNATURE_MARK: (128, 0, 128),
            PageXMLTextTypes.MARGINALIA: (128, 128, 0),
            PageXMLTextTypes.OTHER: (0, 128, 128),
            PageXMLTextTypes.DROP_CAPITAL: (255, 128, 0),
            PageXMLTextTypes.FLOATING: (255, 0, 128),
            PageXMLTextTypes.CAPTION: (128, 255, 0),
            PageXMLTextTypes.ENDNOTE: (0, 255, 128),
            PageXMLTextTypes.IGNORE: (0, 128, 0),
            PageXMLTextTypes.TOCENTRY: (0, 127, 0),
            PageXMLTextTypes.FOOTNOTE: (0, 126, 0),
            PageXMLTextTypes.FOOTNOTE_CONTINUED: (0, 125, 0),
            PageXMLTextTypes.FOOTER: (0, 123, 0),
            PageXMLTextTypes.CREDIT: (0, 133, 0),

        }[self]

    @staticmethod
    def text_types():
        return [x for x in PageXMLTextTypes if x not in PageXMLTextTypes.image_types()]

    @staticmethod
    def image_types():
        return {PageXMLTextTypes.DROP_CAPITAL, PageXMLTextTypes.CAPTION, PageXMLTextTypes.IGNORE}

    @staticmethod
    def ignore_at_load_types():
        return {PageXMLTextTypes.IGNORE}

    def is_text(self):
        return self not in PageXMLTextTypes.image_types()


PageRegionBB = namedtuple('PageRegionBB', 'x1 y1 x2 y2 is_text')


class PolygonRegionBase(abc.ABC):
    @abc.abstractmethod
    def get_polygon(self) -> np.ndarray:
        pass


@dataclass
class TextEquiv:
    index: int
    text: str

    TEXT_ID_GROUND_TRUTH = 0
    TEXT_ID_PREDICTION = 1

    def is_gt(self):
        return self.index == TextEquiv.TEXT_ID_GROUND_TRUTH

    def is_pred(self):
        return self.index == TextEquiv.TEXT_ID_PREDICTION

    def get_escaped(self):
        return xml.sax.saxutils.escape(self.text)



def _scale_pxml_array(array, scale_factor: float):
    return np.array(array * scale_factor + 0.5, dtype=np.int)

@dataclass
class TextLine(PolygonRegionBase):

    id: str
    parent_region_id: str
    polygon: np.array
    baseline: np.array = None
    text_equivs: Dict[int, TextEquiv] = field(default_factory=dict)

    def has_baseline(self) -> np.ndarray:
        return self.baseline is not None

    def get_polygon(self) -> np.ndarray:
        return self.polygon

    def copy(self) -> 'TextLine':
        return TextLine(id=self.id, parent_region_id=self.parent_region_id,polygon=np.copy(self.polygon), baseline=np.copy(self.baseline), text_equivs=dict((t.index, TextEquiv(t.index, t.text)) for t in self.text_equivs.values()))

    def scale(self, scale_factor):
        self.polygon = _scale_pxml_array(self.polygon, scale_factor=scale_factor)
        if self.baseline is not None: # is not None is required because this might be a numpy array
            self.baseline = _scale_pxml_array(self.baseline, scale_factor=scale_factor)

    def get_text_gt(self) -> str:
        return self.text_equivs.get(TextEquiv.TEXT_ID_GROUND_TRUTH, TextEquiv(0,"")).text

    def get_text_pred(self) -> str:
        return self.text_equivs.get(TextEquiv.TEXT_ID_PREDICTION, TextEquiv(1,"")).text

    def get_ordering_heur(self) -> float:
        if self.baseline is not None:
            return float(np.mean(self.baseline[:1]))
        elif self.polygon is not None:
            return float(np.mean(self.polygon[:1]))
        else:
            return float(0)

    def has_gt_text(self):
        return TextEquiv.TEXT_ID_GROUND_TRUTH in self.text_equivs
    def has_pred_text(self):
        return TextEquiv.TEXT_ID_PREDICTION in self.text_equivs

@dataclass
class PageRegion(PolygonRegionBase):
    polygon: np.array
    region_type: PageRegionTypes
    region_text_type: PageXMLTextTypes = None
    region_id: str = None
    text_lines: List[TextLine] = field(default_factory=list)

    def bounding_box(self):
        xmin = int(np.min(self.polygon[:, 0]))
        xmax = int(np.max(self.polygon[:, 0]))
        ymin = int(np.min(self.polygon[:, 1]))
        ymax = int(np.max(self.polygon[:, 1]))
        return PageRegionBB(xmin, ymin, xmax, ymax, self.is_text())

    def shifted(self, x, y):
        # shift the region points by x and y and return as new region
        new_polygon = self.polygon + np.array([x, y])
        return PageRegion(new_polygon, self.region_type, self.region_text_type)

    def coords_str(self):
        # return the coords string as it is written in the pagexml
        parts = []
        for row in self.polygon:
            parts.append("{},{}".format(row[0], row[1]))
        return " ".join(parts)

    def has_negative_coords(self):
        return np.min(self.polygon) < 0

    def is_text(self):
        if self.region_type.is_text():
            return self.region_text_type.is_text()
        else:
            return False

    def is_image(self):
        return not self.region_type.is_text()

    def get_polygon(self):
        return self.polygon

    def copy(self):
        return PageRegion(polygon=np.copy(self.polygon), region_type=self.region_type, region_text_type=self.region_text_type, region_id=self.region_id,
                          text_lines=[t.copy() for t in self.text_lines])

    def scale(self, scale_factor: float, scale_children=True):
        self.polygon = _scale_pxml_array(self.polygon, scale_factor)
        if scale_children:
            for r in self.text_lines:
                r.scale(scale_factor=scale_factor)

    def ordering_heur(self):
        return self.bounding_box().y1 // 10, self.bounding_box().x1

    def textlines_ordered_heur(self):
        return sorted(self.text_lines, key=lambda l: l.get_ordering_heur())


class PageXMLInvalid(Exception):
    pass


class PageXMLParserMode(enum.Enum):
    EASY = enum.auto()
    FAIR = enum.auto()
    STRICT = enum.auto()
    SUPERSTRICT = enum.auto()


@dataclass
class RegionTypeOverrideSettings:
    text: set = dataclasses.field(default_factory=lambda: set())
    image: set = dataclasses.field(default_factory=lambda: set())
    background: set = dataclasses.field(default_factory=lambda: set())

    def as_lowercase(self):
        text = set([x.lower() for x in self.text])
        image = set([x.lower() for x in self.image])
        background = set([x.lower() for x in self.background])
        return RegionTypeOverrideSettings(text, image, background)


class PageLookups:
    def __init__(self, regions: List[PageRegion], text_lines: List[TextLine]):
        self.region_lookup = dict((r.region_id, r) for r in regions)
        self.textline_lookup = dict((l.id, l) for l in text_lines)

@dataclass
class ReadingOrder:
    order: List[str] = field(default_factory=list)  # List of Region IDs in order

class InvalidLookup(Exception):
    pass

@dataclass
class PageXMLPage:
    image_filename: str
    created_on: str
    last_edited: str
    image_width: int
    image_height: int

    regions: List[PageRegion]
    _lookups: PageLookups = None
    reading_order: ReadingOrder = None
    creator: str = None


    def rebuild_lookups(self):
        self._lookups = PageLookups(self.regions, self.text_lines())

    def regions_with_type(self, type: PageRegionTypes):
        return [x for x in self.regions if x.region_type is type]

    def text_regions_with_type(self, type: PageXMLTextTypes):
        # filter by type
        return [x for x in self.regions if x is not None and x is type]

    def contains_images(self):
        for x in self.regions:
            if x.is_text() is False:
                return True
        return False

    def __len__(self):
        return len(self.regions)

    def __iter__(self):
        return iter(self.regions)

    def text_regions(self) -> List[PageRegion]:
        return [r for r in self.regions if r.is_text()]

    def text_lines(self) -> List[TextLine]:
        return list(itertools.chain.from_iterable(r.text_lines for r in self.regions))

    def text_lines_ordered(self) -> List[TextLine]:
        if not hasattr(PageXMLPage, "text_lines_ordered_warning"):
            logger.warning("LibPageXML2: TextLinesOrdered might return wrong result")
            PageXMLPage.text_lines_ordered_warning = True
        return list(itertools.chain.from_iterable(r.textlines_ordered_heur() for r in sorted(self.regions, key=lambda x: x.ordering_heur())))

    # this might skip some textlines, if they are not in the reading order
    def text_lines_by_reading_order(self) -> List[TextLine]:
        return list(itertools.chain.from_iterable(self.region_by_id(r_id).text_lines for r_id in self.reading_order.order))

    def baselines(self) -> List[np.ndarray]:
        return [t.baseline for t in self.text_lines()]

    def region_by_id(self, id):
        try:
            if not self._lookups:
                self.rebuild_lookups()
            return self._lookups.region_lookup[id]
        except KeyError:
            raise InvalidLookup(f"Invalid Region lookup in file {self.image_filename} for region {id}")

    def textline_by_id(self, id):
        if not self._lookups:
            self.rebuild_lookups()
        return self._lookups.textline_lookup[id]

    def copy(self) -> 'PageXMLPage':
        return PageXMLPage(image_filename=str(self.image_filename), image_width=self.image_width, image_height=self.image_height,
                          regions=[p.copy() for p in self.regions], created_on=str(self.created_on), last_edited=str(self.last_edited))

    def rescale(self, scale_factor: float):
        for p in self.regions:
            p.scale(scale_factor)

    def rescale_to(self, target_width: int, target_height: int):
        rescale_factor = float(0.5 * ((target_width / self.image_width) + (target_height / self.image_height)))
        self.rescale(rescale_factor)
       # assert self.image_width == target_width and self.image_height == target_height
        self.image_width = target_width
        self.image_height = target_height

    def rotate(self, angle_deg: float, new_center):
        # rotates the page using the center as the origin
        tx, ty = self.image_width / 2, self.image_height / 2
        cos_phi, sin_phi = math.cos(angle_deg / 360 * 2 * math.pi), math.sin(angle_deg / 360 * 2 * math.pi)
        pbx, pby = new_center
        def rotation_fn(p):
            return rotate_points(p,tx,ty,cos_phi,sin_phi, pbx, pby)

        for region in self.regions:
            if region.polygon is not None:
                region.polygon = rotation_fn(region.polygon)
            for line in region.text_lines:
                if line.polygon is not None:
                    line.polygon = rotation_fn(line.polygon)
                if line.baseline is not None:
                    line.baseline = rotation_fn(line.baseline)

    @staticmethod
    def make_empty(img_filename: str, img_w: int, img_h: int):
        import datetime
        now = datetime.datetime.now()
        return PageXMLPage(img_filename,str(now),str(now),int(img_w), int(img_h),
                                           [],reading_order=ReadingOrder())


class PageXMLParserTextMode(enum.Enum):
    IGNORE = enum.auto()
    CONVERT_PRED = enum.auto()
    CONVERT_GT = enum.auto()
    EQUIV = enum.auto()


# TODO: refactor this to be a pure parser.. this should return a PageXMLPage instance
class PageXMLParser:
    _coords_regex = re.compile(r"([0-9]+)([.]?[0-9]*),([0-9]+)([.]?[0-9]*)")
    _rem_namespace_regex = re.compile("(?:[{][a-zA-Z0-9:/.-]+[}])?([a-zA-Z]+)")
    printed_strict_mode_warning = False
    def is_strict(self):
        return self.strictness is PageXMLParserMode.STRICT or self.strictness is PageXMLParserMode.SUPERSTRICT

    def is_superstrict(self):
        return self.strictness is PageXMLParserMode.SUPERSTRICT

    def _polygon_from_coords_str(self, coords_str: str) -> np.array:
        points = []
        for match in PageXMLParser._coords_regex.finditer(coords_str):
            x, y = int(match.group(1)), int(match.group(3))
            if self.do_rescale:
                x = int(x * self.rescale_factor_x)  # TODO: maybe use round() here !?!
                y = int(y * self.rescale_factor_y)
            points.append((x, y))
        return np.array(points)

    def _remove_namespace_from_tag(self, tagstring):
        # check the validity of the tag
        match = PageXMLParser._rem_namespace_regex.match(tagstring)
        if match is None or match.group(1) is None or match.group(1) == "":
            raise PageXMLInvalid(f"Cannot match Page Element Tag {tagstring}")
        return match.group(1)

    def _parse_textline(self, element:ET.Element, parent_element, parent_id, namespace):
        id = element.attrib["id"]
        if id is None:
            logger.warning(f"Textline with no id")
        coords = element.find(namespace + "Coords")
        if coords is None:
            raise PageXMLInvalid("TextLine must have a coords array")
        if "points" in coords.attrib:
            textline_polygon = self._polygon_from_coords_str(str(coords.attrib["points"]))
        else:
            raise NotImplementedError("Can only parse TextLines with points attribute")
        if textline_polygon.shape[0] < 3:
            raise PageXMLInvalid(f"TextLine {id} has invalid polygon")
        # find a baseline
        baseline = element.find(namespace + "Baseline")
        if baseline is not None and ("points" in baseline.attrib):
            baseline_points = self._polygon_from_coords_str(str(baseline.attrib["points"]))
        else:
            baseline_points = None

        text_equivs = element.findall(namespace+"TextEquiv")
        teq = dict()
        for te in text_equivs:
            uc = te.find(namespace + "Unicode").text
            idx = int(te.attrib["index"]) if "index" in te.attrib else -1
            if idx == -1:
                logger.warning(f"Skipping TextEquiv with no index in {self.image_filename}")
            elif idx != 0 and idx != 1:
                raise PageXMLInvalid("Only supports Text Equiv index of 0 or 1")
            elif uc is None:
                logger.warning(f"TextEquiv has no <Unicode> tag... Loading TextEquivText in {self.image_filename}")
                teq[idx] = TextEquiv(index=idx, text=te.text or "")
            else:
                teq[idx] = TextEquiv(index=idx, text=uc)

        return TextLine(id, parent_id, textline_polygon, baseline_points, text_equivs=teq)


    def _get_textlines(self, region_element, namespace):
        region_id = region_element.get("id", default="")
        textlines = region_element.findall(namespace + "TextLine")

        textlines = [self._parse_textline(line, region_element, region_id, namespace) for line in
                     region_element.findall(namespace + "TextLine")]

        return textlines

    def _parse_region(self, element, tag, namespace):
        region_id = element.get("id", default="")
        if region_id == "" and self.strictness != PageXMLParserMode.EASY:
            raise PageXMLInvalid("Region without id")

        coords = element.find(namespace + "Coords")

        if coords is None:
            logger.warning(f"No coords array for region {region_id} in XML")
            if self.strictness == PageXMLParserMode.EASY:
                logger.warning("Using full page Text region as an alternative")
                return PageRegion(np.array([[0,0],[self.image_width,0],[self.image_width, self.image_height],[0,self.image_height]],dtype=np.float),region_type=PageRegionTypes.TEXT_REGION,region_text_type=PageXMLTextTypes.OTHER,region_id=region_id, text_lines=self._get_textlines(element, namespace))
            return None


        polygon = None
        if "points" in coords.attrib:
            polygon = self._polygon_from_coords_str(str(coords.attrib["points"]))
        else:
            # hopefully there is a
            point_elems = coords.findall(namespace + "Point")
            points = []
            if self.do_rescale:
                for elem in point_elems:
                    points.append((int(float(elem.attrib["x"]) * self.rescale_factor_x), int(float(elem.attrib["y"]) * self.rescale_factor_y)))
                    # TODO: maybe round here !?!
            else:
                for elem in point_elems:
                    points.append((int(elem.attrib["x"]), int(elem.attrib["y"])))
            polygon = np.array(points)
        if not (polygon.shape[0] != 0 and polygon.shape[1] == 2 and len(polygon.shape) == 2):
            if self.strictness is PageXMLParserMode.EASY:
                logger.warning(f"Polygon {region_id} has less than 3 points... skipping")
                return None
            else:
                raise PageXMLInvalid(f"Invalid polygon. Maybe wrong xml format? In region with id: {region_id}")

        if polygon.shape[0] <= 2:
            if self.strictness is PageXMLParserMode.EASY:
                logger.warning(f"Polygon {region_id} has less than 3 points... skipping")
                return None
            else:
                raise PageXMLInvalid(f"Polygon for region {region_id} has less than 3 points.")

        # now we should have the polygon
        # fetch the region type
        try:
            region_type = PageRegionTypes(tag)
        except ValueError as e:
            raise PageXMLInvalid(f"Unknown Region Type encountered: {e.__str__()} in region with id: {region_id}")

        # now we have a proper region type
        # if it is a TextRegion, find the sub type
        if region_type is PageRegionTypes.TEXT_REGION:
            text_type_str = element.get("type")
            if text_type_str is None:
                region_text_type = PageXMLTextTypes.EMPTY
            else:
                try:
                    region_text_type = PageXMLTextTypes(text_type_str.replace("-","_"))  # TODO: this replacement should not be done always
                except ValueError as e:
                    raise PageXMLInvalid(f"Unknown Text Type for Text Region: {e.__str__()} in region with id: {region_id}")
        else:
            region_text_type = None

        # add the textlines of this region
        textlines = self._get_textlines(element, namespace)
        # find the lines in this region
        return PageRegion(
            polygon=polygon,
            region_type=region_type,
            region_text_type=region_text_type,
            region_id=region_id if region_id != "" else None,
            text_lines=textlines
        )

    @staticmethod
    def from_file(filename, rescale=None, strictness: PageXMLParserMode = PageXMLParserMode.SUPERSTRICT):
        with open(filename) as f:
            return PageXMLParser(f.read(), filename, rescale=rescale, strictness=strictness)

    def __init__(self, pagexml, filename=None, rescale=None, strictness:PageXMLParserMode = PageXMLParserMode.SUPERSTRICT,
                 text_mode: PageXMLParserTextMode = PageXMLParserTextMode.EQUIV):
        if rescale is not None:
            logger.warning("Rescaling of all PageXML features is currently unstable")
        tree = ET.ElementTree(ET.fromstring(pagexml))
        root = tree.getroot()
        namespace: str = root.tag.split('}')[0] + "}"
        page: ET.Element = root.find(namespace + "Page")
        self.strictness = strictness
        if not self.is_strict():
            if not PageXMLParser.printed_strict_mode_warning:
                logger.warning(f"Warn: Strict mode is disabled for libpagexml")
                PageXMLParser.printed_strict_mode_warning = True
            # TODO: is superstrict a requirement

        if page is None:
            raise PageXMLInvalid("Cannot find Page Element in the Root of the PageXML")

        # gather some things from the PageXML header
        self.image_filename, self.image_height, self.image_width = \
            str(page.attrib["imageFilename"]), \
            int(page.attrib["imageHeight"]), \
            int(page.attrib["imageWidth"])

        # really stupid hack, but source data is wrong :(
        if self.image_width > self.image_height:
            pass
            # TODO: remove deprecated warning
            #print(f"WARNING! PageXML for file {filename} width greater than height.. Skipping rotate.. but output data might be wrong", file=sys.stderr)
            #self.image_width, self.image_height = self.image_height, self.image_width

        if rescale is None or (rescale[0] == self.image_width and rescale[1] == self.image_height):
            self.do_rescale = False
            self.rescale_factor_x = 1
            self.rescale_factor_y = 1
        else:
            self.do_rescale = True
            self.rescale_factor_x = rescale[0] / self.image_width
            self.rescale_factor_y = rescale[1] / self.image_height
            self.image_width = rescale[0]
            self.image_height = rescale[1]

        self.reading_order = ReadingOrder()
        self.regions = []
        reading_order_query = None
        for element in page:
            tagstr = self._remove_namespace_from_tag(element.tag)
            # check if we ignore this tag
            if PageRegionTypes.should_ignore(tagstr):
                continue
            elif tagstr == "ReadingOrder":
                if reading_order_query is not None:
                    raise PageXMLInvalid("Multiple ReadingOrders")
                else:
                    reading_order_query = (element, tagstr, namespace)
            elif tagstr in PageRegionTypes.supported_regions():
                region = self._parse_region(element, tagstr, namespace)
                if region is None:
                    if self.strictness is not PageXMLParserMode.EASY:
                        raise PageXMLInvalid("Parsing returned None")
                    else:
                        continue

                if region.region_text_type is not None and \
                   region.region_text_type in PageXMLTextTypes.ignore_at_load_types():
                    logger.warning(f"Ignore region {region.region_id} with text type {region.region_text_type} in file {filename}")
                    continue
                self.regions.append(region)
            elif tagstr in {"Relations"}:
                pass # we do not care about the relations type
            else:
                if self.strictness is not PageXMLParserMode.EASY:
                    raise PageXMLInvalid(f"Unsupported XML Tag Type {tagstr} in {filename}")
        if reading_order_query is not None:
            self.reading_order = self.parse_reading_order(*reading_order_query)
        if self.is_superstrict():
            self.check_validity()
        # build the lookup tables
        self.region_lookup = dict((r.region_id, r) for r in self.regions)
        self.textline_lookup = dict((l.id, l) for l in self.text_lines())

        self._fix_textline_by_mode(mode=text_mode, fname=filename)
        self._fix_readingorder(mode=strictness)
    def _fix_readingorder(self, mode: PageXMLParserMode):
        # look for invalid reading order entries and remove them
        new_order = []
        for x in self.reading_order.order:
            if x in self.region_lookup:
                new_order.append(x)
            else:
                if mode == PageXMLParserMode.EASY:
                    logger.warning(f"Region {x} is in reading order but does not exist in file {self.image_filename}")
                else:
                    raise InvalidLookup(f"Region {x} is in reading order but does not exist in file {self.image_filename}")
        self.reading_order.order = new_order

    def parse_reading_order(self, element, tagstr, namespace):
        reading_order = ReadingOrder()

        if len(element) > 1:
            logger.error("More than one Ordered Group in Reading Order in PageXML.")
            raise PageXMLInvalid("More than one Ordered Group in Reading Order in PageXML.")
        for group in element:
            assert group.tag == namespace+"OrderedGroup"
            indexed_reg = []
            for assumed_idx, refidx in enumerate(group):
                if not refidx.tag == namespace+"RegionRefIndexed":
                    raise PageXMLInvalid("OrderedGroup contains something else than refidx tags")
                if not assumed_idx == int(refidx.attrib["index"]):
                    logger.warning(f"Region References are not incrementing in reading Order in file {self.image_filename}")
                indexed_reg.append((int(refidx.attrib["index"]), refidx.attrib["regionRef"]))
            indexed_reg = sorted(indexed_reg,key=lambda x:x[0])

            for i,r in indexed_reg:
                reading_order.order.append(r)

        return reading_order


    def _fix_textline_by_mode(self, mode:PageXMLParserTextMode, fname:str):
        if mode == PageXMLParserTextMode.IGNORE or mode == PageXMLParserTextMode.EQUIV: return

        for tl in self.text_lines():
            if mode == PageXMLParserTextMode.CONVERT_GT:
                if tl.get_text_gt() == "" and tl.get_text_pred() != "":
                    tl.text_equivs[TextEquiv.TEXT_ID_GROUND_TRUTH] = TextEquiv(TextEquiv.TEXT_ID_GROUND_TRUTH, tl.get_text_pred())
                    logger.warning(f"Converting text_line {tl.id} from prediction to ground_truth for file {fname}")
            elif mode == PageXMLParserTextMode.CONVERT_PRED:
                if tl.get_text_pred() == "" and tl.get_text_gt() != "":
                    tl.text_equivs[TextEquiv.TEXT_ID_PREDICTION] = TextEquiv(TextEquiv.TEXT_ID_PREDICTION, tl.get_text_gt())
                    logger.warning(f"Converting text_line {tl.id} from gt to prediction for file {fname}")


    def get_page(self) -> PageXMLPage:
        page = PageXMLPage(image_filename=self.image_filename, image_width=self.image_width, image_height=self.image_height,
                           regions = self.regions, created_on=datetime.datetime.now().isoformat(),
                           last_edited=datetime.datetime.now().isoformat(), reading_order=self.reading_order)
        return page

    def check_validity(self):
        # make sure that all set region are unique
        region_ids = [x.region_id for x in self.regions if x.region_id is not None]
        for rid in region_ids:
            if rid == "":
                raise PageXMLInvalid(f"Region ID is empty")

        if not (len(region_ids) == len(set(region_ids))):
            raise PageXMLInvalid(f"Region IDs are not unique: <{', '.join(region_ids)}>")

        textlines = self.text_lines()
        for tl in  textlines:
            if tl.id == "" or tl.id is None:
                raise PageXMLInvalid("textline has no id")
        tl_ids = set(tl.id for tl in textlines)
        if len(tl_ids) != len(textlines):
            raise PageXMLInvalid("Textline IDs are not unique")

        #  TODO: make sure that no polygons are overlapping, because
        #  that would lead to ambiguous mappings between PageXML and img

        if self.is_superstrict():
            from shapely.geometry import Polygon
            for r1 in self.regions:
                try:
                    p1 = Polygon(r1.polygon)
                except Exception as e:
                    raise PageXMLInvalid(
                        f"Polygon for region {r1.region_id} has invalid geometry.Orig message:{e.__str__()}")
                for r2 in self.regions:
                    if r1 is r2: continue
                    try:
                        p2 = Polygon(r2.polygon)
                    except Exception as e:
                        raise PageXMLInvalid(
                            f"Polygon for region {r1.region_id} has invalid geometry.Orig. message:{e.__str__()}")
                    try:
                        is_intersection = p1.intersects(p2)
                    except Exception as e:
                        raise PageXMLInvalid(
                            f"Polygon Intersection between regions {r1.region_id} "
                            f"and {r2.region_id} fails. Orig msg.:{e.__str__()}")

                    if is_intersection:
                        err_msg = f"Intersecting Polygons between regions {r1.region_id} {r2.region_id}"
                        if self.is_superstrict():
                            raise PageXMLInvalid(err_msg)
                        else:
                            pass

    def regions_with_type(self, type: PageRegionTypes):
        return [x for x in self.regions if x.region_type is type]

    def text_regions_with_type(self, type: PageXMLTextTypes):
        # filter by type
        return [x for x in self.regions if x is not None and x is type]

    def contains_images(self):
        for x in self.regions:
            if x.is_text() is False:
                return True
        return False

    def __len__(self):
        return len(self.regions)

    def __iter__(self):
        return iter(self.regions)

    def text_regions(self) -> List[PageRegion]:
        return [r for r in self.regions if r.is_text()]

    def text_lines(self) -> List[TextLine]:
        return list(itertools.chain.from_iterable(r.text_lines for r in self.regions))


    def baselines(self) -> List[np.ndarray]:
        return [t.baseline for t in self.text_lines()]

    def region_by_id(self, id):
        return self.region_lookup[id]

    def textline_by_id(self, id):
        return self.textline_lookup[id]

def parse_pagexml(fname: str = None, pagexml_str: str = None, strictness: PageXMLParserMode = PageXMLParserMode.EASY):
    if not (fname or pagexml_str):
        raise ValueError("Either fname or pagexml_str must be given!")

    if fname and pagexml_str:
        raise ValueError("Cannot pass both fname and pagexml_str")

    raise NotImplementedError()



class PageXMLBuilder:
    @staticmethod
    def baseline_to_elem(tl: TextLine) -> str:
        stro = [f"{x[0]},{x[1]}" for x in tl.baseline]
        return '<Baseline points="' + " ".join(stro) + '"/>\n'

    @staticmethod
    def points_to_coords_elem(points: np.ndarray) -> str:
        stro = [f"{x[0]},{x[1]}" for x in points]
        return '<Coords points="' + " ".join(stro) + '"/>\n'

    @classmethod
    def push_text_line(cls, out: List[str], tl: TextLine):
        out.append(f'<TextLine id="{tl.id}">\n')
        out.append(cls.points_to_coords_elem(tl.polygon))
        if tl.has_baseline():
            out.append(cls.baseline_to_elem(tl))
        for te in tl.text_equivs.values():
            out.append(f'<TextEquiv index="{te.index}"><Unicode>{te.get_escaped()}</Unicode></TextEquiv>')
        # push the text equiv
        out.append("</TextLine>\n")

    @classmethod
    def push_region(cls, out: List[str], reg: PageRegion):

        #if reg.region_text_type is not None and reg.region_text_type.value != "":
        #    out.append(f'<{reg.region_type.value} id="{reg.region_id}" type="{reg.region_text_type.value}">\n')
        #else: # TODO: why? we should also save the subtype for the region!
        out.append(f'<{reg.region_type.value if reg.region_type else PageRegionTypes.TEXT_REGION} id="{reg.region_id}" type="{reg.region_text_type.value if reg.region_text_type else PageXMLTextTypes.OTHER}">\n')
        out.append(cls.points_to_coords_elem(reg.polygon))
        for tl in reg.text_lines:
            cls.push_text_line(out,tl)
        out.append(f'</{reg.region_type.value}>')

    @classmethod
    def push_order(cls, out: List[str], page: PageXMLPage):
        out.append('<ReadingOrder><OrderedGroup id="ro1" caption="Region reading order">\n')
        for i, x in enumerate(page.reading_order.order):
            out.append(f'<RegionRefIndexed index="{i}" regionRef="{x}" />\n')
        out.append('</OrderedGroup></ReadingOrder>\n')

    @classmethod
    def build_pagexml(cls, page: PageXMLPage, creator: str = None) -> str:
        contents = [f"""\
<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15/pagecontent.xsd">
<Metadata>
<Creator>{creator or page.creator or "LibPageXML2 (generic)"}</Creator>
<Created>{page.created_on}</Created>
<LastChange>{page.last_edited}</LastChange>
<Comments />
</Metadata>
<Page imageFilename="{page.image_filename}" imageHeight="{page.image_height}" imageWidth="{page.image_width}">        
"""]
        if page.reading_order is not None and page.reading_order.order:
            cls.push_order(contents, page)

        for reg in page.regions:
            cls.push_region(contents, reg)
        contents.append("\n</Page></PcGts>\n")
        return "".join(contents)
