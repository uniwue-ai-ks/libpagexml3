from .libpagexml3 import *
import datetime


class PAGE2019Builder:
    @staticmethod
    def baseline_to_elem(tl: TextLine) -> str:
        stro = [f"{round(x[0])},{round(x[1])}" for x in tl.baseline] # in pagexml, all of these have to be integers
        return '<Baseline points="' + " ".join(stro) + '"/>\n'

    @staticmethod
    def points_to_coords_elem(points: np.ndarray) -> str:
        stro = [f"{round(x[0])},{round(x[1])}" for x in points] # in pagexml, all of these have to be integers
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

<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15/pagecontent.xsd">
<Metadata>
<Creator>{creator or page.creator or "LibPageXML2 (generic)"}</Creator>
<Created>{datetime.datetime.now().isoformat()}</Created>
<LastChange>{datetime.datetime.now().isoformat()}</LastChange>
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
