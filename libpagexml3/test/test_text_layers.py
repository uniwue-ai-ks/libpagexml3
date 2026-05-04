import os
import unittest

from libpagexml3.libpagexml3 import (
    BlockTaggedText,
    BlockType,
    FormattedLine,
    FormattedText,
    FormattedTextSegment,
    FulltextTranscription,
    PageXMLBuilder,
    PageXMLPage,
    PageXMLParser,
    PageXMLParserMode,
    TextLayers,
)


DATA_FILE = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "test", "data", "aesofabv_BV046236557_0001.xml",
    )
)


class TextLayersParseTest(unittest.TestCase):
    def test_parses_fulltext_blocks_from_reference_file(self):
        page = PageXMLParser.from_file(
            DATA_FILE, strictness=PageXMLParserMode.EASY
        ).get_page()

        self.assertIsNotNone(page.text_layers)
        self.assertIsNotNone(page.text_layers.fulltext)

        blocks = page.text_layers.fulltext.blocks
        self.assertEqual(len(blocks), 2)

        b0 = blocks[0]
        self.assertIsInstance(b0, BlockTaggedText)
        self.assertEqual(b0.id, "lnmorc22fx0")
        self.assertEqual(b0.tag, "wilhelm_grimm")
        self.assertEqual(b0.type, BlockType.TAGGED_TEXT)
        self.assertEqual(len(b0.text.lines), 1)
        line0 = b0.text.lines[0]
        self.assertEqual(line0.id, "lnmorc22fx1")
        self.assertEqual(
            [(s.text, s.class_) for s in line0.segments],
            [("Hello this is a ", None), ("test", "fmt-strike")],
        )

        b1 = blocks[1]
        self.assertIsInstance(b1, BlockTaggedText)
        self.assertEqual(b1.id, "lnmorq71o11")
        self.assertEqual(b1.tag, "jakob_grimm")
        self.assertEqual(b1.type, BlockType.TAGGED_TEXT)

        expected_lines = [
            (
                "lnmorq71o12",
                [
                    ("This was ", None),
                    ("written", "fmt-underline"),
                    (" by ", None),
                    ("Jakob Grimm", "fmt-correction-jakob"),
                ],
            ),
            ("lnmorq7gjd4", []),
            ("lnmorq7gnb5", [("And it has multiple lines", None)]),
            ("lnmorq7j2v6", []),
            (
                "lnmorq7j677",
                [
                    ("Which should have different ", None),
                    ("IDs", "fmt-underline"),
                ],
            ),
        ]
        self.assertEqual(len(b1.text.lines), len(expected_lines))
        for line, (expected_id, expected_segments) in zip(b1.text.lines, expected_lines):
            self.assertEqual(line.id, expected_id)
            self.assertEqual(
                [(s.text, s.class_) for s in line.segments], expected_segments
            )


class TextLayersBuilderTest(unittest.TestCase):
    @staticmethod
    def _build_sample_page() -> PageXMLPage:
        page = PageXMLPage.make_empty("aesofabv_BV046236557_0001", 2139, 3415)
        page.text_layers = TextLayers(
            fulltext=FulltextTranscription(
                blocks=[
                    BlockTaggedText(
                        id="block_a",
                        tag="wilhelm_grimm",
                        text=FormattedText(
                            lines=[
                                FormattedLine(
                                    id="line_a_1",
                                    segments=[
                                        FormattedTextSegment(text="Hello this is a ", class_=None),
                                        FormattedTextSegment(text="test", class_="fmt-strike"),
                                    ],
                                ),
                            ]
                        ),
                    ),
                    BlockTaggedText(
                        id="block_b",
                        tag="jakob_grimm",
                        text=FormattedText(
                            lines=[
                                FormattedLine(
                                    id="line_b_1",
                                    segments=[
                                        FormattedTextSegment(text="This was ", class_=None),
                                        FormattedTextSegment(text="written", class_="fmt-underline"),
                                        FormattedTextSegment(text=" by ", class_=None),
                                        FormattedTextSegment(text="Jakob Grimm", class_="fmt-correction-jakob"),
                                    ],
                                ),
                                FormattedLine(id="line_b_2", segments=[]),
                                FormattedLine(
                                    id="line_b_3",
                                    segments=[
                                        FormattedTextSegment(text="And it has multiple lines", class_=None),
                                    ],
                                ),
                            ]
                        ),
                    ),
                ]
            )
        )
        return page

    def _assert_pages_equivalent(self, original: PageXMLPage, parsed: PageXMLPage):
        self.assertEqual(original.image_filename, parsed.image_filename)
        self.assertEqual(original.image_width, parsed.image_width)
        self.assertEqual(original.image_height, parsed.image_height)

        self.assertIsNotNone(parsed.text_layers)
        self.assertIsNotNone(parsed.text_layers.fulltext)

        orig_blocks = original.text_layers.fulltext.blocks
        parsed_blocks = parsed.text_layers.fulltext.blocks
        self.assertEqual(len(orig_blocks), len(parsed_blocks))

        for ob, pb in zip(orig_blocks, parsed_blocks):
            self.assertEqual(type(ob), type(pb))
            self.assertEqual(ob.id, pb.id)
            self.assertEqual(ob.type, pb.type)
            self.assertIsInstance(ob, BlockTaggedText)
            self.assertEqual(ob.tag, pb.tag)
            self.assertEqual(len(ob.text.lines), len(pb.text.lines))
            for ol, pl in zip(ob.text.lines, pb.text.lines):
                self.assertEqual(ol.id, pl.id)
                self.assertEqual(
                    [(s.text, s.class_) for s in ol.segments],
                    [(s.text, s.class_) for s in pl.segments],
                )

    def test_build_then_parse_roundtrip(self):
        page = self._build_sample_page()

        xml = PageXMLBuilder.build_pagexml(page)

        # Sanity: the text_layers UserAttribute should have been written.
        self.assertIn('name="text_layers"', xml)

        parsed = PageXMLParser(xml, strictness=PageXMLParserMode.EASY).get_page()
        self._assert_pages_equivalent(page, parsed)

    def test_empty_text_layers_are_not_written(self):
        page = PageXMLPage.make_empty("foo.png", 100, 200)
        xml = PageXMLBuilder.build_pagexml(page)
        self.assertNotIn("text_layers", xml)

        parsed = PageXMLParser(xml, strictness=PageXMLParserMode.EASY).get_page()
        self.assertIsNone(parsed.text_layers.fulltext)


if __name__ == "__main__":
    unittest.main()
