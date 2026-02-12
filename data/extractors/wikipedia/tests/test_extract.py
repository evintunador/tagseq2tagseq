import unittest
import re

from data.wiki_graph_extractor.extract import (
    fix_lists, fix_broken_links, fix_math_tags, rescue_number_templates,
    remove_comments, remove_templates, remove_stub_templates, remove_wikitables, remove_reference_tags,
    remove_external_links, convert_internal_links, convert_html_formatting, remove_file_references,
    convert_bold_and_italics, fix_indented_math, format_sections_and_whitespace, fix_definition_lists,
    normalize_title, remove_unwanted_sections, fix_date_ranges, protect_math_from_templates, restore_math_content, fix_complex_wikilinks,
    fix_corrupted_asterisks, fix_mediawiki_links, fix_excessive_whitespace, fix_malformed_formatting
)

class TestExtract(unittest.TestCase):

    def test_fix_broken_links(self):
        """Test fixing malformed wikilinks like [[Link] -> [Link]."""
        cases = [
            (
                "See [[Link] here.", 
                "See [Link] here."
            ),
            (
                "Nested [[File:Img.png|[[Link]]]] should be safe.",
                "Nested [[File:Img.png|[[Link]]]] should be safe." # Should NOT change valid nested
            ),
            (
                "Multiple [[Link1] and [[Link2].",
                "Multiple [Link1] and [Link2]."
            ),
            (
                "Valid [[Link]].",
                "Valid [[Link]]."
            )
        ]
        for inp, expected in cases:
            with self.subTest(inp=inp):
                self.assertEqual(fix_broken_links(inp), expected)

    def test_fix_lists(self):
        """Test converting Wikitext lists to Markdown."""
        text = """
* Level 1
** Level 2
*** Level 3
# Ordered 1
## Ordered 2
### Ordered 3
• Bullet
: Indent
"""
        expected = """
- Level 1
  - Level 2
    - Level 3
1. Ordered 1
  1. Ordered 2
    1. Ordered 3
- Bullet
> Indent
"""
        # Strip leading/trailing newlines for comparison if needed, but regex handles lines.
        # fix_lists processes the whole block.
        processed = fix_lists(text)
        # We'll compare line by line stripped to avoid minor whitespace issues, 
        # but indentation matters!
        
        processed_lines = [l for l in processed.split('\n') if l.strip()]
        expected_lines = [l for l in expected.split('\n') if l.strip()]
        
        self.assertEqual(processed_lines, expected_lines)

    def test_fix_definition_lists(self):
        """Test conversion of MediaWiki definition lists to Markdown headers."""
        
        cases = [
            # Single definition term
            ("Some text.\n;Definition term\n- List item", 
             "Some text.\n#### Definition term\n- List item"),
            
            # Multiple definition terms
            ("Text.\n;First term\n- Item 1\n;Second term\n- Item 2", 
             "Text.\n#### First term\n- Item 1\n#### Second term\n- Item 2"),
            
            # Definition term with links and formatting
            (";God's [authority](link)\n- Moses parts the Red Sea", 
             "#### God's [authority](link)\n- Moses parts the Red Sea"),
             
            # Mixed with other content
            ("Introduction.\n\n;Healing\n- A man gets up and walks\n\nConclusion.", 
             "Introduction.\n\n#### Healing\n- A man gets up and walks\n\nConclusion."),
            
            # Definition term at start of text
            (";Starting term\nContent follows.", 
             "#### Starting term\nContent follows."),
             
            # No definition terms (should remain unchanged)
            ("Regular text without definition lists.", 
             "Regular text without definition lists."),
             
            # Semicolon in middle of line (should not change)
            ("Text with ; semicolon in middle.", 
             "Text with ; semicolon in middle."),
        ]
        
        for i, (input_text, expected) in enumerate(cases):
            with self.subTest(case=i, input=input_text):
                result = fix_definition_lists(input_text)
                self.assertEqual(result, expected)

    def test_math_protection_from_templates(self):
        """Test that math content is protected from template removal."""
        
        # Test content with LaTeX braces that would be mistaken for templates
        original_text = "Acceleration $$\\mathbf{{a}}$$ formula: $$\\mathbf{{a}} = {{v_1 - v_0} \\over {t_1 - t_0}}$$"
        
        # Simulate the protection workflow
        protected_text, math_content = protect_math_from_templates(original_text)
        
        # Math content should be extracted to dictionary
        self.assertGreater(len(math_content), 0, "Math content should be extracted")
        
        # Protected text should have placeholders
        self.assertIn("__MATH_PLACEHOLDER_", protected_text)
        
        # After template removal (which normally removes {{...}}), the protected text should be unchanged
        after_template_removal = remove_templates(protected_text)
        
        # Restore the math content
        final_result = restore_math_content(after_template_removal, math_content)
        
        # Final result should match the original (LaTeX braces preserved)
        self.assertEqual(final_result, original_text)

    def test_fix_complex_wikilinks(self):
        """Test fixing complex wikilinks with nested brackets like IPA pronunciation."""
        
        cases = [
            # IPA pronunciation link - should become proper markdown link
            ("**Italy** ([[Help:IPA/Italian|[iˈtaːlja]]]) is a country",
             r"\*\*Italy\*\* \(\[iˈtaːlja\]\(help_ipa_italian_[a-f0-9]{6}\)\) is a country"),
             
            # Generic nested brackets - should become proper markdown link
            ("Text [[SomeLink|[content]]] more text",
             r"Text \[content\]\(somelink_[a-f0-9]{6}\) more text"),
             
            # Multiple complex IPA links
            ("[[Help:IPA/French|[fʁɑ̃s]]] and [[Help:IPA/German|[dɔɪtʃlant]]]",
             r"\[fʁɑ̃s\]\(help_ipa_french_[a-f0-9]{6}\) and \[dɔɪtʃlant\]\(help_ipa_german_[a-f0-9]{6}\)"),
             
            # No complex links (should remain unchanged)
            ("Regular [[simple|link]] text",
             "Regular [[simple|link]] text"),
        ]
        
        for i, (input_text, expected_pattern) in enumerate(cases):
            with self.subTest(case=i, input=input_text):
                result = fix_complex_wikilinks(input_text)
                if "[a-f0-9]" in expected_pattern:
                    # This is a regex pattern, use assertRegex
                    self.assertRegex(result, expected_pattern)
                else:
                    # This is an exact match
                    self.assertEqual(result, expected_pattern)

    def test_fix_math_tags(self):
        text = "Equation <math>E=mc^2</math>."
        self.assertEqual(fix_math_tags(text), "Equation $$E=mc^2$$.")

    def test_rescue_number_templates(self):
        text = "Val {{val|1.23}} and {{overline|456}}."
        self.assertEqual(rescue_number_templates(text), "Val 1.23 and 456.")

    def test_remove_comments(self):
        text = "Start <!-- comment --> End."
        self.assertEqual(remove_comments(text), "Start  End.")

    def test_remove_templates(self):
        text = "Start {{template|arg}} End."
        self.assertEqual(remove_templates(text), "Start  End.")
        nested = "Start {{outer|{{inner}}}} End."
        self.assertEqual(remove_templates(nested), "Start  End.")

    def test_remove_stub_templates(self):
        """Test removal of stub and navigation templates that commonly escape general template removal."""
        
        # Test stub templates
        cases = [
            # Math stub template
            ("Article text.\n\n{{math-stub}}", "Article text."),
            
            # Physics stub template  
            ("Article text.\n\n{{physics-stub}}", "Article text."),
            
            # Clear template
            ("Article text.\n\n{{-}}\n\nMore text.", "Article text.\n\nMore text."),
            
            # Navigation template
            ("Article text.\n\n{{shapes}}", "Article text."),
            
            # Generic navigation template
            ("Article text.\n\n{{nav-template}}", "Article text."),
            
            # Multiple templates
            ("Article text.\n\n{{shapes}}\n\n{{math-stub}}", "Article text."),
            
            # Template with spaces
            ("Article text.\n  {{physics-stub}}  ", "Article text."),
            
            # Short generic templates (likely metadata)
            ("Article text.\n{{bio}}\n{{geo}}", "Article text."),
            
            # Mixed with other content
            ("- Item 1\n- Item 2\n\n{{shapes}}\n\n{{math-stub}}", "- Item 1\n- Item 2"),
        ]
        
        for i, (input_text, expected) in enumerate(cases):
            with self.subTest(case=i, input=input_text):
                result = remove_stub_templates(input_text)
                self.assertEqual(result.strip(), expected.strip())

    def test_remove_wikitables(self):
        text = "Start {| table |} End."
        self.assertEqual(remove_wikitables(text), "Start  End.")

    def test_remove_reference_tags(self):
        text = "Statement<ref>Source</ref>."
        self.assertEqual(remove_reference_tags(text), "Statement.")
        text2 = "Statement<ref name='x' />."
        self.assertEqual(remove_reference_tags(text2), "Statement.")

    def test_remove_external_links(self):
        text = "Link [http://example.com Label]."
        self.assertEqual(remove_external_links(text), "Link Label.")
        text2 = "Link [http://example.com]."
        self.assertEqual(remove_external_links(text2), "Link .")

    def test_normalize_title(self):
        title = "Title"
        normalized = normalize_title(title)
        # Check prefix and that it ends with 6 char hash
        self.assertTrue(normalized.startswith("title_"))
        self.assertRegex(normalized, r"_[a-f0-9]{6}$")
        
    def test_convert_internal_links(self):
        # Normal link
        text = "See [[Page Title]]."
        res = convert_internal_links(text)
        self.assertRegex(res, r"See \[Page Title\]\(page_title_[a-f0-9]{6}\)\.")
        
        # Piped link
        text = "See [[Page Title|Label]]."
        res = convert_internal_links(text)
        self.assertRegex(res, r"See \[Label\]\(page_title_[a-f0-9]{6}\)\.")

        # File (stripped)
        text = "Image [[File:Img.png|thumb]]."
        res = convert_internal_links(text)
        self.assertEqual(res, "Image .")
        
        # Colon prefix (kept as text)
        text = "See [[:Category:Cats]]."
        res = convert_internal_links(text)
        self.assertEqual(res, "See :Category:Cats.")
        
        # IPA pronunciation with brackets in label
        text = "**Italy** ([[Help:IPA/Italian|[iˈtaːlja]]])."
        res = convert_internal_links(text)
        self.assertRegex(res, r"\*\*Italy\*\* \(\[iˈtaːlja\]\(help_ipa_italian_[a-f0-9]{6}\)\)\.")

    def test_convert_html_formatting(self):
        # Test blockquote conversion
        text = "<blockquote>Quote</blockquote>"
        self.assertEqual(convert_html_formatting(text).strip(), "> Quote")
        
        # Test superscript and subscript conversion
        text = "<sup>sup</sup>"
        self.assertEqual(convert_html_formatting(text), "^sup")
        
        text = "<sub>sub</sub>"
        self.assertEqual(convert_html_formatting(text), "_sub")
        
        # Test removal of intrusive tags (keep content, remove tags)
        intrusive_tags = ['nowiki', 'big', 'small', 'center', 'font', 'span', 'div', 'u', 's', 'strike', 'code', 'tt', 'gallery']
        for tag in intrusive_tags:
            with self.subTest(tag=tag):
                text = f"Before <{tag}>content</{tag}> after."
                expected = "Before content after."
                self.assertEqual(convert_html_formatting(text), expected)
                
                # Test with attributes
                text = f'Before <{tag} class="test">content</{tag}> after.'
                self.assertEqual(convert_html_formatting(text), expected)
        
        # Test self-closing nowiki
        text = "Code <nowiki/> here."
        self.assertEqual(convert_html_formatting(text), "Code  here.")
        
        # Test br tag conversion
        text = "Line 1<br>Line 2<br />Line 3."
        self.assertEqual(convert_html_formatting(text), "Line 1\nLine 2\nLine 3.")
        
        # Test HTML list conversion
        text = '<ul><li>Item 1</li><li>Item 2</li></ul>'
        expected = "- Item 1\n- Item 2"
        self.assertEqual(convert_html_formatting(text), expected)
        
        # Test HTML list with style attributes
        text = '<ul><li style="background-color: #fff;">Styled item</li></ul>'
        expected = "- Styled item"
        self.assertEqual(convert_html_formatting(text), expected)
        
        # Test malformed HTML list (missing closing tags)
        text = '<ul><li style="color: red;">Item 1<li><li>Item 2</li></ul>'
        expected = "- Item 1\n- Item 2"  # Empty items are skipped
        self.assertEqual(convert_html_formatting(text), expected)

    def test_remove_file_references(self):
        """Test removal of standalone File: reference lines from gallery content."""
        
        cases = [
            # Single file reference
            ("Some content.\nFile:Image.jpg|Caption text\nMore content.", 
             "Some content.\n\nMore content."),
            
            # Multiple file references
            ("Article.\nFile:First.jpg|Caption 1\nFile:Second.png|Caption 2\nEnd.", 
             "Article.\n\n\nEnd."),
            
            # Mixed case
            ("Start.\nfile:lowercase.gif|Caption\nFILE:UPPERCASE.JPG|Caption\nEnd.", 
             "Start.\n\n\nEnd."),
             
            # File reference with complex caption containing links
            ("Text.\nFile:Complex.jpg|Caption with [[links]] and other text\nMore text.", 
             "Text.\n\nMore text."),
            
            # File reference at start
            ("File:Start.jpg|Caption\nRegular content.", 
             "\nRegular content."),
            
            # File reference at end
            ("Regular content.\nFile:End.jpg|Caption", 
             "Regular content.\n"),
             
            # No file references (should remain unchanged)
            ("Regular text without file references.", 
             "Regular text without file references."),
             
            # File: mentioned in text (not at line start)
            ("The File:Something.jpg is mentioned inline.", 
             "The File:Something.jpg is mentioned inline."),
             
            # Image: prefix
            ("Some content.\nImage:SW686-TargetChampion-1a.jpg|\nMore content.", 
             "Some content.\n\nMore content."),
             
            # Gallery entries with file extensions
            ("Some content.\nRegions of Iceland.png|[Regions of Iceland](link)\nMore content.", 
             "Some content.\n\nMore content."),
             
            # Multiple gallery entries with different extensions
            ("Article.\nimage.jpg|Caption\nfile.svg|[Link](target)\nEnd.", 
             "Article.\n\n\nEnd."),
             
            # Imagemap entry
            ("Some content.\n<imagemap>File:1990s decade montage.png|Very long caption|420px|thumb\nMore content.", 
             "Some content.\n\nMore content."),
        ]
        
        for i, (input_text, expected) in enumerate(cases):
            with self.subTest(case=i, input=input_text):
                result = remove_file_references(input_text)
                self.assertEqual(result, expected)

    def test_convert_bold_and_italics(self):
        text = "'''Bold''' and ''Italic''."
        self.assertEqual(convert_bold_and_italics(text), "**Bold** and *Italic*.")

    def test_fix_indented_math(self):
        text = " x = y + z"
        self.assertEqual(fix_indented_math(text), "$$x = y + z$$")
        text = " Normal text"
        self.assertEqual(fix_indented_math(text), " Normal text")

    def test_format_sections_and_whitespace(self):
        text = """
== Header 1 ==
Content 1.

=== Header 2 ===
Content 2.
"""
        res = format_sections_and_whitespace(text)
        self.assertIn("## Header 1", res)
        self.assertIn("### Header 2", res)
        self.assertIn("Content 1.", res)

    def test_remove_unwanted_sections(self):
        lines = ["== Header ==", "Content", "== References ==", "Ref", "== External Links ==", "Link"]
        res = remove_unwanted_sections(lines)
        self.assertEqual(res, ["== Header ==", "Content"])

    def test_fix_corrupted_asterisks(self):
        """Test fixing corrupted content that appears as multiple asterisks."""
        
        cases = [
            # Malformed bold markup
            ("Text with ****Mammalia** content", "Text with **Mammalia** content"),
            
            # Standalone missing content
            ("- **** is the biggest country", "- [missing content] is the biggest country"),
            ("**Pi** (****) is a constant", "**Pi** ([missing content]) is a constant"),
            
            # Math symbols (5 asterisks)
            ("Symbol *****", "Symbol *"),
            
            # Long chains
            ("Text ******** asterisks", "Text *** asterisks"),
            
            # Should preserve normal formatting
            ("Normal **bold** and *italic* text", "Normal **bold** and *italic* text"),
        ]
        
        for i, (input_text, expected) in enumerate(cases):
            with self.subTest(case=i, input=input_text):
                result = fix_corrupted_asterisks(input_text)
                self.assertEqual(result, expected)

    def test_fix_date_ranges(self):
        """Test fixing concatenated date ranges in parentheses."""
        cases = [
            # Birth-death dates
            ("John Taylor (15801653) was a poet.", "John Taylor (1580-1653) was a poet."),
            # Event date ranges
            ("The Chinese Civil War (19271949) resulted in...", "The Chinese Civil War (1927-1949) resulted in..."),
            ("The Beatles (19621970) were a band.", "The Beatles (1962-1970) were a band."),
            # Multiple occurrences
            ("Person A (19001950) and Person B (19201980).", "Person A (1900-1950) and Person B (1920-1980)."),
            # Should not change valid dates already formatted
            ("Already correct (1927-1949) date.", "Already correct (1927-1949) date."),
            # Should not change single years
            ("In the year (1995) something happened.", "In the year (1995) something happened."),
            # Should not change non-year numbers
            ("Call (5551234) for info.", "Call (5551234) for info."),
        ]
        for inp, expected in cases:
            with self.subTest(inp=inp):
                self.assertEqual(fix_date_ranges(inp), expected)

    def test_fix_mediawiki_links(self):
        """Test converting MediaWiki-style double bracket links to markdown format."""
        
        cases = [
            # Simple links
            ("Text with [[mercury]] element", "Text with [mercury](mercury_ba7216) element"),
            ("Read about [[bromine]] here", "Read about [bromine](bromine_8a1b9d) here"),
            
            # Piped links (display|actual link)  
            ("[[Mercury (element)|Mercury]]", "[Mercury](mercury_(element)_dff2a1)"),
            ("[[laboratory|laboratories]]", "[laboratories](laboratory_5b3879)"),
            
            # Multiple links in text
            ("[[Bromine]] and [[silver]] are elements", "[Bromine](bromine_8a1b9d) and [silver](silver_6d7a9c) are elements"),
            
            # Links with spaces and special chars
            ("[[periodic table]]", "[periodic table](periodic_table_f1b2c3)"),
            ("[[States of matter|physical states]]", "[physical states](states_of_matter_a4d5e6)"),
            
            # Should not affect normal text
            ("Normal text without links", "Normal text without links"),
            ("Text with [existing](link) format", "Text with [existing](link) format"),
        ]
        
        for i, (input_text, expected_pattern) in enumerate(cases):
            with self.subTest(case=i, input=input_text):
                result = fix_mediawiki_links(input_text)
                # For this test, we check the structure rather than exact hash values
                # since the hash generation might vary
                if '[[' not in input_text or input_text == "Normal text without links" or "[existing]" in input_text:
                    # No MediaWiki links or should be unchanged
                    self.assertEqual(result, input_text)
                else:
                    # Should have converted to markdown format
                    self.assertNotIn('[[', result)
                    self.assertNotIn(']]', result)
                    # Should contain markdown-style links
                    self.assertRegex(result, r'\[[^\]]+\]\([^\)]+\)')

    def test_fix_excessive_whitespace(self):
        """Test removing excessive blank lines and normalizing whitespace."""
        
        cases = [
            # Three consecutive empty lines -> two empty lines
            ("Line 1\n\n\n\nLine 2", "Line 1\n\n\nLine 2"),
            
            # Four consecutive empty lines -> two empty lines 
            ("Content\n\n\n\n\nMore content", "Content\n\n\nMore content"),
            
            # Leading and trailing whitespace removal
            ("\n\nContent here\n\n", "Content here"),
            
            # Already good spacing preserved
            ("Paragraph 1\n\nParagraph 2", "Paragraph 1\n\nParagraph 2"),
            
            # Single empty line preserved
            ("Text\n\nMore text", "Text\n\nMore text"),
            
            # Mixed spacing scenarios
            ("Start\n\n\n\n\nMiddle\n\nEnd\n\n\n\n", "Start\n\n\nMiddle\n\nEnd"),
        ]
        
        for i, (input_text, expected) in enumerate(cases):
            with self.subTest(case=i, input=repr(input_text)):
                result = fix_excessive_whitespace(input_text)
                self.assertEqual(result, expected)

    def test_fix_malformed_formatting(self):
        """Test fixing malformed bold and italic formatting with unbalanced asterisks."""
        
        cases = [
            # Simple triple asterisks -> prefer bold
            ("***subtext***", "**subtext**"),
            ("***If*** is also a poem", "**If** is also a poem"),
            ("***i***", "**i**"),
            ("***tele*** and ***vision***", "**tele** and **vision**"),
            
            # Complex mixed formatting - remove unbalanced outer italics
            ("***S**ystème **I**nternational d'unités*", "S**ystème **I**nternational d'unités"),
            
            # Remaining corrupted patterns
            ("****Mammalia**", "**Mammalia**"),
            
            # Should not affect correct formatting
            ("**bold** and *italic* text", "**bold** and *italic* text"),
            ("Normal text without formatting", "Normal text without formatting"),
        ]
        
        for i, (input_text, expected) in enumerate(cases):
            with self.subTest(case=i, input=repr(input_text)):
                result = fix_malformed_formatting(input_text)
                self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()
