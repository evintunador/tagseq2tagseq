import re
import html
import hashlib

# ======================================================================
# Main Processing Pipeline
# ======================================================================

def process_wikitext(text):
    """
    The main pipeline for processing raw wikitext into clean Markdown.
    Each step is a separate function for clarity and maintainability.
    """
    # Pre-processing
    text = html.unescape(text)
    text = fix_math_tags(text)
    text = rescue_number_templates(text)

    # The order of these operations is important.
    text = remove_comments(text)
    text, math_content = protect_math_from_templates(text)
    text = remove_templates(text)
    text = restore_math_content(text, math_content)
    text = remove_stub_templates(text)
    text = remove_wikitables(text)
    text = remove_reference_tags(text)
    text = remove_external_links(text)
    text = fix_broken_links(text)
    text = convert_internal_links(text)
    text = fix_mediawiki_links(text)
    text = convert_html_formatting(text)
    text = remove_file_references(text)
    text = fix_date_ranges(text)
    text = fix_lists(text)
    text = fix_definition_lists(text)
    text = convert_bold_and_italics(text)
    text = fix_corrupted_asterisks(text)
    text = fix_malformed_formatting(text)
    text = fix_excessive_whitespace(text)
    
    text = fix_indented_math(text)
    text = format_sections_and_whitespace(text)
    return text

# ======================================================================
# Individual Cleaning Steps
# ======================================================================

def fix_math_tags(text):
    """Converts <math>...</math> to $$...$$."""
    return re.sub(r'<math.*?>(.*?)</math>', r'$$\1$$', text, flags=re.DOTALL)

def protect_math_from_templates(text):
    """
    Temporarily protects math content from template removal by replacing it with placeholders.
    Returns the modified text and a dictionary mapping placeholders to original content.
    """
    import re
    
    math_content = {}
    placeholder_counter = 0
    
    def replace_math(match):
        nonlocal placeholder_counter
        content = match.group(1)
        placeholder = f"__MATH_PLACEHOLDER_{placeholder_counter}__"
        math_content[placeholder] = content
        placeholder_counter += 1
        return f"$${placeholder}$$"
    
    # Find and replace all $$....$$ blocks
    text = re.sub(r'\$\$(.*?)\$\$', replace_math, text, flags=re.DOTALL)
    
    return text, math_content

def restore_math_content(text, math_content):
    """
    Restores the original math content by replacing placeholders.
    """
    for placeholder, content in math_content.items():
        text = text.replace(f"$${placeholder}$$", f"$${content}$$")
    
    return text

def rescue_number_templates(text):
    """
    Preserves content of specific numeric templates before they are removed.
    e.g. {{val|0.999...}} -> 0.999...
    """
    # Handle {{val|...}} - often used for numbers with uncertainty or units
    # We capture the first argument.
    text = re.sub(r'\{\{val\|([^|}]+).*?\}\}', r'\1', text, flags=re.IGNORECASE)
    
    # Handle {{overline|...}} - used for repeating decimals
    text = re.sub(r'\{\{overline\|(.*?)\}\}', r'\1', text, flags=re.IGNORECASE)
    
    return text

def fix_indented_math(text):
    """
    Converts lines starting with a space that look like math to $$...$$
    """
    lines = text.split('\n')
    new_lines = []
    for line in lines:
        # Check if line starts with space and has some math-like content
        # Heuristic: starts with space, contains = or + or - or \ or numbers
        # And is not a list item (* or -)
        # Also check for common math symbols including unescaped entities
        if line.startswith(' ') and not line.strip().startswith(('*', '-', '#')):
            stripped = line.strip()
            # Basic check for math symbols: =, +, − (unicode), \, ×, ÷
            if any(x in stripped for x in ['=', '+', '−', '\\', '×', '÷']):
                 new_lines.append(f"$${stripped}$$")
                 continue
        new_lines.append(line)
    return '\n'.join(new_lines)

def remove_comments(text):
    """Removes HTML-style comments."""
    return re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

def remove_templates(text):
    """Removes template invocations (e.g., {{...}})."""
    return drop_nested(text, r'{{', r'}}')

def remove_stub_templates(text):
    """
    Removes specific stub and navigation templates that commonly escape 
    the general template removal, especially at the end of articles.
    These include stub classification templates and navigation boxes.
    """
    # Pattern to match standalone templates at the end of lines or in their own paragraphs
    # This catches templates that might escape the general removal due to formatting
    stub_patterns = [
        # Stub templates
        r'^\s*\{\{[^}]*stub[^}]*\}\}\s*$',
        # Navigation and formatting templates  
        r'^\s*\{\{-\}\}\s*$',          # Clear template
        r'^\s*\{\{clear\}\}\s*$',      # Clear template variant
        r'^\s*\{\{shapes\}\}\s*$',     # Navigation templates
        r'^\s*\{\{[^}]*nav[^}]*\}\}\s*$',  # Navigation boxes
        # Generic short templates that are likely metadata
        r'^\s*\{\{[a-zA-Z-]{1,20}\}\}\s*$',  # Short template names (likely stubs/nav)
    ]
    
    for pattern in stub_patterns:
        text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    return text

def remove_wikitables(text):
    """Removes wikitable syntax ({|...|})."""
    return drop_nested(text, r'{\|', r'\|}')

def remove_reference_tags(text):
    """Removes reference tags (e.g. <ref>...</ref>, <ref name="..."/>)."""
    return re.sub(r'<ref.*?>.*?</ref>|<ref.*?>', '', text, flags=re.DOTALL)

def remove_external_links(text):
    """
    Removes external links, keeping the anchor text if present.
    e.g., "[http://example.com link text]" -> "link text"
    """
    text = re.sub(r'\[https?://[^ ]+\s+(.*?)\]', r'\1', text)
    return re.sub(r'\[https?://[^ ]+\]', '', text)

def fix_broken_links(text):
    """
    Fixes malformed wikilinks that break the parser.
    e.g. [[Link] -> [Link]
    """
    return re.sub(r'\[\[([^\[\]]*?)\](?!\])', r'[\1]', text)

def fix_complex_wikilinks(text):
    """
    Handles complex wikilinks that contain nested brackets before main link processing.
    Specifically targets IPA pronunciation links like [[Help:IPA/Italian|[iˈtaːlja]]].
    Converts them to proper markdown links preserving both the content and target.
    """
    def convert_ipa_link(match):
        full_title = match.group(1)  # e.g., "Help:IPA/Italian"
        content = match.group(2)     # e.g., "iˈtaːlja"
        
        # Normalize the title to create the target URL
        normalized_title = normalize_title(full_title)
        
        # Return the proper markdown link
        return f'[{content}]({normalized_title})'
    
    # Pattern for IPA pronunciation links: [[Help:IPA/Language|[pronunciation]]]
    # Convert to proper markdown links: [pronunciation](help_ipa_language_hash)
    text = re.sub(
        r'\[\[(Help:IPA/[^|]+)\|\[([^\]]+)\]\]\]',
        convert_ipa_link,
        text,
        flags=re.IGNORECASE
    )
    
    # More general pattern for any link with nested brackets in the display text
    # [[link|[content]]] -> [content](link_normalized)
    def convert_general_link(match):
        title = match.group(1)
        content = match.group(2)
        normalized_title = normalize_title(title)
        return f'[{content}]({normalized_title})'
    
    text = re.sub(
        r'\[\[([^|]+)\|\[([^\]]+)\]\]\]',
        convert_general_link,
        text
    )
    
    return text

def convert_internal_links(text):
    """
    Replaces internal links: [[link|text]] -> [text](link)
    """
    # First handle complex IPA-style links with nested brackets
    text = fix_complex_wikilinks(text)
    
    cur = 0
    res = ''
    # match tail after wikilink, e.g. the 's' in [[apple]]s
    tail_re = re.compile(r'\w+')
    
    for s, e in find_balanced(text, ['[['], [']]']):
        m = tail_re.match(text, e)
        trail = m.group(0) if m else ''
        end = m.end() if m else e
        
        inner = text[s + 2:e - 2]
        pipe = inner.rfind('|')
        
        title = inner[:pipe].rstrip() if pipe > -1 else inner
        label = inner[pipe + 1:].strip() if pipe > -1 else title
        
        # Clean brackets from label if it's enclosed (like IPA pronunciation)
        if label.startswith('[') and label.endswith(']'):
            label = label[1:-1]
        
        # Clean title for checking prefixes
        clean_title = title.strip().lower()

        # Handle wikt: prefix
        if clean_title.startswith('wikt:'):
            # preserve original casing/spacing after prefix if needed, 
            # but usually we just want the content.
            # Find where the prefix ends in the original title
            prefix_match = re.match(r'\s*wikt:', title, re.IGNORECASE)
            if prefix_match:
                title = title[prefix_match.end():]
            
            if pipe == -1:
                label = title

        # Check for standard prefixes (embedded images/files/categories)
        # These are typically stripped entirely.
        if any(clean_title.startswith(p) for p in ['file:', 'image:', 'category:', 'media:']):
            res += text[cur:s] 
        
        # Check for colon prefixes (text links to files/images/categories)
        # e.g. [[:Image:Foo.jpg|Label]] -> Label
        elif any(clean_title.startswith(':' + p) for p in ['file:', 'image:', 'category:', 'media:']):
            res += f"{text[cur:s]}{label}"

        else:
            encoded_title = normalize_title(title)
            res += f"{text[cur:s]}[{label}]({encoded_title}){trail}"
            
        cur = end
        
    return res + text[cur:]

def clean_leftover_wikilinks(text):
    """
    Cleans up any remaining Wikipedia-style [[link]] syntax that escaped the main converter.
    This is a safety net for edge cases like nested brackets or malformed links.
    """
    # Simple approach: remove any remaining [[ ]] brackets
    # These should only be leftover artifacts from failed conversions
    text = re.sub(r'\[\[', '', text)
    text = re.sub(r'\]\]', '', text)
    
    return text

def convert_html_formatting(text):
    """
    Converts HTML tags to Markdown or removes them.
    """
    # Blockquotes
    def quote_repl(match):
        content = match.group(1)
        return '\n' + '\n'.join(f"> {line.strip()}" for line in content.splitlines() if line.strip()) + '\n'
    
    text = re.sub(r'<blockquote.*?>(.*?)</blockquote>', quote_repl, text, flags=re.DOTALL | re.IGNORECASE)
    
    # Superscripts and Subscripts
    text = re.sub(r'<sup.*?>(.*?)</sup>', r'^\1', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<sub.*?>(.*?)</sub>', r'_\1', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove intrusive tags but keep content
    # Includes nowiki, big, small, center, font, span, div, u, s, strike, code, tt, gallery
    text = re.sub(r'</?(?:nowiki|big|small|center|font|span|div|u|s|strike|code|tt|gallery)\b.*?>', '', text, flags=re.IGNORECASE)
    
    # Remove self-closing nowiki and br
    text = re.sub(r'<nowiki\s*/>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    
    # Convert HTML lists to Markdown
    text = convert_html_lists(text)
    
    return text

def convert_html_lists(text):
    """
    Converts HTML ul/ol and li tags to Markdown list format.
    Removes style attributes and other HTML attributes.
    Handles both well-formed and malformed HTML.
    """
    def process_list(match):
        list_content = match.group(1)
        list_items = []
        
        # Split by <li> tags to find all items, handling both well-formed and malformed
        # This approach handles cases where </li> might be missing or malformed
        li_splits = re.split(r'<li(?:\s[^>]*)?>', list_content, flags=re.IGNORECASE)
        
        # The first element is usually empty (before the first <li>)
        for item_content in li_splits[1:]:  # Skip the first empty part
            # Remove any closing tags and clean up
            clean_item = re.sub(r'</li>|<li\b[^>]*>', '', item_content, flags=re.IGNORECASE).strip()
            if clean_item:
                list_items.append(f"- {clean_item}")
        
        return '\n'.join(list_items)
    
    # Handle unordered lists
    text = re.sub(r'<ul\b[^>]*>(.*?)</ul>', process_list, text, flags=re.DOTALL | re.IGNORECASE)
    
    # Handle ordered lists (convert to unordered for simplicity)
    text = re.sub(r'<ol\b[^>]*>(.*?)</ol>', process_list, text, flags=re.DOTALL | re.IGNORECASE)
    
    return text

def remove_file_references(text):
    """
    Removes standalone file reference lines that remain from gallery processing.
    These are lines like 'File:Image.jpg|Caption text' that are image metadata
    and not useful content for LLM training.
    """
    # Remove lines that start with File: (case insensitive)
    # These are typically from gallery content where the gallery tags were removed
    # but the file references remained
    text = re.sub(r'^File:[^\n]*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    # Remove lines that start with Image: (alternative prefix)
    text = re.sub(r'^Image:[^\n]*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    # Remove lines with image file extensions followed by pipe (leftover gallery entries)
    # Examples: "filename.png|caption", "image.jpg|[Link](target)"
    image_extensions = r'(?:png|jpg|jpeg|gif|svg|webp|bmp|tiff)'
    text = re.sub(rf'^[^|\n]*\.{image_extensions}\|[^\n]*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    # Remove imagemap entries (complex gallery syntax)
    # Example: "<imagemap>File:filename.ext|very long caption|420px|thumb"
    text = re.sub(r'^<imagemap>[^\n]*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    return text

def fix_corrupted_asterisks(text):
    """
    Fixes corrupted content that appears as multiple asterisks.
    These are usually placeholders for missing symbols or malformed markup.
    """
    # Fix malformed bold markup: ****text** -> **text**
    text = re.sub(r'\*\*\*\*([^*]+)\*\*(?!\*)', r'**\1**', text)
    
    # Clean up long chains of asterisks first (6+ asterisks) -> reduce to 3
    text = re.sub(r'(?<!\*)\*{6,}(?!\*)', '***', text)
    
    # Fix ***** (5 asterisks) which might be math symbols
    text = re.sub(r'\*\*\*\*\*', '*', text)  # Replace with single asterisk (common math symbol)
    
    # Replace standalone **** with placeholder text for missing content
    # This should now only catch actual 4-asterisk patterns
    text = re.sub(r'(?<!\*)(\*{4})(?!\*)', '[missing content]', text)
    
    return text

def fix_mediawiki_links(text):
    """
    Converts remaining MediaWiki-style double bracket links to markdown format.
    Converts [[link]] to [link](link_hash) and [[display|link]] to [display](link_hash).
    Removes File/Image references which should have been handled earlier.
    """
    import hashlib
    
    def generate_link_id(link_text):
        """Generate a hash-based ID for the link, similar to existing link format"""
        # Clean the link text and create a hash
        clean_text = link_text.lower().strip().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
        hash_obj = hashlib.md5(clean_text.encode())
        return clean_text + '_' + hash_obj.hexdigest()[:6]
    
    # First, remove File/Image references that weren't caught earlier
    # These are complex and often contain nested links, so we need to handle them carefully
    # Use a more sophisticated approach for File links that might span multiple lines or have nested content
    
    def remove_file_links(text):
        # Handle nested File/Image links by finding balanced brackets
        result = []
        i = 0
        while i < len(text):
            # Look for [[File: or [[Image: pattern
            if i <= len(text) - 8 and text[i:i+7] == '[[File:':
                # Find the matching closing ]]
                bracket_count = 2  # We've seen [[
                j = i + 2
                while j < len(text) - 1 and bracket_count > 0:
                    if text[j:j+2] == '[[':
                        bracket_count += 2
                        j += 2
                    elif text[j:j+2] == ']]':
                        bracket_count -= 2
                        j += 2
                    else:
                        j += 1
                
                if bracket_count == 0:
                    # Skip the entire File link
                    i = j
                    continue
            elif i <= len(text) - 9 and text[i:i+8] == '[[Image:':
                # Same logic for Image links
                bracket_count = 2
                j = i + 2
                while j < len(text) - 1 and bracket_count > 0:
                    if text[j:j+2] == '[[':
                        bracket_count += 2
                        j += 2
                    elif text[j:j+2] == ']]':
                        bracket_count -= 2
                        j += 2
                    else:
                        j += 1
                
                if bracket_count == 0:
                    # Skip the entire Image link
                    i = j
                    continue
            
            result.append(text[i])
            i += 1
        
        return ''.join(result)
    
    text = remove_file_links(text)
    
    # Handle piped links first: [[display text|actual link]] -> [display text](actual_link_hash)
    def replace_piped_link(match):
        full_content = match.group(1)
        
        # Skip if this looks like a remaining file reference
        if full_content.startswith(('File:', 'Image:')):
            return ''
            
        if '|' in full_content:
            display_text, link_text = full_content.split('|', 1)
            link_id = generate_link_id(link_text)
            return f'[{display_text}]({link_id})'
        else:
            # Simple link: [[text]] -> [text](text_hash)
            link_id = generate_link_id(full_content)
            return f'[{full_content}]({link_id})'
    
    # Replace all MediaWiki-style links
    text = re.sub(r'\[\[([^\]]+)\]\]', replace_piped_link, text)
    
    return text

def fix_excessive_whitespace(text):
    """
    Removes excessive blank lines and normalizes whitespace.
    Converts 3+ consecutive empty lines to 2 empty lines maximum.
    """
    # Split into lines and process
    lines = text.split('\n')
    result = []
    empty_line_count = 0
    
    for line in lines:
        if line.strip() == '':
            empty_line_count += 1
            # Allow maximum of 2 consecutive empty lines
            if empty_line_count <= 2:
                result.append(line)
        else:
            empty_line_count = 0
            result.append(line)
    
    # Also clean up whitespace at the very beginning and end
    while result and result[0].strip() == '':
        result.pop(0)
    while result and result[-1].strip() == '':
        result.pop()
    
    return '\n'.join(result)

def fix_malformed_formatting(text):
    """
    Fixes malformed bold and italic formatting with unbalanced asterisks.
    Common issues:
    - ***word*** -> **word** (prefer bold for emphasis)  
    - ***S**ystème **I**nternational* -> **S**ystème **I**nternational (remove unbalanced outer italics)
    - ****word** -> **word** (catch any remaining corrupted patterns)
    """
    # First, fix complex mixed formatting patterns before simpler ones  
    # Pattern: ***content with **bold** inside ending with single *
    # Example: ***S**ystème **I**nternational d'unités* -> S**ystème **I**nternational d'unités
    # More specific pattern - matches content with multiple ** pairs inside
    # Example: ***S**ystème **I**nternational d'unités* has 3 ** pairs
    text = re.sub(r'\*\*\*([^*]+(?:\*\*[^*]+)+)\*(?!\*)', r'\1', text)
    
    # Fix simple triple asterisks around words (prefer bold for emphasis)
    # ***word*** -> **word**
    text = re.sub(r'\*\*\*([^*]+)\*\*\*', r'**\1**', text)
    
    # Clean up any remaining 4+ asterisk patterns that might have been missed
    text = re.sub(r'\*{4,}([^*]+)\*\*', r'**\1**', text)
    
    return text

def fix_date_ranges(text):
    """
    Fixes concatenated date ranges in parentheses.
    e.g., (19271949) -> (1927-1949)
    Commonly appears for birth-death dates or event date ranges.
    """
    # Match 4-digit year pairs in parentheses and insert a dash
    text = re.sub(r'\((\d{4})(\d{4})\)', r'(\1-\2)', text)
    return text

def normalize_title(title):
    """
    Normalizes a title for use in filenames and link targets.
    Strict normalization to ensure alignment:
    - Lowercase
    - Replace spaces and special characters with underscores
    - Limit length
    - Appends a hash of the pre-stripped title to ensure distinct documents
      (e.g. 'A+B' vs 'A-B') don't collide.
    """
    # Decode HTML entities
    title = html.unescape(title)
    
    # Pre-normalization canonicalization (soft) to determine identity
    # This handles case-insensitivity and space/underscore equivalence
    canonical = title.lower().strip().replace(' ', '_')
    
    # Compute hash of the canonical form to distinguish different symbols
    # e.g. "a+b" vs "a-b" which both normalize to "a_b" below.
    # We use MD5 and take the first 6 chars for a compact suffix.
    title_hash = hashlib.md5(canonical.encode('utf-8')).hexdigest()[:6]

    # Apply strict normalization for the filename part
    clean_title = canonical
    
    # Replace invalid chars with underscores (keep only alphanumeric, hyphen, underscore)
    clean_title = re.sub(r'[^a-z0-9\-_]', '_', clean_title)
    # Collapse underscores
    clean_title = re.sub(r'__+', '_', clean_title)
    # Strip leading/trailing underscores
    clean_title = clean_title.strip('_')
    
    # Limit length (leave room for hash and separator)
    # 200 - 1 (separator) - 6 (hash) = 193
    if len(clean_title) > 193:
        clean_title = clean_title[:193]
        
    return f"{clean_title}_{title_hash}"

def convert_bold_and_italics(text):
    """Converts wikitext bold/italics to Markdown."""
    text = re.sub(r"'''(.*?)'''", r'**\1**', text) # Bold
    return re.sub(r"''(.*?)''", r'*\1*', text)   # Italics

def fix_lists(text):
    """
    Converts wikitext list items to Markdown.
    - Converts • bullets to -
    - Converts : indentation/definition lists to > blockquotes
    - Converts * lists to - (handling nesting)
    - Converts # lists to 1. (handling nesting)
    """
    # Fix bullet points using •
    text = re.sub(r'^•\s*', '- ', text, flags=re.MULTILINE)
    
    # Fix definition lists/indentation with :
    text = re.sub(r'^:\s*', '> ', text, flags=re.MULTILINE)
    
    # Fix nested lists (process deepest first)
    # Wikitext lists (*, **, ***)
    text = re.sub(r'^\*{3}\s*', '    - ', text, flags=re.MULTILINE)
    text = re.sub(r'^\*{2}\s*', '  - ', text, flags=re.MULTILINE)
    text = re.sub(r'^\*\s*', '- ', text, flags=re.MULTILINE)
    
    # Wikitext numbered lists (#, ##, ###)
    text = re.sub(r'^#{3}\s*', '    1. ', text, flags=re.MULTILINE)
    text = re.sub(r'^#{2}\s*', '  1. ', text, flags=re.MULTILINE)
    text = re.sub(r'^#\s*', '1. ', text, flags=re.MULTILINE)
    
    return text

def fix_definition_lists(text):
    """
    Converts MediaWiki definition lists to Markdown subheadings.
    Definition terms starting with ';' become level 4 headers (####).
    This provides better structure for LLM training than raw semicolon terms.
    """
    # Convert lines starting with ; to #### headers
    # These are MediaWiki definition list terms that should be proper subheadings
    text = re.sub(r'^;(.+)$', r'#### \1', text, flags=re.MULTILINE)
    
    return text

def format_sections_and_whitespace(text):
    """
    Removes unwanted sections, handles section headers, cleans up whitespace,
    and removes empty sections.
    """
    lines = text.split('\n')
    lines = remove_unwanted_sections(lines)

    # Group lines into sections. The first "section" is the intro.
    sections = []
    current_lines = []
    for line in lines:
        if re.match(r'^==+.*==+$', line.strip()):
            if current_lines:
                sections.append(current_lines)
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines:
        sections.append(current_lines)

    # Process and filter sections
    output_parts = []
    for section_lines in sections:
        # Check for content. A section is empty if it has no text lines with content.
        first_line_is_header = re.match(r'^==+.*==+$', section_lines[0].strip())
        content_lines = section_lines[1:] if first_line_is_header else section_lines
        
        if not any(line.strip() for line in content_lines):
            continue # Skip empty sections

        # If not empty, process the section
        # Strip trailing whitespace from all lines
        processed_section_lines = [l.rstrip() for l in section_lines]

        if first_line_is_header:
            header_line = processed_section_lines.pop(0)
            m = re.match(r'^(==+)\s*(.*?)\s*\1$', header_line)
            if m:
                # This is a well-formed header
                title = m.group(2)
                level = len(m.group(1))
                # Format header and join with its content
                output_parts.append(f"{'#' * level} {title}\n" + '\n'.join(processed_section_lines))
            else:
                # This is a malformed header, treat it as plain text to avoid crashing
                output_parts.append(header_line + '\n' + '\n'.join(processed_section_lines))
        else:
            # This is the intro content
            output_parts.append('\n'.join(processed_section_lines))

    # Join everything with proper spacing
    full_text = '\n\n'.join(part.strip() for part in output_parts if part.strip())
    return full_text

# ======================================================================
# Helper Functions
# ======================================================================

def remove_unwanted_sections(lines):
    """
    Removes sections that are not useful, like 'References' and 'See also'.
    """
    unwanted_headers = [
        "references", "see also", "sources", "citations", 
        "external links", "further reading", "bibliography",
        "notes", "footnotes", "other websites"
    ]
    
    result_lines = []
    skip_section = False
    
    for line in lines:
        header_match = re.match(r'^(==+)\s*(.*?)\s*\1$', line)
        
        if header_match:
            header_text = header_match.group(2).strip().lower()
            skip_section = any(unwanted in header_text for unwanted in unwanted_headers)
        
        if not skip_section:
            result_lines.append(line)
    
    return result_lines

def drop_nested(text, open_delim, close_delim):
    """
    Removes all occurrences of nested delimited text blocks.
    e.g., drop_nested("a {{b {{c}} d}} e", r'{{', r'}}') -> "a  e"
    """
    open_re = re.compile(open_delim)
    close_re = re.compile(close_delim)
    
    events = sorted(
        [(m.start(), 1, m.end() - m.start()) for m in open_re.finditer(text)] +
        [(m.start(), -1, m.end() - m.start()) for m in close_re.finditer(text)]
    )
    
    level = 0
    start = -1
    spans_to_drop = []

    for pos, type, length in events:
        if level == 0 and type == 1:
            start = pos
        level += type
        if level == 0 and start != -1:
            spans_to_drop.append((start, pos + length))
            start = -1
            
    for s, e in reversed(spans_to_drop):
        text = text[:s] + text[e:]
            
    return text

def find_balanced(text, open_delim_list, close_delim_list):
    """
    An iterator which identifies balanced opening and closing delimiters.
    """
    open_pat = '|'.join([re.escape(x) for x in open_delim_list])
    after_pat = {o: re.compile(open_pat + '|' + c, re.DOTALL) for o, c in zip(open_delim_list, close_delim_list)}
    
    stack = []
    start = 0
    cur = 0
    start_set = False
    start_pat = re.compile(open_pat)
    next_pat = start_pat
    
    while True:
        next_match = next_pat.search(text, cur)
        if not next_match:
            return
            
        if not start_set:
            start = next_match.start()
            start_set = True
            
        delim = next_match.group(0)
        
        if delim in open_delim_list:
            stack.append(delim)
            next_pat = after_pat[delim]
        else:
            if not stack:
                cur = next_match.end()
                continue

            stack.pop()
            
            if not stack:
                yield start, next_match.end()
                next_pat = start_pat
                start_set = False
        
        cur = next_match.end()
