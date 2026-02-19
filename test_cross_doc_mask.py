"""
Test script for cross-document link mask detection and matching.
"""

import torch
import tiktoken
from cross_doc_mask import CrossDocLinkMaskCreator
from dataclasses import dataclass
from typing import List

@dataclass
class MockDocSpan:
    """Mock DocSpan for testing."""
    doc_id: int
    title: str
    clean_title: str
    start: int
    end: int
    truncated: bool = False
    outgoing_titles: List[str] = None

def test_link_detection_and_matching():
    """Test link detection and matching to doc_spans."""

    # Initialize tokenizer and mask creator
    enc = tiktoken.get_encoding('gpt2')
    mask_creator = CrossDocLinkMaskCreator(tokenizer_decode_fn=enc.decode)

    # Create a test sequence simulating a packed batch:
    # Doc 0 (target_doc): "This is the target document."
    # Doc 1: "Here is [a link](target_doc) to the first doc."
    # Doc 2: "Another [ref](target_doc) and [nonexistent](missing_doc)."

    doc0_text = "This is the target document."
    doc1_text = " Here is [a link](target_doc) to the first doc."
    doc2_text = " Another [ref](target_doc) and [nonexistent](missing_doc)."

    full_text = doc0_text + doc1_text + doc2_text
    tokens = enc.encode(full_text)

    print("Full Text:", full_text)
    print("Tokens:", tokens)
    print()

    # Calculate doc spans
    doc0_tokens = enc.encode(doc0_text)
    doc1_tokens = enc.encode(doc1_text)
    doc2_tokens = enc.encode(doc2_text)

    doc_spans = [
        MockDocSpan(doc_id=0, title="target_doc", clean_title="target_doc",
                   start=0, end=len(doc0_tokens)),
        MockDocSpan(doc_id=1, title="linker_doc", clean_title="linker_doc",
                   start=len(doc0_tokens), end=len(doc0_tokens) + len(doc1_tokens)),
        MockDocSpan(doc_id=2, title="another_doc", clean_title="another_doc",
                   start=len(doc0_tokens) + len(doc1_tokens), end=len(tokens)),
    ]

    print("Document Spans:")
    for span in doc_spans:
        print(f"  Doc {span.doc_id} ('{span.clean_title}'): [{span.start}, {span.end})")
    print()

    # Convert to tensor
    input_ids = torch.tensor(tokens, dtype=torch.long)

    # Detect links
    links = mask_creator._detect_links(input_ids)
    print(f"Found {len(links)} link(s):")
    for link in links:
        target_text = enc.decode(input_ids[link.target_start:link.target_end].tolist())
        print(f"  Link at pos {link.link_end_pos} -> target: '{target_text}'")
    print()

    # Match links to docs
    link_to_target = mask_creator._match_links_to_docs(links, input_ids, doc_spans)
    print(f"Matched {len(link_to_target)} links to documents:")
    for link_pos, target_doc_id in link_to_target.items():
        print(f"  Link at pos {link_pos} -> Doc {target_doc_id}")
    print()

    # Build the cross-doc mask
    cross_doc_mask = mask_creator._build_cross_doc_mask(
        len(input_ids), doc_spans, link_to_target, torch.device('cpu')
    )
    print(f"Cross-doc mask shape: {cross_doc_mask.shape}")
    print()

    # Check specific positions
    print("Access check examples:")
    # Token right after first link (pos 16) should be able to access tokens in doc 0
    if cross_doc_mask[16, 2]:  # pos 16 -> pos 2 (in doc 0)
        print(f"  ✓ Position 16 (after first link) can access position 2 in doc 0")
    else:
        print(f"  ✗ Position 16 (after first link) CANNOT access position 2 in doc 0")

    # Token before first link (pos 10) should NOT be able to access doc 0 via link
    if not cross_doc_mask[10, 2]:
        print(f"  ✓ Position 10 (before first link) cannot access position 2 in doc 0 via link")
    else:
        print(f"  ✗ Position 10 (before first link) CAN access position 2 in doc 0 (unexpected)")

    # Token in doc 0 should not have cross-doc link access to doc 1
    if not cross_doc_mask[2, 10]:
        print(f"  ✓ Position 2 (in doc 0) has no cross-doc link to position 10 (doc 1)")
    else:
        print(f"  ✗ Position 2 (in doc 0) has unexpected cross-doc link access")
    print()

    # Test full mask creation
    print("Creating full FlexAttention mask...")
    tokens_2d = torch.cat([input_ids, torch.tensor([0])]).unsqueeze(0)  # Add target token, make 2D
    try:
        block_mask = mask_creator(tokens_2d, doc_spans)
        print(f"  ✓ Successfully created BlockMask")
    except Exception as e:
        print(f"  ✗ Failed to create BlockMask: {e}")

if __name__ == "__main__":
    test_link_detection_and_matching()
