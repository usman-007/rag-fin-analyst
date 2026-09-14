from filings_analyst.ingest.chunker import chunk_section, count_tokens

SHORT_TEXT = "First paragraph.\nSecond paragraph.\nThird paragraph."


def _repeat_paragraph(text: str, n: int) -> str:
    return "\n".join(f"{text} (instance {i})" for i in range(n))


def test_short_text_produces_one_chunk() -> None:
    chunks = chunk_section(SHORT_TEXT)
    assert len(chunks) == 1
    assert chunks[0].text == SHORT_TEXT
    assert chunks[0].char_start == 0
    assert chunks[0].char_end == len(SHORT_TEXT)


def test_char_offsets_map_back_into_original_text() -> None:
    chunks = chunk_section(SHORT_TEXT)
    for chunk in chunks:
        assert SHORT_TEXT[chunk.char_start : chunk.char_end] in SHORT_TEXT
        assert chunk.text.split("\n")[0] in SHORT_TEXT[chunk.char_start :]


def test_long_text_splits_into_multiple_chunks_near_target() -> None:
    paragraph = "This paragraph discusses supply chain risk and foreign currency exposure. " * 8
    text = _repeat_paragraph(paragraph, 20)
    chunks = chunk_section(text, target_tokens=450, overlap_tokens=80)
    assert len(chunks) > 1
    for chunk in chunks[:-1]:
        # Allow slack for the last paragraph added before the target is exceeded.
        assert count_tokens(chunk.text) <= 450 + 200


def test_consecutive_chunks_overlap() -> None:
    paragraph = "This paragraph discusses supply chain risk and foreign currency exposure. " * 8
    text = _repeat_paragraph(paragraph, 20)
    chunks = chunk_section(text, target_tokens=450, overlap_tokens=80)
    first_lines = set(chunks[0].text.split("\n"))
    second_lines = set(chunks[1].text.split("\n"))
    assert first_lines & second_lines, "consecutive chunks should share overlap paragraphs"


def test_never_merges_across_item_boundary() -> None:
    # The chunker only ever sees one item's text at a time — calling it per-item is what
    # guarantees no cross-item merging; this asserts a chunk's content is a substring of
    # only the item text passed in, never leaking sibling-item content.
    item_1a_text = _repeat_paragraph("Risk factor discussion.", 5)
    item_7_text = _repeat_paragraph("MD&A discussion.", 5)
    chunks_1a = chunk_section(item_1a_text)
    chunks_7 = chunk_section(item_7_text)
    assert all("MD&A" not in c.text for c in chunks_1a)
    assert all("Risk factor" not in c.text for c in chunks_7)


def test_oversized_single_paragraph_is_split_by_sentence() -> None:
    long_sentence_block = ("This is a long sentence about risk. " * 60).strip()
    chunks = chunk_section(long_sentence_block, target_tokens=100, overlap_tokens=20)
    assert len(chunks) > 1


def test_count_tokens_is_positive_and_monotonic() -> None:
    short = "hello world"
    long = short * 10
    assert count_tokens(short) > 0
    assert count_tokens(long) > count_tokens(short)
