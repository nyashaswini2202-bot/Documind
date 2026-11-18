from app.chunker import chunk_pages, split_text


def test_short_text_is_single_chunk():
    assert split_text("hello world", 100, 10) == ["hello world"]


def test_chunks_respect_size_and_cover_text():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 40 for i in range(12))
    chunks = split_text(text, 300, 60)
    assert len(chunks) > 3
    assert all(len(c) <= 300 for c in chunks)
    assert "Paragraph 0." in chunks[0] and "Paragraph 11." in chunks[-1]


def test_overlap_repeats_content():
    text = " ".join(f"w{i}" for i in range(300))
    chunks = split_text(text, 200, 50)
    assert len(chunks) > 2
    overlap_words = set(chunks[0].split()) & set(chunks[1].split())
    assert overlap_words  # consecutive chunks share some words


def test_very_long_word_is_hard_split():
    chunks = split_text("x" * 1000, 200, 20)
    assert all(len(c) <= 200 for c in chunks) and len(chunks) >= 5


def test_chunk_pages_keeps_page_numbers():
    pages = [(1, "alpha " * 100), (3, "beta " * 100)]
    chunks = chunk_pages(pages, 200, 20)
    assert {c["page"] for c in chunks} == {1, 3}
    assert all("beta" not in c["text"] for c in chunks if c["page"] == 1)
