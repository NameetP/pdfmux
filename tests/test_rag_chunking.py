"""Tests for RAG-optimized chunking."""

from __future__ import annotations

from pdfmux.chunking import chunk_for_rag, estimate_tokens


class TestChunkForRag:
    def test_empty_text(self):
        assert chunk_for_rag("") == []
        assert chunk_for_rag("   ") == []

    def test_single_section_under_limit(self):
        text = "# Title\n\nShort paragraph."
        chunks = chunk_for_rag(text, max_tokens=500)
        assert len(chunks) == 1
        assert chunks[0].title == "Title"

    def test_respects_max_tokens(self):
        # Create a section with lots of text
        long_para = "This is a test sentence with several words. " * 50
        text = f"# Long Section\n\n{long_para}"
        chunks = chunk_for_rag(text, max_tokens=100, overlap_tokens=0)
        # Without overlap, chunks should be reasonably close to max
        for chunk in chunks:
            assert chunk.tokens <= 200, f"Chunk has {chunk.tokens} tokens, expected <= ~200"
        # Should produce multiple chunks (original was 549 tokens)
        assert len(chunks) > 3

    def test_multiple_sections(self):
        text = "# Section 1\n\nContent one.\n\n# Section 2\n\nContent two.\n\n# Section 3\n\nContent three."
        chunks = chunk_for_rag(text, max_tokens=500)
        assert len(chunks) == 3
        titles = [c.title for c in chunks]
        assert "Section 1" in titles
        assert "Section 2" in titles
        assert "Section 3" in titles

    def test_overlap_adds_context(self):
        # Need longer sections so overlap is meaningful
        para_a = "First section paragraph one. " * 20
        para_b = "Second section paragraph two. " * 20
        text = f"# Part A\n\n{para_a}\n\n# Part B\n\n{para_b}"
        chunks_no_overlap = chunk_for_rag(text, max_tokens=500, overlap_tokens=0)
        chunks_with_overlap = chunk_for_rag(text, max_tokens=500, overlap_tokens=10)

        if len(chunks_with_overlap) > 1:
            # Overlapped chunk should be longer than non-overlapped
            assert chunks_with_overlap[1].tokens >= chunks_no_overlap[1].tokens

    def test_zero_overlap(self):
        text = "# A\n\nContent A.\n\n# B\n\nContent B."
        chunks = chunk_for_rag(text, overlap_tokens=0)
        if len(chunks) > 1:
            assert not chunks[1].text.startswith("...")

    def test_preserves_confidence(self):
        text = "# Heading\n\nSome text here."
        chunks = chunk_for_rag(text, confidence=0.85)
        assert all(c.confidence == 0.85 for c in chunks)

    def test_preserves_extractor(self):
        text = "# Heading\n\nSome text here."
        chunks = chunk_for_rag(text, extractor="pymupdf4llm")
        assert all(c.extractor == "pymupdf4llm" for c in chunks)

    def test_no_headings_falls_back_to_pages(self):
        text = "Just plain text without any headings at all."
        chunks = chunk_for_rag(text, max_tokens=500)
        assert len(chunks) >= 1

    def test_large_document_chunking(self):
        # Simulate a 20-section document
        sections = []
        for i in range(20):
            para = f"Content for section {i}. " * 20
            sections.append(f"## Section {i}\n\n{para}")
        text = "# Document\n\n" + "\n\n".join(sections)

        chunks = chunk_for_rag(text, max_tokens=200)
        assert len(chunks) > 10  # should split into many chunks
        for chunk in chunks:
            assert chunk.tokens > 0
            assert chunk.title


class TestEstimateTokens:
    def test_empty(self):
        assert estimate_tokens("") == 1  # min 1

    def test_approximation(self):
        # "hello world" = 11 chars → ~2-3 tokens
        tokens = estimate_tokens("hello world")
        assert 1 <= tokens <= 5

    def test_longer_text(self):
        text = "word " * 100  # 500 chars → ~125 tokens
        tokens = estimate_tokens(text)
        assert 100 <= tokens <= 150


class TestChunkPublicAPI:
    """Test the pdfmux.chunk() public API function exists and is importable."""

    def test_chunk_is_importable(self):
        from pdfmux import chunk

        assert callable(chunk)

    def test_chunk_in_all(self):
        import pdfmux

        assert "chunk" in pdfmux.__all__
