"""
Unit tests for GenerationResult data structures.
"""
import pytest
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from experiments.dagseq2dagseq.model.generation_result import (
    GeneratedDocument,
    GenerationResult,
)


class TestGeneratedDocument:
    """Tests for GeneratedDocument dataclass."""
    
    def test_basic_creation_with_tokens(self):
        """Test creating a document with tokens."""
        tokens = np.array([1, 2, 3, 4, 5])
        doc = GeneratedDocument(
            title="Test Document",
            title_normalized="test_document_abc123",
            tokens=tokens,
            text=None,
            source="generated",
            is_root=True,
            parent_title=None,
        )
        
        assert doc.title == "Test Document"
        assert doc.title_normalized == "test_document_abc123"
        assert np.array_equal(doc.tokens, tokens)
        assert doc.text is None
        assert doc.source == "generated"
        assert doc.is_root is True
        assert doc.parent_title is None
    
    def test_basic_creation_with_text(self):
        """Test creating a document with text."""
        doc = GeneratedDocument(
            title="Test Document",
            title_normalized="test_document_abc123",
            tokens=None,
            text="This is test text",
            source="corpus",
            is_root=False,
            parent_title="Parent Document",
        )
        
        assert doc.title == "Test Document"
        assert doc.text == "This is test text"
        assert doc.tokens is None
        assert doc.source == "corpus"
        assert doc.is_root is False
        assert doc.parent_title == "Parent Document"
    
    def test_creation_with_both_tokens_and_text(self):
        """Test creating a document with both tokens and text."""
        tokens = np.array([1, 2, 3])
        doc = GeneratedDocument(
            title="Test",
            title_normalized="test_abc123",
            tokens=tokens,
            text="Test text",
            source="generated",
            is_root=True,
            parent_title=None,
        )
        
        assert doc.tokens is not None
        assert doc.text is not None
    
    def test_source_as_path(self):
        """Test creating a document with Path as source."""
        doc = GeneratedDocument(
            title="Test",
            title_normalized="test_abc123",
            tokens=None,
            text="Text",
            source=Path("/path/to/corpus"),
            is_root=False,
            parent_title="Root",
        )
        
        assert isinstance(doc.source, Path)
        assert doc.source == Path("/path/to/corpus")
    
    def test_validation_both_none(self):
        """Test that having both tokens and text as None raises error."""
        with pytest.raises(ValueError, match="must have at least one"):
            GeneratedDocument(
                title="Test",
                title_normalized="test_abc123",
                tokens=None,
                text=None,
                source="generated",
                is_root=True,
                parent_title=None,
            )


class TestGenerationResult:
    """Tests for GenerationResult dataclass."""
    
    def setup_method(self):
        """Set up test documents."""
        self.root_doc = GeneratedDocument(
            title="Root Document",
            title_normalized="root_document_abc123",
            tokens=np.array([1, 2, 3]),
            text="Root text",
            source="generated",
            is_root=True,
            parent_title=None,
        )
        
        self.aux_doc1 = GeneratedDocument(
            title="Auxiliary 1",
            title_normalized="auxiliary_1_def456",
            tokens=np.array([4, 5, 6]),
            text="Aux 1 text",
            source="generated",
            is_root=False,
            parent_title="Root Document",
        )
        
        self.aux_doc2 = GeneratedDocument(
            title="Auxiliary 2",
            title_normalized="auxiliary_2_ghi789",
            tokens=np.array([7, 8, 9]),
            text="Aux 2 text",
            source="corpus",
            is_root=False,
            parent_title="Root Document",
        )
    
    def test_basic_creation(self):
        """Test creating a GenerationResult."""
        result = GenerationResult(
            root_document=self.root_doc,
            auxiliary_documents=[self.aux_doc1, self.aux_doc2],
            generation_config={"temperature": 1.0},
        )
        
        assert result.root_document == self.root_doc
        assert len(result.auxiliary_documents) == 2
        assert result.generation_config == {"temperature": 1.0}
    
    def test_get_all_documents(self):
        """Test get_all_documents method."""
        result = GenerationResult(
            root_document=self.root_doc,
            auxiliary_documents=[self.aux_doc1, self.aux_doc2],
            generation_config={},
        )
        
        all_docs = result.get_all_documents()
        
        assert len(all_docs) == 3
        assert all_docs[0] == self.root_doc
        assert all_docs[1] == self.aux_doc1
        assert all_docs[2] == self.aux_doc2
    
    def test_get_document_by_title_original(self):
        """Test getting document by original title."""
        result = GenerationResult(
            root_document=self.root_doc,
            auxiliary_documents=[self.aux_doc1, self.aux_doc2],
            generation_config={},
        )
        
        doc = result.get_document_by_title("Auxiliary 1")
        assert doc == self.aux_doc1
    
    def test_get_document_by_title_normalized(self):
        """Test getting document by normalized title."""
        result = GenerationResult(
            root_document=self.root_doc,
            auxiliary_documents=[self.aux_doc1, self.aux_doc2],
            generation_config={},
        )
        
        doc = result.get_document_by_title("auxiliary_1_def456")
        assert doc == self.aux_doc1
    
    def test_get_document_by_title_not_found(self):
        """Test getting document that doesn't exist."""
        result = GenerationResult(
            root_document=self.root_doc,
            auxiliary_documents=[self.aux_doc1],
            generation_config={},
        )
        
        doc = result.get_document_by_title("Nonexistent")
        assert doc is None
    
    def test_get_generated_documents(self):
        """Test getting only generated documents."""
        result = GenerationResult(
            root_document=self.root_doc,
            auxiliary_documents=[self.aux_doc1, self.aux_doc2],
            generation_config={},
        )
        
        generated_docs = result.get_generated_documents()
        
        assert len(generated_docs) == 2
        assert self.root_doc in generated_docs
        assert self.aux_doc1 in generated_docs
        assert self.aux_doc2 not in generated_docs
    
    def test_get_corpus_documents(self):
        """Test getting only corpus documents."""
        result = GenerationResult(
            root_document=self.root_doc,
            auxiliary_documents=[self.aux_doc1, self.aux_doc2],
            generation_config={},
        )
        
        corpus_docs = result.get_corpus_documents()
        
        assert len(corpus_docs) == 1
        assert self.aux_doc2 in corpus_docs
    
    def test_get_corpus_documents_with_path(self):
        """Test getting corpus documents when source is a Path."""
        path_doc = GeneratedDocument(
            title="Path Doc",
            title_normalized="path_doc_xyz000",
            tokens=None,
            text="Text",
            source=Path("/corpus/path"),
            is_root=False,
            parent_title="Root",
        )
        
        result = GenerationResult(
            root_document=self.root_doc,
            auxiliary_documents=[path_doc],
            generation_config={},
        )
        
        corpus_docs = result.get_corpus_documents()
        assert len(corpus_docs) == 1
        assert path_doc in corpus_docs
    
    def test_get_document_count(self):
        """Test getting total document count."""
        result = GenerationResult(
            root_document=self.root_doc,
            auxiliary_documents=[self.aux_doc1, self.aux_doc2],
            generation_config={},
        )
        
        assert result.get_document_count() == 3
    
    def test_get_document_count_no_auxiliary(self):
        """Test document count with no auxiliary documents."""
        result = GenerationResult(
            root_document=self.root_doc,
            auxiliary_documents=[],
            generation_config={},
        )
        
        assert result.get_document_count() == 1
    
    def test_repr(self):
        """Test string representation."""
        result = GenerationResult(
            root_document=self.root_doc,
            auxiliary_documents=[self.aux_doc1, self.aux_doc2],
            generation_config={},
        )
        
        repr_str = repr(result)
        
        assert "GenerationResult" in repr_str
        assert "total_docs=3" in repr_str
        assert "generated=2" in repr_str
        assert "corpus=1" in repr_str
        assert "Root Document" in repr_str
    
    def test_empty_auxiliary_documents(self):
        """Test result with no auxiliary documents."""
        result = GenerationResult(
            root_document=self.root_doc,
            auxiliary_documents=[],
            generation_config={},
        )
        
        assert len(result.get_all_documents()) == 1
        assert result.get_document_count() == 1
        assert len(result.get_generated_documents()) == 1


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_complex_document_graph(self):
        """Test a more complex document graph."""
        root = GeneratedDocument(
            title="Main Article",
            title_normalized="main_article_aaa111",
            tokens=np.array([1, 2, 3]),
            text="Main content",
            source="generated",
            is_root=True,
            parent_title=None,
        )
        
        aux1 = GeneratedDocument(
            title="Reference 1",
            title_normalized="reference_1_bbb222",
            tokens=np.array([4, 5]),
            text="Ref 1",
            source="corpus",
            is_root=False,
            parent_title="Main Article",
        )
        
        aux2 = GeneratedDocument(
            title="Reference 2",
            title_normalized="reference_2_ccc333",
            tokens=np.array([6, 7]),
            text="Ref 2",
            source="generated",
            is_root=False,
            parent_title="Reference 1",
        )
        
        result = GenerationResult(
            root_document=root,
            auxiliary_documents=[aux1, aux2],
            generation_config={"max_link_depth": 2},
        )
        
        # Test various queries
        assert result.get_document_count() == 3
        assert len(result.get_generated_documents()) == 2
        assert len(result.get_corpus_documents()) == 1
        
        # Test finding documents
        assert result.get_document_by_title("Main Article") == root
        assert result.get_document_by_title("reference_1_bbb222") == aux1
        assert result.get_document_by_title("Reference 2") == aux2
        
        # Test ordering
        all_docs = result.get_all_documents()
        assert all_docs[0] == root
        assert all_docs[1] == aux1
        assert all_docs[2] == aux2
