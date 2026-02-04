# SPDX-License-Identifier: MIT
"""Tests for Neumann data types."""

import pytest

from neumann.types import (
    AggregateResult,
    BatchOperationResult,
    BlobStats,
    CentralityItem,
    CentralityResult,
    ChainBlockInfo,
    ChainCodebookInfo,
    ChainCommitted,
    ChainConflictResolution,
    ChainDrift,
    ChainHeight,
    ChainHistory,
    ChainHistoryEntry,
    ChainMergeResult,
    ChainRolledBack,
    ChainSimilar,
    ChainSimilarItem,
    ChainTip,
    ChainTransactionBegun,
    ChainTransitionAnalysis,
    CheckpointInfo,
    CommunitiesResult,
    CommunityItem,
    ConstraintItem,
    ConstraintsResult,
    Edge,
    EdgeBinding,
    GraphIndexesResult,
    Node,
    NodeBinding,
    PageRankItem,
    PageRankResult,
    Path,
    PathBinding,
    PathSegment,
    PatternMatchResult,
    PatternMatchStats,
    QueryResult,
    QueryResultType,
    Row,
    ScalarType,
    SimilarItem,
    UnifiedItem,
    UnifiedResult,
    Value,
)


class TestValue:
    """Tests for Value class."""

    def test_null_value(self) -> None:
        """Test null value creation."""
        v = Value.null()
        assert v.type == ScalarType.NULL
        assert v.data is None
        assert v.as_python() is None

    def test_int_value(self) -> None:
        """Test integer value creation."""
        v = Value.int_(42)
        assert v.type == ScalarType.INT
        assert v.data == 42
        assert v.as_python() == 42

    def test_float_value(self) -> None:
        """Test float value creation."""
        v = Value.float_(3.14)
        assert v.type == ScalarType.FLOAT
        assert v.data == 3.14
        assert v.as_python() == 3.14

    def test_string_value(self) -> None:
        """Test string value creation."""
        v = Value.string("hello")
        assert v.type == ScalarType.STRING
        assert v.data == "hello"
        assert v.as_python() == "hello"

    def test_bool_value(self) -> None:
        """Test boolean value creation."""
        v = Value.bool_(True)
        assert v.type == ScalarType.BOOL
        assert v.data is True
        assert v.as_python() is True

    def test_bytes_value(self) -> None:
        """Test bytes value creation."""
        v = Value.bytes_(b"data")
        assert v.type == ScalarType.BYTES
        assert v.data == b"data"
        assert v.as_python() == b"data"

    def test_value_is_frozen(self) -> None:
        """Test that Value is immutable."""
        v = Value.int_(42)
        with pytest.raises(AttributeError):
            v.data = 100  # type: ignore


class TestRow:
    """Tests for Row class."""

    def test_empty_row(self) -> None:
        """Test empty row creation."""
        row = Row()
        assert len(row.values) == 0
        assert row.get("missing") is None

    def test_row_with_values(self) -> None:
        """Test row with values."""
        row = Row(
            values={
                "name": Value.string("Alice"),
                "age": Value.int_(30),
            }
        )
        assert row.get("name") == Value.string("Alice")
        assert row.get_string("name") == "Alice"
        assert row.get_int("age") == 30

    def test_row_to_dict(self) -> None:
        """Test row conversion to dict."""
        row = Row(
            values={
                "name": Value.string("Bob"),
                "active": Value.bool_(True),
            }
        )
        d = row.to_dict()
        assert d == {"name": "Bob", "active": True}

    def test_row_get_typed_methods(self) -> None:
        """Test typed getter methods."""
        row = Row(
            values={
                "int_val": Value.int_(100),
                "float_val": Value.float_(1.5),
                "str_val": Value.string("test"),
                "bool_val": Value.bool_(False),
            }
        )
        assert row.get_int("int_val") == 100
        assert row.get_float("float_val") == 1.5
        assert row.get_string("str_val") == "test"
        assert row.get_bool("bool_val") is False

        # Wrong type returns None
        assert row.get_int("str_val") is None
        assert row.get_string("int_val") is None


class TestNode:
    """Tests for Node class."""

    def test_node_creation(self) -> None:
        """Test node creation."""
        node = Node(
            id="n1",
            label="Person",
            properties={"name": Value.string("Alice")},
        )
        assert node.id == "n1"
        assert node.label == "Person"
        assert node.get_property("name") == Value.string("Alice")

    def test_node_to_dict(self) -> None:
        """Test node conversion to dict."""
        node = Node(
            id="n2",
            label="User",
            properties={"age": Value.int_(25)},
        )
        d = node.to_dict()
        assert d == {
            "id": "n2",
            "label": "User",
            "properties": {"age": 25},
        }


class TestEdge:
    """Tests for Edge class."""

    def test_edge_creation(self) -> None:
        """Test edge creation."""
        edge = Edge(
            id="e1",
            edge_type="KNOWS",
            source="n1",
            target="n2",
            properties={"since": Value.int_(2020)},
        )
        assert edge.id == "e1"
        assert edge.edge_type == "KNOWS"
        assert edge.source == "n1"
        assert edge.target == "n2"
        assert edge.get_property("since") == Value.int_(2020)

    def test_edge_to_dict(self) -> None:
        """Test edge conversion to dict."""
        edge = Edge(
            id="e2",
            edge_type="FOLLOWS",
            source="a",
            target="b",
        )
        d = edge.to_dict()
        assert d == {
            "id": "e2",
            "type": "FOLLOWS",
            "source": "a",
            "target": "b",
            "properties": {},
        }


class TestPath:
    """Tests for Path class."""

    def test_empty_path(self) -> None:
        """Test empty path."""
        path = Path()
        assert len(path) == 0
        assert path.nodes == []
        assert path.edges == []

    def test_path_with_segments(self) -> None:
        """Test path with segments."""
        n1 = Node(id="1", label="A")
        n2 = Node(id="2", label="B")
        e1 = Edge(id="e1", edge_type="R", source="1", target="2")

        path = Path(
            segments=[
                PathSegment(node=n1, edge=e1),
                PathSegment(node=n2),
            ]
        )

        assert len(path) == 2
        assert path.nodes == [n1, n2]
        assert path.edges == [e1]


class TestSimilarItem:
    """Tests for SimilarItem class."""

    def test_similar_item(self) -> None:
        """Test similar item creation."""
        item = SimilarItem(
            key="vec1",
            score=0.95,
            metadata={"source": Value.string("doc1")},
        )
        assert item.key == "vec1"
        assert item.score == 0.95
        assert item.metadata["source"] == Value.string("doc1")


class TestQueryResult:
    """Tests for QueryResult class."""

    def test_empty_result(self) -> None:
        """Test empty result."""
        result = QueryResult(QueryResultType.EMPTY)
        assert result.is_empty
        assert not result.is_error
        assert result.rows == []

    def test_rows_result(self) -> None:
        """Test rows result."""
        rows = [Row(values={"x": Value.int_(1)})]
        result = QueryResult(QueryResultType.ROWS, rows)
        assert result.type == QueryResultType.ROWS
        assert result.rows == rows
        assert result.nodes == []

    def test_count_result(self) -> None:
        """Test count result."""
        result = QueryResult(QueryResultType.COUNT, 42)
        assert result.count == 42

    def test_error_result(self) -> None:
        """Test error result."""
        result = QueryResult(QueryResultType.ERROR, "something went wrong")
        assert result.is_error
        assert result.error_message == "something went wrong"

    def test_ids_result(self) -> None:
        """Test IDs result."""
        result = QueryResult(QueryResultType.IDS, ["id1", "id2", "id3"])
        assert result.ids == ["id1", "id2", "id3"]

    def test_table_list_result(self) -> None:
        """Test table list result."""
        result = QueryResult(QueryResultType.TABLE_LIST, ["users", "products"])
        assert result.table_names == ["users", "products"]

    def test_nodes_result(self) -> None:
        """Test nodes result."""
        nodes = [Node(id="n1", label="Person")]
        result = QueryResult(QueryResultType.NODES, nodes)
        assert result.nodes == nodes

    def test_edges_result(self) -> None:
        """Test edges result."""
        edges = [Edge(id="e1", edge_type="KNOWS", source="n1", target="n2")]
        result = QueryResult(QueryResultType.EDGES, edges)
        assert result.edges == edges

    def test_paths_result(self) -> None:
        """Test paths result."""
        n1 = Node(id="n1", label="A")
        path = Path(segments=[PathSegment(node=n1)])
        result = QueryResult(QueryResultType.PATHS, [path])
        assert result.paths == [path]

    def test_similar_items_result(self) -> None:
        """Test similar items result."""
        items = [SimilarItem(key="k1", score=0.9)]
        result = QueryResult(QueryResultType.SIMILAR, items)
        assert result.similar_items == items

    def test_blob_data_result(self) -> None:
        """Test blob data result."""
        result = QueryResult(QueryResultType.BLOB, b"binary data")
        assert result.blob_data == b"binary data"

    def test_blob_info_result(self) -> None:
        """Test blob info result."""
        from neumann.types import ArtifactInfo

        info = ArtifactInfo(
            artifact_id="art-1",
            filename="file.txt",
            size=100,
            checksum="abc",
            content_type="text/plain",
            created_at=1234567890,
            tags=["tag1"],
        )
        result = QueryResult(QueryResultType.BLOB_INFO, info)
        assert result.blob_info == info

    def test_result_wrong_type_accessors(self) -> None:
        """Test accessors return empty/None for wrong types."""
        # COUNT result
        count_result = QueryResult(QueryResultType.COUNT, 42)
        assert count_result.nodes == []
        assert count_result.edges == []
        assert count_result.paths == []
        assert count_result.similar_items == []
        assert count_result.ids == []
        assert count_result.table_names == []
        assert count_result.blob_data is None
        assert count_result.blob_info is None
        assert count_result.error_message is None

    def test_result_none_data_accessors(self) -> None:
        """Test accessors return empty/None when data is None."""
        # ROWS with None data
        rows_result = QueryResult(QueryResultType.ROWS, None)
        assert rows_result.rows == []

        # COUNT with None data
        count_result = QueryResult(QueryResultType.COUNT, None)
        assert count_result.count is None


class TestRowGetters:
    """Tests for Row getter methods with edge cases."""

    def test_get_int_wrong_type(self) -> None:
        """Test get_int returns None for wrong type."""
        row = Row(values={"name": Value.string("Alice")})
        assert row.get_int("name") is None

    def test_get_float_wrong_type(self) -> None:
        """Test get_float returns None for wrong type."""
        row = Row(values={"name": Value.string("Alice")})
        assert row.get_float("name") is None

    def test_get_string_wrong_type(self) -> None:
        """Test get_string returns None for wrong type."""
        row = Row(values={"count": Value.int_(42)})
        assert row.get_string("count") is None

    def test_get_bool_wrong_type(self) -> None:
        """Test get_bool returns None for wrong type."""
        row = Row(values={"name": Value.string("Alice")})
        assert row.get_bool("name") is None

    def test_get_missing_column(self) -> None:
        """Test getters return None for missing columns."""
        row = Row(values={})
        assert row.get_int("missing") is None
        assert row.get_float("missing") is None
        assert row.get_string("missing") is None
        assert row.get_bool("missing") is None


class TestUnifiedTypes:
    """Tests for UnifiedItem and UnifiedResult."""

    def test_unified_item_creation(self) -> None:
        """Test UnifiedItem creation."""
        item = UnifiedItem(
            entity_type="node",
            key="n1",
            fields={"name": "Alice"},
            score=0.95,
        )
        assert item.entity_type == "node"
        assert item.key == "n1"
        assert item.fields == {"name": "Alice"}
        assert item.score == 0.95

    def test_unified_item_default_fields(self) -> None:
        """Test UnifiedItem with default fields."""
        item = UnifiedItem(entity_type="edge", key="e1")
        assert item.fields == {}
        assert item.score is None

    def test_unified_item_is_frozen(self) -> None:
        """Test UnifiedItem is immutable."""
        item = UnifiedItem(entity_type="node", key="n1")
        with pytest.raises(AttributeError):
            item.key = "n2"  # type: ignore

    def test_unified_result_creation(self) -> None:
        """Test UnifiedResult creation."""
        items = [UnifiedItem(entity_type="node", key="n1")]
        result = UnifiedResult(description="Found 1 item", items=items)
        assert result.description == "Found 1 item"
        assert len(result.items) == 1
        assert result.items[0].key == "n1"

    def test_unified_result_default_items(self) -> None:
        """Test UnifiedResult with default items."""
        result = UnifiedResult(description="Empty result")
        assert result.items == []


class TestBlobStats:
    """Tests for BlobStats."""

    def test_blob_stats_creation(self) -> None:
        """Test BlobStats creation."""
        stats = BlobStats(
            artifact_count=100,
            chunk_count=500,
            total_bytes=1024000,
            unique_bytes=512000,
            dedup_ratio=2.0,
            orphaned_chunks=5,
        )
        assert stats.artifact_count == 100
        assert stats.chunk_count == 500
        assert stats.total_bytes == 1024000
        assert stats.unique_bytes == 512000
        assert stats.dedup_ratio == 2.0
        assert stats.orphaned_chunks == 5

    def test_blob_stats_is_frozen(self) -> None:
        """Test BlobStats is immutable."""
        stats = BlobStats(
            artifact_count=100,
            chunk_count=500,
            total_bytes=1024000,
            unique_bytes=512000,
            dedup_ratio=2.0,
            orphaned_chunks=5,
        )
        with pytest.raises(AttributeError):
            stats.artifact_count = 200  # type: ignore


class TestCheckpointInfo:
    """Tests for CheckpointInfo."""

    def test_checkpoint_info_creation(self) -> None:
        """Test CheckpointInfo creation."""
        cp = CheckpointInfo(
            id="cp-1",
            name="backup-2024",
            created_at=1704067200,
            is_auto=False,
        )
        assert cp.id == "cp-1"
        assert cp.name == "backup-2024"
        assert cp.created_at == 1704067200
        assert cp.is_auto is False

    def test_checkpoint_info_auto(self) -> None:
        """Test CheckpointInfo with auto flag."""
        cp = CheckpointInfo(
            id="cp-2",
            name="auto-backup",
            created_at=1704153600,
            is_auto=True,
        )
        assert cp.is_auto is True


class TestChainTypes:
    """Tests for chain-related types."""

    def test_chain_transaction_begun(self) -> None:
        """Test ChainTransactionBegun creation."""
        begun = ChainTransactionBegun(tx_id="tx-123")
        assert begun.tx_id == "tx-123"

    def test_chain_committed(self) -> None:
        """Test ChainCommitted creation."""
        committed = ChainCommitted(block_hash="abc123", height=100)
        assert committed.block_hash == "abc123"
        assert committed.height == 100

    def test_chain_rolled_back(self) -> None:
        """Test ChainRolledBack creation."""
        rolled = ChainRolledBack(to_height=50)
        assert rolled.to_height == 50

    def test_chain_history_entry(self) -> None:
        """Test ChainHistoryEntry creation."""
        entry = ChainHistoryEntry(
            height=10,
            transaction_type="PUT",
            data=b"payload",
        )
        assert entry.height == 10
        assert entry.transaction_type == "PUT"
        assert entry.data == b"payload"

    def test_chain_history_entry_no_data(self) -> None:
        """Test ChainHistoryEntry without data."""
        entry = ChainHistoryEntry(height=5, transaction_type="DELETE")
        assert entry.data is None

    def test_chain_history(self) -> None:
        """Test ChainHistory creation."""
        entries = [ChainHistoryEntry(height=1, transaction_type="PUT")]
        history = ChainHistory(entries=entries)
        assert len(history.entries) == 1

    def test_chain_similar_item(self) -> None:
        """Test ChainSimilarItem creation."""
        item = ChainSimilarItem(
            block_hash="xyz789",
            height=42,
            similarity=0.92,
        )
        assert item.block_hash == "xyz789"
        assert item.height == 42
        assert item.similarity == 0.92

    def test_chain_similar(self) -> None:
        """Test ChainSimilar creation."""
        items = [ChainSimilarItem(block_hash="a", height=1, similarity=0.9)]
        similar = ChainSimilar(items=items)
        assert len(similar.items) == 1

    def test_chain_drift(self) -> None:
        """Test ChainDrift creation."""
        drift = ChainDrift(
            from_height=0,
            to_height=100,
            total_drift=15.5,
            avg_drift_per_block=0.155,
            max_drift=2.3,
        )
        assert drift.from_height == 0
        assert drift.to_height == 100
        assert drift.total_drift == 15.5
        assert drift.avg_drift_per_block == 0.155
        assert drift.max_drift == 2.3

    def test_chain_height(self) -> None:
        """Test ChainHeight creation."""
        height = ChainHeight(height=500)
        assert height.height == 500

    def test_chain_tip(self) -> None:
        """Test ChainTip creation."""
        tip = ChainTip(hash="head123", height=500)
        assert tip.hash == "head123"
        assert tip.height == 500

    def test_chain_block_info(self) -> None:
        """Test ChainBlockInfo creation."""
        block = ChainBlockInfo(
            height=100,
            hash="block100",
            prev_hash="block99",
            timestamp=1704067200,
            transaction_count=50,
            proposer="validator1",
        )
        assert block.height == 100
        assert block.hash == "block100"
        assert block.prev_hash == "block99"
        assert block.timestamp == 1704067200
        assert block.transaction_count == 50
        assert block.proposer == "validator1"

    def test_chain_codebook_info(self) -> None:
        """Test ChainCodebookInfo creation."""
        codebook = ChainCodebookInfo(
            scope="global",
            entry_count=256,
            dimension=128,
            domain="embeddings",
        )
        assert codebook.scope == "global"
        assert codebook.entry_count == 256
        assert codebook.dimension == 128
        assert codebook.domain == "embeddings"

    def test_chain_codebook_info_no_domain(self) -> None:
        """Test ChainCodebookInfo without domain."""
        codebook = ChainCodebookInfo(
            scope="local",
            entry_count=64,
            dimension=32,
        )
        assert codebook.domain is None

    def test_chain_transition_analysis(self) -> None:
        """Test ChainTransitionAnalysis creation."""
        analysis = ChainTransitionAnalysis(
            total_transitions=100,
            valid_transitions=95,
            invalid_transitions=5,
            avg_validity_score=0.95,
        )
        assert analysis.total_transitions == 100
        assert analysis.valid_transitions == 95
        assert analysis.invalid_transitions == 5
        assert analysis.avg_validity_score == 0.95

    def test_chain_conflict_resolution(self) -> None:
        """Test ChainConflictResolution creation."""
        resolution = ChainConflictResolution(
            strategy="semantic",
            conflicts_resolved=10,
        )
        assert resolution.strategy == "semantic"
        assert resolution.conflicts_resolved == 10

    def test_chain_merge_result(self) -> None:
        """Test ChainMergeResult creation."""
        merge = ChainMergeResult(success=True, merged_count=25)
        assert merge.success is True
        assert merge.merged_count == 25

    def test_chain_merge_result_failed(self) -> None:
        """Test ChainMergeResult with failure."""
        merge = ChainMergeResult(success=False, merged_count=0)
        assert merge.success is False
        assert merge.merged_count == 0


class TestQueryResultNewAccessors:
    """Tests for new QueryResult property accessors."""

    def test_unified_result_accessor(self) -> None:
        """Test unified_result accessor."""
        unified = UnifiedResult(
            description="test",
            items=[UnifiedItem(entity_type="node", key="n1")],
        )
        result = QueryResult(QueryResultType.UNIFIED, unified)
        assert result.unified_result == unified
        assert result.unified_result.description == "test"

    def test_unified_items_accessor(self) -> None:
        """Test unified_items accessor."""
        items = [UnifiedItem(entity_type="node", key="n1")]
        unified = UnifiedResult(description="test", items=items)
        result = QueryResult(QueryResultType.UNIFIED, unified)
        assert result.unified_items == items

    def test_unified_description_accessor(self) -> None:
        """Test unified_description accessor."""
        unified = UnifiedResult(description="Found 5 items", items=[])
        result = QueryResult(QueryResultType.UNIFIED, unified)
        assert result.unified_description == "Found 5 items"

    def test_artifact_ids_accessor(self) -> None:
        """Test artifact_ids accessor."""
        result = QueryResult(QueryResultType.ARTIFACT_LIST, ["art1", "art2"])
        assert result.artifact_ids == ["art1", "art2"]

    def test_blob_stats_accessor(self) -> None:
        """Test blob_stats accessor."""
        stats = BlobStats(
            artifact_count=10,
            chunk_count=50,
            total_bytes=1000,
            unique_bytes=500,
            dedup_ratio=2.0,
            orphaned_chunks=1,
        )
        result = QueryResult(QueryResultType.BLOB_STATS, stats)
        assert result.blob_stats == stats

    def test_checkpoints_accessor(self) -> None:
        """Test checkpoints accessor."""
        checkpoints = [CheckpointInfo(id="1", name="cp1", created_at=100, is_auto=False)]
        result = QueryResult(QueryResultType.CHECKPOINT_LIST, checkpoints)
        assert result.checkpoints == checkpoints

    def test_chain_transaction_begun_accessor(self) -> None:
        """Test chain_transaction_begun accessor."""
        begun = ChainTransactionBegun(tx_id="tx-1")
        result = QueryResult(QueryResultType.CHAIN_TRANSACTION_BEGUN, begun)
        assert result.chain_transaction_begun == begun

    def test_chain_committed_accessor(self) -> None:
        """Test chain_committed accessor."""
        committed = ChainCommitted(block_hash="hash1", height=100)
        result = QueryResult(QueryResultType.CHAIN_COMMITTED, committed)
        assert result.chain_committed == committed

    def test_chain_rolled_back_accessor(self) -> None:
        """Test chain_rolled_back accessor."""
        rolled = ChainRolledBack(to_height=50)
        result = QueryResult(QueryResultType.CHAIN_ROLLED_BACK, rolled)
        assert result.chain_rolled_back == rolled

    def test_chain_history_accessor(self) -> None:
        """Test chain_history accessor."""
        history = ChainHistory(entries=[])
        result = QueryResult(QueryResultType.CHAIN_HISTORY, history)
        assert result.chain_history == history

    def test_chain_similar_accessor(self) -> None:
        """Test chain_similar accessor."""
        similar = ChainSimilar(items=[])
        result = QueryResult(QueryResultType.CHAIN_SIMILAR, similar)
        assert result.chain_similar == similar

    def test_chain_drift_accessor(self) -> None:
        """Test chain_drift accessor."""
        drift = ChainDrift(
            from_height=0,
            to_height=100,
            total_drift=10.0,
            avg_drift_per_block=0.1,
            max_drift=1.0,
        )
        result = QueryResult(QueryResultType.CHAIN_DRIFT, drift)
        assert result.chain_drift == drift

    def test_chain_height_accessor(self) -> None:
        """Test chain_height accessor."""
        height = ChainHeight(height=500)
        result = QueryResult(QueryResultType.CHAIN_HEIGHT, height)
        assert result.chain_height == height

    def test_chain_tip_accessor(self) -> None:
        """Test chain_tip accessor."""
        tip = ChainTip(hash="abc", height=100)
        result = QueryResult(QueryResultType.CHAIN_TIP, tip)
        assert result.chain_tip == tip

    def test_chain_block_accessor(self) -> None:
        """Test chain_block accessor."""
        block = ChainBlockInfo(
            height=1,
            hash="a",
            prev_hash="b",
            timestamp=0,
            transaction_count=1,
            proposer="p",
        )
        result = QueryResult(QueryResultType.CHAIN_BLOCK, block)
        assert result.chain_block == block

    def test_chain_codebook_accessor(self) -> None:
        """Test chain_codebook accessor."""
        codebook = ChainCodebookInfo(scope="global", entry_count=10, dimension=128)
        result = QueryResult(QueryResultType.CHAIN_CODEBOOK, codebook)
        assert result.chain_codebook == codebook

    def test_chain_transition_analysis_accessor(self) -> None:
        """Test chain_transition_analysis accessor."""
        analysis = ChainTransitionAnalysis(
            total_transitions=10,
            valid_transitions=9,
            invalid_transitions=1,
            avg_validity_score=0.9,
        )
        result = QueryResult(QueryResultType.CHAIN_TRANSITION_ANALYSIS, analysis)
        assert result.chain_transition_analysis == analysis

    def test_chain_conflict_resolution_accessor(self) -> None:
        """Test chain_conflict_resolution accessor."""
        resolution = ChainConflictResolution(strategy="auto", conflicts_resolved=5)
        result = QueryResult(QueryResultType.CHAIN_CONFLICT_RESOLUTION, resolution)
        assert result.chain_conflict_resolution == resolution

    def test_chain_merge_accessor(self) -> None:
        """Test chain_merge accessor."""
        merge = ChainMergeResult(success=True, merged_count=10)
        result = QueryResult(QueryResultType.CHAIN_MERGE, merge)
        assert result.chain_merge == merge

    def test_new_accessors_wrong_type(self) -> None:
        """Test new accessors return None/empty for wrong types."""
        count_result = QueryResult(QueryResultType.COUNT, 42)
        assert count_result.unified_result is None
        assert count_result.unified_items == []
        assert count_result.unified_description is None
        assert count_result.artifact_ids == []
        assert count_result.blob_stats is None
        assert count_result.checkpoints == []
        assert count_result.chain_transaction_begun is None
        assert count_result.chain_committed is None
        assert count_result.chain_rolled_back is None
        assert count_result.chain_history is None
        assert count_result.chain_similar is None
        assert count_result.chain_drift is None
        assert count_result.chain_height is None
        assert count_result.chain_tip is None
        assert count_result.chain_block is None
        assert count_result.chain_codebook is None
        assert count_result.chain_transition_analysis is None
        assert count_result.chain_conflict_resolution is None
        assert count_result.chain_merge is None

    def test_new_accessors_none_data(self) -> None:
        """Test new accessors with None data."""
        unified_result = QueryResult(QueryResultType.UNIFIED, None)
        assert unified_result.unified_result is None
        assert unified_result.unified_items == []
        assert unified_result.unified_description is None

        artifact_result = QueryResult(QueryResultType.ARTIFACT_LIST, None)
        assert artifact_result.artifact_ids == []

        blob_stats_result = QueryResult(QueryResultType.BLOB_STATS, None)
        assert blob_stats_result.blob_stats is None

        checkpoint_result = QueryResult(QueryResultType.CHECKPOINT_LIST, None)
        assert checkpoint_result.checkpoints == []


class TestPageRankResult:
    """Tests for PageRank result types."""

    def test_page_rank_item(self) -> None:
        """Test PageRankItem creation."""
        item = PageRankItem(node_id=42, score=0.95)
        assert item.node_id == 42
        assert item.score == 0.95

    def test_page_rank_result(self) -> None:
        """Test PageRankResult creation."""
        items = [PageRankItem(node_id=1, score=0.5), PageRankItem(node_id=2, score=0.3)]
        result = PageRankResult(items=items, iterations=10, convergence=0.001, converged=True)
        assert len(result.items) == 2
        assert result.iterations == 10
        assert result.convergence == 0.001
        assert result.converged is True

    def test_page_rank_result_defaults(self) -> None:
        """Test PageRankResult default values."""
        result = PageRankResult()
        assert result.items == []
        assert result.iterations is None
        assert result.convergence is None
        assert result.converged is None


class TestCentralityResult:
    """Tests for Centrality result types."""

    def test_centrality_item(self) -> None:
        """Test CentralityItem creation."""
        item = CentralityItem(node_id=1, score=0.75)
        assert item.node_id == 1
        assert item.score == 0.75

    def test_centrality_result(self) -> None:
        """Test CentralityResult creation."""
        items = [CentralityItem(node_id=1, score=0.9)]
        result = CentralityResult(
            items=items,
            centrality_type="betweenness",
            iterations=5,
            converged=True,
            sample_count=100,
        )
        assert len(result.items) == 1
        assert result.centrality_type == "betweenness"
        assert result.iterations == 5
        assert result.converged is True
        assert result.sample_count == 100


class TestCommunitiesResult:
    """Tests for Communities result types."""

    def test_community_item(self) -> None:
        """Test CommunityItem creation."""
        item = CommunityItem(node_id=5, community_id=2)
        assert item.node_id == 5
        assert item.community_id == 2

    def test_communities_result(self) -> None:
        """Test CommunitiesResult creation."""
        items = [
            CommunityItem(node_id=1, community_id=0),
            CommunityItem(node_id=2, community_id=0),
            CommunityItem(node_id=3, community_id=1),
        ]
        result = CommunitiesResult(
            items=items,
            community_count=2,
            modularity=0.45,
            passes=3,
            iterations=10,
            communities=[[1, 2], [3]],
        )
        assert len(result.items) == 3
        assert result.community_count == 2
        assert result.modularity == 0.45
        assert result.passes == 3
        assert result.iterations == 10
        assert result.communities == [[1, 2], [3]]


class TestConstraintsResult:
    """Tests for Constraints result types."""

    def test_constraint_item(self) -> None:
        """Test ConstraintItem creation."""
        item = ConstraintItem(
            name="unique_email",
            target="user",
            property="email",
            constraint_type="unique",
        )
        assert item.name == "unique_email"
        assert item.target == "user"
        assert item.property == "email"
        assert item.constraint_type == "unique"

    def test_constraints_result(self) -> None:
        """Test ConstraintsResult creation."""
        items = [
            ConstraintItem(name="pk", target="users", property="id", constraint_type="primary_key")
        ]
        result = ConstraintsResult(items=items)
        assert len(result.items) == 1


class TestAggregateResult:
    """Tests for Aggregate result types."""

    def test_aggregate_result_count(self) -> None:
        """Test AggregateResult with count."""
        result = AggregateResult(value=42, aggregate_type="count")
        assert result.value == 42
        assert result.aggregate_type == "count"

    def test_aggregate_result_sum(self) -> None:
        """Test AggregateResult with sum."""
        result = AggregateResult(value=1234.56, aggregate_type="sum")
        assert result.value == 1234.56
        assert result.aggregate_type == "sum"


class TestBatchOperationResult:
    """Tests for BatchOperation result types."""

    def test_batch_operation_result(self) -> None:
        """Test BatchOperationResult creation."""
        result = BatchOperationResult(
            operation="INSERT", affected_count=100, created_ids=[1, 2, 3, 4, 5]
        )
        assert result.operation == "INSERT"
        assert result.affected_count == 100
        assert len(result.created_ids) == 5

    def test_batch_operation_result_defaults(self) -> None:
        """Test BatchOperationResult default values."""
        result = BatchOperationResult(operation="DELETE", affected_count=50)
        assert result.created_ids == []


class TestGraphIndexesResult:
    """Tests for GraphIndexes result types."""

    def test_graph_indexes_result(self) -> None:
        """Test GraphIndexesResult creation."""
        result = GraphIndexesResult(indexes=["idx_user_name", "idx_user_email"])
        assert len(result.indexes) == 2
        assert "idx_user_name" in result.indexes

    def test_graph_indexes_result_empty(self) -> None:
        """Test GraphIndexesResult with no indexes."""
        result = GraphIndexesResult()
        assert result.indexes == []


class TestPatternMatchResult:
    """Tests for PatternMatch result types."""

    def test_node_binding(self) -> None:
        """Test NodeBinding creation."""
        binding = NodeBinding(id=42, label="Person")
        assert binding.id == 42
        assert binding.label == "Person"

    def test_edge_binding(self) -> None:
        """Test EdgeBinding creation."""
        binding = EdgeBinding(id=1, edge_type="KNOWS", from_id=10, to_id=20)
        assert binding.id == 1
        assert binding.edge_type == "KNOWS"
        assert binding.from_id == 10
        assert binding.to_id == 20

    def test_path_binding(self) -> None:
        """Test PathBinding creation."""
        binding = PathBinding(nodes=[1, 2, 3], edges=[10, 20], length=2)
        assert binding.nodes == [1, 2, 3]
        assert binding.edges == [10, 20]
        assert binding.length == 2

    def test_path_binding_defaults(self) -> None:
        """Test PathBinding default values."""
        binding = PathBinding()
        assert binding.nodes == []
        assert binding.edges == []
        assert binding.length == 0

    def test_pattern_match_stats(self) -> None:
        """Test PatternMatchStats creation."""
        stats = PatternMatchStats(
            matches_found=100,
            nodes_evaluated=500,
            edges_evaluated=1000,
            truncated=False,
        )
        assert stats.matches_found == 100
        assert stats.nodes_evaluated == 500
        assert stats.edges_evaluated == 1000
        assert stats.truncated is False

    def test_pattern_match_result(self) -> None:
        """Test PatternMatchResult creation."""
        node = NodeBinding(id=1, label="Person")
        matches = [{"person": node}]
        stats = PatternMatchStats(
            matches_found=1, nodes_evaluated=10, edges_evaluated=5, truncated=False
        )
        result = PatternMatchResult(matches=matches, stats=stats)
        assert len(result.matches) == 1
        assert result.stats is not None
        assert result.stats.matches_found == 1


class TestQueryResultGraphAlgorithms:
    """Tests for QueryResult with graph algorithm types."""

    def test_page_rank_accessor(self) -> None:
        """Test page_rank accessor."""
        items = [PageRankItem(node_id=1, score=0.9)]
        pr_result = PageRankResult(items=items, iterations=10)
        result = QueryResult(QueryResultType.PAGE_RANK, pr_result)
        assert result.page_rank is not None
        assert len(result.page_rank.items) == 1
        assert result.page_rank.iterations == 10

    def test_centrality_accessor(self) -> None:
        """Test centrality accessor."""
        items = [CentralityItem(node_id=1, score=0.5)]
        c_result = CentralityResult(items=items, centrality_type="betweenness")
        result = QueryResult(QueryResultType.CENTRALITY, c_result)
        assert result.centrality is not None
        assert len(result.centrality.items) == 1
        assert result.centrality.centrality_type == "betweenness"

    def test_communities_accessor(self) -> None:
        """Test communities accessor."""
        items = [CommunityItem(node_id=1, community_id=0)]
        comm_result = CommunitiesResult(items=items, community_count=1)
        result = QueryResult(QueryResultType.COMMUNITIES, comm_result)
        assert result.communities is not None
        assert len(result.communities.items) == 1
        assert result.communities.community_count == 1

    def test_constraints_accessor(self) -> None:
        """Test constraints accessor."""
        items = [
            ConstraintItem(name="pk", target="users", property="id", constraint_type="primary_key")
        ]
        c_result = ConstraintsResult(items=items)
        result = QueryResult(QueryResultType.CONSTRAINTS, c_result)
        assert result.constraints is not None
        assert len(result.constraints.items) == 1

    def test_aggregate_accessor(self) -> None:
        """Test aggregate accessor."""
        agg_result = AggregateResult(value=42, aggregate_type="count")
        result = QueryResult(QueryResultType.AGGREGATE, agg_result)
        assert result.aggregate is not None
        assert result.aggregate.value == 42
        assert result.aggregate.aggregate_type == "count"

    def test_batch_operation_accessor(self) -> None:
        """Test batch_operation accessor."""
        batch_result = BatchOperationResult(
            operation="INSERT", affected_count=10, created_ids=[1, 2, 3]
        )
        result = QueryResult(QueryResultType.BATCH_OPERATION, batch_result)
        assert result.batch_operation is not None
        assert result.batch_operation.operation == "INSERT"
        assert result.batch_operation.affected_count == 10

    def test_graph_indexes_accessor(self) -> None:
        """Test graph_indexes accessor."""
        idx_result = GraphIndexesResult(indexes=["idx_name"])
        result = QueryResult(QueryResultType.GRAPH_INDEXES, idx_result)
        assert result.graph_indexes is not None
        assert "idx_name" in result.graph_indexes.indexes

    def test_pattern_match_accessor(self) -> None:
        """Test pattern_match accessor."""
        node = NodeBinding(id=1, label="Person")
        matches = [{"p": node}]
        stats = PatternMatchStats(
            matches_found=1, nodes_evaluated=5, edges_evaluated=2, truncated=False
        )
        pm_result = PatternMatchResult(matches=matches, stats=stats)
        result = QueryResult(QueryResultType.PATTERN_MATCH, pm_result)
        assert result.pattern_match is not None
        assert len(result.pattern_match.matches) == 1
        assert result.pattern_match.stats is not None

    def test_graph_algorithm_accessors_wrong_type(self) -> None:
        """Test graph algorithm accessors return None for wrong types."""
        result = QueryResult(QueryResultType.COUNT, 42)
        assert result.page_rank is None
        assert result.centrality is None
        assert result.communities is None
        assert result.constraints is None
        assert result.aggregate is None
        assert result.batch_operation is None
        assert result.graph_indexes is None
        assert result.pattern_match is None

    def test_graph_algorithm_accessors_none_data(self) -> None:
        """Test graph algorithm accessors with None data."""
        assert QueryResult(QueryResultType.PAGE_RANK, None).page_rank is None
        assert QueryResult(QueryResultType.CENTRALITY, None).centrality is None
        assert QueryResult(QueryResultType.COMMUNITIES, None).communities is None
        assert QueryResult(QueryResultType.CONSTRAINTS, None).constraints is None
        assert QueryResult(QueryResultType.AGGREGATE, None).aggregate is None
        assert QueryResult(QueryResultType.BATCH_OPERATION, None).batch_operation is None
        assert QueryResult(QueryResultType.GRAPH_INDEXES, None).graph_indexes is None
        assert QueryResult(QueryResultType.PATTERN_MATCH, None).pattern_match is None
