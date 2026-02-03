"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.loadProto = loadProto;
exports.loadVectorProto = loadVectorProto;
exports.getQueryServiceClient = getQueryServiceClient;
exports.getBlobServiceClient = getBlobServiceClient;
exports.getHealthClient = getHealthClient;
exports.getPointsServiceClient = getPointsServiceClient;
exports.getCollectionsServiceClient = getCollectionsServiceClient;
exports.cleanup = cleanup;
// SPDX-License-Identifier: MIT
/**
 * gRPC module for Node.js environments.
 *
 * This module provides factory functions for creating gRPC service clients.
 * It uses dynamic proto loading with a portable path resolution that works
 * in both ESM and CommonJS builds.
 */
const grpc = __importStar(require("@grpc/grpc-js"));
const protoLoader = __importStar(require("@grpc/proto-loader"));
const path = __importStar(require("path"));
const fs = __importStar(require("fs"));
const os = __importStar(require("os"));
// Inline proto definitions to avoid filesystem dependencies
const NEUMANN_PROTO = `
syntax = "proto3";
package neumann.v1;

service QueryService {
  rpc Execute(QueryRequest) returns (QueryResponse);
  rpc ExecuteStream(QueryRequest) returns (stream QueryResponseChunk);
  rpc ExecuteBatch(BatchQueryRequest) returns (BatchQueryResponse);
  rpc ExecutePaginated(PaginatedQueryRequest) returns (PaginatedQueryResponse);
  rpc CloseCursor(CloseCursorRequest) returns (CloseCursorResponse);
}

service BlobService {
  rpc Upload(stream BlobUploadRequest) returns (BlobUploadResponse);
  rpc Download(BlobDownloadRequest) returns (stream BlobDownloadChunk);
  rpc Delete(BlobDeleteRequest) returns (BlobDeleteResponse);
  rpc GetMetadata(BlobMetadataRequest) returns (ArtifactInfo);
}

service Health {
  rpc Check(HealthCheckRequest) returns (HealthCheckResponse);
}

message QueryRequest {
  string query = 1;
  optional string identity = 2;
}

message QueryResponse {
  oneof result {
    EmptyResult empty = 1;
    StringValue value = 2;
    CountResult count = 3;
    IdsResult ids = 4;
    RowsResult rows = 5;
    NodesResult nodes = 6;
    EdgesResult edges = 7;
    PathResult path = 8;
    SimilarResult similar = 9;
    UnifiedQueryResult unified = 10;
    TableListResult table_list = 11;
    BlobResult blob = 12;
    ArtifactInfo artifact_info = 13;
    ArtifactListResult artifact_list = 14;
    BlobStatsResult blob_stats = 15;
    CheckpointListResult checkpoint_list = 16;
    ChainQueryResult chain = 17;
    PageRankResult page_rank = 18;
    CentralityResult centrality = 19;
    CommunitiesResult communities = 21;
    ConstraintsResult constraints = 22;
    AggregateResult aggregate = 23;
    BatchOperationResult batch_operation = 24;
    GraphIndexesResult graph_indexes = 25;
    PatternMatchResult pattern_match = 26;
  }
  optional ErrorInfo error = 20;
}

message QueryResponseChunk {
  oneof chunk {
    RowChunk row = 1;
    NodeChunk node = 2;
    EdgeChunk edge = 3;
    SimilarChunk similar_item = 4;
    bytes blob_data = 5;
    ErrorInfo error = 10;
  }
  bool is_final = 15;
  optional StreamCursorInfo cursor_info = 16;
  optional uint64 sequence_number = 17;
}

message BatchQueryRequest { repeated QueryRequest queries = 1; }
message BatchQueryResponse { repeated QueryResponse results = 1; }
message PaginatedQueryRequest {
  string query = 1;
  optional string identity = 2;
  optional string cursor = 3;
  optional uint32 page_size = 4;
  optional bool count_total = 5;
  optional uint32 cursor_ttl_secs = 6;
}
message PaginatedQueryResponse {
  QueryResponse result = 1;
  optional string next_cursor = 2;
  optional string prev_cursor = 3;
  optional uint64 total_count = 4;
  bool has_more = 5;
  uint32 page_size = 6;
}
message CloseCursorRequest { string cursor = 1; }
message CloseCursorResponse { bool success = 1; }
message StreamCursorInfo {
  string cursor = 1;
  uint64 items_sent = 2;
  optional uint64 total_count = 3;
}
message EmptyResult {}
message StringValue { string value = 1; }
message CountResult { uint64 count = 1; }
message IdsResult { repeated uint64 ids = 1; }
message RowsResult { repeated Row rows = 1; }
message Row { uint64 id = 1; repeated ColumnValue values = 2; }
message ColumnValue { string name = 1; Value value = 2; }
message Value {
  oneof kind {
    bool null = 1;
    int64 int_value = 2;
    double float_value = 3;
    string string_value = 4;
    bool bool_value = 5;
  }
}
message NodesResult { repeated Node nodes = 1; }
message Node { uint64 id = 1; string label = 2; map<string, string> properties = 3; }
message NodeChunk { Node node = 1; }
message EdgesResult { repeated Edge edges = 1; }
message Edge { uint64 id = 1; uint64 from = 2; uint64 to = 3; string label = 4; }
message EdgeChunk { Edge edge = 1; }
message RowChunk { Row row = 1; }
message PathResult { repeated uint64 node_ids = 1; }
message SimilarResult { repeated SimilarItem items = 1; }
message SimilarItem { string key = 1; float score = 2; }
message SimilarChunk { SimilarItem item = 1; }
message PageRankResult {
  repeated PageRankItem items = 1;
  optional uint32 iterations = 2;
  optional double convergence = 3;
  optional bool converged = 4;
}
message PageRankItem { uint64 node_id = 1; double score = 2; }
message CentralityResult {
  repeated CentralityItem items = 1;
  optional CentralityType centrality_type = 2;
  optional uint32 iterations = 3;
  optional bool converged = 4;
  optional uint32 sample_count = 5;
}
message CentralityItem { uint64 node_id = 1; double score = 2; }
enum CentralityType {
  CENTRALITY_TYPE_UNSPECIFIED = 0;
  CENTRALITY_TYPE_BETWEENNESS = 1;
  CENTRALITY_TYPE_CLOSENESS = 2;
  CENTRALITY_TYPE_EIGENVECTOR = 3;
}
message CommunitiesResult {
  repeated CommunityItem items = 1;
  optional uint32 community_count = 2;
  optional double modularity = 3;
  optional uint32 passes = 4;
  optional uint32 iterations = 5;
  repeated CommunityMemberList communities = 6;
}
message CommunityItem { uint64 node_id = 1; uint64 community_id = 2; }
message CommunityMemberList { uint64 community_id = 1; repeated uint64 member_node_ids = 2; }
message ConstraintsResult { repeated ConstraintItem items = 1; }
message ConstraintItem { string name = 1; string target = 2; string property = 3; string constraint_type = 4; }
message AggregateResult {
  oneof value {
    uint64 count = 1;
    double sum = 2;
    double avg = 3;
    double min = 4;
    double max = 5;
  }
}
message BatchOperationResult { string operation = 1; uint64 affected_count = 2; repeated uint64 created_ids = 3; }
message GraphIndexesResult { repeated string indexes = 1; }
message PatternMatchResult { repeated PatternMatchBinding matches = 1; PatternMatchStats stats = 2; }
message PatternMatchBinding { repeated BindingEntry bindings = 1; }
message BindingEntry { string variable = 1; BindingValue value = 2; }
message BindingValue {
  oneof value {
    NodeBinding node = 1;
    EdgeBinding edge = 2;
    PathBinding path = 3;
  }
}
message NodeBinding { uint64 id = 1; string label = 2; }
message EdgeBinding { uint64 id = 1; string edge_type = 2; uint64 from = 3; uint64 to = 4; }
message PathBinding { repeated uint64 nodes = 1; repeated uint64 edges = 2; uint64 length = 3; }
message PatternMatchStats { uint64 matches_found = 1; uint64 nodes_evaluated = 2; uint64 edges_evaluated = 3; bool truncated = 4; }
message UnifiedQueryResult { string description = 1; repeated UnifiedItem items = 2; }
message UnifiedItem { string entity_type = 1; string key = 2; map<string, string> fields = 3; optional float score = 4; }
message TableListResult { repeated string tables = 1; }
message BlobResult { bytes data = 1; }
message ArtifactInfo {
  string id = 1;
  string filename = 2;
  string content_type = 3;
  uint64 size = 4;
  string checksum = 5;
  uint64 chunk_count = 6;
  uint64 created = 7;
  uint64 modified = 8;
  string created_by = 9;
  repeated string tags = 10;
  repeated string linked_to = 11;
  map<string, string> custom = 12;
}
message ArtifactListResult { repeated string artifact_ids = 1; }
message BlobStatsResult {
  uint64 artifact_count = 1;
  uint64 chunk_count = 2;
  uint64 total_bytes = 3;
  uint64 unique_bytes = 4;
  double dedup_ratio = 5;
  uint64 orphaned_chunks = 6;
}
message CheckpointListResult { repeated CheckpointInfo checkpoints = 1; }
message CheckpointInfo { string id = 1; string name = 2; uint64 created_at = 3; bool is_auto = 4; }
message ChainQueryResult {
  oneof result {
    ChainTransactionBegun transaction_begun = 1;
    ChainCommitted committed = 2;
    ChainRolledBack rolled_back = 3;
    ChainHistory history = 4;
    ChainSimilar similar = 5;
    ChainDrift drift = 6;
    ChainHeight height = 7;
    ChainTip tip = 8;
    ChainBlockInfo block = 9;
    ChainCodebookInfo codebook = 10;
    ChainTransitionAnalysis transition_analysis = 11;
    ChainConflictResolution conflict_resolution = 12;
    ChainMergeResult merge = 13;
  }
}
message ChainTransactionBegun { string tx_id = 1; }
message ChainCommitted { string block_hash = 1; uint64 height = 2; }
message ChainRolledBack { uint64 to_height = 1; }
message ChainHistory { repeated ChainHistoryEntry entries = 1; }
message ChainHistoryEntry { uint64 height = 1; string transaction_type = 2; optional bytes data = 3; }
message ChainSimilar { repeated ChainSimilarItem items = 1; }
message ChainSimilarItem { string block_hash = 1; uint64 height = 2; float similarity = 3; }
message ChainDrift { uint64 from_height = 1; uint64 to_height = 2; float total_drift = 3; float avg_drift_per_block = 4; float max_drift = 5; }
message ChainHeight { uint64 height = 1; }
message ChainTip { string hash = 1; uint64 height = 2; }
message ChainBlockInfo { uint64 height = 1; string hash = 2; string prev_hash = 3; uint64 timestamp = 4; uint64 transaction_count = 5; string proposer = 6; }
message ChainCodebookInfo { string scope = 1; uint64 entry_count = 2; uint64 dimension = 3; optional string domain = 4; }
message ChainTransitionAnalysis { uint64 total_transitions = 1; uint64 valid_transitions = 2; uint64 invalid_transitions = 3; float avg_validity_score = 4; }
message ChainConflictResolution { string strategy = 1; uint64 conflicts_resolved = 2; }
message ChainMergeResult { bool success = 1; uint64 merged_count = 2; }
message ErrorInfo { ErrorCode code = 1; string message = 2; optional string details = 3; }
enum ErrorCode {
  ERROR_CODE_UNSPECIFIED = 0;
  ERROR_CODE_INVALID_QUERY = 1;
  ERROR_CODE_NOT_FOUND = 2;
  ERROR_CODE_PERMISSION_DENIED = 3;
  ERROR_CODE_ALREADY_EXISTS = 4;
  ERROR_CODE_INTERNAL = 5;
  ERROR_CODE_UNAVAILABLE = 6;
  ERROR_CODE_INVALID_ARGUMENT = 7;
  ERROR_CODE_UNAUTHENTICATED = 8;
}
message BlobUploadRequest {
  oneof request {
    BlobUploadMetadata metadata = 1;
    bytes chunk = 2;
  }
}
message BlobUploadMetadata {
  string filename = 1;
  optional string content_type = 2;
  optional string created_by = 3;
  repeated string tags = 4;
  repeated string linked_to = 5;
  map<string, string> custom = 6;
}
message BlobUploadResponse { string artifact_id = 1; uint64 size = 2; string checksum = 3; }
message BlobDownloadRequest { string artifact_id = 1; }
message BlobDownloadChunk { bytes data = 1; bool is_final = 2; }
message BlobDeleteRequest { string artifact_id = 1; }
message BlobDeleteResponse { bool success = 1; }
message BlobMetadataRequest { string artifact_id = 1; }
message HealthCheckRequest { optional string service = 1; }
message HealthCheckResponse { ServingStatus status = 1; }
enum ServingStatus {
  SERVING_STATUS_UNSPECIFIED = 0;
  SERVING_STATUS_SERVING = 1;
  SERVING_STATUS_NOT_SERVING = 2;
}
`;
const VECTOR_PROTO = `
syntax = "proto3";
package neumann.vector.v1;

service PointsService {
  rpc Upsert(UpsertPointsRequest) returns (UpsertPointsResponse);
  rpc Get(GetPointsRequest) returns (GetPointsResponse);
  rpc Delete(DeletePointsRequest) returns (DeletePointsResponse);
  rpc Query(QueryPointsRequest) returns (QueryPointsResponse);
  rpc Scroll(ScrollPointsRequest) returns (ScrollPointsResponse);
}

service CollectionsService {
  rpc Create(CreateCollectionRequest) returns (CreateCollectionResponse);
  rpc Get(GetCollectionRequest) returns (GetCollectionResponse);
  rpc Delete(DeleteCollectionRequest) returns (DeleteCollectionResponse);
  rpc List(ListCollectionsRequest) returns (ListCollectionsResponse);
}

message Point { string id = 1; repeated float vector = 2; map<string, bytes> payload = 3; }
message ScoredPoint { string id = 1; float score = 2; map<string, bytes> payload = 3; repeated float vector = 4; }
message UpsertPointsRequest { string collection = 1; repeated Point points = 2; }
message UpsertPointsResponse { uint64 upserted = 1; }
message GetPointsRequest { string collection = 1; repeated string ids = 2; bool with_payload = 3; bool with_vector = 4; }
message GetPointsResponse { repeated Point points = 1; }
message DeletePointsRequest { string collection = 1; repeated string ids = 2; }
message DeletePointsResponse { uint64 deleted = 1; }
message QueryPointsRequest {
  string collection = 1;
  repeated float vector = 2;
  uint32 limit = 3;
  uint32 offset = 4;
  optional float score_threshold = 5;
  bool with_payload = 6;
  bool with_vector = 7;
}
message QueryPointsResponse { repeated ScoredPoint results = 1; }
message ScrollPointsRequest {
  string collection = 1;
  optional string offset_id = 2;
  uint32 limit = 3;
  bool with_payload = 4;
  bool with_vector = 5;
}
message ScrollPointsResponse { repeated Point points = 1; optional string next_offset = 2; }
message CreateCollectionRequest { string name = 1; uint32 dimension = 2; string distance = 3; }
message CreateCollectionResponse { bool created = 1; }
message GetCollectionRequest { string name = 1; }
message GetCollectionResponse { string name = 1; uint64 points_count = 2; uint32 dimension = 3; string distance = 4; }
message DeleteCollectionRequest { string name = 1; }
message DeleteCollectionResponse { bool deleted = 1; }
message ListCollectionsRequest {}
message ListCollectionsResponse { repeated string collections = 1; }
`;
const LOADER_OPTIONS = {
    keepCase: false,
    longs: Number,
    enums: String,
    defaults: true,
    oneofs: true,
};
// Cached proto objects
let _neumannProto = null;
let _vectorProto = null;
let _tempDir = null;
let _cleanupRegistered = false;
/**
 * Synchronously clean up temp proto files.
 * Called on process exit.
 */
function cleanupSync() {
    if (_tempDir) {
        try {
            fs.rmSync(_tempDir, { recursive: true, force: true });
        }
        catch {
            // Ignore cleanup errors
        }
        _tempDir = null;
    }
}
/**
 * Register cleanup handlers for process exit/signals.
 */
function registerCleanupHandlers() {
    if (_cleanupRegistered) {
        return;
    }
    _cleanupRegistered = true;
    // Normal exit
    process.on('exit', cleanupSync);
    // Handle signals - exit with appropriate code
    process.on('SIGINT', () => {
        cleanupSync();
        process.exit(130);
    });
    process.on('SIGTERM', () => {
        cleanupSync();
        process.exit(143);
    });
}
/**
 * Get or create temp directory for proto files.
 */
function getTempDir() {
    if (_tempDir) {
        return _tempDir;
    }
    registerCleanupHandlers();
    _tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'neumann-proto-'));
    return _tempDir;
}
/**
 * Write proto content to a temp file and load it.
 */
function loadProtoFromString(content, filename) {
    const tempDir = getTempDir();
    const protoPath = path.join(tempDir, filename);
    fs.writeFileSync(protoPath, content, 'utf8');
    return protoLoader.loadSync(protoPath, LOADER_OPTIONS);
}
/**
 * Load the Neumann proto definition.
 * The proto is cached after the first load.
 */
function loadProto() {
    if (_neumannProto) {
        return Promise.resolve(_neumannProto);
    }
    const packageDefinition = loadProtoFromString(NEUMANN_PROTO, 'neumann.proto');
    const grpcObj = grpc.loadPackageDefinition(packageDefinition);
    _neumannProto = grpcObj.neumann.v1;
    return Promise.resolve(_neumannProto);
}
/**
 * Load the Vector proto definition.
 * The proto is cached after the first load.
 */
function loadVectorProto() {
    if (_vectorProto) {
        return Promise.resolve(_vectorProto);
    }
    const packageDefinition = loadProtoFromString(VECTOR_PROTO, 'vector.proto');
    const grpcObj = grpc.loadPackageDefinition(packageDefinition);
    _vectorProto = grpcObj.neumann.vector
        .v1;
    return Promise.resolve(_vectorProto);
}
/**
 * Create a QueryService client.
 */
function getQueryServiceClient(proto, address, credentials) {
    const QueryService = proto.QueryService;
    return new QueryService(address, credentials);
}
/**
 * Create a BlobService client.
 */
function getBlobServiceClient(proto, address, credentials) {
    const BlobService = proto.BlobService;
    return new BlobService(address, credentials);
}
/**
 * Create a Health client.
 */
function getHealthClient(proto, address, credentials) {
    const Health = proto.Health;
    return new Health(address, credentials);
}
/**
 * Create a PointsService client.
 */
function getPointsServiceClient(proto, address, credentials) {
    const PointsService = proto.PointsService;
    return new PointsService(address, credentials);
}
/**
 * Create a CollectionsService client.
 */
function getCollectionsServiceClient(proto, address, credentials) {
    const CollectionsService = proto.CollectionsService;
    return new CollectionsService(address, credentials);
}
/**
 * Clean up temp proto files.
 */
function cleanup() {
    if (_tempDir) {
        try {
            fs.rmSync(_tempDir, { recursive: true, force: true });
        }
        catch {
            // Ignore cleanup errors
        }
        _tempDir = null;
    }
    _neumannProto = null;
    _vectorProto = null;
}
//# sourceMappingURL=grpc.js.map