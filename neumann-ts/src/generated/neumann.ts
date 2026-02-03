// SPDX-License-Identifier: MIT
/**
 * Static TypeScript definitions for neumann.proto
 *
 * This file provides static type definitions that are compatible with both
 * ESM and CommonJS builds, eliminating the need for runtime proto loading.
 */

import type * as grpc from '@grpc/grpc-js';

// === Enums ===

export enum ErrorCode {
  UNSPECIFIED = 0,
  INVALID_QUERY = 1,
  NOT_FOUND = 2,
  PERMISSION_DENIED = 3,
  ALREADY_EXISTS = 4,
  INTERNAL = 5,
  UNAVAILABLE = 6,
  INVALID_ARGUMENT = 7,
  UNAUTHENTICATED = 8,
}

export enum ServingStatus {
  UNSPECIFIED = 0,
  SERVING = 1,
  NOT_SERVING = 2,
}

export enum CentralityType {
  UNSPECIFIED = 0,
  BETWEENNESS = 1,
  CLOSENESS = 2,
  EIGENVECTOR = 3,
}

// === Value Types ===

export interface ProtoValue {
  null?: boolean;
  intValue?: number;
  floatValue?: number;
  stringValue?: string;
  boolValue?: boolean;
}

export interface ColumnValue {
  name: string;
  value?: ProtoValue;
}

// === Row Types ===

export interface ProtoRow {
  id: number;
  values?: ColumnValue[];
}

export interface RowsResult {
  rows: ProtoRow[];
}

export interface RowChunk {
  row?: ProtoRow;
}

// === Graph Types ===

export interface ProtoNode {
  id: number;
  label: string;
  properties?: Record<string, string>;
}

export interface ProtoEdge {
  id: number;
  from: number;
  to: number;
  label: string;
}

export interface NodesResult {
  nodes: ProtoNode[];
}

export interface EdgesResult {
  edges: ProtoEdge[];
}

export interface NodeChunk {
  node?: ProtoNode;
}

export interface EdgeChunk {
  edge?: ProtoEdge;
}

export interface PathResult {
  nodeIds: number[];
}

// === Graph Algorithm Types ===

export interface PageRankItem {
  nodeId: number;
  score: number;
}

export interface PageRankResult {
  items: PageRankItem[];
  iterations?: number;
  convergence?: number;
  converged?: boolean;
}

export interface CentralityItem {
  nodeId: number;
  score: number;
}

export interface CentralityResult {
  items: CentralityItem[];
  centralityType?: CentralityType;
  iterations?: number;
  converged?: boolean;
  sampleCount?: number;
}

export interface CommunityItem {
  nodeId: number;
  communityId: number;
}

export interface CommunityMemberList {
  communityId: number;
  memberNodeIds: number[];
}

export interface CommunitiesResult {
  items: CommunityItem[];
  communityCount?: number;
  modularity?: number;
  passes?: number;
  iterations?: number;
  communities?: CommunityMemberList[];
}

export interface ConstraintItem {
  name: string;
  target: string;
  property: string;
  constraintType: string;
}

export interface ConstraintsResult {
  items: ConstraintItem[];
}

export interface AggregateResult {
  count?: number;
  sum?: number;
  avg?: number;
  min?: number;
  max?: number;
}

export interface BatchOperationResult {
  operation: string;
  affectedCount: number;
  createdIds: number[];
}

export interface GraphIndexesResult {
  indexes: string[];
}

// === Pattern Match Types ===

export interface NodeBinding {
  id: number;
  label: string;
}

export interface EdgeBinding {
  id: number;
  edgeType: string;
  from: number;
  to: number;
}

export interface PathBinding {
  nodes: number[];
  edges: number[];
  length: number;
}

export interface BindingValue {
  node?: NodeBinding;
  edge?: EdgeBinding;
  path?: PathBinding;
}

export interface BindingEntry {
  variable: string;
  value?: BindingValue;
}

export interface PatternMatchBinding {
  bindings: BindingEntry[];
}

export interface PatternMatchStats {
  matchesFound: number;
  nodesEvaluated: number;
  edgesEvaluated: number;
  truncated: boolean;
}

export interface PatternMatchResult {
  matches: PatternMatchBinding[];
  stats?: PatternMatchStats;
}

// === Vector/Similarity Types ===

export interface SimilarItem {
  key: string;
  score: number;
}

export interface SimilarResult {
  items: SimilarItem[];
}

export interface SimilarChunk {
  item?: SimilarItem;
}

// === Unified Query Types ===

export interface UnifiedItem {
  entityType: string;
  key: string;
  fields?: Record<string, string>;
  score?: number;
}

export interface UnifiedQueryResult {
  description: string;
  items: UnifiedItem[];
}

// === Table/List Types ===

export interface TableListResult {
  tables: string[];
}

export interface IdsResult {
  ids: number[];
}

export interface CountResult {
  count: number;
}

export interface StringValue {
  value: string;
}

// === Blob Types ===

export interface BlobResult {
  data: Uint8Array;
}

export interface ArtifactInfo {
  id: string;
  filename: string;
  contentType: string;
  size: number;
  checksum: string;
  chunkCount: number;
  created: number;
  modified: number;
  createdBy: string;
  tags: string[];
  linkedTo: string[];
  custom?: Record<string, string>;
}

export interface ArtifactListResult {
  artifactIds: string[];
}

export interface BlobStatsResult {
  artifactCount: number;
  chunkCount: number;
  totalBytes: number;
  uniqueBytes: number;
  dedupRatio: number;
  orphanedChunks: number;
}

// === Checkpoint Types ===

export interface CheckpointInfo {
  id: string;
  name: string;
  createdAt: number;
  isAuto: boolean;
}

export interface CheckpointListResult {
  checkpoints: CheckpointInfo[];
}

// === Chain Types ===

export interface ChainTransactionBegun {
  txId: string;
}

export interface ChainCommitted {
  blockHash: string;
  height: number;
}

export interface ChainRolledBack {
  toHeight: number;
}

export interface ChainHistoryEntry {
  height: number;
  transactionType: string;
  data?: Uint8Array;
}

export interface ChainHistory {
  entries: ChainHistoryEntry[];
}

export interface ChainSimilarItem {
  blockHash: string;
  height: number;
  similarity: number;
}

export interface ChainSimilar {
  items: ChainSimilarItem[];
}

export interface ChainDrift {
  fromHeight: number;
  toHeight: number;
  totalDrift: number;
  avgDriftPerBlock: number;
  maxDrift: number;
}

export interface ChainHeight {
  height: number;
}

export interface ChainTip {
  hash: string;
  height: number;
}

export interface ChainBlockInfo {
  height: number;
  hash: string;
  prevHash: string;
  timestamp: number;
  transactionCount: number;
  proposer: string;
}

export interface ChainCodebookInfo {
  scope: string;
  entryCount: number;
  dimension: number;
  domain?: string;
}

export interface ChainTransitionAnalysis {
  totalTransitions: number;
  validTransitions: number;
  invalidTransitions: number;
  avgValidityScore: number;
}

export interface ChainConflictResolution {
  strategy: string;
  conflictsResolved: number;
}

export interface ChainMergeResult {
  success: boolean;
  mergedCount: number;
}

export interface ChainQueryResult {
  transactionBegun?: ChainTransactionBegun;
  committed?: ChainCommitted;
  rolledBack?: ChainRolledBack;
  history?: ChainHistory;
  similar?: ChainSimilar;
  drift?: ChainDrift;
  height?: ChainHeight;
  tip?: ChainTip;
  block?: ChainBlockInfo;
  codebook?: ChainCodebookInfo;
  transitionAnalysis?: ChainTransitionAnalysis;
  conflictResolution?: ChainConflictResolution;
  merge?: ChainMergeResult;
}

// === Error Types ===

export interface ErrorInfo {
  code: ErrorCode;
  message: string;
  details?: string;
}

// === Query Request/Response ===

export interface QueryRequest {
  query: string;
  identity?: string;
}

export interface QueryResponse {
  empty?: object;
  value?: StringValue;
  count?: CountResult;
  ids?: IdsResult;
  rows?: RowsResult;
  nodes?: NodesResult;
  edges?: EdgesResult;
  path?: PathResult;
  similar?: SimilarResult;
  unified?: UnifiedQueryResult;
  tableList?: TableListResult;
  blob?: BlobResult;
  artifactInfo?: ArtifactInfo;
  artifactList?: ArtifactListResult;
  blobStats?: BlobStatsResult;
  checkpointList?: CheckpointListResult;
  chain?: ChainQueryResult;
  pageRank?: PageRankResult;
  centrality?: CentralityResult;
  communities?: CommunitiesResult;
  constraints?: ConstraintsResult;
  aggregate?: AggregateResult;
  batchOperation?: BatchOperationResult;
  graphIndexes?: GraphIndexesResult;
  patternMatch?: PatternMatchResult;
  error?: ErrorInfo;
}

export interface StreamCursorInfo {
  cursor: string;
  itemsSent: number;
  totalCount?: number;
}

export interface QueryResponseChunk {
  row?: RowChunk;
  node?: NodeChunk;
  edge?: EdgeChunk;
  similarItem?: SimilarChunk;
  blobData?: Uint8Array;
  error?: ErrorInfo;
  isFinal: boolean;
  cursorInfo?: StreamCursorInfo;
  sequenceNumber?: number;
}

export interface BatchQueryRequest {
  queries: QueryRequest[];
}

export interface BatchQueryResponse {
  results: QueryResponse[];
}

export interface PaginatedQueryRequest {
  query: string;
  identity?: string;
  cursor?: string;
  pageSize?: number;
  countTotal?: boolean;
  cursorTtlSecs?: number;
}

export interface PaginatedQueryResponse {
  result?: QueryResponse;
  nextCursor?: string;
  prevCursor?: string;
  totalCount?: number;
  hasMore: boolean;
  pageSize: number;
}

export interface CloseCursorRequest {
  cursor: string;
}

export interface CloseCursorResponse {
  success: boolean;
}

// === Blob Service Types ===

export interface BlobUploadMetadata {
  filename: string;
  contentType?: string | undefined;
  createdBy?: string | undefined;
  tags?: string[] | undefined;
  linkedTo?: string[] | undefined;
  custom?: Record<string, string> | undefined;
}

export interface BlobUploadRequest {
  metadata?: BlobUploadMetadata;
  chunk?: Uint8Array;
}

export interface BlobUploadResponse {
  artifactId: string;
  size: number;
  checksum: string;
}

export interface BlobDownloadRequest {
  artifactId: string;
}

export interface BlobDownloadChunk {
  data: Uint8Array;
  isFinal: boolean;
}

export interface BlobDeleteRequest {
  artifactId: string;
}

export interface BlobDeleteResponse {
  success: boolean;
}

export interface BlobMetadataRequest {
  artifactId: string;
}

// === Health Service Types ===

export interface HealthCheckRequest {
  service?: string | undefined;
}

export interface HealthCheckResponse {
  status: ServingStatus;
}

// === gRPC Service Interfaces ===

export type GrpcCallback<T> = (err: grpc.ServiceError | null, response: T) => void;

export interface QueryServiceClient extends grpc.Client {
  Execute(
    request: QueryRequest,
    metadata: grpc.Metadata,
    callback: GrpcCallback<QueryResponse>
  ): grpc.ClientUnaryCall;
  ExecuteStream(
    request: QueryRequest,
    metadata: grpc.Metadata
  ): grpc.ClientReadableStream<QueryResponseChunk>;
  ExecuteBatch(
    request: BatchQueryRequest,
    metadata: grpc.Metadata,
    callback: GrpcCallback<BatchQueryResponse>
  ): grpc.ClientUnaryCall;
  ExecutePaginated(
    request: PaginatedQueryRequest,
    metadata: grpc.Metadata,
    callback: GrpcCallback<PaginatedQueryResponse>
  ): grpc.ClientUnaryCall;
  CloseCursor(
    request: CloseCursorRequest,
    metadata: grpc.Metadata,
    callback: GrpcCallback<CloseCursorResponse>
  ): grpc.ClientUnaryCall;
}

export interface BlobServiceClient extends grpc.Client {
  Upload(
    metadata: grpc.Metadata,
    callback: GrpcCallback<BlobUploadResponse>
  ): grpc.ClientWritableStream<BlobUploadRequest>;
  Download(
    request: BlobDownloadRequest,
    metadata: grpc.Metadata
  ): grpc.ClientReadableStream<BlobDownloadChunk>;
  Delete(
    request: BlobDeleteRequest,
    metadata: grpc.Metadata,
    callback: GrpcCallback<BlobDeleteResponse>
  ): grpc.ClientUnaryCall;
  GetMetadata(
    request: BlobMetadataRequest,
    metadata: grpc.Metadata,
    callback: GrpcCallback<ArtifactInfo>
  ): grpc.ClientUnaryCall;
}

export interface HealthClient extends grpc.Client {
  Check(
    request: HealthCheckRequest,
    metadata: grpc.Metadata,
    callback: GrpcCallback<HealthCheckResponse>
  ): grpc.ClientUnaryCall;
}
