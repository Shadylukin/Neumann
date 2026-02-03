import type { Value } from './value.js';
/**
 * A row from a relational query result.
 */
export interface Row {
    readonly values: ReadonlyMap<string, Value>;
}
/**
 * A graph node.
 */
export interface Node {
    readonly id: string;
    readonly label: string;
    readonly properties: ReadonlyMap<string, Value>;
}
/**
 * A graph edge.
 */
export interface Edge {
    readonly id: string;
    readonly edgeType: string;
    readonly source: string;
    readonly target: string;
    readonly properties: ReadonlyMap<string, Value>;
}
/**
 * A segment in a graph path.
 */
export interface PathSegment {
    node: Node;
    edge?: Edge;
}
/**
 * A path through a graph.
 */
export interface Path {
    segments: PathSegment[];
}
/**
 * A similarity search result.
 */
export interface SimilarItem {
    readonly key: string;
    readonly score: number;
    readonly metadata?: ReadonlyMap<string, Value>;
}
/**
 * Metadata for a blob artifact.
 */
export interface ArtifactInfo {
    artifactId: string;
    filename: string;
    size: number;
    checksum: string;
    contentType: string;
    createdAt: number;
    tags: string[];
}
/**
 * Types of query results.
 */
export type QueryResultType = 'empty' | 'value' | 'count' | 'rows' | 'nodes' | 'edges' | 'paths' | 'similar' | 'ids' | 'tableList' | 'blob' | 'blobInfo' | 'artifactList' | 'blobStats' | 'checkpointList' | 'pageRank' | 'centrality' | 'communities' | 'patternMatch' | 'constraints' | 'aggregate' | 'batchOperation' | 'graphIndexes' | 'chain' | 'unified' | 'error';
/**
 * Empty query result.
 */
export interface EmptyResult {
    type: 'empty';
}
/**
 * Single value result.
 */
export interface ValueResult {
    type: 'value';
    value: string;
}
/**
 * Count result.
 */
export interface CountResult {
    type: 'count';
    count: number;
}
/**
 * Rows result from relational query.
 */
export interface RowsResult {
    type: 'rows';
    rows: Row[];
}
/**
 * Nodes result from graph query.
 */
export interface NodesResult {
    type: 'nodes';
    nodes: Node[];
}
/**
 * Edges result from graph query.
 */
export interface EdgesResult {
    type: 'edges';
    edges: Edge[];
}
/**
 * Paths result from graph traversal.
 */
export interface PathsResult {
    type: 'paths';
    paths: Path[];
}
/**
 * Similar items result from vector search.
 */
export interface SimilarResult {
    type: 'similar';
    items: SimilarItem[];
}
/**
 * List of IDs result.
 */
export interface IdsResult {
    type: 'ids';
    ids: string[];
}
/**
 * Table list result.
 */
export interface TableListResult {
    type: 'tableList';
    names: string[];
}
/**
 * Blob data result.
 */
export interface BlobResult {
    type: 'blob';
    data: Uint8Array;
}
/**
 * Blob info result.
 */
export interface BlobInfoResult {
    type: 'blobInfo';
    info: ArtifactInfo;
}
/**
 * Error result.
 */
export interface ErrorResult {
    type: 'error';
    code: number;
    message: string;
}
/**
 * Artifact list result.
 */
export interface ArtifactListResult {
    type: 'artifactList';
    artifactIds: string[];
}
/**
 * Blob statistics result.
 */
export interface BlobStatsResult {
    type: 'blobStats';
    artifactCount: number;
    chunkCount: number;
    totalBytes: number;
    uniqueBytes: number;
    dedupRatio: number;
    orphanedChunks: number;
}
/**
 * Checkpoint information.
 */
export interface CheckpointInfo {
    id: string;
    name: string;
    createdAt: number;
    isAuto: boolean;
}
/**
 * Checkpoint list result.
 */
export interface CheckpointListResult {
    type: 'checkpointList';
    checkpoints: CheckpointInfo[];
}
/**
 * PageRank item.
 */
export interface PageRankItem {
    nodeId: string;
    score: number;
}
/**
 * PageRank result.
 */
export interface PageRankResult {
    type: 'pageRank';
    items: PageRankItem[];
    iterations?: number;
    convergence?: number;
    converged?: boolean;
}
/**
 * Centrality type enum.
 */
export type CentralityType = 'betweenness' | 'closeness' | 'eigenvector';
/**
 * Centrality item.
 */
export interface CentralityItem {
    nodeId: string;
    score: number;
}
/**
 * Centrality result.
 */
export interface CentralityResult {
    type: 'centrality';
    items: CentralityItem[];
    centralityType?: CentralityType;
    iterations?: number;
    converged?: boolean;
    sampleCount?: number;
}
/**
 * Community item.
 */
export interface CommunityItem {
    nodeId: string;
    communityId: string;
}
/**
 * Community member list.
 */
export interface CommunityMemberList {
    communityId: string;
    memberNodeIds: string[];
}
/**
 * Communities result.
 */
export interface CommunitiesResult {
    type: 'communities';
    items: CommunityItem[];
    communityCount?: number;
    modularity?: number;
    passes?: number;
    iterations?: number;
    communities?: CommunityMemberList[];
}
/**
 * Pattern match node binding.
 */
export interface PatternNodeBinding {
    id: string;
    label: string;
}
/**
 * Pattern match edge binding.
 */
export interface PatternEdgeBinding {
    id: string;
    edgeType: string;
    from: string;
    to: string;
}
/**
 * Pattern match path binding.
 */
export interface PatternPathBinding {
    nodes: string[];
    edges: string[];
    length: number;
}
/**
 * Pattern match binding value.
 */
export type PatternBindingValue = {
    type: 'node';
    value: PatternNodeBinding;
} | {
    type: 'edge';
    value: PatternEdgeBinding;
} | {
    type: 'path';
    value: PatternPathBinding;
};
/**
 * Pattern match binding entry.
 */
export interface PatternBindingEntry {
    variable: string;
    value: PatternBindingValue;
}
/**
 * Pattern match binding.
 */
export interface PatternMatchBinding {
    bindings: PatternBindingEntry[];
}
/**
 * Pattern match statistics.
 */
export interface PatternMatchStats {
    matchesFound: number;
    nodesEvaluated: number;
    edgesEvaluated: number;
    truncated: boolean;
}
/**
 * Pattern match result.
 */
export interface PatternMatchResult {
    type: 'patternMatch';
    matches: PatternMatchBinding[];
    stats?: PatternMatchStats;
}
/**
 * Constraint item.
 */
export interface ConstraintItem {
    name: string;
    target: string;
    property: string;
    constraintType: string;
}
/**
 * Constraints result.
 */
export interface ConstraintsResult {
    type: 'constraints';
    items: ConstraintItem[];
}
/**
 * Aggregate value type.
 */
export type AggregateValue = {
    type: 'count';
    value: number;
} | {
    type: 'sum';
    value: number;
} | {
    type: 'avg';
    value: number;
} | {
    type: 'min';
    value: number;
} | {
    type: 'max';
    value: number;
};
/**
 * Aggregate result.
 */
export interface AggregateResult {
    type: 'aggregate';
    value: AggregateValue;
}
/**
 * Batch operation result.
 */
export interface BatchOperationResult {
    type: 'batchOperation';
    operation: string;
    affectedCount: number;
    createdIds: string[];
}
/**
 * Graph indexes result.
 */
export interface GraphIndexesResult {
    type: 'graphIndexes';
    indexes: string[];
}
/**
 * Chain transaction begun.
 */
export interface ChainTransactionBegun {
    txId: string;
}
/**
 * Chain committed.
 */
export interface ChainCommitted {
    blockHash: string;
    height: number;
}
/**
 * Chain rolled back.
 */
export interface ChainRolledBack {
    toHeight: number;
}
/**
 * Chain history entry.
 */
export interface ChainHistoryEntry {
    height: number;
    transactionType: string;
    data?: Uint8Array;
}
/**
 * Chain history.
 */
export interface ChainHistory {
    entries: ChainHistoryEntry[];
}
/**
 * Chain similar item.
 */
export interface ChainSimilarItem {
    blockHash: string;
    height: number;
    similarity: number;
}
/**
 * Chain similar.
 */
export interface ChainSimilar {
    items: ChainSimilarItem[];
}
/**
 * Chain drift.
 */
export interface ChainDrift {
    fromHeight: number;
    toHeight: number;
    totalDrift: number;
    avgDriftPerBlock: number;
    maxDrift: number;
}
/**
 * Chain height.
 */
export interface ChainHeight {
    height: number;
}
/**
 * Chain tip.
 */
export interface ChainTip {
    hash: string;
    height: number;
}
/**
 * Chain block info.
 */
export interface ChainBlockInfo {
    height: number;
    hash: string;
    prevHash: string;
    timestamp: number;
    transactionCount: number;
    proposer: string;
}
/**
 * Chain codebook info.
 */
export interface ChainCodebookInfo {
    scope: string;
    entryCount: number;
    dimension: number;
    domain?: string;
}
/**
 * Chain transition analysis.
 */
export interface ChainTransitionAnalysis {
    totalTransitions: number;
    validTransitions: number;
    invalidTransitions: number;
    avgValidityScore: number;
}
/**
 * Chain conflict resolution.
 */
export interface ChainConflictResolution {
    strategy: string;
    conflictsResolved: number;
}
/**
 * Chain merge result.
 */
export interface ChainMergeResult {
    success: boolean;
    mergedCount: number;
}
/**
 * Chain query sub-result type.
 */
export type ChainSubResult = {
    type: 'transactionBegun';
    value: ChainTransactionBegun;
} | {
    type: 'committed';
    value: ChainCommitted;
} | {
    type: 'rolledBack';
    value: ChainRolledBack;
} | {
    type: 'history';
    value: ChainHistory;
} | {
    type: 'similar';
    value: ChainSimilar;
} | {
    type: 'drift';
    value: ChainDrift;
} | {
    type: 'height';
    value: ChainHeight;
} | {
    type: 'tip';
    value: ChainTip;
} | {
    type: 'block';
    value: ChainBlockInfo;
} | {
    type: 'codebook';
    value: ChainCodebookInfo;
} | {
    type: 'transitionAnalysis';
    value: ChainTransitionAnalysis;
} | {
    type: 'conflictResolution';
    value: ChainConflictResolution;
} | {
    type: 'merge';
    value: ChainMergeResult;
};
/**
 * Chain query result.
 */
export interface ChainQueryResult {
    type: 'chain';
    result: ChainSubResult;
}
/**
 * Unified item.
 */
export interface UnifiedItem {
    readonly entityType: string;
    readonly key: string;
    readonly fields: ReadonlyMap<string, Value>;
    readonly score?: number;
}
/**
 * Unified result.
 */
export interface UnifiedResult {
    type: 'unified';
    description: string;
    items: UnifiedItem[];
}
/**
 * Discriminated union of all query result types.
 */
export type QueryResult = EmptyResult | ValueResult | CountResult | RowsResult | NodesResult | EdgesResult | PathsResult | SimilarResult | IdsResult | TableListResult | BlobResult | BlobInfoResult | ArtifactListResult | BlobStatsResult | CheckpointListResult | PageRankResult | CentralityResult | CommunitiesResult | PatternMatchResult | ConstraintsResult | AggregateResult | BatchOperationResult | GraphIndexesResult | ChainQueryResult | UnifiedResult | ErrorResult;
/**
 * Type guard for empty result.
 */
export declare function isEmptyResult(result: QueryResult): result is EmptyResult;
/**
 * Type guard for rows result.
 */
export declare function isRowsResult(result: QueryResult): result is RowsResult;
/**
 * Type guard for nodes result.
 */
export declare function isNodesResult(result: QueryResult): result is NodesResult;
/**
 * Type guard for edges result.
 */
export declare function isEdgesResult(result: QueryResult): result is EdgesResult;
/**
 * Type guard for paths result.
 */
export declare function isPathsResult(result: QueryResult): result is PathsResult;
/**
 * Type guard for similar result.
 */
export declare function isSimilarResult(result: QueryResult): result is SimilarResult;
/**
 * Type guard for error result.
 */
export declare function isErrorResult(result: QueryResult): result is ErrorResult;
/**
 * Type guard for value result.
 */
export declare function isValueResult(result: QueryResult): result is ValueResult;
/**
 * Type guard for count result.
 */
export declare function isCountResult(result: QueryResult): result is CountResult;
/**
 * Type guard for ids result.
 */
export declare function isIdsResult(result: QueryResult): result is IdsResult;
/**
 * Type guard for table list result.
 */
export declare function isTableListResult(result: QueryResult): result is TableListResult;
/**
 * Type guard for blob result.
 */
export declare function isBlobResult(result: QueryResult): result is BlobResult;
/**
 * Type guard for blob info result.
 */
export declare function isBlobInfoResult(result: QueryResult): result is BlobInfoResult;
/**
 * Type guard for artifact list result.
 */
export declare function isArtifactListResult(result: QueryResult): result is ArtifactListResult;
/**
 * Type guard for blob stats result.
 */
export declare function isBlobStatsResult(result: QueryResult): result is BlobStatsResult;
/**
 * Type guard for checkpoint list result.
 */
export declare function isCheckpointListResult(result: QueryResult): result is CheckpointListResult;
/**
 * Type guard for page rank result.
 */
export declare function isPageRankResult(result: QueryResult): result is PageRankResult;
/**
 * Type guard for centrality result.
 */
export declare function isCentralityResult(result: QueryResult): result is CentralityResult;
/**
 * Type guard for communities result.
 */
export declare function isCommunitiesResult(result: QueryResult): result is CommunitiesResult;
/**
 * Type guard for pattern match result.
 */
export declare function isPatternMatchResult(result: QueryResult): result is PatternMatchResult;
/**
 * Type guard for constraints result.
 */
export declare function isConstraintsResult(result: QueryResult): result is ConstraintsResult;
/**
 * Type guard for aggregate result.
 */
export declare function isAggregateResult(result: QueryResult): result is AggregateResult;
/**
 * Type guard for batch operation result.
 */
export declare function isBatchOperationResult(result: QueryResult): result is BatchOperationResult;
/**
 * Type guard for graph indexes result.
 */
export declare function isGraphIndexesResult(result: QueryResult): result is GraphIndexesResult;
/**
 * Type guard for chain query result.
 */
export declare function isChainQueryResult(result: QueryResult): result is ChainQueryResult;
/**
 * Type guard for unified result.
 */
export declare function isUnifiedResult(result: QueryResult): result is UnifiedResult;
/**
 * Convert a Row to a plain object.
 */
export declare function rowToObject(row: Row): Record<string, unknown>;
/**
 * Convert a Node to a plain object.
 */
export declare function nodeToObject(node: Node): Record<string, unknown>;
/**
 * Convert an Edge to a plain object.
 */
export declare function edgeToObject(edge: Edge): Record<string, unknown>;
/**
 * Create a mutable copy of row values.
 * Use this when you need to modify values from a Row.
 */
export declare function copyRowValues(row: Row): Map<string, Value>;
/**
 * Create a mutable copy of node properties.
 * Use this when you need to modify properties from a Node.
 */
export declare function copyNodeProperties(node: Node): Map<string, Value>;
/**
 * Create a mutable copy of edge properties.
 * Use this when you need to modify properties from an Edge.
 */
export declare function copyEdgeProperties(edge: Edge): Map<string, Value>;
/**
 * Create a mutable copy of similar item metadata.
 * Use this when you need to modify metadata from a SimilarItem.
 */
export declare function copySimilarItemMetadata(item: SimilarItem): Map<string, Value> | undefined;
/**
 * Create a mutable copy of unified item fields.
 * Use this when you need to modify fields from a UnifiedItem.
 */
export declare function copyUnifiedItemFields(item: UnifiedItem): Map<string, Value>;
//# sourceMappingURL=query-result.d.ts.map