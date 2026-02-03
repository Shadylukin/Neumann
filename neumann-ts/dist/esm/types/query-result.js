/**
 * Type guard for empty result.
 */
export function isEmptyResult(result) {
    return result.type === 'empty';
}
/**
 * Type guard for rows result.
 */
export function isRowsResult(result) {
    return result.type === 'rows';
}
/**
 * Type guard for nodes result.
 */
export function isNodesResult(result) {
    return result.type === 'nodes';
}
/**
 * Type guard for edges result.
 */
export function isEdgesResult(result) {
    return result.type === 'edges';
}
/**
 * Type guard for paths result.
 */
export function isPathsResult(result) {
    return result.type === 'paths';
}
/**
 * Type guard for similar result.
 */
export function isSimilarResult(result) {
    return result.type === 'similar';
}
/**
 * Type guard for error result.
 */
export function isErrorResult(result) {
    return result.type === 'error';
}
/**
 * Type guard for value result.
 */
export function isValueResult(result) {
    return result.type === 'value';
}
/**
 * Type guard for count result.
 */
export function isCountResult(result) {
    return result.type === 'count';
}
/**
 * Type guard for ids result.
 */
export function isIdsResult(result) {
    return result.type === 'ids';
}
/**
 * Type guard for table list result.
 */
export function isTableListResult(result) {
    return result.type === 'tableList';
}
/**
 * Type guard for blob result.
 */
export function isBlobResult(result) {
    return result.type === 'blob';
}
/**
 * Type guard for blob info result.
 */
export function isBlobInfoResult(result) {
    return result.type === 'blobInfo';
}
/**
 * Type guard for artifact list result.
 */
export function isArtifactListResult(result) {
    return result.type === 'artifactList';
}
/**
 * Type guard for blob stats result.
 */
export function isBlobStatsResult(result) {
    return result.type === 'blobStats';
}
/**
 * Type guard for checkpoint list result.
 */
export function isCheckpointListResult(result) {
    return result.type === 'checkpointList';
}
/**
 * Type guard for page rank result.
 */
export function isPageRankResult(result) {
    return result.type === 'pageRank';
}
/**
 * Type guard for centrality result.
 */
export function isCentralityResult(result) {
    return result.type === 'centrality';
}
/**
 * Type guard for communities result.
 */
export function isCommunitiesResult(result) {
    return result.type === 'communities';
}
/**
 * Type guard for pattern match result.
 */
export function isPatternMatchResult(result) {
    return result.type === 'patternMatch';
}
/**
 * Type guard for constraints result.
 */
export function isConstraintsResult(result) {
    return result.type === 'constraints';
}
/**
 * Type guard for aggregate result.
 */
export function isAggregateResult(result) {
    return result.type === 'aggregate';
}
/**
 * Type guard for batch operation result.
 */
export function isBatchOperationResult(result) {
    return result.type === 'batchOperation';
}
/**
 * Type guard for graph indexes result.
 */
export function isGraphIndexesResult(result) {
    return result.type === 'graphIndexes';
}
/**
 * Type guard for chain query result.
 */
export function isChainQueryResult(result) {
    return result.type === 'chain';
}
/**
 * Type guard for unified result.
 */
export function isUnifiedResult(result) {
    return result.type === 'unified';
}
/**
 * Convert a Row to a plain object.
 */
export function rowToObject(row) {
    const obj = {};
    for (const [key, value] of row.values) {
        obj[key] = value.data;
    }
    return obj;
}
/**
 * Convert a Node to a plain object.
 */
export function nodeToObject(node) {
    const props = {};
    for (const [key, value] of node.properties) {
        props[key] = value.data;
    }
    return {
        id: node.id,
        label: node.label,
        properties: props,
    };
}
/**
 * Convert an Edge to a plain object.
 */
export function edgeToObject(edge) {
    const props = {};
    for (const [key, value] of edge.properties) {
        props[key] = value.data;
    }
    return {
        id: edge.id,
        type: edge.edgeType,
        source: edge.source,
        target: edge.target,
        properties: props,
    };
}
/**
 * Create a mutable copy of row values.
 * Use this when you need to modify values from a Row.
 */
export function copyRowValues(row) {
    return new Map(row.values);
}
/**
 * Create a mutable copy of node properties.
 * Use this when you need to modify properties from a Node.
 */
export function copyNodeProperties(node) {
    return new Map(node.properties);
}
/**
 * Create a mutable copy of edge properties.
 * Use this when you need to modify properties from an Edge.
 */
export function copyEdgeProperties(edge) {
    return new Map(edge.properties);
}
/**
 * Create a mutable copy of similar item metadata.
 * Use this when you need to modify metadata from a SimilarItem.
 */
export function copySimilarItemMetadata(item) {
    return item.metadata ? new Map(item.metadata) : undefined;
}
/**
 * Create a mutable copy of unified item fields.
 * Use this when you need to modify fields from a UnifiedItem.
 */
export function copyUnifiedItemFields(item) {
    return new Map(item.fields);
}
//# sourceMappingURL=query-result.js.map