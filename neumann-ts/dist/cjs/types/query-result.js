"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.isEmptyResult = isEmptyResult;
exports.isRowsResult = isRowsResult;
exports.isNodesResult = isNodesResult;
exports.isEdgesResult = isEdgesResult;
exports.isPathsResult = isPathsResult;
exports.isSimilarResult = isSimilarResult;
exports.isErrorResult = isErrorResult;
exports.isValueResult = isValueResult;
exports.isCountResult = isCountResult;
exports.isIdsResult = isIdsResult;
exports.isTableListResult = isTableListResult;
exports.isBlobResult = isBlobResult;
exports.isBlobInfoResult = isBlobInfoResult;
exports.isArtifactListResult = isArtifactListResult;
exports.isBlobStatsResult = isBlobStatsResult;
exports.isCheckpointListResult = isCheckpointListResult;
exports.isPageRankResult = isPageRankResult;
exports.isCentralityResult = isCentralityResult;
exports.isCommunitiesResult = isCommunitiesResult;
exports.isPatternMatchResult = isPatternMatchResult;
exports.isConstraintsResult = isConstraintsResult;
exports.isAggregateResult = isAggregateResult;
exports.isBatchOperationResult = isBatchOperationResult;
exports.isGraphIndexesResult = isGraphIndexesResult;
exports.isChainQueryResult = isChainQueryResult;
exports.isUnifiedResult = isUnifiedResult;
exports.rowToObject = rowToObject;
exports.nodeToObject = nodeToObject;
exports.edgeToObject = edgeToObject;
exports.copyRowValues = copyRowValues;
exports.copyNodeProperties = copyNodeProperties;
exports.copyEdgeProperties = copyEdgeProperties;
exports.copySimilarItemMetadata = copySimilarItemMetadata;
exports.copyUnifiedItemFields = copyUnifiedItemFields;
/**
 * Type guard for empty result.
 */
function isEmptyResult(result) {
    return result.type === 'empty';
}
/**
 * Type guard for rows result.
 */
function isRowsResult(result) {
    return result.type === 'rows';
}
/**
 * Type guard for nodes result.
 */
function isNodesResult(result) {
    return result.type === 'nodes';
}
/**
 * Type guard for edges result.
 */
function isEdgesResult(result) {
    return result.type === 'edges';
}
/**
 * Type guard for paths result.
 */
function isPathsResult(result) {
    return result.type === 'paths';
}
/**
 * Type guard for similar result.
 */
function isSimilarResult(result) {
    return result.type === 'similar';
}
/**
 * Type guard for error result.
 */
function isErrorResult(result) {
    return result.type === 'error';
}
/**
 * Type guard for value result.
 */
function isValueResult(result) {
    return result.type === 'value';
}
/**
 * Type guard for count result.
 */
function isCountResult(result) {
    return result.type === 'count';
}
/**
 * Type guard for ids result.
 */
function isIdsResult(result) {
    return result.type === 'ids';
}
/**
 * Type guard for table list result.
 */
function isTableListResult(result) {
    return result.type === 'tableList';
}
/**
 * Type guard for blob result.
 */
function isBlobResult(result) {
    return result.type === 'blob';
}
/**
 * Type guard for blob info result.
 */
function isBlobInfoResult(result) {
    return result.type === 'blobInfo';
}
/**
 * Type guard for artifact list result.
 */
function isArtifactListResult(result) {
    return result.type === 'artifactList';
}
/**
 * Type guard for blob stats result.
 */
function isBlobStatsResult(result) {
    return result.type === 'blobStats';
}
/**
 * Type guard for checkpoint list result.
 */
function isCheckpointListResult(result) {
    return result.type === 'checkpointList';
}
/**
 * Type guard for page rank result.
 */
function isPageRankResult(result) {
    return result.type === 'pageRank';
}
/**
 * Type guard for centrality result.
 */
function isCentralityResult(result) {
    return result.type === 'centrality';
}
/**
 * Type guard for communities result.
 */
function isCommunitiesResult(result) {
    return result.type === 'communities';
}
/**
 * Type guard for pattern match result.
 */
function isPatternMatchResult(result) {
    return result.type === 'patternMatch';
}
/**
 * Type guard for constraints result.
 */
function isConstraintsResult(result) {
    return result.type === 'constraints';
}
/**
 * Type guard for aggregate result.
 */
function isAggregateResult(result) {
    return result.type === 'aggregate';
}
/**
 * Type guard for batch operation result.
 */
function isBatchOperationResult(result) {
    return result.type === 'batchOperation';
}
/**
 * Type guard for graph indexes result.
 */
function isGraphIndexesResult(result) {
    return result.type === 'graphIndexes';
}
/**
 * Type guard for chain query result.
 */
function isChainQueryResult(result) {
    return result.type === 'chain';
}
/**
 * Type guard for unified result.
 */
function isUnifiedResult(result) {
    return result.type === 'unified';
}
/**
 * Convert a Row to a plain object.
 */
function rowToObject(row) {
    const obj = {};
    for (const [key, value] of row.values) {
        obj[key] = value.data;
    }
    return obj;
}
/**
 * Convert a Node to a plain object.
 */
function nodeToObject(node) {
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
function edgeToObject(edge) {
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
function copyRowValues(row) {
    return new Map(row.values);
}
/**
 * Create a mutable copy of node properties.
 * Use this when you need to modify properties from a Node.
 */
function copyNodeProperties(node) {
    return new Map(node.properties);
}
/**
 * Create a mutable copy of edge properties.
 * Use this when you need to modify properties from an Edge.
 */
function copyEdgeProperties(edge) {
    return new Map(edge.properties);
}
/**
 * Create a mutable copy of similar item metadata.
 * Use this when you need to modify metadata from a SimilarItem.
 */
function copySimilarItemMetadata(item) {
    return item.metadata ? new Map(item.metadata) : undefined;
}
/**
 * Create a mutable copy of unified item fields.
 * Use this when you need to modify fields from a UnifiedItem.
 */
function copyUnifiedItemFields(item) {
    return new Map(item.fields);
}
//# sourceMappingURL=query-result.js.map