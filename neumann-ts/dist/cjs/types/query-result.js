"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.isEmptyResult = isEmptyResult;
exports.isRowsResult = isRowsResult;
exports.isNodesResult = isNodesResult;
exports.isEdgesResult = isEdgesResult;
exports.isPathsResult = isPathsResult;
exports.isSimilarResult = isSimilarResult;
exports.isErrorResult = isErrorResult;
exports.rowToObject = rowToObject;
exports.nodeToObject = nodeToObject;
exports.edgeToObject = edgeToObject;
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
//# sourceMappingURL=query-result.js.map