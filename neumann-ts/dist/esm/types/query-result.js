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
//# sourceMappingURL=query-result.js.map