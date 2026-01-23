import type { Value } from './value.js';

/**
 * A row from a relational query result.
 */
export interface Row {
  values: Map<string, Value>;
}

/**
 * A graph node.
 */
export interface Node {
  id: string;
  label: string;
  properties: Map<string, Value>;
}

/**
 * A graph edge.
 */
export interface Edge {
  id: string;
  edgeType: string;
  source: string;
  target: string;
  properties: Map<string, Value>;
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
  key: string;
  score: number;
  metadata?: Map<string, Value>;
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
export type QueryResultType =
  | 'empty'
  | 'value'
  | 'count'
  | 'rows'
  | 'nodes'
  | 'edges'
  | 'paths'
  | 'similar'
  | 'ids'
  | 'tableList'
  | 'blob'
  | 'blobInfo'
  | 'chainCommitted'
  | 'chainTransaction'
  | 'unified'
  | 'error';

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
 * Discriminated union of all query result types.
 */
export type QueryResult =
  | EmptyResult
  | ValueResult
  | CountResult
  | RowsResult
  | NodesResult
  | EdgesResult
  | PathsResult
  | SimilarResult
  | IdsResult
  | TableListResult
  | BlobResult
  | BlobInfoResult
  | ErrorResult;

/**
 * Type guard for empty result.
 */
export function isEmptyResult(result: QueryResult): result is EmptyResult {
  return result.type === 'empty';
}

/**
 * Type guard for rows result.
 */
export function isRowsResult(result: QueryResult): result is RowsResult {
  return result.type === 'rows';
}

/**
 * Type guard for nodes result.
 */
export function isNodesResult(result: QueryResult): result is NodesResult {
  return result.type === 'nodes';
}

/**
 * Type guard for edges result.
 */
export function isEdgesResult(result: QueryResult): result is EdgesResult {
  return result.type === 'edges';
}

/**
 * Type guard for paths result.
 */
export function isPathsResult(result: QueryResult): result is PathsResult {
  return result.type === 'paths';
}

/**
 * Type guard for similar result.
 */
export function isSimilarResult(result: QueryResult): result is SimilarResult {
  return result.type === 'similar';
}

/**
 * Type guard for error result.
 */
export function isErrorResult(result: QueryResult): result is ErrorResult {
  return result.type === 'error';
}

/**
 * Convert a Row to a plain object.
 */
export function rowToObject(row: Row): Record<string, unknown> {
  const obj: Record<string, unknown> = {};
  for (const [key, value] of row.values) {
    obj[key] = value.data;
  }
  return obj;
}

/**
 * Convert a Node to a plain object.
 */
export function nodeToObject(node: Node): Record<string, unknown> {
  const props: Record<string, unknown> = {};
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
export function edgeToObject(edge: Edge): Record<string, unknown> {
  const props: Record<string, unknown> = {};
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
