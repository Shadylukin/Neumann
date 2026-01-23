/**
 * @neumann/client - TypeScript SDK for Neumann database
 *
 * This package provides a TypeScript client for the Neumann database with
 * support for both Node.js (gRPC) and browser (gRPC-Web) environments.
 *
 * @example
 * ```typescript
 * import { NeumannClient } from '@neumann/client';
 *
 * // Connect to a remote server
 * const client = await NeumannClient.connect('localhost:9200', {
 *   apiKey: 'your-api-key',
 * });
 *
 * // Execute a query
 * const result = await client.execute('SELECT users');
 *
 * if (result.type === 'rows') {
 *   for (const row of result.rows) {
 *     console.log(rowToObject(row));
 *   }
 * }
 *
 * // Close the connection
 * client.close();
 * ```
 *
 * @packageDocumentation
 */
// Client
export { NeumannClient } from './client.js';
// Conversion utilities
export { convertProtoValue, convertProtoRow, convertProtoNode, convertProtoEdge, convertProtoPath, convertProtoSimilarItem, convertProtoArtifactInfo, } from './client.js';
export { nullValue, intValue, floatValue, stringValue, boolValue, bytesValue, valueToNative, valueFromNative, } from './types/value.js';
export { isEmptyResult, isRowsResult, isNodesResult, isEdgesResult, isPathsResult, isSimilarResult, isErrorResult, rowToObject, nodeToObject, edgeToObject, } from './types/query-result.js';
// Errors
export { ErrorCode, NeumannError, ConnectionError, AuthenticationError, PermissionDeniedError, NotFoundError, InvalidArgumentError, ParseError, QueryError, InternalError, errorFromCode, } from './types/errors.js';
//# sourceMappingURL=index.js.map