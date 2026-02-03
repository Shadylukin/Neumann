// SPDX-License-Identifier: MIT
/**
 * @neumann/client - TypeScript SDK for Neumann database
 *
 * This package provides a TypeScript client for the Neumann database with
 * support for both Node.js (gRPC) and browser (gRPC-Web) environments.
 *
 * @example
 * ```typescript
 * import { NeumannClient, VectorClient } from '@neumann/client';
 *
 * // Connect to a remote server
 * const client = await NeumannClient.connect('localhost:9200', {
 *   apiKey: 'your-api-key',
 * });
 *
 * // Execute a query
 * const result = await client.query('SELECT users');
 *
 * if (result.type === 'rows') {
 *   for (const row of result.rows) {
 *     console.log(rowToObject(row));
 *   }
 * }
 *
 * // Close the connection
 * client.close();
 *
 * // Vector operations
 * const vectors = await VectorClient.connect('localhost:9200');
 * await vectors.createCollection('docs', 384, 'cosine');
 * await vectors.upsertPoints('docs', [
 *   { id: 'doc1', vector: [...], payload: { title: 'Hello' } },
 * ]);
 * const results = await vectors.queryPoints('docs', queryVector, { limit: 10 });
 * vectors.close();
 * ```
 *
 * @packageDocumentation
 */
// Main client
export { NeumannClient } from './client.js';
// Retry and configuration
export { withRetry, withRetryWrapper, isRetryable, calculateBackoff, mergeRetryConfig, DEFAULT_RETRY_CONFIG, RETRYABLE_STATUS_CODES, } from './retry.js';
export { mergeClientConfig, noRetryConfig, fastFailConfig, highLatencyConfig, toGrpcChannelOptions, DEFAULT_TIMEOUT_CONFIG, DEFAULT_KEEPALIVE_CONFIG, DEFAULT_CLIENT_CONFIG, } from './config.js';
// Vector client
export { VectorClient } from './vector-client.js';
// Conversion utilities
export { convertProtoValue, convertProtoRow, convertProtoNode, convertProtoEdge, convertProtoPath, convertProtoSimilarItem, convertProtoArtifactInfo, convertProtoCheckpoint, convertProtoPageRankItem, convertProtoCentralityType, convertProtoCentralityItem, convertProtoCommunityItem, convertProtoCommunityMemberList, convertProtoPatternMatchBinding, convertProtoBindingEntry, convertProtoBindingValue, convertProtoPatternMatchStats, convertProtoConstraintItem, convertProtoAggregateValue, convertProtoChainResult, convertProtoUnifiedItem, } from './client.js';
export { nullValue, intValue, floatValue, stringValue, boolValue, bytesValue, valueToNative, valueFromNative, } from './types/value.js';
export { isEmptyResult, isRowsResult, isNodesResult, isEdgesResult, isPathsResult, isSimilarResult, isErrorResult, isValueResult, isCountResult, isIdsResult, isTableListResult, isBlobResult, isBlobInfoResult, isArtifactListResult, isBlobStatsResult, isCheckpointListResult, isPageRankResult, isCentralityResult, isCommunitiesResult, isPatternMatchResult, isConstraintsResult, isAggregateResult, isBatchOperationResult, isGraphIndexesResult, isChainQueryResult, isUnifiedResult, rowToObject, nodeToObject, edgeToObject, copyRowValues, copyNodeProperties, copyEdgeProperties, copySimilarItemMetadata, copyUnifiedItemFields, } from './types/query-result.js';
// Validation utilities
export { MAX_STRING_LENGTH, MAX_BYTES_LENGTH, validateIntValue, validateFloatValue, validateStringValue, validateBytesValue, safeIdToString, safeIdsToStrings, } from './types/validation.js';
// Errors
export { ErrorCode, NeumannError, ConnectionError, AuthenticationError, PermissionDeniedError, NotFoundError, InvalidArgumentError, ParseError, QueryError, InternalError, errorFromCode, } from './types/errors.js';
// Services
export { BlobClient, HealthClient, HealthStatus, PointsClient, CollectionsClient, } from './services/index.js';
//# sourceMappingURL=index.js.map