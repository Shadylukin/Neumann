"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.isCommunitiesResult = exports.isCentralityResult = exports.isPageRankResult = exports.isCheckpointListResult = exports.isBlobStatsResult = exports.isArtifactListResult = exports.isBlobInfoResult = exports.isBlobResult = exports.isTableListResult = exports.isIdsResult = exports.isCountResult = exports.isValueResult = exports.isErrorResult = exports.isSimilarResult = exports.isPathsResult = exports.isEdgesResult = exports.isNodesResult = exports.isRowsResult = exports.isEmptyResult = exports.valueFromNative = exports.valueToNative = exports.bytesValue = exports.boolValue = exports.stringValue = exports.floatValue = exports.intValue = exports.nullValue = exports.convertProtoUnifiedItem = exports.convertProtoChainResult = exports.convertProtoAggregateValue = exports.convertProtoConstraintItem = exports.convertProtoPatternMatchStats = exports.convertProtoBindingValue = exports.convertProtoBindingEntry = exports.convertProtoPatternMatchBinding = exports.convertProtoCommunityMemberList = exports.convertProtoCommunityItem = exports.convertProtoCentralityItem = exports.convertProtoCentralityType = exports.convertProtoPageRankItem = exports.convertProtoCheckpoint = exports.convertProtoArtifactInfo = exports.convertProtoSimilarItem = exports.convertProtoPath = exports.convertProtoEdge = exports.convertProtoNode = exports.convertProtoRow = exports.convertProtoValue = exports.VectorClient = exports.NeumannClient = void 0;
exports.CollectionsClient = exports.PointsClient = exports.HealthStatus = exports.HealthClient = exports.BlobClient = exports.errorFromCode = exports.InternalError = exports.QueryError = exports.ParseError = exports.InvalidArgumentError = exports.NotFoundError = exports.PermissionDeniedError = exports.AuthenticationError = exports.ConnectionError = exports.NeumannError = exports.ErrorCode = exports.safeIdsToStrings = exports.safeIdToString = exports.validateBytesValue = exports.validateStringValue = exports.validateFloatValue = exports.validateIntValue = exports.MAX_BYTES_LENGTH = exports.MAX_STRING_LENGTH = exports.copyUnifiedItemFields = exports.copySimilarItemMetadata = exports.copyEdgeProperties = exports.copyNodeProperties = exports.copyRowValues = exports.edgeToObject = exports.nodeToObject = exports.rowToObject = exports.isUnifiedResult = exports.isChainQueryResult = exports.isGraphIndexesResult = exports.isBatchOperationResult = exports.isAggregateResult = exports.isConstraintsResult = exports.isPatternMatchResult = void 0;
// Main client
var client_js_1 = require("./client.js");
Object.defineProperty(exports, "NeumannClient", { enumerable: true, get: function () { return client_js_1.NeumannClient; } });
// Vector client
var vector_client_js_1 = require("./vector-client.js");
Object.defineProperty(exports, "VectorClient", { enumerable: true, get: function () { return vector_client_js_1.VectorClient; } });
// Conversion utilities
var client_js_2 = require("./client.js");
Object.defineProperty(exports, "convertProtoValue", { enumerable: true, get: function () { return client_js_2.convertProtoValue; } });
Object.defineProperty(exports, "convertProtoRow", { enumerable: true, get: function () { return client_js_2.convertProtoRow; } });
Object.defineProperty(exports, "convertProtoNode", { enumerable: true, get: function () { return client_js_2.convertProtoNode; } });
Object.defineProperty(exports, "convertProtoEdge", { enumerable: true, get: function () { return client_js_2.convertProtoEdge; } });
Object.defineProperty(exports, "convertProtoPath", { enumerable: true, get: function () { return client_js_2.convertProtoPath; } });
Object.defineProperty(exports, "convertProtoSimilarItem", { enumerable: true, get: function () { return client_js_2.convertProtoSimilarItem; } });
Object.defineProperty(exports, "convertProtoArtifactInfo", { enumerable: true, get: function () { return client_js_2.convertProtoArtifactInfo; } });
Object.defineProperty(exports, "convertProtoCheckpoint", { enumerable: true, get: function () { return client_js_2.convertProtoCheckpoint; } });
Object.defineProperty(exports, "convertProtoPageRankItem", { enumerable: true, get: function () { return client_js_2.convertProtoPageRankItem; } });
Object.defineProperty(exports, "convertProtoCentralityType", { enumerable: true, get: function () { return client_js_2.convertProtoCentralityType; } });
Object.defineProperty(exports, "convertProtoCentralityItem", { enumerable: true, get: function () { return client_js_2.convertProtoCentralityItem; } });
Object.defineProperty(exports, "convertProtoCommunityItem", { enumerable: true, get: function () { return client_js_2.convertProtoCommunityItem; } });
Object.defineProperty(exports, "convertProtoCommunityMemberList", { enumerable: true, get: function () { return client_js_2.convertProtoCommunityMemberList; } });
Object.defineProperty(exports, "convertProtoPatternMatchBinding", { enumerable: true, get: function () { return client_js_2.convertProtoPatternMatchBinding; } });
Object.defineProperty(exports, "convertProtoBindingEntry", { enumerable: true, get: function () { return client_js_2.convertProtoBindingEntry; } });
Object.defineProperty(exports, "convertProtoBindingValue", { enumerable: true, get: function () { return client_js_2.convertProtoBindingValue; } });
Object.defineProperty(exports, "convertProtoPatternMatchStats", { enumerable: true, get: function () { return client_js_2.convertProtoPatternMatchStats; } });
Object.defineProperty(exports, "convertProtoConstraintItem", { enumerable: true, get: function () { return client_js_2.convertProtoConstraintItem; } });
Object.defineProperty(exports, "convertProtoAggregateValue", { enumerable: true, get: function () { return client_js_2.convertProtoAggregateValue; } });
Object.defineProperty(exports, "convertProtoChainResult", { enumerable: true, get: function () { return client_js_2.convertProtoChainResult; } });
Object.defineProperty(exports, "convertProtoUnifiedItem", { enumerable: true, get: function () { return client_js_2.convertProtoUnifiedItem; } });
var value_js_1 = require("./types/value.js");
Object.defineProperty(exports, "nullValue", { enumerable: true, get: function () { return value_js_1.nullValue; } });
Object.defineProperty(exports, "intValue", { enumerable: true, get: function () { return value_js_1.intValue; } });
Object.defineProperty(exports, "floatValue", { enumerable: true, get: function () { return value_js_1.floatValue; } });
Object.defineProperty(exports, "stringValue", { enumerable: true, get: function () { return value_js_1.stringValue; } });
Object.defineProperty(exports, "boolValue", { enumerable: true, get: function () { return value_js_1.boolValue; } });
Object.defineProperty(exports, "bytesValue", { enumerable: true, get: function () { return value_js_1.bytesValue; } });
Object.defineProperty(exports, "valueToNative", { enumerable: true, get: function () { return value_js_1.valueToNative; } });
Object.defineProperty(exports, "valueFromNative", { enumerable: true, get: function () { return value_js_1.valueFromNative; } });
var query_result_js_1 = require("./types/query-result.js");
Object.defineProperty(exports, "isEmptyResult", { enumerable: true, get: function () { return query_result_js_1.isEmptyResult; } });
Object.defineProperty(exports, "isRowsResult", { enumerable: true, get: function () { return query_result_js_1.isRowsResult; } });
Object.defineProperty(exports, "isNodesResult", { enumerable: true, get: function () { return query_result_js_1.isNodesResult; } });
Object.defineProperty(exports, "isEdgesResult", { enumerable: true, get: function () { return query_result_js_1.isEdgesResult; } });
Object.defineProperty(exports, "isPathsResult", { enumerable: true, get: function () { return query_result_js_1.isPathsResult; } });
Object.defineProperty(exports, "isSimilarResult", { enumerable: true, get: function () { return query_result_js_1.isSimilarResult; } });
Object.defineProperty(exports, "isErrorResult", { enumerable: true, get: function () { return query_result_js_1.isErrorResult; } });
Object.defineProperty(exports, "isValueResult", { enumerable: true, get: function () { return query_result_js_1.isValueResult; } });
Object.defineProperty(exports, "isCountResult", { enumerable: true, get: function () { return query_result_js_1.isCountResult; } });
Object.defineProperty(exports, "isIdsResult", { enumerable: true, get: function () { return query_result_js_1.isIdsResult; } });
Object.defineProperty(exports, "isTableListResult", { enumerable: true, get: function () { return query_result_js_1.isTableListResult; } });
Object.defineProperty(exports, "isBlobResult", { enumerable: true, get: function () { return query_result_js_1.isBlobResult; } });
Object.defineProperty(exports, "isBlobInfoResult", { enumerable: true, get: function () { return query_result_js_1.isBlobInfoResult; } });
Object.defineProperty(exports, "isArtifactListResult", { enumerable: true, get: function () { return query_result_js_1.isArtifactListResult; } });
Object.defineProperty(exports, "isBlobStatsResult", { enumerable: true, get: function () { return query_result_js_1.isBlobStatsResult; } });
Object.defineProperty(exports, "isCheckpointListResult", { enumerable: true, get: function () { return query_result_js_1.isCheckpointListResult; } });
Object.defineProperty(exports, "isPageRankResult", { enumerable: true, get: function () { return query_result_js_1.isPageRankResult; } });
Object.defineProperty(exports, "isCentralityResult", { enumerable: true, get: function () { return query_result_js_1.isCentralityResult; } });
Object.defineProperty(exports, "isCommunitiesResult", { enumerable: true, get: function () { return query_result_js_1.isCommunitiesResult; } });
Object.defineProperty(exports, "isPatternMatchResult", { enumerable: true, get: function () { return query_result_js_1.isPatternMatchResult; } });
Object.defineProperty(exports, "isConstraintsResult", { enumerable: true, get: function () { return query_result_js_1.isConstraintsResult; } });
Object.defineProperty(exports, "isAggregateResult", { enumerable: true, get: function () { return query_result_js_1.isAggregateResult; } });
Object.defineProperty(exports, "isBatchOperationResult", { enumerable: true, get: function () { return query_result_js_1.isBatchOperationResult; } });
Object.defineProperty(exports, "isGraphIndexesResult", { enumerable: true, get: function () { return query_result_js_1.isGraphIndexesResult; } });
Object.defineProperty(exports, "isChainQueryResult", { enumerable: true, get: function () { return query_result_js_1.isChainQueryResult; } });
Object.defineProperty(exports, "isUnifiedResult", { enumerable: true, get: function () { return query_result_js_1.isUnifiedResult; } });
Object.defineProperty(exports, "rowToObject", { enumerable: true, get: function () { return query_result_js_1.rowToObject; } });
Object.defineProperty(exports, "nodeToObject", { enumerable: true, get: function () { return query_result_js_1.nodeToObject; } });
Object.defineProperty(exports, "edgeToObject", { enumerable: true, get: function () { return query_result_js_1.edgeToObject; } });
Object.defineProperty(exports, "copyRowValues", { enumerable: true, get: function () { return query_result_js_1.copyRowValues; } });
Object.defineProperty(exports, "copyNodeProperties", { enumerable: true, get: function () { return query_result_js_1.copyNodeProperties; } });
Object.defineProperty(exports, "copyEdgeProperties", { enumerable: true, get: function () { return query_result_js_1.copyEdgeProperties; } });
Object.defineProperty(exports, "copySimilarItemMetadata", { enumerable: true, get: function () { return query_result_js_1.copySimilarItemMetadata; } });
Object.defineProperty(exports, "copyUnifiedItemFields", { enumerable: true, get: function () { return query_result_js_1.copyUnifiedItemFields; } });
// Validation utilities
var validation_js_1 = require("./types/validation.js");
Object.defineProperty(exports, "MAX_STRING_LENGTH", { enumerable: true, get: function () { return validation_js_1.MAX_STRING_LENGTH; } });
Object.defineProperty(exports, "MAX_BYTES_LENGTH", { enumerable: true, get: function () { return validation_js_1.MAX_BYTES_LENGTH; } });
Object.defineProperty(exports, "validateIntValue", { enumerable: true, get: function () { return validation_js_1.validateIntValue; } });
Object.defineProperty(exports, "validateFloatValue", { enumerable: true, get: function () { return validation_js_1.validateFloatValue; } });
Object.defineProperty(exports, "validateStringValue", { enumerable: true, get: function () { return validation_js_1.validateStringValue; } });
Object.defineProperty(exports, "validateBytesValue", { enumerable: true, get: function () { return validation_js_1.validateBytesValue; } });
Object.defineProperty(exports, "safeIdToString", { enumerable: true, get: function () { return validation_js_1.safeIdToString; } });
Object.defineProperty(exports, "safeIdsToStrings", { enumerable: true, get: function () { return validation_js_1.safeIdsToStrings; } });
// Errors
var errors_js_1 = require("./types/errors.js");
Object.defineProperty(exports, "ErrorCode", { enumerable: true, get: function () { return errors_js_1.ErrorCode; } });
Object.defineProperty(exports, "NeumannError", { enumerable: true, get: function () { return errors_js_1.NeumannError; } });
Object.defineProperty(exports, "ConnectionError", { enumerable: true, get: function () { return errors_js_1.ConnectionError; } });
Object.defineProperty(exports, "AuthenticationError", { enumerable: true, get: function () { return errors_js_1.AuthenticationError; } });
Object.defineProperty(exports, "PermissionDeniedError", { enumerable: true, get: function () { return errors_js_1.PermissionDeniedError; } });
Object.defineProperty(exports, "NotFoundError", { enumerable: true, get: function () { return errors_js_1.NotFoundError; } });
Object.defineProperty(exports, "InvalidArgumentError", { enumerable: true, get: function () { return errors_js_1.InvalidArgumentError; } });
Object.defineProperty(exports, "ParseError", { enumerable: true, get: function () { return errors_js_1.ParseError; } });
Object.defineProperty(exports, "QueryError", { enumerable: true, get: function () { return errors_js_1.QueryError; } });
Object.defineProperty(exports, "InternalError", { enumerable: true, get: function () { return errors_js_1.InternalError; } });
Object.defineProperty(exports, "errorFromCode", { enumerable: true, get: function () { return errors_js_1.errorFromCode; } });
// Services
var index_js_1 = require("./services/index.js");
Object.defineProperty(exports, "BlobClient", { enumerable: true, get: function () { return index_js_1.BlobClient; } });
Object.defineProperty(exports, "HealthClient", { enumerable: true, get: function () { return index_js_1.HealthClient; } });
Object.defineProperty(exports, "HealthStatus", { enumerable: true, get: function () { return index_js_1.HealthStatus; } });
Object.defineProperty(exports, "PointsClient", { enumerable: true, get: function () { return index_js_1.PointsClient; } });
Object.defineProperty(exports, "CollectionsClient", { enumerable: true, get: function () { return index_js_1.CollectionsClient; } });
//# sourceMappingURL=index.js.map