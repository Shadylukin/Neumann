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
Object.defineProperty(exports, "__esModule", { value: true });
exports.errorFromCode = exports.InternalError = exports.QueryError = exports.ParseError = exports.InvalidArgumentError = exports.NotFoundError = exports.PermissionDeniedError = exports.AuthenticationError = exports.ConnectionError = exports.NeumannError = exports.ErrorCode = exports.edgeToObject = exports.nodeToObject = exports.rowToObject = exports.isErrorResult = exports.isSimilarResult = exports.isPathsResult = exports.isEdgesResult = exports.isNodesResult = exports.isRowsResult = exports.isEmptyResult = exports.valueFromNative = exports.valueToNative = exports.bytesValue = exports.boolValue = exports.stringValue = exports.floatValue = exports.intValue = exports.nullValue = exports.convertProtoArtifactInfo = exports.convertProtoSimilarItem = exports.convertProtoPath = exports.convertProtoEdge = exports.convertProtoNode = exports.convertProtoRow = exports.convertProtoValue = exports.NeumannClient = void 0;
// Client
var client_js_1 = require("./client.js");
Object.defineProperty(exports, "NeumannClient", { enumerable: true, get: function () { return client_js_1.NeumannClient; } });
// Conversion utilities
var client_js_2 = require("./client.js");
Object.defineProperty(exports, "convertProtoValue", { enumerable: true, get: function () { return client_js_2.convertProtoValue; } });
Object.defineProperty(exports, "convertProtoRow", { enumerable: true, get: function () { return client_js_2.convertProtoRow; } });
Object.defineProperty(exports, "convertProtoNode", { enumerable: true, get: function () { return client_js_2.convertProtoNode; } });
Object.defineProperty(exports, "convertProtoEdge", { enumerable: true, get: function () { return client_js_2.convertProtoEdge; } });
Object.defineProperty(exports, "convertProtoPath", { enumerable: true, get: function () { return client_js_2.convertProtoPath; } });
Object.defineProperty(exports, "convertProtoSimilarItem", { enumerable: true, get: function () { return client_js_2.convertProtoSimilarItem; } });
Object.defineProperty(exports, "convertProtoArtifactInfo", { enumerable: true, get: function () { return client_js_2.convertProtoArtifactInfo; } });
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
Object.defineProperty(exports, "rowToObject", { enumerable: true, get: function () { return query_result_js_1.rowToObject; } });
Object.defineProperty(exports, "nodeToObject", { enumerable: true, get: function () { return query_result_js_1.nodeToObject; } });
Object.defineProperty(exports, "edgeToObject", { enumerable: true, get: function () { return query_result_js_1.edgeToObject; } });
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
//# sourceMappingURL=index.js.map