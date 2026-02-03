"use strict";
// SPDX-License-Identifier: MIT
/**
 * Static TypeScript definitions for neumann.proto
 *
 * This file provides static type definitions that are compatible with both
 * ESM and CommonJS builds, eliminating the need for runtime proto loading.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.CentralityType = exports.ServingStatus = exports.ErrorCode = void 0;
// === Enums ===
var ErrorCode;
(function (ErrorCode) {
    ErrorCode[ErrorCode["UNSPECIFIED"] = 0] = "UNSPECIFIED";
    ErrorCode[ErrorCode["INVALID_QUERY"] = 1] = "INVALID_QUERY";
    ErrorCode[ErrorCode["NOT_FOUND"] = 2] = "NOT_FOUND";
    ErrorCode[ErrorCode["PERMISSION_DENIED"] = 3] = "PERMISSION_DENIED";
    ErrorCode[ErrorCode["ALREADY_EXISTS"] = 4] = "ALREADY_EXISTS";
    ErrorCode[ErrorCode["INTERNAL"] = 5] = "INTERNAL";
    ErrorCode[ErrorCode["UNAVAILABLE"] = 6] = "UNAVAILABLE";
    ErrorCode[ErrorCode["INVALID_ARGUMENT"] = 7] = "INVALID_ARGUMENT";
    ErrorCode[ErrorCode["UNAUTHENTICATED"] = 8] = "UNAUTHENTICATED";
})(ErrorCode || (exports.ErrorCode = ErrorCode = {}));
var ServingStatus;
(function (ServingStatus) {
    ServingStatus[ServingStatus["UNSPECIFIED"] = 0] = "UNSPECIFIED";
    ServingStatus[ServingStatus["SERVING"] = 1] = "SERVING";
    ServingStatus[ServingStatus["NOT_SERVING"] = 2] = "NOT_SERVING";
})(ServingStatus || (exports.ServingStatus = ServingStatus = {}));
var CentralityType;
(function (CentralityType) {
    CentralityType[CentralityType["UNSPECIFIED"] = 0] = "UNSPECIFIED";
    CentralityType[CentralityType["BETWEENNESS"] = 1] = "BETWEENNESS";
    CentralityType[CentralityType["CLOSENESS"] = 2] = "CLOSENESS";
    CentralityType[CentralityType["EIGENVECTOR"] = 3] = "EIGENVECTOR";
})(CentralityType || (exports.CentralityType = CentralityType = {}));
//# sourceMappingURL=neumann.js.map