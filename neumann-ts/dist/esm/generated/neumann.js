// SPDX-License-Identifier: MIT
/**
 * Static TypeScript definitions for neumann.proto
 *
 * This file provides static type definitions that are compatible with both
 * ESM and CommonJS builds, eliminating the need for runtime proto loading.
 */
// === Enums ===
export var ErrorCode;
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
})(ErrorCode || (ErrorCode = {}));
export var ServingStatus;
(function (ServingStatus) {
    ServingStatus[ServingStatus["UNSPECIFIED"] = 0] = "UNSPECIFIED";
    ServingStatus[ServingStatus["SERVING"] = 1] = "SERVING";
    ServingStatus[ServingStatus["NOT_SERVING"] = 2] = "NOT_SERVING";
})(ServingStatus || (ServingStatus = {}));
export var CentralityType;
(function (CentralityType) {
    CentralityType[CentralityType["UNSPECIFIED"] = 0] = "UNSPECIFIED";
    CentralityType[CentralityType["BETWEENNESS"] = 1] = "BETWEENNESS";
    CentralityType[CentralityType["CLOSENESS"] = 2] = "CLOSENESS";
    CentralityType[CentralityType["EIGENVECTOR"] = 3] = "EIGENVECTOR";
})(CentralityType || (CentralityType = {}));
//# sourceMappingURL=neumann.js.map