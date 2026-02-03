"use strict";
// SPDX-License-Identifier: MIT
/**
 * Service clients for Neumann.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.CollectionsClient = exports.PointsClient = exports.HealthStatus = exports.HealthClient = exports.BlobClient = void 0;
var blob_js_1 = require("./blob.js");
Object.defineProperty(exports, "BlobClient", { enumerable: true, get: function () { return blob_js_1.BlobClient; } });
var health_js_1 = require("./health.js");
Object.defineProperty(exports, "HealthClient", { enumerable: true, get: function () { return health_js_1.HealthClient; } });
Object.defineProperty(exports, "HealthStatus", { enumerable: true, get: function () { return health_js_1.HealthStatus; } });
var vector_js_1 = require("./vector.js");
Object.defineProperty(exports, "PointsClient", { enumerable: true, get: function () { return vector_js_1.PointsClient; } });
Object.defineProperty(exports, "CollectionsClient", { enumerable: true, get: function () { return vector_js_1.CollectionsClient; } });
//# sourceMappingURL=index.js.map