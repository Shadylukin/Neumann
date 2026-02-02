// SPDX-License-Identifier: MIT
/**
 * Service clients for Neumann.
 */

export { BlobClient } from './blob.js';
export type {
  BlobUploadOptions,
  BlobUploadResult,
  ArtifactMetadata,
} from './blob.js';

export { HealthClient, HealthStatus } from './health.js';
export type { HealthCheckResult } from './health.js';

export {
  PointsClient,
  CollectionsClient,
} from './vector.js';
export type {
  VectorPoint,
  ScoredVectorPoint,
  UpsertOptions,
  GetPointsOptions,
  QueryOptions,
  ScrollOptions,
  ScrollResult,
  CollectionInfo,
  DistanceMetric,
} from './vector.js';
