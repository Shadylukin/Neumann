// SPDX-License-Identifier: BSL-1.1
/**
 * Static TypeScript definitions for vector.proto
 *
 * This file provides static type definitions for the Vector services,
 * compatible with both ESM and CommonJS builds.
 */

import type * as grpc from '@grpc/grpc-js';

// === Point Types ===

export interface Point {
  id: string;
  vector: number[];
  payload?: Record<string, Uint8Array> | undefined;
}

export interface ScoredPoint {
  id: string;
  score: number;
  payload?: Record<string, Uint8Array> | undefined;
  vector?: number[] | undefined;
}

// === Points Service Request/Response Types ===

export interface UpsertPointsRequest {
  collection: string;
  points: Point[];
}

export interface UpsertPointsResponse {
  upserted: number;
}

export interface GetPointsRequest {
  collection: string;
  ids: string[];
  withPayload?: boolean | undefined;
  withVector?: boolean | undefined;
}

export interface GetPointsResponse {
  points: Point[];
}

export interface DeletePointsRequest {
  collection: string;
  ids: string[];
}

export interface DeletePointsResponse {
  deleted: number;
}

export interface QueryPointsRequest {
  collection: string;
  vector: number[];
  limit: number;
  offset?: number | undefined;
  scoreThreshold?: number | undefined;
  withPayload?: boolean | undefined;
  withVector?: boolean | undefined;
}

export interface QueryPointsResponse {
  results: ScoredPoint[];
}

export interface ScrollPointsRequest {
  collection: string;
  offsetId?: string | undefined;
  limit: number;
  withPayload?: boolean | undefined;
  withVector?: boolean | undefined;
}

export interface ScrollPointsResponse {
  points: Point[];
  nextOffset?: string | undefined;
}

// === Collections Service Request/Response Types ===

export interface CreateCollectionRequest {
  name: string;
  dimension: number;
  distance: string;
}

export interface CreateCollectionResponse {
  created: boolean;
}

export interface GetCollectionRequest {
  name: string;
}

export interface GetCollectionResponse {
  name: string;
  pointsCount: number;
  dimension: number;
  distance: string;
}

export interface DeleteCollectionRequest {
  name: string;
}

export interface DeleteCollectionResponse {
  deleted: boolean;
}

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface ListCollectionsRequest {}

export interface ListCollectionsResponse {
  collections: string[];
}

// === gRPC Service Interfaces ===

export type GrpcCallback<T> = (err: grpc.ServiceError | null, response: T) => void;

export interface PointsServiceClient extends grpc.Client {
  Upsert(
    request: UpsertPointsRequest,
    metadata: grpc.Metadata,
    callback: GrpcCallback<UpsertPointsResponse>
  ): grpc.ClientUnaryCall;
  Get(
    request: GetPointsRequest,
    metadata: grpc.Metadata,
    callback: GrpcCallback<GetPointsResponse>
  ): grpc.ClientUnaryCall;
  Delete(
    request: DeletePointsRequest,
    metadata: grpc.Metadata,
    callback: GrpcCallback<DeletePointsResponse>
  ): grpc.ClientUnaryCall;
  Query(
    request: QueryPointsRequest,
    metadata: grpc.Metadata,
    callback: GrpcCallback<QueryPointsResponse>
  ): grpc.ClientUnaryCall;
  Scroll(
    request: ScrollPointsRequest,
    metadata: grpc.Metadata,
    callback: GrpcCallback<ScrollPointsResponse>
  ): grpc.ClientUnaryCall;
}

export interface CollectionsServiceClient extends grpc.Client {
  Create(
    request: CreateCollectionRequest,
    metadata: grpc.Metadata,
    callback: GrpcCallback<CreateCollectionResponse>
  ): grpc.ClientUnaryCall;
  Get(
    request: GetCollectionRequest,
    metadata: grpc.Metadata,
    callback: GrpcCallback<GetCollectionResponse>
  ): grpc.ClientUnaryCall;
  Delete(
    request: DeleteCollectionRequest,
    metadata: grpc.Metadata,
    callback: GrpcCallback<DeleteCollectionResponse>
  ): grpc.ClientUnaryCall;
  List(
    request: ListCollectionsRequest,
    metadata: grpc.Metadata,
    callback: GrpcCallback<ListCollectionsResponse>
  ): grpc.ClientUnaryCall;
}
