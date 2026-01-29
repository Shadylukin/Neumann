// SPDX-License-Identifier: MIT
/**
 * gRPC proto loading module for Node.js environments.
 *
 * This module handles dynamic loading of the Neumann proto definition
 * and provides factory functions for creating gRPC service clients.
 */
import * as protoLoader from '@grpc/proto-loader';
import * as grpc from '@grpc/grpc-js';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PROTO_PATH = path.resolve(__dirname, '../../neumann_server/proto/neumann.proto');

const LOADER_OPTIONS: protoLoader.Options = {
  keepCase: false,
  longs: Number,
  enums: String,
  defaults: true,
  oneofs: true,
};

let _packageDefinition: protoLoader.PackageDefinition | null = null;
let _neumannProto: grpc.GrpcObject | null = null;

/**
 * Load the Neumann proto definition.
 *
 * The proto is cached after the first load.
 */
export async function loadProto(): Promise<grpc.GrpcObject> {
  if (_neumannProto) {
    return _neumannProto;
  }

  _packageDefinition = await protoLoader.load(PROTO_PATH, LOADER_OPTIONS);
  const grpcObj = grpc.loadPackageDefinition(_packageDefinition);
  _neumannProto = (grpcObj.neumann as grpc.GrpcObject).v1 as grpc.GrpcObject;
  return _neumannProto;
}

/**
 * Create a QueryService client.
 *
 * @param proto - The loaded proto object from loadProto().
 * @param address - The server address.
 * @param credentials - The gRPC channel credentials.
 * @returns A QueryService client instance.
 */
export function getQueryServiceClient(
  proto: grpc.GrpcObject,
  address: string,
  credentials: grpc.ChannelCredentials
): grpc.Client {
  const QueryService = proto.QueryService as grpc.ServiceClientConstructor;
  return new QueryService(address, credentials);
}

/**
 * Callback function for gRPC unary calls.
 */
type GrpcCallback = (err: grpc.ServiceError | null, response: unknown) => void;

/**
 * Interface for QueryService client methods.
 */
export interface QueryServiceClient extends grpc.Client {
  Execute: (
    request: unknown,
    metadata: grpc.Metadata,
    callback: GrpcCallback
  ) => grpc.ClientUnaryCall;
  ExecuteStream: (
    request: unknown,
    metadata: grpc.Metadata
  ) => grpc.ClientReadableStream<unknown>;
  ExecuteBatch: (
    request: unknown,
    metadata: grpc.Metadata,
    callback: GrpcCallback
  ) => grpc.ClientUnaryCall;
}
