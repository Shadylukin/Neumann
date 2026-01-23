/**
 * Type declarations for grpc-web module.
 * Used for browser-based gRPC connections.
 */
declare module 'grpc-web' {
  export interface GrpcWebClientBaseOptions {
    format?: 'text' | 'binary';
  }

  export class GrpcWebClientBase {
    constructor(options?: GrpcWebClientBaseOptions);
  }
}
