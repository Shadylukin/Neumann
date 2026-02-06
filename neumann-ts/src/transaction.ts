// SPDX-License-Identifier: BSL-1.1
/**
 * Transaction support for Neumann database.
 *
 * Provides automatic transaction management with commit/rollback semantics.
 *
 * @example
 * ```typescript
 * // Using the transaction helper method
 * await client.withTransaction(async (tx) => {
 *   await tx.execute("INSERT users name='Alice'");
 *   await tx.execute("INSERT users name='Bob'");
 *   // Auto-commits on success, auto-rollbacks on error
 * });
 *
 * // Manual transaction management
 * const tx = client.beginTransaction();
 * try {
 *   await tx.begin();
 *   await tx.execute("INSERT users name='Alice'");
 *   await tx.commit();
 * } catch (e) {
 *   await tx.rollback();
 *   throw e;
 * }
 * ```
 */

import type { NeumannClient } from './client.js';
import type { QueryResult } from './types/query-result.js';
import { isChainQueryResult } from './types/query-result.js';
import { NeumannError, InvalidArgumentError } from './types/errors.js';

/**
 * Options for creating a transaction.
 */
export interface TransactionOptions {
  /** Identity for vault access. */
  identity?: string;
  /** Whether to auto-commit on successful completion (default: true). */
  autoCommit?: boolean;
}

/**
 * Result of a successful commit operation.
 */
export interface CommitResult {
  /** The block hash of the committed transaction. */
  blockHash: string;
  /** The blockchain height after commit. */
  height: number;
}

/**
 * Result of a rollback operation.
 */
export interface RollbackResult {
  /** The blockchain height after rollback. */
  toHeight: number;
}

/**
 * A database transaction with automatic commit/rollback support.
 *
 * Transactions provide ACID guarantees for a sequence of operations.
 * Use the helper methods on NeumannClient for easier transaction management.
 */
export class Transaction {
  private readonly client: NeumannClient;
  private readonly identity?: string;
  private readonly _autoCommit: boolean;
  private _txId: string | null = null;
  private _active = false;
  private _committed = false;
  private _rolledBack = false;

  /**
   * Create a new transaction.
   *
   * @param client - The NeumannClient to use for the transaction.
   * @param options - Transaction options.
   */
  constructor(client: NeumannClient, options: TransactionOptions = {}) {
    this.client = client;
    if (options.identity !== undefined) {
      this.identity = options.identity;
    }
    this._autoCommit = options.autoCommit ?? true;
  }

  /**
   * Get the transaction ID (available after begin()).
   */
  get txId(): string | null {
    return this._txId;
  }

  /**
   * Check if the transaction is currently active.
   */
  get isActive(): boolean {
    return this._active;
  }

  /**
   * Check if the transaction was committed.
   */
  get isCommitted(): boolean {
    return this._committed;
  }

  /**
   * Check if the transaction was rolled back.
   */
  get isRolledBack(): boolean {
    return this._rolledBack;
  }

  /**
   * Whether auto-commit is enabled for this transaction.
   */
  get autoCommit(): boolean {
    return this._autoCommit;
  }

  /**
   * Begin the transaction.
   *
   * @returns The transaction ID.
   * @throws {NeumannError} If the transaction is already active or begin fails.
   */
  async begin(): Promise<string> {
    if (this._active) {
      throw new InvalidArgumentError('Transaction already active');
    }

    const result = await this.client.execute('CHAIN BEGIN', { ...(this.identity !== undefined && { identity: this.identity }) });

    if (isChainQueryResult(result)) {
      const chainResult = result.result;
      if (chainResult.type === 'transactionBegun') {
        this._txId = chainResult.value.txId;
        this._active = true;
        return this._txId;
      }
    }

    throw new NeumannError('Failed to begin transaction: unexpected result type');
  }

  /**
   * Execute a query within this transaction.
   *
   * @param query - The query to execute.
   * @returns The query result.
   * @throws {NeumannError} If the transaction is not active.
   */
  async execute(query: string): Promise<QueryResult> {
    if (!this._active) {
      throw new InvalidArgumentError('Transaction is not active');
    }

    return this.client.execute(query, { ...(this.identity !== undefined && { identity: this.identity }) });
  }

  /**
   * Commit the transaction.
   *
   * @returns The commit result with block hash and height.
   * @throws {NeumannError} If the transaction is not active or commit fails.
   */
  async commit(): Promise<CommitResult> {
    if (!this._active) {
      throw new InvalidArgumentError('Transaction is not active');
    }

    const result = await this.client.execute('CHAIN COMMIT', { ...(this.identity !== undefined && { identity: this.identity }) });

    if (isChainQueryResult(result)) {
      const chainResult = result.result;
      if (chainResult.type === 'committed') {
        this._active = false;
        this._committed = true;
        return {
          blockHash: chainResult.value.blockHash,
          height: chainResult.value.height,
        };
      }
    }

    throw new NeumannError('Failed to commit transaction: unexpected result type');
  }

  /**
   * Rollback the transaction.
   *
   * @returns The rollback result with the height rolled back to.
   * @throws {NeumannError} If the transaction is not active or rollback fails.
   */
  async rollback(): Promise<RollbackResult> {
    if (!this._active) {
      throw new InvalidArgumentError('Transaction is not active');
    }

    const result = await this.client.execute('CHAIN ROLLBACK', { ...(this.identity !== undefined && { identity: this.identity }) });

    if (isChainQueryResult(result)) {
      const chainResult = result.result;
      if (chainResult.type === 'rolledBack') {
        this._active = false;
        this._rolledBack = true;
        return {
          toHeight: chainResult.value.toHeight,
        };
      }
    }

    throw new NeumannError('Failed to rollback transaction: unexpected result type');
  }

  /**
   * Run a function within this transaction with automatic commit/rollback.
   *
   * If the function completes successfully, the transaction is committed.
   * If the function throws an error, the transaction is rolled back.
   *
   * @param fn - The function to execute within the transaction.
   * @returns The result of the function.
   * @throws The error thrown by the function (after rollback).
   */
  async run<T>(fn: (tx: Transaction) => Promise<T>): Promise<T> {
    await this.begin();

    try {
      const result = await fn(this);

      if (this._autoCommit && this._active) {
        await this.commit();
      }

      return result;
    } catch (error) {
      // Try to rollback, but don't mask the original error
      if (this._active) {
        try {
          await this.rollback();
        } catch {
          // Ignore rollback errors
        }
      }
      throw error;
    }
  }
}

/**
 * Builder for creating transactions with custom options.
 *
 * @example
 * ```typescript
 * const tx = new TransactionBuilder(client)
 *   .withIdentity('user:alice')
 *   .withAutoCommit(false)
 *   .build();
 *
 * await tx.begin();
 * await tx.execute("INSERT users name='Alice'");
 * await tx.commit(); // Manual commit required
 * ```
 */
export class TransactionBuilder {
  private readonly client: NeumannClient;
  private identity?: string;
  private autoCommit = true;

  /**
   * Create a new transaction builder.
   *
   * @param client - The NeumannClient to use for the transaction.
   */
  constructor(client: NeumannClient) {
    this.client = client;
  }

  /**
   * Set the identity for vault access.
   *
   * @param identity - The identity to use.
   * @returns This builder for chaining.
   */
  withIdentity(identity: string): TransactionBuilder {
    this.identity = identity;
    return this;
  }

  /**
   * Set whether to auto-commit on successful completion.
   *
   * @param autoCommit - Whether to auto-commit.
   * @returns This builder for chaining.
   */
  withAutoCommit(autoCommit: boolean): TransactionBuilder {
    this.autoCommit = autoCommit;
    return this;
  }

  /**
   * Build the transaction.
   *
   * @returns The configured Transaction.
   */
  build(): Transaction {
    return new Transaction(this.client, {
      ...(this.identity !== undefined && { identity: this.identity }),
      ...(this.autoCommit !== undefined && { autoCommit: this.autoCommit }),
    });
  }
}
