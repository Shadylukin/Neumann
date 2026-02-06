// SPDX-License-Identifier: BSL-1.1
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { Transaction, TransactionBuilder } from './transaction.js';
import type { NeumannClient } from './client.js';
import type { QueryResult, ChainQueryResult } from './types/query-result.js';

// Mock NeumannClient
function createMockClient(): NeumannClient {
  return {
    execute: vi.fn(),
  } as unknown as NeumannClient;
}

describe('Transaction', () => {
  let mockClient: NeumannClient;

  beforeEach(() => {
    mockClient = createMockClient();
  });

  describe('constructor', () => {
    it('should create transaction with default options', () => {
      const tx = new Transaction(mockClient);
      expect(tx.txId).toBeNull();
      expect(tx.isActive).toBe(false);
      expect(tx.isCommitted).toBe(false);
      expect(tx.isRolledBack).toBe(false);
      expect(tx.autoCommit).toBe(true);
    });

    it('should create transaction with custom options', () => {
      const tx = new Transaction(mockClient, {
        identity: 'user:alice',
        autoCommit: false,
      });
      expect(tx.autoCommit).toBe(false);
    });
  });

  describe('begin', () => {
    it('should begin transaction successfully', async () => {
      const chainResult: ChainQueryResult = {
        type: 'chain',
        result: { type: 'transactionBegun', value: { txId: 'tx-123' } },
      };
      vi.mocked(mockClient.execute).mockResolvedValue(chainResult);

      const tx = new Transaction(mockClient);
      const txId = await tx.begin();

      expect(txId).toBe('tx-123');
      expect(tx.txId).toBe('tx-123');
      expect(tx.isActive).toBe(true);
      expect(mockClient.execute).toHaveBeenCalledWith('CHAIN BEGIN', {});
    });

    it('should begin transaction with identity', async () => {
      const chainResult: ChainQueryResult = {
        type: 'chain',
        result: { type: 'transactionBegun', value: { txId: 'tx-456' } },
      };
      vi.mocked(mockClient.execute).mockResolvedValue(chainResult);

      const tx = new Transaction(mockClient, { identity: 'user:alice' });
      await tx.begin();

      expect(mockClient.execute).toHaveBeenCalledWith('CHAIN BEGIN', { identity: 'user:alice' });
    });

    it('should throw if transaction already active', async () => {
      const chainResult: ChainQueryResult = {
        type: 'chain',
        result: { type: 'transactionBegun', value: { txId: 'tx-123' } },
      };
      vi.mocked(mockClient.execute).mockResolvedValue(chainResult);

      const tx = new Transaction(mockClient);
      await tx.begin();

      await expect(tx.begin()).rejects.toThrow('Transaction already active');
    });

    it('should throw on unexpected result type', async () => {
      const emptyResult: QueryResult = { type: 'empty' };
      vi.mocked(mockClient.execute).mockResolvedValue(emptyResult);

      const tx = new Transaction(mockClient);
      await expect(tx.begin()).rejects.toThrow('Failed to begin transaction');
    });
  });

  describe('execute', () => {
    it('should execute query within transaction', async () => {
      const beginResult: ChainQueryResult = {
        type: 'chain',
        result: { type: 'transactionBegun', value: { txId: 'tx-123' } },
      };
      const queryResult: QueryResult = { type: 'count', count: 1 };
      vi.mocked(mockClient.execute)
        .mockResolvedValueOnce(beginResult)
        .mockResolvedValueOnce(queryResult);

      const tx = new Transaction(mockClient);
      await tx.begin();
      const result = await tx.execute("INSERT users name='Alice'");

      expect(result).toEqual(queryResult);
    });

    it('should throw if transaction not active', async () => {
      const tx = new Transaction(mockClient);
      await expect(tx.execute('SELECT 1')).rejects.toThrow('Transaction is not active');
    });
  });

  describe('commit', () => {
    it('should commit transaction successfully', async () => {
      const beginResult: ChainQueryResult = {
        type: 'chain',
        result: { type: 'transactionBegun', value: { txId: 'tx-123' } },
      };
      const commitResult: ChainQueryResult = {
        type: 'chain',
        result: { type: 'committed', value: { blockHash: 'hash-abc', height: 42 } },
      };
      vi.mocked(mockClient.execute)
        .mockResolvedValueOnce(beginResult)
        .mockResolvedValueOnce(commitResult);

      const tx = new Transaction(mockClient);
      await tx.begin();
      const result = await tx.commit();

      expect(result.blockHash).toBe('hash-abc');
      expect(result.height).toBe(42);
      expect(tx.isActive).toBe(false);
      expect(tx.isCommitted).toBe(true);
    });

    it('should throw if transaction not active', async () => {
      const tx = new Transaction(mockClient);
      await expect(tx.commit()).rejects.toThrow('Transaction is not active');
    });

    it('should throw on unexpected result type', async () => {
      const beginResult: ChainQueryResult = {
        type: 'chain',
        result: { type: 'transactionBegun', value: { txId: 'tx-123' } },
      };
      const emptyResult: QueryResult = { type: 'empty' };
      vi.mocked(mockClient.execute)
        .mockResolvedValueOnce(beginResult)
        .mockResolvedValueOnce(emptyResult);

      const tx = new Transaction(mockClient);
      await tx.begin();
      await expect(tx.commit()).rejects.toThrow('Failed to commit transaction');
    });
  });

  describe('rollback', () => {
    it('should rollback transaction successfully', async () => {
      const beginResult: ChainQueryResult = {
        type: 'chain',
        result: { type: 'transactionBegun', value: { txId: 'tx-123' } },
      };
      const rollbackResult: ChainQueryResult = {
        type: 'chain',
        result: { type: 'rolledBack', value: { toHeight: 41 } },
      };
      vi.mocked(mockClient.execute)
        .mockResolvedValueOnce(beginResult)
        .mockResolvedValueOnce(rollbackResult);

      const tx = new Transaction(mockClient);
      await tx.begin();
      const result = await tx.rollback();

      expect(result.toHeight).toBe(41);
      expect(tx.isActive).toBe(false);
      expect(tx.isRolledBack).toBe(true);
    });

    it('should throw if transaction not active', async () => {
      const tx = new Transaction(mockClient);
      await expect(tx.rollback()).rejects.toThrow('Transaction is not active');
    });

    it('should throw on unexpected result type', async () => {
      const beginResult: ChainQueryResult = {
        type: 'chain',
        result: { type: 'transactionBegun', value: { txId: 'tx-123' } },
      };
      const emptyResult: QueryResult = { type: 'empty' };
      vi.mocked(mockClient.execute)
        .mockResolvedValueOnce(beginResult)
        .mockResolvedValueOnce(emptyResult);

      const tx = new Transaction(mockClient);
      await tx.begin();
      await expect(tx.rollback()).rejects.toThrow('Failed to rollback transaction');
    });
  });

  describe('run', () => {
    it('should auto-commit on success', async () => {
      const beginResult: ChainQueryResult = {
        type: 'chain',
        result: { type: 'transactionBegun', value: { txId: 'tx-123' } },
      };
      const queryResult: QueryResult = { type: 'count', count: 1 };
      const commitResult: ChainQueryResult = {
        type: 'chain',
        result: { type: 'committed', value: { blockHash: 'hash-abc', height: 42 } },
      };
      vi.mocked(mockClient.execute)
        .mockResolvedValueOnce(beginResult)
        .mockResolvedValueOnce(queryResult)
        .mockResolvedValueOnce(commitResult);

      const tx = new Transaction(mockClient);
      const result = await tx.run(async (t) => {
        await t.execute("INSERT users name='Alice'");
        return 'success';
      });

      expect(result).toBe('success');
      expect(tx.isCommitted).toBe(true);
    });

    it('should not auto-commit when disabled', async () => {
      const beginResult: ChainQueryResult = {
        type: 'chain',
        result: { type: 'transactionBegun', value: { txId: 'tx-123' } },
      };
      vi.mocked(mockClient.execute).mockResolvedValueOnce(beginResult);

      const tx = new Transaction(mockClient, { autoCommit: false });
      await tx.run(async () => {
        return 'success';
      });

      expect(tx.isActive).toBe(true);
      expect(tx.isCommitted).toBe(false);
    });

    it('should rollback on error', async () => {
      const beginResult: ChainQueryResult = {
        type: 'chain',
        result: { type: 'transactionBegun', value: { txId: 'tx-123' } },
      };
      const rollbackResult: ChainQueryResult = {
        type: 'chain',
        result: { type: 'rolledBack', value: { toHeight: 41 } },
      };
      vi.mocked(mockClient.execute)
        .mockResolvedValueOnce(beginResult)
        .mockResolvedValueOnce(rollbackResult);

      const tx = new Transaction(mockClient);

      await expect(
        tx.run(async () => {
          throw new Error('Test error');
        })
      ).rejects.toThrow('Test error');

      expect(tx.isRolledBack).toBe(true);
    });

    it('should suppress rollback errors', async () => {
      const beginResult: ChainQueryResult = {
        type: 'chain',
        result: { type: 'transactionBegun', value: { txId: 'tx-123' } },
      };
      vi.mocked(mockClient.execute)
        .mockResolvedValueOnce(beginResult)
        .mockRejectedValueOnce(new Error('Rollback failed'));

      const tx = new Transaction(mockClient);

      await expect(
        tx.run(async () => {
          throw new Error('Original error');
        })
      ).rejects.toThrow('Original error');
    });
  });
});

describe('TransactionBuilder', () => {
  let mockClient: NeumannClient;

  beforeEach(() => {
    mockClient = createMockClient();
  });

  it('should build transaction with defaults', () => {
    const tx = new TransactionBuilder(mockClient).build();
    expect(tx.autoCommit).toBe(true);
  });

  it('should build transaction with identity', async () => {
    const beginResult: ChainQueryResult = {
      type: 'chain',
      result: { type: 'transactionBegun', value: { txId: 'tx-123' } },
    };
    vi.mocked(mockClient.execute).mockResolvedValue(beginResult);

    const tx = new TransactionBuilder(mockClient).withIdentity('user:alice').build();

    // Trigger begin to verify identity is passed
    await tx.begin();
    expect(mockClient.execute).toHaveBeenCalledWith('CHAIN BEGIN', { identity: 'user:alice' });
  });

  it('should build transaction with auto-commit disabled', () => {
    const tx = new TransactionBuilder(mockClient).withAutoCommit(false).build();
    expect(tx.autoCommit).toBe(false);
  });

  it('should support method chaining', () => {
    const tx = new TransactionBuilder(mockClient)
      .withIdentity('user:bob')
      .withAutoCommit(false)
      .build();

    expect(tx.autoCommit).toBe(false);
  });

  it('should build transaction without identity (omits identity from options)', async () => {
    const beginResult: ChainQueryResult = {
      type: 'chain',
      result: { type: 'transactionBegun', value: { txId: 'tx-789' } },
    };
    vi.mocked(mockClient.execute).mockResolvedValue(beginResult);

    const tx = new TransactionBuilder(mockClient).withAutoCommit(false).build();
    await tx.begin();
    expect(mockClient.execute).toHaveBeenCalledWith('CHAIN BEGIN', {});
  });
});

describe('Transaction identity propagation', () => {
  let mockClient: NeumannClient;

  beforeEach(() => {
    mockClient = createMockClient();
  });

  it('should pass identity to execute when set', async () => {
    const beginResult: ChainQueryResult = {
      type: 'chain',
      result: { type: 'transactionBegun', value: { txId: 'tx-100' } },
    };
    const queryResult: QueryResult = { type: 'count', count: 1 };
    vi.mocked(mockClient.execute)
      .mockResolvedValueOnce(beginResult)
      .mockResolvedValueOnce(queryResult);

    const tx = new Transaction(mockClient, { identity: 'user:alice' });
    await tx.begin();
    await tx.execute('SELECT 1');

    expect(mockClient.execute).toHaveBeenCalledWith('SELECT 1', { identity: 'user:alice' });
  });

  it('should omit identity from execute when not set', async () => {
    const beginResult: ChainQueryResult = {
      type: 'chain',
      result: { type: 'transactionBegun', value: { txId: 'tx-101' } },
    };
    const queryResult: QueryResult = { type: 'count', count: 1 };
    vi.mocked(mockClient.execute)
      .mockResolvedValueOnce(beginResult)
      .mockResolvedValueOnce(queryResult);

    const tx = new Transaction(mockClient);
    await tx.begin();
    await tx.execute('SELECT 1');

    expect(mockClient.execute).toHaveBeenCalledWith('SELECT 1', {});
  });

  it('should pass identity to commit when set', async () => {
    const beginResult: ChainQueryResult = {
      type: 'chain',
      result: { type: 'transactionBegun', value: { txId: 'tx-102' } },
    };
    const commitResult: ChainQueryResult = {
      type: 'chain',
      result: { type: 'committed', value: { blockHash: 'hash', height: 1 } },
    };
    vi.mocked(mockClient.execute)
      .mockResolvedValueOnce(beginResult)
      .mockResolvedValueOnce(commitResult);

    const tx = new Transaction(mockClient, { identity: 'user:bob' });
    await tx.begin();
    await tx.commit();

    expect(mockClient.execute).toHaveBeenCalledWith('CHAIN COMMIT', { identity: 'user:bob' });
  });

  it('should pass identity to rollback when set', async () => {
    const beginResult: ChainQueryResult = {
      type: 'chain',
      result: { type: 'transactionBegun', value: { txId: 'tx-103' } },
    };
    const rollbackResult: ChainQueryResult = {
      type: 'chain',
      result: { type: 'rolledBack', value: { toHeight: 0 } },
    };
    vi.mocked(mockClient.execute)
      .mockResolvedValueOnce(beginResult)
      .mockResolvedValueOnce(rollbackResult);

    const tx = new Transaction(mockClient, { identity: 'user:carol' });
    await tx.begin();
    await tx.rollback();

    expect(mockClient.execute).toHaveBeenCalledWith('CHAIN ROLLBACK', { identity: 'user:carol' });
  });

  it('should handle begin with non-transactionBegun chain result', async () => {
    const chainResult: ChainQueryResult = {
      type: 'chain',
      result: { type: 'committed', value: { blockHash: 'hash', height: 1 } },
    };
    vi.mocked(mockClient.execute).mockResolvedValue(chainResult);

    const tx = new Transaction(mockClient);
    await expect(tx.begin()).rejects.toThrow('Failed to begin transaction');
  });

  it('should handle commit with non-committed chain result', async () => {
    const beginResult: ChainQueryResult = {
      type: 'chain',
      result: { type: 'transactionBegun', value: { txId: 'tx-104' } },
    };
    const wrongResult: ChainQueryResult = {
      type: 'chain',
      result: { type: 'transactionBegun', value: { txId: 'tx-wrong' } },
    };
    vi.mocked(mockClient.execute)
      .mockResolvedValueOnce(beginResult)
      .mockResolvedValueOnce(wrongResult);

    const tx = new Transaction(mockClient);
    await tx.begin();
    await expect(tx.commit()).rejects.toThrow('Failed to commit transaction');
  });

  it('should handle rollback with non-rolledBack chain result', async () => {
    const beginResult: ChainQueryResult = {
      type: 'chain',
      result: { type: 'transactionBegun', value: { txId: 'tx-105' } },
    };
    const wrongResult: ChainQueryResult = {
      type: 'chain',
      result: { type: 'transactionBegun', value: { txId: 'tx-wrong' } },
    };
    vi.mocked(mockClient.execute)
      .mockResolvedValueOnce(beginResult)
      .mockResolvedValueOnce(wrongResult);

    const tx = new Transaction(mockClient);
    await tx.begin();
    await expect(tx.rollback()).rejects.toThrow('Failed to rollback transaction');
  });
});
