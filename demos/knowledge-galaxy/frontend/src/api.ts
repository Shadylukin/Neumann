const GALAXY_API = 'http://localhost:9000';
const SPATIAL_API = 'http://localhost:8080';

interface GalaxyResponse {
  type: string;
  items: Record<string, unknown>[];
  error?: string;
}

interface Spatial3DResult {
  key: string;
  distance: number;
  x: number;
  y: number;
  z: number;
}

async function queryGalaxy(sql: string): Promise<GalaxyResponse> {
  const res = await fetch(`${GALAXY_API}/api/galaxy`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: sql }),
  });
  if (!res.ok) {
    throw new Error(`Galaxy API error: ${res.status} ${res.statusText}`);
  }
  const raw = await res.json();
  // Flatten nested unified response: items[0].items[].data -> flat items
  return flattenResponse(raw);
}

/** Unwrap a PropertyValue enum wrapper like {String: "val"} or {Int: 42}. */
function unwrapPropValue(val: unknown): unknown {
  if (val && typeof val === 'object' && !Array.isArray(val)) {
    const obj = val as Record<string, unknown>;
    return obj['String'] ?? obj['Int'] ?? obj['Float'] ?? obj['Bool'] ?? val;
  }
  return val;
}

/** Flatten graph node properties map into a simple key-value object. */
function flattenNode(node: Record<string, unknown>): Record<string, unknown> {
  const result: Record<string, unknown> = {
    _id: node['id'],
    label: node['label'],
  };
  const props = node['properties'] as Record<string, unknown> | undefined;
  if (props) {
    for (const [key, val] of Object.entries(props)) {
      result[key] = unwrapPropValue(val);
    }
  }
  return result;
}

/** Flatten the nested unified/graph response into a flat items array. */
function flattenResponse(raw: Record<string, unknown>): GalaxyResponse {
  const error = raw['error'] as string | undefined;
  const type_ = String(raw['type'] ?? 'unknown');
  const topItems = raw['items'] as Record<string, unknown>[] | undefined;

  if (!topItems || topItems.length === 0) {
    return { type: type_, items: [], error };
  }

  const flatItems: Record<string, unknown>[] = [];

  if (type_ === 'nodes') {
    // NODE LIST response: items are graph nodes with {id, label, properties}
    for (const node of topItems) {
      flatItems.push(flattenNode(node));
    }
  } else if (type_ === 'unified') {
    // Unified response: each top item has a nested "items" array
    for (const top of topItems) {
      const nested = top['items'] as Record<string, unknown>[] | undefined;
      if (nested) {
        for (const entry of nested) {
          const data = entry['data'] as Record<string, unknown> | undefined;
          if (data) {
            flatItems.push({
              ...data,
              _id: entry['id'],
              _score: entry['score'],
              _source: entry['source'],
            });
          } else {
            flatItems.push(entry);
          }
        }
      } else {
        flatItems.push(top);
      }
    }
  } else {
    // Already flat (rows, etc.)
    for (const item of topItems) {
      flatItems.push(item);
    }
  }

  return { type: type_, items: flatItems, error };
}

async function spatialNearest3D(
  x: number,
  y: number,
  z: number,
  k: number,
): Promise<Spatial3DResult[]> {
  const res = await fetch(
    `${SPATIAL_API}/collections/galaxy/spatial3d/nearest`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ x, y, z, limit: k }),
    },
  );
  if (!res.ok) {
    throw new Error(`Spatial API error: ${res.status} ${res.statusText}`);
  }
  const data = await res.json();
  return data.results;
}

async function spatialRegion3D(
  min: [number, number, number],
  max: [number, number, number],
): Promise<Spatial3DResult[]> {
  const res = await fetch(
    `${SPATIAL_API}/collections/galaxy/spatial3d/region`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ min, max }),
    },
  );
  if (!res.ok) {
    throw new Error(`Spatial API error: ${res.status} ${res.statusText}`);
  }
  const data = await res.json();
  return data.results;
}

export { queryGalaxy, spatialNearest3D, spatialRegion3D, GALAXY_API, SPATIAL_API };
export type { GalaxyResponse, Spatial3DResult };
