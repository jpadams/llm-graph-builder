import { OptionType, NodeSpec, RelSpec, PatternSpec, PropertySpec, SchemaSpec } from '../types';

/**
 * Build a typed SchemaSpec from the current FileContext state. The property
 * type info is preserved end-to-end (TTL xsd:integer / xsd:dateTime / etc. and
 * Data Importer typed property entries flow through ``dbNodeProperties`` /
 * ``dbRelProperties`` as ``PropertySpec[]``).
 */
export const buildSchemaSpec = (
  selectedNodes: readonly OptionType[],
  selectedRels: readonly OptionType[],
  combinedPatterns: readonly string[],
  dbNodeProperties: Record<string, PropertySpec[]>,
  dbRelProperties: Record<string, PropertySpec[]>
): SchemaSpec | null => {
  if (!selectedNodes.length && !selectedRels.length && !combinedPatterns.length) {
    return null;
  }

  const nodeLabels = new Set<string>();
  selectedNodes.forEach((n) => n.value && nodeLabels.add(n.value));

  const relLabels = new Set<string>();
  selectedRels.forEach((r) => {
    const parts = r.value.split(',');
    if (parts.length >= 2 && parts[1]) {
      relLabels.add(parts[1]);
    }
  });

  const patterns: PatternSpec[] = [];
  for (const p of combinedPatterns) {
    const m = p.match(/^(.+?) -\[:(.+?)\]-> (.+)$/);
    if (!m) continue;
    const [, sourceLabel, relLabel, targetLabel] = m;
    nodeLabels.add(sourceLabel);
    nodeLabels.add(targetLabel);
    relLabels.add(relLabel);
    patterns.push({ sourceLabel, relLabel, targetLabel });
  }

  const nodes: NodeSpec[] = Array.from(nodeLabels)
    .sort()
    .map((label) => ({
      label,
      properties: dbNodeProperties[label] ?? [],
    }));

  const relationships: RelSpec[] = Array.from(relLabels)
    .sort()
    .map((label) => ({
      label,
      properties: dbRelProperties[label] ?? [],
    }));

  return {
    source: 'db',
    nodes,
    relationships,
    patterns,
  };
};
