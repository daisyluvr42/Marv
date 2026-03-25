import type { DatabaseSync } from "node:sqlite";
import { createSubsystemLogger } from "../../logging/subsystem.js";

const log = createSubsystemLogger("memory");

export function deleteStaleIndexedPaths(params: {
  db: DatabaseSync;
  source: string;
  activePaths: Set<string>;
  vectorTable: string;
  ftsTable: string;
  ftsEnabled: boolean;
  ftsAvailable: boolean;
  model: string;
}) {
  const staleRows = params.db
    .prepare(`SELECT path FROM files WHERE source = ?`)
    .all(params.source) as Array<{ path: string }>;

  for (const stale of staleRows) {
    if (params.activePaths.has(stale.path)) {
      continue;
    }
    params.db
      .prepare(`DELETE FROM files WHERE path = ? AND source = ?`)
      .run(stale.path, params.source);
    try {
      params.db
        .prepare(
          `DELETE FROM ${params.vectorTable} WHERE id IN (SELECT id FROM chunks WHERE path = ? AND source = ?)`,
        )
        .run(stale.path, params.source);
    } catch (err) {
      log.warn(
        `DELETE from ${params.vectorTable} failed for stale path ${stale.path}: ${String(err)}`,
      );
    }
    params.db
      .prepare(`DELETE FROM chunks WHERE path = ? AND source = ?`)
      .run(stale.path, params.source);
    if (params.ftsEnabled && params.ftsAvailable) {
      try {
        params.db
          .prepare(`DELETE FROM ${params.ftsTable} WHERE path = ? AND source = ? AND model = ?`)
          .run(stale.path, params.source, params.model);
      } catch (err) {
        log.warn(
          `DELETE from ${params.ftsTable} failed for stale path ${stale.path}: ${String(err)}`,
        );
      }
    }
  }
}
