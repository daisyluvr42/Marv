import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

const repoRoot = path.dirname(fileURLToPath(import.meta.url));
const isCI = process.env.CI === "true" || process.env.GITHUB_ACTIONS === "true";
const cpuCount = os.cpus().length;
const defaultWorkers = isCI
  ? Math.min(4, Math.max(2, Math.floor(cpuCount * 0.5)))
  : Math.min(8, Math.max(4, Math.floor(cpuCount * 0.6)));
const requestedWorkers = Number.parseInt(process.env.MARV_E2E_WORKERS ?? "", 10);
const e2eWorkers =
  Number.isFinite(requestedWorkers) && requestedWorkers > 0
    ? Math.min(16, requestedWorkers)
    : defaultWorkers;
const verboseE2E = process.env.MARV_E2E_VERBOSE === "1";

export default defineConfig({
  resolve: {
    alias: [
      {
        find: "marv/plugin-sdk/account-id",
        replacement: path.join(repoRoot, "src", "plugin-sdk", "account-id.ts"),
      },
      {
        find: "marv/plugin-sdk",
        replacement: path.join(repoRoot, "src", "plugin-sdk", "index.ts"),
      },
    ],
  },
  test: {
    testTimeout: 120_000,
    hookTimeout: process.platform === "win32" ? 180_000 : 120_000,
    unstubEnvs: true,
    unstubGlobals: true,
    pool: "vmForks",
    maxWorkers: e2eWorkers,
    silent: !verboseE2E,
    include: ["test/**/*.e2e.test.ts", "src/**/*.e2e.test.ts"],
    setupFiles: ["test/setup.ts"],
    exclude: ["dist/**", "apps/**", "**/node_modules/**", "**/vendor/**", "dist/Marv.app/**"],
  },
});
