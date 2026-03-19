import { readFileSync, readdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { defineConfig } from "tsdown";

const env = {
  NODE_ENV: "production",
};

/**
 * Post-build hook: rolldown leaks its `__exportAll` runtime helper into the
 * plugin-sdk entry's export list (as a minified alias like `t`).  When jiti
 * transpiles the ESM chunks to CJS, the cross-chunk import of that alias
 * fails.  Strip the leaked export from the entry and inline the helper into
 * every chunk that references it so the module graph stays self-contained.
 */
function patchPluginSdkExportAll(outDir: string) {
  const indexPath = join(outDir, "index.js");
  let src = readFileSync(indexPath, "utf-8");

  // Find the alias used for __exportAll (e.g. `__exportAll as t`)
  const aliasMatch = src.match(/__exportAll as (\w+)/);
  if (!aliasMatch) {
    return;
  }
  const alias = aliasMatch[1];

  // Extract the rolldown runtime block (includes __defProp + __exportAll)
  const runtimeMatch = src.match(/\/\/#region \\0rolldown\/runtime\.js\n([\s\S]*?)\/\/#endregion/);
  if (!runtimeMatch) {
    return;
  }
  const runtimeBlock = runtimeMatch[1].trim();

  // Remove `__exportAll as <alias>` from the export statement
  src = src.replace(new RegExp(`,?\\s*__exportAll as ${alias}`), "");
  src = src.replace(new RegExp(`__exportAll as ${alias},?\\s*`), "");
  writeFileSync(indexPath, src);

  // Inline the helper into chunks that import it
  for (const file of readdirSync(outDir)) {
    if (file === "index.js" || !file.endsWith(".js")) {
      continue;
    }
    const chunkPath = join(outDir, file);
    let chunk = readFileSync(chunkPath, "utf-8");
    const importPattern = new RegExp(
      `import\\s*\\{\\s*${alias} as __exportAll\\s*\\}\\s*from\\s*["']\\.\\/index\\.js["'];?\\n?`,
    );
    if (!importPattern.test(chunk)) {
      continue;
    }
    // Only inline the helpers needed by chunks (__defProp + __exportAll).
    // Filter out __require/createRequire which are index.js-only concerns.
    const chunkRuntime = runtimeBlock
      .split("\n")
      .filter((l: string) => !l.includes("createRequire") && !l.includes("__require"))
      .join("\n");
    chunk = chunk.replace(importPattern, `${chunkRuntime}\n`);
    writeFileSync(chunkPath, chunk);
  }
}

export default defineConfig([
  {
    entry: "src/index.ts",
    env,
    fixedExtension: false,
    platform: "node",
  },
  {
    entry: "src/entry.ts",
    env,
    fixedExtension: false,
    platform: "node",
  },
  {
    // Ensure this module is bundled as an entry so legacy CLI shims can resolve its exports.
    entry: "src/cli/daemon-cli.ts",
    env,
    fixedExtension: false,
    platform: "node",
  },
  {
    entry: "src/infra/warning-filter.ts",
    env,
    fixedExtension: false,
    platform: "node",
  },
  {
    entry: "src/plugin-sdk/index.ts",
    outDir: "dist/plugin-sdk",
    env,
    fixedExtension: false,
    platform: "node",
    onSuccess() {
      patchPluginSdkExportAll("dist/plugin-sdk");
    },
  },
  {
    entry: "src/plugin-sdk/account-id.ts",
    outDir: "dist/plugin-sdk",
    env,
    fixedExtension: false,
    platform: "node",
  },
  {
    entry: "src/extensionAPI.ts",
    env,
    fixedExtension: false,
    platform: "node",
  },
]);
