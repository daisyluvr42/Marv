import { type CommandOptions, runCommandWithTimeout } from "../../process/exec.js";
import { VERSION } from "../version.js";

export type VersionDriftResult = {
  cliVersion: string;
  appVersion: string | null;
  drifted: boolean;
  message: string | null;
};

const MACOS_APP_PATHS = [
  "/Applications/Marv.app/Contents/Info.plist",
  `${process.env.HOME}/Applications/Marv.app/Contents/Info.plist`,
];

/**
 * Read the macOS app version from its Info.plist using `defaults read`.
 * Returns null if the app is not installed or on non-macOS platforms.
 */
async function readMacAppVersion(
  runCommand?: (
    argv: string[],
    opts: CommandOptions,
  ) => Promise<{ stdout: string; code: number | null }>,
): Promise<string | null> {
  if (process.platform !== "darwin") {
    return null;
  }

  const run =
    runCommand ??
    (async (argv: string[], opts: CommandOptions) => {
      const res = await runCommandWithTimeout(argv, opts);
      return { stdout: res.stdout, code: res.code };
    });

  for (const plistPath of MACOS_APP_PATHS) {
    const appPath = plistPath.replace("/Contents/Info.plist", "");
    const res = await run(
      ["defaults", "read", `${appPath}/Contents/Info`, "CFBundleShortVersionString"],
      { timeoutMs: 3000 },
    ).catch(() => null);
    if (res && res.code === 0) {
      const version = res.stdout.trim();
      if (version) {
        return version;
      }
    }
  }
  return null;
}

/**
 * Check if the CLI version and macOS app version are in sync.
 */
export async function checkVersionDrift(params?: {
  runCommand?: (
    argv: string[],
    opts: CommandOptions,
  ) => Promise<{ stdout: string; code: number | null }>;
}): Promise<VersionDriftResult> {
  const cliVersion = VERSION;
  const appVersion = await readMacAppVersion(params?.runCommand);

  if (!appVersion) {
    return { cliVersion, appVersion: null, drifted: false, message: null };
  }

  const drifted = cliVersion !== appVersion;
  const message = drifted
    ? `Version drift: CLI is ${cliVersion}, macOS app is ${appVersion}. Consider updating the ${cliVersion > appVersion ? "macOS app" : "CLI"}.`
    : null;

  return { cliVersion, appVersion, drifted, message };
}
