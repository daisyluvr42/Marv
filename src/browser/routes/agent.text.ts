import type { BrowserRouteContext } from "../server-context.js";
import { resolveTargetIdFromQuery, withPlaywrightRouteContext } from "./agent.shared.js";
import type { BrowserRouteRegistrar } from "./types.js";
import { toNumber, toStringOrEmpty } from "./utils.js";

export function registerBrowserAgentTextRoutes(
  app: BrowserRouteRegistrar,
  ctx: BrowserRouteContext,
) {
  app.get("/text", async (req, res) => {
    const targetId = resolveTargetIdFromQuery(req.query);
    const ref = toStringOrEmpty(req.query.ref) || undefined;
    const maxChars = toNumber(req.query.maxChars) ?? undefined;
    const timeoutMs = toNumber(req.query.timeoutMs) ?? undefined;

    await withPlaywrightRouteContext({
      req,
      res,
      ctx,
      targetId,
      feature: "text extraction",
      run: async ({ cdpUrl, tab, pw }) => {
        const result = await pw.extractTextViaPlaywright({
          cdpUrl,
          targetId: tab.targetId,
          ref,
          maxChars,
          timeoutMs,
        });
        res.json({
          ok: true,
          targetId: tab.targetId,
          url: result.url,
          title: result.title,
          text: result.text,
          truncated: result.truncated,
        });
      },
    });
  });
}
