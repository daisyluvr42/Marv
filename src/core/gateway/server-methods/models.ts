import { resolveRuntimeModelPlan } from "../../../agents/model/model-pool.js";
import { loadConfig } from "../../config/config.js";
import {
  ErrorCodes,
  errorShape,
  formatValidationErrors,
  validateModelsListParams,
} from "../protocol/index.js";
import type { GatewayRequestHandlers } from "./types.js";

export const modelsHandlers: GatewayRequestHandlers = {
  "models.list": async ({ params, respond, context }) => {
    if (!validateModelsListParams(params)) {
      respond(
        false,
        undefined,
        errorShape(
          ErrorCodes.INVALID_REQUEST,
          `invalid models.list params: ${formatValidationErrors(validateModelsListParams.errors)}`,
        ),
      );
      return;
    }
    try {
      const catalog = await context.loadGatewayModelCatalog();
      const cfg = loadConfig();
      const plan = resolveRuntimeModelPlan({ cfg });
      // Only return models that are in the configured pool and available.
      const poolRefs = new Set(plan.candidates.map((c) => c.ref));
      const filtered = catalog.filter((entry) => poolRefs.has(`${entry.provider}/${entry.id}`));
      respond(true, { models: filtered.length > 0 ? filtered : catalog }, undefined);
    } catch (err) {
      respond(false, undefined, errorShape(ErrorCodes.UNAVAILABLE, String(err)));
    }
  },
};
