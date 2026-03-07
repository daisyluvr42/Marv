import { z } from "zod";

const PrivacyCategorySchema = z.union([
  z.literal("api_keys"),
  z.literal("passwords"),
  z.literal("tokens"),
  z.literal("private_keys"),
  z.literal("env_secrets"),
  z.literal("personal_info"),
  z.literal("internal_urls"),
  z.literal("config_secrets"),
]);

const AutonomyPrivacySchema = z
  .object({
    enabled: z.boolean().optional(),
    categories: z.array(PrivacyCategorySchema).optional(),
    outputScan: z.boolean().optional(),
  })
  .strict();

const AutonomyEscalationSchema = z
  .object({
    enabled: z.boolean().optional(),
    taskScoped: z.literal(true).optional(),
    approvalTimeoutSeconds: z.number().int().positive().optional(),
  })
  .strict();

const AutonomyDiscoverySchema = z
  .object({
    enabled: z.boolean().optional(),
    scope: z.union([z.literal("bundled"), z.literal("managed"), z.literal("all")]).optional(),
    installApproval: z.union([z.literal("per-skill"), z.literal("batch")]).optional(),
  })
  .strict();

export const AutonomySchema = z
  .object({
    mode: z.union([z.literal("full"), z.literal("supervised"), z.literal("minimal")]).optional(),
    approvalMode: z.union([z.literal("strict"), z.literal("relaxed")]).optional(),
    skills: z.union([z.literal("all"), z.literal("eligible"), z.literal("manual")]).optional(),
    toolProfile: z
      .union([z.literal("minimal"), z.literal("coding"), z.literal("messaging"), z.literal("full")])
      .optional(),
    autoInstallSkills: z.boolean().optional(),
    escalation: AutonomyEscalationSchema.optional(),
    discovery: AutonomyDiscoverySchema.optional(),
    privacy: AutonomyPrivacySchema.optional(),
  })
  .strict()
  .optional();
