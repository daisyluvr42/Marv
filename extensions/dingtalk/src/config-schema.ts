import { z } from "zod";

const DingTalkAccountConfigSchema = z.object({
  enabled: z.boolean().optional(),
  name: z.string().optional(),
  clientId: z.string().optional(),
  clientSecret: z.string().optional(),
  robotCode: z.string().optional(),
  dmPolicy: z.enum(["open", "allowlist", "disabled"]).optional(),
  allowFrom: z.array(z.union([z.string(), z.number()])).optional(),
  groupPolicy: z.enum(["open", "allowlist", "disabled"]).optional(),
  groupAllowFrom: z.array(z.union([z.string(), z.number()])).optional(),
});

export const DingTalkConfigSchema = DingTalkAccountConfigSchema.extend({
  accounts: z.record(z.string(), DingTalkAccountConfigSchema).optional(),
});
