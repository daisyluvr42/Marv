import { z } from "zod";

const WeChatAccountConfigSchema = z.object({
  enabled: z.boolean().optional(),
  name: z.string().optional(),
  puppet: z.string().optional(),
  puppetToken: z.string().optional(),
  dmPolicy: z.enum(["pairing", "open", "allowlist", "disabled"]).optional(),
  allowFrom: z.array(z.union([z.string(), z.number()])).optional(),
  groupPolicy: z.enum(["open", "allowlist", "disabled"]).optional(),
  groupAllowFrom: z.array(z.union([z.string(), z.number()])).optional(),
  groups: z
    .record(
      z.string(),
      z.object({
        requireMention: z.boolean().optional(),
      }),
    )
    .optional(),
});

export const WeChatConfigSchema = WeChatAccountConfigSchema.extend({
  accounts: z.record(z.string(), WeChatAccountConfigSchema).optional(),
});
