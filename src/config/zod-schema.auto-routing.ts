import { z } from "zod";

const ThinkLevelSchema = z.union([
  z.literal("off"),
  z.literal("minimal"),
  z.literal("low"),
  z.literal("medium"),
  z.literal("high"),
  z.literal("xhigh"),
]);

const AutoRoutingComplexitySchema = z.union([
  z.literal("simple"),
  z.literal("moderate"),
  z.literal("complex"),
  z.literal("expert"),
]);

export const AutoRoutingSchema = z
  .object({
    enabled: z.boolean().optional(),
    classifier: z.union([z.literal("rules"), z.literal("llm")]).optional(),
    classifierModel: z.string().optional(),
    rules: z
      .array(
        z
          .object({
            complexity: AutoRoutingComplexitySchema,
            model: z.string().min(1),
            thinking: ThinkLevelSchema.optional(),
          })
          .strict(),
      )
      .optional(),
    thresholds: z
      .object({
        simpleMaxChars: z.number().int().positive().optional(),
        moderateMaxChars: z.number().int().positive().optional(),
        complexMaxChars: z.number().int().positive().optional(),
        complexPatterns: z.array(z.string()).optional(),
      })
      .strict()
      .optional(),
  })
  .strict()
  .optional();
