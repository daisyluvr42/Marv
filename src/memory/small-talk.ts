/**
 * Shared small-talk detection regex used by runtime ingest,
 * memory search precheck, and write heuristics.
 */
export const SMALL_TALK_RE =
  /^(hi|hello|hey|yo|thanks|thank you|ok|okay|cool|nice|great|sounds good|good morning|good night|bye|goodbye|你好|嗨|谢谢|好的|再见)[.!? ]*$/i;

export function isSmallTalk(input: string): boolean {
  return SMALL_TALK_RE.test(input.trim());
}
