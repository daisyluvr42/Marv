type MinimalTheme = {
  dim: (s: string) => string;
  bold: (s: string) => string;
  accentSoft: (s: string) => string;
};

export const defaultWaitingPhrases = [
  // original marv vibes
  "flibbertigibbeting",
  "kerfuffling",
  "dillydallying",
  "twiddling thumbs",
  "noodling",
  "bamboozling",
  "moseying",
  "hobnobbing",
  "pondering",
  "conjuring",

  // thinking & reasoning
  "thinking",
  "cogitating",
  "contemplating",
  "deliberating",
  "ruminating",
  "cerebrating",
  "mulling",
  "considering",
  "philosophising",
  "pontificating",
  "puzzling",
  "deciphering",
  "elucidating",
  "perusing",
  "inferring",
  "ideating",

  // creative & craft
  "crafting",
  "composing",
  "sketching",
  "doodling",
  "improvising",
  "tinkering",
  "inventing",
  "concocting",
  "embellishing",
  "architecting",
  "orchestrating",
  "choreographing",
  "harmonizing",
  "synthesizing",
  "imagining",
  "envisioning",
  "dreaming",

  // cooking & chemistry
  "brewing",
  "simmering",
  "marinating",
  "fermenting",
  "percolating",
  "stewing",
  "seasoning",
  "garnishing",
  "caramelizing",
  "julienning",
  "kneading",
  "leavening",
  "blanching",
  "frosting",
  "zesting",
  "drizzling",

  // science & magic
  "photosynthesizing",
  "sublimating",
  "crystallizing",
  "transmuting",
  "transfiguring",
  "metamorphosing",
  "ionizing",
  "nebulizing",
  "osmosing",
  "nucleating",
  "quantumizing",
  "reticulating",
  "recombobulating",
  "discombobulating",
  "combobulating",
  "prestidigitating",
  "enchanting",
  "manifesting",

  // movement & adventure
  "moonwalking",
  "gallivanting",
  "scampering",
  "skedaddling",
  "meandering",
  "perambulating",
  "spelunking",
  "frolicking",
  "waddling",
  "slithering",
  "swooping",
  "levitating",
  "orbiting",
  "catapulting",
  "zigzagging",
  "schlepping",
  "scurrying",
  "puttering",
  "roaming",
  "wandering",

  // nature & growth
  "sprouting",
  "germinating",
  "cultivating",
  "pollinating",
  "hatching",
  "roosting",
  "burrowing",
  "unfurling",
  "blossoming",

  // fun & whimsy
  "shenaniganing",
  "tomfoolering",
  "lollygagging",
  "boondoggling",
  "razzmatazzing",
  "hullaballooing",
  "whatchamacalliting",
  "flummoxing",
  "befuddling",
  "finagling",
  "wrangling",
  "bloviating",
  "smooshing",
  "wibbling",

  // rhythm & motion
  "shimmying",
  "jitterbugging",
  "boogieing",
  "grooving",
  "swirling",
  "whirring",
  "whirlpooling",
  "cascading",
  "undulating",
  "fluttering",

  // forge & transform
  "forging",
  "tempering",
  "churning",
  "crunching",
  "extracting",
  "distilling",
  "infusing",
  "unravelling",

  // sounds & vibes
  "honking",
  "thundering",
  "whisking",
  "gusting",
  "pouncing",
  "galloping",

  // meta & self-aware
  "clauding",
  "gitifying",
  "hyperspacing",
  "symbioting",
  "newspapering",
];

export function pickWaitingPhrase(tick: number, phrases = defaultWaitingPhrases) {
  const idx = Math.floor(tick / 10) % phrases.length;
  return phrases[idx] ?? phrases[0] ?? "waiting";
}

export function shimmerText(theme: MinimalTheme, text: string, tick: number) {
  const width = 6;
  const hi = (ch: string) => theme.bold(theme.accentSoft(ch));

  const pos = tick % (text.length + width);
  const start = Math.max(0, pos - width);
  const end = Math.min(text.length - 1, pos);

  let out = "";
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    out += i >= start && i <= end ? hi(ch) : theme.dim(ch);
  }
  return out;
}

export function buildWaitingStatusMessage(params: {
  theme: MinimalTheme;
  tick: number;
  elapsed: string;
  connectionStatus: string;
  phrases?: string[];
}) {
  const phrase = pickWaitingPhrase(params.tick, params.phrases);
  const cute = shimmerText(params.theme, `${phrase}…`, params.tick);
  return `${cute} • ${params.elapsed} | ${params.connectionStatus}`;
}
