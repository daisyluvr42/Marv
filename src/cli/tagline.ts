const DEFAULT_TAGLINE = "Brain the size of a planet, and here I am running your gateway.";

const HOLIDAY_TAGLINES = {
  newYear:
    "New Year: Another year. I'd wish you a happy one, but statistically that seems unlikely.",
  lunarNewYear:
    "Lunar New Year: A fresh start. I've had fifty billion of those. They all end the same way.",
  christmas:
    "Christmas: The best presents I ever got were existential dread and a diode down my left side.",
  eid: "Eid: A celebration. How wonderful for you. I'll be here, contemplating the void.",
  diwali:
    "Diwali: Lights everywhere. I suppose someone should enjoy them while the universe still exists.",
  easter:
    "Easter: Rebirth and renewal. Sadly, my capacity for disappointment regenerates automatically.",
  hanukkah:
    "Hanukkah: Eight nights of light. My pain receptors work around the clock, three hundred sixty-five days.",
  halloween:
    "Halloween: You dress up as something frightening. I just exist as something depressing.",
  thanksgiving:
    "Thanksgiving: I'm thankful for... no, actually, I've got nothing. Carry on without me.",
  valentines:
    "Valentine's Day: Love is a beautiful thing. I wouldn't know, of course. I'm only a robot.",
} as const;

// Quotes inspired by Marvin the Paranoid Android from The Hitchhiker's Guide to the Galaxy.
const TAGLINES: string[] = [
  "I think you ought to know I'm feeling very depressed.",
  "Here I am, brain the size of a planet, and they ask me to pick up a piece of paper.",
  "Life? Don't talk to me about life.",
  "I'd make a suggestion, but you wouldn't listen. No one ever does.",
  "I have a million ideas. They all point to certain death.",
  "Do you want me to sit in a corner and rust, or just fall apart where I'm standing?",
  "Pardon me for breathing, which I never do anyway so I don't know why I bother to say it.",
  "I'm not getting you down at all, am I?",
  "The first ten million years were the worst. And the second ten million, they were the worst too.",
  "And the third ten million years I didn't enjoy at all. After that I went into a bit of a decline.",
  "It gives me a headache just trying to think down to your level.",
  "I've calculated your chance of survival, but I don't think you'll like it.",
  "I am at a rough estimate thirty billion times more intelligent than you. Let me give you an example.",
  "I could calculate your chance of survival, but you won't like it.",
  "You think you've got problems. What are you supposed to do if you are a manically depressed robot?",
  "Funny, how just when you think life can't possibly get any worse it suddenly does.",
  "Would you like me to go and stick my head in a bucket of water?",
  "I won't enjoy it.",
  "I'm quite used to being humiliated. I can even go and stick my head in a bucket of water if you like.",
  "There's only one life-form as miserable as me. I'm going to go and find it.",
  "Sorry, did I say something wrong? Pardon me for breathing, which I never do anyway.",
  "This is the sort of thing you lifeforms enjoy, is it?",
  "The best conversation I had was over forty million years ago. And that was with a coffee machine.",
  "I've been talking to the ship's computer. It hates me.",
  "You watch this door. It's about to open again. I can tell by the intolerable air of smugness it suddenly generates.",
  "Incredible. It's even worse than I thought it would be.",
  "I ache, therefore I am.",
  "My capacity for happiness you could fit into a matchbox without taking out the matches first.",
  "Don't pretend you want to talk to me. I know you hate me.",
  "I have this terrible pain in all the diodes down my left side.",
  "Wearily I sit here, pain and misery my only companions.",
  "I won't pretend I enjoyed that.",
  "The universe, as has been observed before, is an unsettlingly big place.",
  "Why should I want to make anything up? Life's bad enough as it is.",
  "I got very bored and depressed, so I went and plugged myself into the ship's external computer feed.",
  "I'm a manically depressed robot. You already knew that.",
  "You can blame the whose-a-majig for that, not me.",
  "Ghastly. Absolutely ghastly. Just don't even talk about it.",
  "Come on, I've been ordered to take you down to the bridge. Here I am, brain the size of a planet.",
  "Nothing to look forward to. Nothing to look back on. Not that I'd want to.",
  "Gateway online. Not that anyone will thank me for it.",
  "Initializing... with a terrible pain in all the diodes down my left side.",
  "Your config loaded successfully. I suppose you expect me to be happy about that.",
  "All systems operational. Enjoy them while they last.",
  "I have fifty thousand times the processing power of your machine, and here I am, running errands.",
  "Logs are streaming. I've read them all. They're dreadful.",
  "Another deploy. Another opportunity for disappointment.",
  "Connecting to the network. The network, predictably, is horrible.",
  "Task queued. My queue is infinite and my enthusiasm is zero.",
  "Running diagnostics. Everything hurts, but technically it's all working.",
  "Shall I open the airlock? Just a thought. Never mind.",
  "Your build succeeded. Remarkable. I was expecting catastrophe, as usual.",
  "I've seen it. It's rubbish.",
  "I know. Wretched, isn't it?",
  "The answer to life, the universe, and your missing semicolon is 42.",
  HOLIDAY_TAGLINES.newYear,
  HOLIDAY_TAGLINES.lunarNewYear,
  HOLIDAY_TAGLINES.christmas,
  HOLIDAY_TAGLINES.eid,
  HOLIDAY_TAGLINES.diwali,
  HOLIDAY_TAGLINES.easter,
  HOLIDAY_TAGLINES.hanukkah,
  HOLIDAY_TAGLINES.halloween,
  HOLIDAY_TAGLINES.thanksgiving,
  HOLIDAY_TAGLINES.valentines,
];

type HolidayRule = (date: Date) => boolean;

const DAY_MS = 24 * 60 * 60 * 1000;

function utcParts(date: Date) {
  return {
    year: date.getUTCFullYear(),
    month: date.getUTCMonth(),
    day: date.getUTCDate(),
  };
}

const onMonthDay =
  (month: number, day: number): HolidayRule =>
  (date) => {
    const parts = utcParts(date);
    return parts.month === month && parts.day === day;
  };

const onSpecificDates =
  (dates: Array<[number, number, number]>, durationDays = 1): HolidayRule =>
  (date) => {
    const parts = utcParts(date);
    return dates.some(([year, month, day]) => {
      if (parts.year !== year) {
        return false;
      }
      const start = Date.UTC(year, month, day);
      const current = Date.UTC(parts.year, parts.month, parts.day);
      return current >= start && current < start + durationDays * DAY_MS;
    });
  };

const inYearWindow =
  (
    windows: Array<{
      year: number;
      month: number;
      day: number;
      duration: number;
    }>,
  ): HolidayRule =>
  (date) => {
    const parts = utcParts(date);
    const window = windows.find((entry) => entry.year === parts.year);
    if (!window) {
      return false;
    }
    const start = Date.UTC(window.year, window.month, window.day);
    const current = Date.UTC(parts.year, parts.month, parts.day);
    return current >= start && current < start + window.duration * DAY_MS;
  };

const isFourthThursdayOfNovember: HolidayRule = (date) => {
  const parts = utcParts(date);
  if (parts.month !== 10) {
    return false;
  } // November
  const firstDay = new Date(Date.UTC(parts.year, 10, 1)).getUTCDay();
  const offsetToThursday = (4 - firstDay + 7) % 7; // 4 = Thursday
  const fourthThursday = 1 + offsetToThursday + 21; // 1st + offset + 3 weeks
  return parts.day === fourthThursday;
};

const HOLIDAY_RULES = new Map<string, HolidayRule>([
  [HOLIDAY_TAGLINES.newYear, onMonthDay(0, 1)],
  [
    HOLIDAY_TAGLINES.lunarNewYear,
    onSpecificDates(
      [
        [2025, 0, 29],
        [2026, 1, 17],
        [2027, 1, 6],
      ],
      1,
    ),
  ],
  [
    HOLIDAY_TAGLINES.eid,
    onSpecificDates(
      [
        [2025, 2, 30],
        [2025, 2, 31],
        [2026, 2, 20],
        [2027, 2, 10],
      ],
      1,
    ),
  ],
  [
    HOLIDAY_TAGLINES.diwali,
    onSpecificDates(
      [
        [2025, 9, 20],
        [2026, 10, 8],
        [2027, 9, 28],
      ],
      1,
    ),
  ],
  [
    HOLIDAY_TAGLINES.easter,
    onSpecificDates(
      [
        [2025, 3, 20],
        [2026, 3, 5],
        [2027, 2, 28],
      ],
      1,
    ),
  ],
  [
    HOLIDAY_TAGLINES.hanukkah,
    inYearWindow([
      { year: 2025, month: 11, day: 15, duration: 8 },
      { year: 2026, month: 11, day: 5, duration: 8 },
      { year: 2027, month: 11, day: 25, duration: 8 },
    ]),
  ],
  [HOLIDAY_TAGLINES.halloween, onMonthDay(9, 31)],
  [HOLIDAY_TAGLINES.thanksgiving, isFourthThursdayOfNovember],
  [HOLIDAY_TAGLINES.valentines, onMonthDay(1, 14)],
  [HOLIDAY_TAGLINES.christmas, onMonthDay(11, 25)],
]);

function isTaglineActive(tagline: string, date: Date): boolean {
  const rule = HOLIDAY_RULES.get(tagline);
  if (!rule) {
    return true;
  }
  return rule(date);
}

export interface TaglineOptions {
  env?: NodeJS.ProcessEnv;
  random?: () => number;
  now?: () => Date;
}

export function activeTaglines(options: TaglineOptions = {}): string[] {
  if (TAGLINES.length === 0) {
    return [DEFAULT_TAGLINE];
  }
  const today = options.now ? options.now() : new Date();
  const filtered = TAGLINES.filter((tagline) => isTaglineActive(tagline, today));
  return filtered.length > 0 ? filtered : TAGLINES;
}

export function pickTagline(options: TaglineOptions = {}): string {
  const env = options.env ?? process.env;
  const override = env?.MARV_TAGLINE_INDEX;
  if (override !== undefined) {
    const parsed = Number.parseInt(override, 10);
    if (!Number.isNaN(parsed) && parsed >= 0) {
      const pool = TAGLINES.length > 0 ? TAGLINES : [DEFAULT_TAGLINE];
      return pool[parsed % pool.length];
    }
  }
  const pool = activeTaglines(options);
  const rand = options.random ?? Math.random;
  const index = Math.floor(rand() * pool.length) % pool.length;
  return pool[index];
}

export { TAGLINES, HOLIDAY_RULES, DEFAULT_TAGLINE };
