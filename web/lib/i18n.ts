"use client";

// Interface language.
//
// The app is English-only in a product built for India, where the people doing the annotating are far more
// likely to read Hindi, Kannada or Tamil than the person who wrote the labels. That is not a cosmetic gap:
// an annotator who has to decode "quarantine" or "promotion blocked" in a second language is slower and
// makes different mistakes, and those mistakes land in the corpus.
//
// Deliberately not react-intl or next-intl. Both bring a provider, a message-extraction step and a build
// plugin, and this app needs none of that: the strings are chrome, they are known at build time, and the
// data (class names, session ids, model versions) is never translated because renaming a class in the
// interface would make it impossible to talk about with the ontology.
//
// Missing keys fall through to English rather than rendering a key. A half-translated interface should read
// as partly English, not as `nav.review.queue`.

export type Locale = "en" | "hi" | "kn" | "ta";

export const LOCALES: { code: Locale; label: string; native: string }[] = [
  { code: "en", label: "English", native: "English" },
  { code: "hi", label: "Hindi", native: "हिन्दी" },
  { code: "kn", label: "Kannada", native: "ಕನ್ನಡ" },
  { code: "ta", label: "Tamil", native: "தமிழ்" },
];

const KEY = "lbx_locale";

// Only the strings an annotator actually reads while working are translated. The governance and platform
// surfaces stay English on purpose: they are read by operators who configured the deployment, and a
// half-translated compliance page is worse than an English one because it invites the reader to trust a
// phrasing nobody reviewed for legal meaning.
type Dict = Record<string, string>;

const EN: Dict = {
  "action.accept": "accept",
  "action.reject": "reject",
  "action.reclassify": "reclassify",
  "action.skip": "skip",
  "action.undo": "undo",
  "action.save": "save",
  "action.cancel": "cancel",
  "action.confirm": "confirm frame",
  "action.next": "next",
  "action.previous": "previous",
  "review.queue": "review queue",
  "review.rapid": "rapid review",
  "review.empty": "nothing in the queue",
  "review.decided": "decided",
  "editor.objects": "objects",
  "editor.lanes": "lanes",
  "editor.pose": "pose",
  "editor.review": "review",
  "editor.class": "class",
  "editor.confidence": "confidence",
  "editor.saved": "saved",
  "editor.unsaved": "unsaved changes",
  "editor.pick_label": "pick a label first",
  "nav.home": "home",
  "nav.activity": "activity",
  "nav.profile": "your account",
  "notify.empty": "nothing yet",
  "notify.mark_all": "mark all read",
  "onboarding.welcome": "Welcome to LabeloxAV",
  "onboarding.skip": "skip the tour",
  "onboarding.next": "next",
  "onboarding.done": "start working",
};

const HI: Dict = {
  "action.accept": "स्वीकार करें",
  "action.reject": "अस्वीकार करें",
  "action.reclassify": "वर्ग बदलें",
  "action.skip": "छोड़ें",
  "action.undo": "पूर्ववत करें",
  "action.save": "सहेजें",
  "action.cancel": "रद्द करें",
  "action.confirm": "फ़्रेम की पुष्टि करें",
  "action.next": "अगला",
  "action.previous": "पिछला",
  "review.queue": "समीक्षा सूची",
  "review.rapid": "त्वरित समीक्षा",
  "review.empty": "सूची में कुछ नहीं है",
  "review.decided": "निर्णय लिए गए",
  "editor.objects": "वस्तुएँ",
  "editor.lanes": "लेन",
  "editor.pose": "मुद्रा",
  "editor.review": "समीक्षा",
  "editor.class": "वर्ग",
  "editor.confidence": "विश्वास",
  "editor.saved": "सहेजा गया",
  "editor.unsaved": "असहेजे बदलाव",
  "editor.pick_label": "पहले एक लेबल चुनें",
  "nav.home": "होम",
  "nav.activity": "गतिविधि",
  "nav.profile": "आपका खाता",
  "notify.empty": "अभी कुछ नहीं",
  "notify.mark_all": "सभी पढ़ा हुआ चिह्नित करें",
  "onboarding.welcome": "LabeloxAV में आपका स्वागत है",
  "onboarding.skip": "परिचय छोड़ें",
  "onboarding.next": "अगला",
  "onboarding.done": "काम शुरू करें",
};

const KN: Dict = {
  "action.accept": "ಸ್ವೀಕರಿಸಿ",
  "action.reject": "ತಿರಸ್ಕರಿಸಿ",
  "action.reclassify": "ವರ್ಗ ಬದಲಿಸಿ",
  "action.skip": "ಬಿಟ್ಟುಬಿಡಿ",
  "action.undo": "ರದ್ದುಗೊಳಿಸಿ",
  "action.save": "ಉಳಿಸಿ",
  "action.cancel": "ರದ್ದು",
  "action.confirm": "ಫ್ರೇಮ್ ದೃಢೀಕರಿಸಿ",
  "action.next": "ಮುಂದೆ",
  "action.previous": "ಹಿಂದೆ",
  "review.queue": "ಪರಿಶೀಲನಾ ಸಾಲು",
  "review.rapid": "ತ್ವರಿತ ಪರಿಶೀಲನೆ",
  "review.empty": "ಸಾಲಿನಲ್ಲಿ ಏನೂ ಇಲ್ಲ",
  "review.decided": "ನಿರ್ಧರಿಸಲಾಗಿದೆ",
  "editor.objects": "ವಸ್ತುಗಳು",
  "editor.lanes": "ಪಥಗಳು",
  "editor.pose": "ಭಂಗಿ",
  "editor.review": "ಪರಿಶೀಲನೆ",
  "editor.class": "ವರ್ಗ",
  "editor.confidence": "ವಿಶ್ವಾಸ",
  "editor.saved": "ಉಳಿಸಲಾಗಿದೆ",
  "editor.unsaved": "ಉಳಿಸದ ಬದಲಾವಣೆಗಳು",
  "editor.pick_label": "ಮೊದಲು ಲೇಬಲ್ ಆಯ್ಕೆಮಾಡಿ",
  "nav.home": "ಮುಖಪುಟ",
  "nav.activity": "ಚಟುವಟಿಕೆ",
  "nav.profile": "ನಿಮ್ಮ ಖಾತೆ",
  "notify.empty": "ಇನ್ನೂ ಏನೂ ಇಲ್ಲ",
  "notify.mark_all": "ಎಲ್ಲವನ್ನೂ ಓದಿದೆ ಎಂದು ಗುರುತಿಸಿ",
  "onboarding.welcome": "LabeloxAV ಗೆ ಸ್ವಾಗತ",
  "onboarding.skip": "ಪರಿಚಯ ಬಿಟ್ಟುಬಿಡಿ",
  "onboarding.next": "ಮುಂದೆ",
  "onboarding.done": "ಕೆಲಸ ಪ್ರಾರಂಭಿಸಿ",
};

const TA: Dict = {
  "action.accept": "ஏற்கவும்",
  "action.reject": "நிராகரிக்கவும்",
  "action.reclassify": "வகை மாற்று",
  "action.skip": "தவிர்",
  "action.undo": "செயல்தவிர்",
  "action.save": "சேமி",
  "action.cancel": "ரத்து",
  "action.confirm": "சட்டகத்தை உறுதிப்படுத்து",
  "action.next": "அடுத்து",
  "action.previous": "முந்தைய",
  "review.queue": "மதிப்பாய்வு வரிசை",
  "review.rapid": "விரைவு மதிப்பாய்வு",
  "review.empty": "வரிசையில் எதுவும் இல்லை",
  "review.decided": "முடிவு செய்யப்பட்டது",
  "editor.objects": "பொருட்கள்",
  "editor.lanes": "பாதைகள்",
  "editor.pose": "நிலை",
  "editor.review": "மதிப்பாய்வு",
  "editor.class": "வகை",
  "editor.confidence": "நம்பிக்கை",
  "editor.saved": "சேமிக்கப்பட்டது",
  "editor.unsaved": "சேமிக்கப்படாத மாற்றங்கள்",
  "editor.pick_label": "முதலில் ஒரு லேபிளைத் தேர்ந்தெடுக்கவும்",
  "nav.home": "முகப்பு",
  "nav.activity": "செயல்பாடு",
  "nav.profile": "உங்கள் கணக்கு",
  "notify.empty": "இன்னும் எதுவும் இல்லை",
  "notify.mark_all": "அனைத்தையும் படித்ததாகக் குறி",
  "onboarding.welcome": "LabeloxAV க்கு வரவேற்கிறோம்",
  "onboarding.skip": "அறிமுகத்தைத் தவிர்",
  "onboarding.next": "அடுத்து",
  "onboarding.done": "வேலையைத் தொடங்கு",
};

const DICTS: Record<Locale, Dict> = { en: EN, hi: HI, kn: KN, ta: TA };

let _locale: Locale | null = null;

export function getLocale(): Locale {
  if (_locale) return _locale;
  if (typeof window === "undefined") return "en";
  const stored = localStorage.getItem(KEY) as Locale | null;
  if (stored && stored in DICTS) { _locale = stored; return stored; }
  // The browser's own preference, which is a better first guess than English for a user who never opens
  // the language menu because they did not know it was there.
  const nav = (navigator.language || "en").slice(0, 2) as Locale;
  _locale = nav in DICTS ? nav : "en";
  return _locale;
}

export function setLocale(locale: Locale): void {
  _locale = locale;
  if (typeof window !== "undefined") {
    localStorage.setItem(KEY, locale);
    document.documentElement.lang = locale;
    // A full reload rather than a re-render. Strings are read at call time all over the tree, and a
    // context that re-renders only its consumers would leave half the interface in the previous language,
    // which is more confusing than either language alone.
    window.location.reload();
  }
}

/** Translate a key. Falls through to English, then to the key, so nothing ever renders blank. */
export function t(key: string, fallback?: string): string {
  const dict = DICTS[getLocale()];
  return dict[key] ?? EN[key] ?? fallback ?? key;
}

/** Every key, for the test that keeps the dictionaries from drifting apart. */
export function allKeys(): string[] {
  return Object.keys(EN).sort();
}

export function dictFor(locale: Locale): Dict {
  return DICTS[locale];
}
