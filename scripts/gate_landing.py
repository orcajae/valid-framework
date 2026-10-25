#!/usr/bin/env python3
"""Pre-push gates for the landing page (docs/).

Scope of the text gates is what a visitor can actually read:
  * HTML text nodes and the user-facing attributes (title, meta content, alt,
    aria-label) -- never tag names, class names or href targets
  * string literals in scorer.js -- not identifiers, so `return` is not a hit
  * the extracted text of the one-page checklist PDF

Rule 48 normalisation is applied to every extracted string before matching:
newlines collapsed, unicode math letters folded to ASCII, NBSP folded to space.

Exit status is non-zero if any gate fails; every failure prints the offending
line verbatim.

Service-page gates (October 2026). Every piece of body text sits in a region
marked data-gate="service" (new copy: the hero, the service sections, the
footer; the head counts as one) or data-gate="record" (the correction box and
the scorer, whose wording follows the CHANGELOG entries and the published
checklist). The attribute may sit on any element.
  1b  banned words in service copy: priority and market-gap claims, assurance
      words, performance words, journal names
  1c  numbers in service copy: digits outside the named label contexts must be
      result numbers inside their own claim block; number words are banned
      outside an explicit phrase list
  1d  each data-claim block matches its reviewed text exactly, once
  1e  no em-dash in the page text or in scorer.js strings (JS escapes decoded;
      the one-page PDF is a fixed record and is not scanned for this)
  1f  no prices or currencies anywhere on the page
  1g  wording the service copy must carry (nature of a review, citation form,
      verdict wording, disclosure, contact)
  1h  no body text outside a service or record region
  1i  at most one call to action (rule 73): a.btn and button.btn outside the
      record region, plus any mailto link outside the footer
  1j  exactly one record region, not nested in another region, holding the
      sections #correction and #scorer, whose normalised text and link
      targets have the sha256
      in RECORD_SHA (a changelog package that edits the correction box or a
      change to the scorer updates RECORD_SHA in the same commit)
  5c  anchors that outside links rely on exist (parsed ids), no duplicate ids
selftest() runs first: every gate must pass the clean example and fire on each
of its bad examples, or the script stops with exit status 2.
"""
import html as htmllib
import os
import re
import sys
import unicodedata
from html.parser import HTMLParser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(ROOT, "docs", "index.html")
SCORER = os.path.join(ROOT, "docs", "scorer.js")
ONEPAGE = os.path.join(ROOT, "docs", "assets", "valid_checklist_onepage.pdf")
README = os.path.join(ROOT, "README.md")

# ---------------------------------------------------------------- gate 1
BANNED = r"guarantee|refund|ROI|returns|discount|% off|limited time|only \d+ left"

# ---------------------------------------------------------------- gate 2
RETIRED = (r"of 75|75 papers|0/75|25/75|64/75|72%|x\.com/jwquant|@jwquant"
           # withdrawn in CHANGELOG_CR.md E13 (October 2026)
           r"|2\.5 ?/ ?12|2\.5 of 12|0/74|74 empirical|9 ?/ ?12|9 of 12|21\.7"
           r"|27% → 0%|PBO ?= ?1\.0|4\.4%|Request an audit|27[–-]30%"
           # restated in CHANGELOG_CR.md E16
           r"|52% of our 340"
           # workshop name: the official page (sites.google.com/view/kdd-mlf-2026)
           # does not say ACM SIGKDD (INBOX #26). "SIGKDD" alone also catches the
           # name split across a line break, since the raw scan reads line by line
           # (INBOX #29; case-sensitive; a hyphenated split such as "SIG-" / "KDD"
           # is not caught)
           r"|SIGKDD")

# ---------------------------------------------------------------- gate 3
# October 2026 correction state (CHANGELOG_CR.md, E13). Each entry must
# appear at least once in the landing text.
CANON = [
    (r"Correction, October 2026", "correction notice on the page"),
    (r"\bE13\b", "link to CHANGELOG E13"),
    (r"\bE14\b", "link to CHANGELOG E14 (pre-registered re-measurement)"),
    (r"\bE15\b", "link to CHANGELOG E15 (pre-registered robustness study)"),
    (r"\bE16\b", "link to CHANGELOG E16 (corpus re-count)"),
    (r"\bE17\b", "link to CHANGELOG E17 (PBO thresholds across designs)"),
    (r"\bE18\b", "link to CHANGELOG E18 (probability of loss and embargo)"),
    (r"\b278\b", "corrected corpus count 165 of 278 (E16)"),
    (r"26\.5%", "AUC alone 27.0/30.0/26.5% (stands, E13)"),
]

# ---------------------------------------------------------------- gate 1b
# The sample line of brief v1.2 section 12, required wherever the sample report
# is shown. Gate 1b removes this exact sentence before it scans (it says
# "only"); any other wording with "only" still fails.
SAMPLE_LINE = ("SAMPLE: fictional fund, synthetic data, illustrative only. "
               "Findings, no assurance.")
SERVICE_BANNED = (
    r"first|only|signals?|returns|alpha|profit\w*|outperform\w*|guarantee\w*"
    r"|verif(?!ier)\w*|validated|validates|certif\w*|audit(?! opinion)\w*"
    r"|(?<!not an )audit opinion|(?<!no )(?<!opinion or )assur\w*"
    r"|independent reviewer|regulated|licensed|accredit\w*"
    r"|(?<!US )(?<!pre-)registered(?! series)|(?<=US )registered(?! investment advisers)"
    r"|endors(?!ement of past results)\w*"
    r"|book a|book your|order now|get started|sign up|subscribe\w*"
    r"|award\w*|credited|compliant|approved|passed VALID|ROI"
    r"|JFDS|Journal of|Financial Data Science|Portfolio Management Research"
    r"|Risk Model Validation|SR 11-7|SR 26-2"
    # market-gap and priority claims (PRICING_RESEARCH 2026-10-24, section 1)
    r"|nobody|no one|no other|gaps?|white space|underserved|overlooked"
    r"|neglected|less attention|unique\w*|novel|no prior|no standard"
    r"|unmatched|unparalleled|best-in-class"
    # superlatives and borrowed standing (v2 review, MINOR 13)
    r"|leading|most trusted|popular|world-class|published in")

# ---------------------------------------------------------------- gate 1c
# Label contexts are removed before numbers are read; what is left must be a
# result number inside its own claim block. Results come from brief section 6
# and were re-checked against CHANGELOG_CR.md at the base commit.
LABEL_CONTEXTS = [
    r"AUC > 0\.55", r"PBO < 0\.20", r"18 bp", r"95% intervals", r"N of 12",
    r"KDD-MLF 2026", r"KDD 2026", r"October 2026", r"9th Workshop",
    r"Paper 18", r"[Pp]aper 6508779", r"\bT[123]\b",
    r"\bTier [123]\b", r"\bE1[3-8]\b", r"version 3\.4", r"15-minute", r"1-hour",
    r"jwim1101", r"jaewookkim1101", r"VALID v1\.0",
    # engagement, contact and who-does-the-work copy (INBOX #29)
    r"1–2 weeks", r"3–4 weeks", r"6–8 weeks", r"2 business days",
    r"UTC\+9",
]
# Any digit counts, glued to letters or after a decimal point too.
NUM_TOKEN = re.compile(r"\.?\d[\d.,]*(?:–\d[\d.,]*)?%?")
CLAIM_NUMBERS = {
    "auc": {"27.0%", "30.0%", "26.5%", "200"},               # E13, E14
    "pbo-designs": {"3.1–23.7%", "216"},                     # E17 D1
    "cost": {"165", "278", "59%", "52–60%", "9", "116", "118", "98%"},  # E16
}
NUMBER_WORDS = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven"
    r"|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen"
    r"|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred"
    r"|thousand|million|billion|half|halves|quarter|third|dozen|percent"
    r"|second|fourth|fifth|sixth|seventh|eighth|ninth|tenth|twice|thrice"
    r"|double\w*|triple\w*|quadruple\w*)\b", re.I)
NUMBER_WORD_PHRASES = [
    "one backtest", "one gate", "two or three", "one or two",
    "twelve VALID items", "Three levels",
    "zero average performance", "below zero",
    "more than one way", "best-looking one", "a planted one", "to suggest one", "third party",
    "third-party",
]

# ---------------------------------------------------------------- gate 1d
# The reviewed text of each result block (normalised). Any rewording, swapped
# scope word or added negation fails; changing a block means changing it here.
CLAIMS = {
    "auc": "27.0%, 30.0%, 26.5% of resampled BTC, ETH and SOL daily histories "
           "with nothing to find beyond each coin's average move passed a "
           "prediction check. AUC > 0.55 used alone; 200 daily series per asset, "
           "days drawn at random with repeats. E14",
    "pbo-designs": "3.1–23.7% of simulated groups of strategy settings that all "
                   "had zero average performance passed an overfitting cutoff, "
                   "depending on the setup. "
                   "PBO < 0.20 under canonical CSCV, sets with equal means, across "
                   "216 synthetic designs. PBO thresholds are not test levels. E17",
    "cost": "165 of 278 strategy versions in our own archive came out negative "
            "after costs (59%). Net Sharpe ratio below zero at a nominal 18 bp, "
            "each result counted once; 52–60% across the definitions we "
            "published; costs charged differently across strategy families. "
            "The shortest bars fared worst: 15-minute bars 9 of 9, 1-hour bars "
            "116 of 118 (98%). E16",
}

_CL = "https://github.com/orcajae/valid-framework/blob/main/CHANGELOG_CR.md#"
CLAIM_HREFS = {
    "auc": [_CL + "e14-pbo-and-the-permutation-test-re-measured-under-a-pre-registered-design"],
    "pbo-designs": [_CL + "e17-pbo-thresholds-across-designs-under-a-pre-registered-monte-carlo"],
    "cost": [_CL + "e16-the-340-variant-corpus-re-counted-e13s-corpus-figures-restated"],
}

# ---------------------------------------------------------------- gate 1j
# sha256 of the normalised text of the record region: the correction box as
# published at 1ca480f and the scorer section of 1ca480f with its four
# em-dashes replaced (": " or ", ").
RECORD_SHA = "36e87acaabb54797dbafeda198b5e89c8314ef4cc64412cabfb229fa997e4d74"
RECORD_SECTIONS = ["correction", "scorer"]

# ---------------------------------------------------------------- gate 1f
PRICES = (r"[$€£₩]\s?\d[\d,.]*|\b(?:USD|EUR|GBP|KRW)\b"
          r"|\bpric(?:e|es|ed|ing)\b|per hour|per day")

# ---------------------------------------------------------------- gate 1g
# Each phrase must appear in the service copy. Phrases in REQUIRED_OPEN must
# also appear outside a collapsed <details> body (a <summary> counts as open),
# so a visitor sees them without opening anything (v2 review, MAJOR 6).
REQUIRED_SERVICE = [
    "not an audit opinion or assurance, and not investment advice",
    "VALID checklist score N of 12 (findings, no assurance)",
    "The rule does not require a third-party review",
    "PBO thresholds are not test levels",
    "A low PBO alone does not show that any configuration has a positive edge",
    "Reviews are carried out by Jaewook Kim alone",
    "develops systematic trading strategies of his own",
    "Client code, data and results are never used in his own trading",
    "requires the written consent of the client, and of any manager under "
    "review, and is limited to anonymised statistics that the client approves "
    "before publication",
    "jwim1101@gmail.com",
    "linkedin.com/in/jaewookkim1101",
    SAMPLE_LINE,          # the sample-report link on the T2 card
]
REQUIRED_OPEN = [REQUIRED_SERVICE[i] for i in (0, 3, 4, 5, 6, 7, 9, 10, 11)]

REQUIRED_IDS = ["scorer", "correction", "contact", "review"]

VISIBLE_ATTRS = {"content", "alt", "aria-label", "title", "value", "placeholder"}
SKIP_TAGS = {"script", "style"}


VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input",
             "link", "meta", "source", "track", "wbr"}


class Visible(HTMLParser):
    """Text a visitor can read: text nodes plus user-facing attributes.

    For the service-page gates it also records, for each string, the region it
    sits in (data-gate on the nearest enclosing element that has one; the head
    is a service region) and the data-claim block it belongs to, the full text
    of each claim block, the ids, and structural errors."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out = []
        self.service = []         # [text, claim name or None, hidden in <details>]
        self.unregioned = []
        self.claims = []          # [name, text]
        self.ids = []
        self.errors = []
        self.records = []         # {"text", "sections"}
        self.ctas = 0
        self._skip = 0
        self._stack = []          # [tag, data-gate, claim index, record index, vis]
        self._in_body = False

    def _region(self):
        for _, gate, _, _, _ in reversed(self._stack):
            if gate:
                return gate
        return None

    def _claim(self):
        for _, _, idx, _, _ in reversed(self._stack):
            if idx is not None:
                return idx
        return None

    def _record(self):
        for _, _, _, rec, _ in reversed(self._stack):
            if rec is not None:
                return rec
        return None

    def _hidden(self):
        """True when a visitor cannot read the text without acting: inside an
        element hidden by attribute, style, class or <template>, or in the body
        of a closed <details> (its <summary> stays readable)."""
        for i, e in enumerate(self._stack):
            if e[4] == "gone":
                return True
            # only the closed element's own <summary> (its direct child) is readable
            if e[4] == "closed" and not (i + 1 < len(self._stack)
                                         and self._stack[i + 1][0] == "summary"):
                return True
        return False

    @staticmethod
    def _vis(tag, a):
        style = re.sub(r"\s+", "", (a.get("style") or "").lower())
        classes = set((a.get("class") or "").split())
        if (tag == "template" or "hidden" in a
                or (a.get("aria-hidden") or "").lower() == "true"
                or re.search(r"display:none|visibility:hidden|opacity:0(?![.\d]*[1-9])"
                             r"|font-size:0(?![.\d]*[1-9])", style)
                or classes & {"sr-only", "visually-hidden"}):
            return "gone"
        if tag == "details" and "open" not in a:
            return "closed"
        return None

    def _add(self, text):
        self.out.append(text)
        region = self._region()
        idx = self._claim()
        if region == "service":
            self.service.append([text, None if idx is None else self.claims[idx][0],
                                 self._hidden()])
        elif region != "record" and self._in_body:
            self.unregioned.append(text)
        if idx is not None:
            self.claims[idx][1] += " " + text
        rec = self._record()
        if rec is not None:
            self.records[rec]["text"] += " " + text

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in SKIP_TAGS:
            self._skip += 1
        if tag == "body":
            self._in_body = True
        if a.get("id"):
            self.ids.append(a["id"])
        gate = "service" if tag == "head" else a.get("data-gate")
        if gate not in (None, "service", "record"):
            self.errors.append("unknown data-gate %r" % gate)
        idx = None
        if a.get("data-claim"):
            if self._claim() is not None:
                self.errors.append("nested data-claim %r" % a["data-claim"])
            self.claims.append([a["data-claim"], "", []])
            idx = len(self.claims) - 1
        rec = None
        if gate == "record":
            if self._region() is not None:
                self.errors.append("record region inside a %s region" % self._region())
            self.records.append({"text": "", "sections": []})
            rec = len(self.records) - 1
        if tag == "section" and self._record() is not None:
            self.records[self._record()]["sections"].append(a.get("id"))
        is_btn = "btn" in (a.get("class") or "").split()
        if tag in ("a", "button") and is_btn and self._region() != "record":
            self.ctas += 1
        if (tag == "a" and (a.get("href") or "").startswith("mailto:")
                and not any(e[0] == "footer" for e in self._stack)):
            self.ctas += 1
        if tag == "a" and self._record() is not None and a.get("href"):
            self.records[self._record()]["text"] += " <%s>" % a["href"]
        if tag == "a":
            if self._claim() is not None:
                self.claims[self._claim()][2].append(a.get("href"))
        if tag not in VOID_TAGS:
            self._stack.append([tag, gate, idx, rec, self._vis(tag, a)])
        if tag == "meta" and a.get("name") == "viewport":
            return                # layout hint, not text a visitor reads
        for k, v in attrs:
            if k in VISIBLE_ATTRS and v:
                self._add(v)

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS and self._skip:
            self._skip -= 1
        if tag == "body":
            self._in_body = False
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i][0] == tag:
                del self._stack[i:]
                break

    def handle_data(self, data):
        if not self._skip and data.strip():
            self._add(data)


def norm(s):
    """Rule 48: collapse newlines, fold unicode math letters and NBSP."""
    s = s.replace(" ", " ")
    s = "".join(
        unicodedata.normalize("NFKC", ch) if ord(ch) > 0x7F else ch for ch in s
    )
    return re.sub(r"\s+", " ", s).strip()


def js_strings(path):
    src = open(path, encoding="utf-8").read()
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    src = re.sub(r"(?m)^\s*//.*$", " ", src)
    return [
        js_decode(htmllib.unescape(m.group(1) if m.group(1) is not None
                                   else m.group(2)))
        for m in re.finditer(r'"((?:[^"\\]|\\.)*)"|\'((?:[^\'\\]|\\.)*)\'', src)
    ]


def js_decode(s):
    """Decode \\uXXXX and \\xXX escapes so an escaped character is seen."""
    s = re.sub(r"\\u\{([0-9a-fA-F]+)\}", lambda m: chr(int(m.group(1), 16)), s)
    s = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), s)
    return re.sub(r"\\x([0-9a-fA-F]{2})", lambda m: chr(int(m.group(1), 16)), s)


def pdf_text(path):
    try:
        import fitz
    except ImportError:
        return None
    with fitz.open(path) as d:
        return "\n".join(p.get_text() for p in d)


def fetch(base):
    """Pull the three user-facing files off a deployed site into a temp dir."""
    import tempfile
    import urllib.request
    d = tempfile.mkdtemp(prefix="gate-live-")
    got = {}
    for name, rel in (("index", ""), ("scorer", "scorer.js"),
                      ("onepage", "assets/valid_checklist_onepage.pdf")):
        url = base.rstrip("/") + "/" + rel
        req = urllib.request.Request(url, headers={"Cache-Control": "no-cache"})
        with urllib.request.urlopen(req, timeout=45) as r:
            if r.status != 200:
                sys.exit("live fetch %s -> HTTP %s" % (url, r.status))
            body = r.read()
        path = os.path.join(d, rel.split("/")[-1] or "index.html")
        with open(path, "wb") as fh:
            fh.write(body)
        got[name] = path
        print("  fetched %-46s %d bytes" % (url, len(body)))
    return got


def collect(src=None):
    """(label, text) pairs making up the user-visible surface."""
    index = src["index"] if src else INDEX
    scorer = src["scorer"] if src else SCORER
    onepage = src["onepage"] if src else ONEPAGE
    p = parse_html(open(index, encoding="utf-8").read())
    items = [("index.html", t) for t in p.out]
    items += [("scorer.js", t) for t in js_strings(scorer)]
    t = pdf_text(onepage)
    if t is None:
        print("  ! PyMuPDF missing: the one-page PDF was NOT scanned")
    else:
        items += [("valid_checklist_onepage.pdf", t)]
    return [(lbl, norm(s)) for lbl, s in items if norm(s)]


def scan(items, pattern, name, flags=re.I):
    rx = re.compile(r"(?<![\w%])(?:" + pattern + r")(?![\w])", flags)
    hits = [(lbl, s, m.group(0)) for lbl, s in items for m in [rx.search(s)] if m]
    print("\n[%s] pattern: %s" % (name, pattern))
    if hits:
        for lbl, s, m in hits:
            print("  FAIL %-32s %r  in: %s" % (lbl, m, s[:150]))
        return False
    print("  PASS  0 hits across %d visible strings" % len(items))
    return True


def canon(items):
    blob = " ".join(s for _, s in items)
    print("\n[gate 3] October 2026 correction state")
    ok = True
    for rx, label in CANON:
        m = re.search(rx, blob)
        print("  %s %-32s %s" % ("PASS" if m else "FAIL", label,
                                 repr(m.group(0)) if m else "MISSING"))
        ok = ok and bool(m)
    return ok


def links():
    """Local hrefs/srcs in the landing page and README must resolve."""
    print("\n[gate 5] local links resolve")
    ok = True
    targets = []
    src = open(INDEX, encoding="utf-8").read()
    for m in re.finditer(r'(?:href|src)="([^"]+)"', src):
        u = m.group(1)
        if not re.match(r"^(https?:|mailto:|#|//)", u):
            targets.append(("docs/index.html", u,
                            os.path.join(ROOT, "docs", u.split("#")[0])))
    rd = open(README, encoding="utf-8").read()
    for m in re.finditer(r"\]\(([^)\s]+)\)", rd):
        u = m.group(1)
        if not re.match(r"^(https?:|mailto:|#|//)", u):
            targets.append(("README.md", u, os.path.join(ROOT, u.split("#")[0])))
    for where, u, path in targets:
        good = os.path.exists(path)
        ok = ok and good
        print("  %s %-14s %s" % ("PASS" if good else "FAIL", where, u))
    return ok


def anchors():
    """Every in-page #anchor has a matching id."""
    src = open(INDEX, encoding="utf-8").read()
    ids = set(re.findall(r'id="([^"]+)"', src))
    print("\n[gate 5b] in-page anchors")
    ok = True
    for m in re.finditer(r'href="#([^"]+)"', src):
        good = m.group(1) in ids
        ok = ok and good
        print("  %s #%s" % ("PASS" if good else "FAIL", m.group(1)))
    return ok


def raw_sources():
    """Whole-file scan. Retired values must not survive even inside an href."""
    out = []
    for path in (INDEX, SCORER, README, os.path.join(ROOT, "CITATION.cff")):
        rel = os.path.relpath(path, ROOT)
        for i, line in enumerate(open(path, encoding="utf-8"), 1):
            if line.strip():
                out.append(("%s:%d" % (rel, i), norm(line)))
    return out


def parse_html(doc):
    p = Visible()
    p.feed(doc)
    p.close()
    return p


def strip_sample_line(t):
    return norm(t).replace(SAMPLE_LINE, " ").strip()


def service_items(p):
    return [("index.html[service]", strip_sample_line(t)) for t, _, _ in p.service
            if strip_sample_line(t)]


def numbers(p):
    """Gate 1c: digits and number words in service copy."""
    print("\n[gate 1c] numbers in service copy")
    bad = []
    for t, claim, _ in p.service:
        t = norm(t)
        rest = t
        for rx in LABEL_CONTEXTS:
            rest = re.sub(rx, " ", rest)
        for tok in (t.rstrip(".,") for t in NUM_TOKEN.findall(rest)):
            if claim is None or tok not in CLAIM_NUMBERS.get(claim, set()):
                bad.append((tok, claim, t))
        words = t
        for ph in NUMBER_WORD_PHRASES:
            words = re.sub(re.escape(ph), " ", words, flags=re.I)
        for m in NUMBER_WORDS.finditer(words):
            bad.append((m.group(0), claim, t))
    for tok, claim, t in bad:
        print("  FAIL %r (claim %s) in: %s" % (tok, claim, t[:150]))
    if not bad:
        print("  PASS  every number is a label or a result in its own block")
    return not bad


def claims(p):
    """Gate 1d: each claim block matches its reviewed text, exactly once."""
    print("\n[gate 1d] claim blocks match their reviewed text")
    ok = True
    for e in p.errors:
        if e.startswith("nested"):
            print("  FAIL %s" % e)
            ok = False
    seen = {}
    for name, text, hrefs in p.claims:
        seen[name] = seen.get(name, 0) + 1
        want = CLAIMS.get(name)
        got = norm(text)
        good = want is not None and got == want and hrefs == CLAIM_HREFS.get(name)
        if want is not None and got == want and not good:
            got = "links %s" % hrefs
        ok = ok and good
        print("  %s %-12s %s" % ("PASS" if good else "FAIL", name,
                                 "exact" if good else "got: %s" % got[:150]))
    for name in CLAIMS:
        if seen.get(name, 0) != 1:
            print("  FAIL %r appears %d times (want 1)" % (name, seen.get(name, 0)))
            ok = False
    return ok


def emdash(items):
    """Gate 1e: no em-dash in the page text or in scorer.js strings."""
    print("\n[gate 1e] no em-dash (index.html, scorer.js)")
    hits = [(lbl, s) for lbl, s in items
            if lbl in ("index.html", "scorer.js") and "—" in s]
    for lbl, s in hits:
        print("  FAIL %-12s %s" % (lbl, s[:150]))
    if not hits:
        print("  PASS  0 hits")
    return not hits


def required(p):
    """Gate 1g: wording the service copy must carry."""
    blob = " ".join(norm(t) for t, _, _ in p.service)
    shown = " ".join(norm(t) for t, _, hid in p.service if not hid)
    print("\n[gate 1g] required service wording (open = outside collapsed details)")
    ok = True
    for ph in REQUIRED_SERVICE:
        if ph not in blob:
            verdict = "FAIL MISSING"
        elif ph in REQUIRED_OPEN and ph not in shown:
            verdict = "FAIL HIDDEN "
        else:
            verdict = "PASS open" if ph in REQUIRED_OPEN else "PASS"
        ok = ok and verdict.startswith("PASS")
        print("  %s %s" % (verdict, ph))
    return ok


def regions(p):
    """Gate 1h: no body text outside a service or record region."""
    print("\n[gate 1h] body text sits in a service or record region")
    for t in p.unregioned:
        print("  FAIL unregioned: %s" % norm(t)[:150])
    for e in p.errors:
        if e.startswith("unknown data-gate"):
            print("  FAIL %s" % e)
    bad = bool(p.unregioned) or any(e.startswith("unknown") for e in p.errors)
    if not bad:
        print("  PASS")
    return not bad


def ids_present(p):
    """Gate 5c: required anchors exist as parsed ids; no duplicate ids."""
    print("\n[gate 5c] required anchors, unique ids")
    ok = True
    for i in REQUIRED_IDS:
        good = i in p.ids
        ok = ok and good
        print("  %s #%s" % ("PASS" if good else "FAIL", i))
    dup = sorted({i for i in p.ids if p.ids.count(i) > 1})
    for i in dup:
        print("  FAIL duplicate id %r" % i)
    return ok and not dup


def ctas(p):
    """Gate 1i: at most one call-to-action button."""
    print("\n[gate 1i] call-to-action buttons: %d (at most 1)" % p.ctas)
    print("  %s" % ("PASS" if p.ctas <= 1 else "FAIL"))
    return p.ctas <= 1


def records(p, want_sha):
    """Gate 1j: one record region, top level, #correction and #scorer, pinned text."""
    import hashlib
    print("\n[gate 1j] record region")
    ok = True
    for e in p.errors:
        if e.startswith("record region"):
            print("  FAIL %s" % e)
            ok = False
    if len(p.records) != 1:
        print("  FAIL %d record regions (want 1)" % len(p.records))
        return False
    r = p.records[0]
    if r["sections"] != RECORD_SECTIONS:
        print("  FAIL sections %s (want %s)" % (r["sections"], RECORD_SECTIONS))
        ok = False
    got = hashlib.sha256(norm(r["text"]).encode("utf-8")).hexdigest()
    if got != want_sha:
        print("  FAIL text sha256 %s (want %s)" % (got, want_sha))
        ok = False
    if ok:
        print("  PASS sections %s, text sha256 %s" % (r["sections"], got[:12]))
    return ok


SERVICE_GATES = ["1b", "1c", "1d", "1e", "1f", "1g", "1h", "1i", "1j", "5c"]


def run_service(p, items, record_sha=None):
    return [
        scan(service_items(p), SERVICE_BANNED, "gate 1b banned in service copy"),
        numbers(p),
        claims(p),
        emdash(items),
        scan(items, PRICES, "gate 1f no prices"),
        required(p),
        regions(p),
        ctas(p),
        records(p, RECORD_SHA if record_sha is None else record_sha),
        ids_present(p),
    ]


def _good():
    req = " ".join(ph + "." for ph in REQUIRED_SERVICE)
    return (
        "<html><head><title>VALID review</title></head><body>"
        "<header data-gate='service'><p>A plain sentence.</p><p>" + req + "</p></header>"
        "<main><section id='review' data-gate='service'><ul>"
        + "".join("<li data-claim='%s'><span>%s</span> <a href='%s'>%s</a></li>"
                  % (k, htmllib.escape(v.rsplit(" ", 1)[0], quote=False),
                     CLAIM_HREFS[k][0], v.rsplit(" ", 1)[1])
                  for k, v in CLAIMS.items())
        + "</ul><p>Tier 2 gives 95% intervals.</p>"
        "<p><a class='btn' href='#contact'>Ask</a></p></section>"
        "<div data-gate='record'><section id='correction'><p>the first and only "
        "signal</p></section><section id='scorer'></section></div></main>"
        "<footer id='contact' data-gate='service'><p>Contact.</p></footer>"
        "</body></html>")


def _bad():
    g = _good()
    plain = "A plain sentence."
    sub = lambda a, b: g.replace(a, b, 1)
    return [
        ({"1b"}, sub(plain, "The first plain sentence.")),
        ({"1b"}, sub(plain, "A verified plain sentence.")),
        ({"1b"}, sub(plain, "An audit of the sentence.")),
        ({"1b"}, sub(plain, "Outside auditors check it.")),
        ({"1b"}, sub(plain, "Verification is offered.")),
        ({"1b"}, sub(plain, "Nobody else does this.")),
        ({"1b"}, sub(plain, "There is a gap in the market.")),
        ({"1b"}, sub("<title>VALID review</title>", "<title>The first review</title>")),
        ({"1b"}, sub("</head>", "<meta name='description' content='the only review'></head>")),
        ({"1c"}, sub(plain, "A plain sentence on 42% of series.")),
        ({"1c"}, sub(plain, "A plain sentence: nine of twelve.")),
        ({"1c"}, sub(plain, "Then 95% of backtests fail.")),
        ({"1c"}, sub(plain, "We served 12 clients.")),
        ({"1c"}, sub(plain, "Of 278 variants.")),
        ({"1c", "1d"}, sub("200 daily", "98% daily")),
        ({"1d"}, sub("under canonical CSCV", "not under canonical CSCV")),
        ({"1d"}, sub("beyond each coin's average move", "at all")),
        ({"1c", "1d"}, sub("data-claim='cost'", "data-claim='costs'")),
        ({"1d"}, sub("<span>165 of 278", "<span><b data-claim='auc'>x</b>165 of 278")),
        ({"1e"}, sub(plain, "A plain sentence — with a dash.")),
        ({"1f", "1j"}, sub("only signal", "only signal at $500")),
        ({"1f", "1j"}, sub("only signal", "only signal, pricing on request")),
        ({"1g"}, sub(REQUIRED_SERVICE[7] + ".", "")),
        ({"1h"}, sub("<main>", "<main><p>Loose text.</p>")),
        ({"1h", "1j"}, sub("<div data-gate='record'>", "<div data-gate='recrd'>")),
        ({"5c", "1j"}, sub("id='scorer'", "class='scorer'")),
        ({"5c"}, sub("<p>Contact.</p>", "<p id='review'>Contact.</p>")),
        ({"1b"}, sub(plain, "It ends in an audit opinion.")),
        ({"1b"}, sub(plain, "It gives reasonable assurance.")),
        ({"1b"}, sub(plain, "An independent reviewer found the errors.")),
        ({"1b"}, sub(plain, "SEC-registered and FINRA-regulated.")),
        ({"1b"}, sub(plain, "Our pre-registered plan is first.")),
        ({"1c"}, sub(plain, "Our SR2.02 and CAGR124% stand.")),
        ({"1c"}, sub(plain, "An AUC of .95 here.")),
        ({"1c"}, sub(plain, "It passes twice as often.")),
        ({"1d"}, sub(CLAIM_HREFS["auc"][0], _CL + "e13-elsewhere")),
        ({"1i"}, sub("<p>Contact.</p>", "<p><a class='btn' href='#x'>Buy</a></p>")),
        ({"1j"}, sub("only signal", "only signal changed")),
        ({"1j"}, sub("<p>A plain sentence.</p>",
              "<p>A plain sentence.</p><span data-gate='record'>The first audit: "
              "95% fail.</span>")),
        ({"1j", "5c"}, sub("<section id='scorer'></section>", "")),
        ({"1j"}, sub("<section id='correction'><p>", "<section id='correction'><p><a href='https://example.com'>known issues</a> ")),
        ({"1b"}, sub(plain, "We are US registered.")),
        ({"1b"}, sub(plain, "We are fully licensed.")),
        ({"1b"}, sub(plain, "Book a review now.")),
        ({"1i"}, sub("<p>Contact.</p>", "<p><button class='btn'>Go</button></p>")),
        ({"1i"}, sub("<p>A plain sentence.</p>", "<p>A plain sentence.</p><p><a href='mailto:x@y.z'>Mail</a></p>")),
        ({"1b"}, sub(plain, "The leading review service.")),
        ({"1b"}, sub(plain, "A popular overfitting cutoff.")),
        ({"1b"}, sub(plain, "Work published in a journal.")),
        ({"1b"}, sub(plain, "The only plain sentence.")),
        ({"1b"}, sub(plain, "It is illustrative only.")),
        ({"1b"}, sub(plain, "Real fund, real data, illustrative only. Findings, no assurance.")),
        ({"1g"}, sub(REQUIRED_OPEN[5] + ".", "").replace(
            "<p>Contact.</p>", "<details><summary>Q</summary><p>"
            + REQUIRED_OPEN[5] + ".</p></details><p>Contact.</p>", 1)),
        (set(), sub(REQUIRED_SERVICE[2] + ".", "").replace(
            "<p>Contact.</p>", "<details><summary>Q</summary><p>"
            + REQUIRED_SERVICE[2] + ".</p></details><p>Contact.</p>", 1)),
        (set(), sub(REQUIRED_OPEN[5] + ".", "").replace(
            "<p>Contact.</p>", "<details><summary>" + REQUIRED_OPEN[5]
            + ".</summary><p>A.</p></details><p>Contact.</p>", 1)),
        (set(), sub(REQUIRED_OPEN[5] + ".", "").replace(
            "<p>Contact.</p>", "<details open><summary>Q</summary><p>"
            + REQUIRED_OPEN[5] + ".</p></details><p>Contact.</p>", 1)),
    ] + [({"1g"}, sub(REQUIRED_OPEN[5] + ".", "").replace(
            "<p>Contact.</p>", wrap % (REQUIRED_OPEN[5] + ".") + "<p>Contact.</p>", 1))
         for wrap in ("<p hidden>%s</p>", "<p style='display: none'>%s</p>",
                      "<p class='note sr-only'>%s</p>", "<template><p>%s</p></template>",
                      "<div aria-hidden='true'><p>%s</p></div>",
                      "<p class='visually-hidden'>%s</p>", "<p style='opacity: 0'>%s</p>",
                      "<p style='font-size:0px'>%s</p>",
                      "<details><summary>Q</summary><details><summary>%s</summary>"
                      "</details></details>")
    ] + [(set(), sub(REQUIRED_OPEN[5] + ".", "").replace(
            "<p>Contact.</p>", wrap % (REQUIRED_OPEN[5] + ".") + "<p>Contact.</p>", 1))
         for wrap in ("<p style='opacity:0.9'>%s</p>", "<p style='font-size:0.95rem'>%s</p>")]


def selftest():
    """Every service gate passes the clean example and fires on its bad ones."""
    import contextlib
    import io

    import hashlib
    good_rec = parse_html(_good()).records
    good_sha = hashlib.sha256(norm(good_rec[0]["text"]).encode("utf-8")).hexdigest()

    def verdicts(doc):
        p = parse_html(doc)
        items = [("index.html", norm(t)) for t in p.out if norm(t)]
        with contextlib.redirect_stdout(io.StringIO()):
            r = run_service(p, items, record_sha=good_sha)
        return {k for k, v in zip(SERVICE_GATES, r) if not v}

    print("\n[selftest] service gates against built-in examples")
    ok = True
    fired = verdicts(_good())
    if fired:
        print("  FAIL clean example rejected by %s" % sorted(fired))
        ok = False
    for want, doc in _bad():
        if doc == _good():
            print("  FAIL a bad example equals the clean one (its anchor text is gone)")
            ok = False
            continue
        got = verdicts(doc)
        good = got == want
        ok = ok and good
        print("  %s want %-10s got %s" % ("PASS" if good else "FAIL",
                                          ",".join(sorted(want)),
                                          ",".join(sorted(got)) or "-"))
    lines = [("README.md:1", "the 9th ACM"), ("README.md:2", "SIGKDD Workshop on ML")]
    with contextlib.redirect_stdout(io.StringIO()):
        caught = not scan(lines, RETIRED, "x", flags=0)
    ok = ok and caught
    print("  %s RETIRED catches the workshop name split across a line break"
          % ("PASS" if caught else "FAIL"))
    dec = js_decode('a \\u2014 b')
    good = "—" in dec
    ok = ok and good
    print("  %s JS escape \\u2014 decoded" % ("PASS" if good else "FAIL"))
    return ok


def dump_service():
    """Print the service copy, one string per line (for canon_check.py)."""
    p = parse_html(open(INDEX, encoding="utf-8").read())
    for t, _, _ in p.service:
        if norm(t):
            print(norm(t))
    return 0


def live_match():
    """The deployed page must be the working tree's page, byte for byte."""
    print("\n[gate 6] deployed bytes == working tree")
    ok = True
    for label, local, remote in (
        ("index.html", INDEX, LIVE_SRC["index"]),
        ("scorer.js", SCORER, LIVE_SRC["scorer"]),
        ("valid_checklist_onepage.pdf", ONEPAGE, LIVE_SRC["onepage"]),
    ):
        a = open(local, "rb").read()
        b = open(remote, "rb").read()
        good = a == b
        ok = ok and good
        print("  %s %-30s local %d B / live %d B"
              % ("PASS" if good else "FAIL", label, len(a), len(b)))
    return ok


def main(argv):
    global LIVE_SRC
    LIVE_SRC = None
    if len(argv) > 1 and argv[1] == "--live":
        base = argv[2] if len(argv) > 2 else "https://orcajae.github.io/valid-framework"
        print("live mode: %s" % base)
        LIVE_SRC = fetch(base)

    if len(argv) > 1 and argv[1] == "--dump-service":
        return dump_service()
    if not selftest():
        print("\nSELFTEST FAILED: the service gates cannot be trusted; stop.")
        return 2
    items = collect(LIVE_SRC)
    index = LIVE_SRC["index"] if LIVE_SRC else INDEX
    results = [
        scan(items, BANNED, "gate 1 banned marketing terms"),
        scan(items, RETIRED, "gate 2 retired values", flags=0),
        canon(items),
    ]
    results += run_service(parse_html(open(index, encoding="utf-8").read()), items)
    if LIVE_SRC:
        results.append(live_match())
    else:
        results += [
            scan(raw_sources(), RETIRED, "gate 2b retired values, raw source",
                 flags=0),
            links(),
            anchors(),
        ]
    print("\n%s  (%d/%d gates)" % ("ALL PASS" if all(results) else "FAILED",
                                   sum(results), len(results)))
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
