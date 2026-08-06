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
RETIRED = r"of 75|75 papers|0/75|25/75|64/75|72%|x\.com/jwquant|@jwquant"

# ---------------------------------------------------------------- gate 3
# n=74 canon. Each entry must appear at least once in the landing text.
CANON = [
    (r"\b74\b", "74 empirical papers"),
    (r"\b80\b", "80 surveyed"),
    (r"2\.5\s*/\s*12|2\.5 of 12", "median 2.5 / 12"),
    (r"\b340\b", "340 variants"),
    (r"52%", "52% net-negative"),
    (r"4\.4%", "4.4% exceed the benchmark"),
    (r"0\.917", "benchmark 0.917"),
    (r"9\s*/\s*12|9 of 12", "self-audit 9 / 12"),
]

VISIBLE_ATTRS = {"content", "alt", "aria-label", "title", "value", "placeholder"}
SKIP_TAGS = {"script", "style"}


class Visible(HTMLParser):
    """Text a visitor can read: text nodes plus user-facing attributes."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in SKIP_TAGS:
            self._skip += 1
        for k, v in attrs:
            if k in VISIBLE_ATTRS and v:
                self.out.append(v)

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip and data.strip():
            self.out.append(data)


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
        htmllib.unescape(m.group(1) if m.group(1) is not None else m.group(2))
        for m in re.finditer(r'"((?:[^"\\]|\\.)*)"|\'((?:[^\'\\]|\\.)*)\'', src)
    ]


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
    p = Visible()
    p.feed(open(index, encoding="utf-8").read())
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
    print("\n[gate 3] n=74 canon")
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
    for path in (INDEX, SCORER, README):
        rel = os.path.relpath(path, ROOT)
        for i, line in enumerate(open(path, encoding="utf-8"), 1):
            if line.strip():
                out.append(("%s:%d" % (rel, i), norm(line)))
    return out


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

    items = collect(LIVE_SRC)
    results = [
        scan(items, BANNED, "gate 1 banned marketing terms"),
        scan(items, RETIRED, "gate 2 retired values", flags=0),
        canon(items),
    ]
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
