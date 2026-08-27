#!/usr/bin/env python3
"""Pre-push gates for the landing page (docs/).

This file used to carry the banned-term, retired-value and canon regexes as
literals. It no longer carries any: the lists live in forbidden.json and the
scanning is done by scripts/gate_canon.sh CHECK B, which this script shells out
to. Keeping a second copy of those lists here is the duplication the canon gate
exists to remove — and a scanner holding its own patterns matched itself, which
is why the old gate needed a whitelist.

What stays here is the work no other script does, because it needs the landing
page's structure rather than its bytes:

  * canon presence over the text a visitor can actually read — HTML text nodes
    and user-facing attributes, string literals in scorer.js, and the extracted
    text of the one-page checklist PDF
  * local links and in-page anchors resolve
  * --live: the deployed bytes are the working tree's bytes

Rule 48 normalisation is applied to every extracted string before matching:
newlines collapsed, unicode math letters folded to ASCII, NBSP folded to space.

Exit status is non-zero if any gate fails; every failure prints the offending
line verbatim.
"""
import html as htmllib
import json
import os
import re
import subprocess
import sys
import unicodedata
from html.parser import HTMLParser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(ROOT, "docs", "index.html")
SCORER = os.path.join(ROOT, "docs", "scorer.js")
ONEPAGE = os.path.join(ROOT, "docs", "assets", "valid_checklist_onepage.pdf")
README = os.path.join(ROOT, "README.md")
FORBIDDEN = os.path.join(ROOT, "forbidden.json")
GATE_CANON = os.path.join(ROOT, "scripts", "gate_canon.sh")

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
    s = s.replace(" ", " ")
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


def token_gates():
    """Delegate banned/retired scanning to the canon gate.

    Its CHECK B covers every tracked file, forbidden.json holds the patterns,
    and its landing-scoped rules cover exactly the surface this script used to
    scan on its own.
    """
    print("\n[gate 1+2] banned and retired tokens -> scripts/gate_canon.sh b")
    rc = subprocess.call(["bash", GATE_CANON, "b"])
    print("  %s gate_canon.sh CHECK B exit %d" % ("PASS" if rc == 0 else "FAIL", rc))
    return rc == 0


def canon(items):
    """Every canonical figure the landing page is expected to carry is present.

    The list, and the canonical key each entry stands for, come from
    forbidden.json; the expected value is cross-checked against canonical.json so
    a figure cannot drift here without CHECK A noticing it there.
    """
    rules = json.load(open(FORBIDDEN, encoding="utf-8"))["canon_required"]
    blob = " ".join(s for _, s in items)
    print("\n[gate 3] canon presence (forbidden.json:canon_required)")
    ok = True
    for entry in rules:
        m = re.search(entry["regex"], blob)
        print("  %s %-32s %s" % ("PASS" if m else "FAIL", entry["label"],
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


def live_match(live_src):
    """The deployed page must be the working tree's page, byte for byte."""
    print("\n[gate 6] deployed bytes == working tree")
    ok = True
    for label, local, remote in (
        ("index.html", INDEX, live_src["index"]),
        ("scorer.js", SCORER, live_src["scorer"]),
        ("valid_checklist_onepage.pdf", ONEPAGE, live_src["onepage"]),
    ):
        a = open(local, "rb").read()
        b = open(remote, "rb").read()
        good = a == b
        ok = ok and good
        print("  %s %-30s local %d B / live %d B"
              % ("PASS" if good else "FAIL", label, len(a), len(b)))
    return ok


def main(argv):
    live_src = None
    if len(argv) > 1 and argv[1] == "--live":
        base = argv[2] if len(argv) > 2 else "https://orcajae.github.io/valid-framework"
        print("live mode: %s" % base)
        live_src = fetch(base)

    items = collect(live_src)
    results = [canon(items)]
    if live_src:
        results.append(live_match(live_src))
    else:
        # Token scanning reads the working tree, so it is meaningless against a
        # fetched copy; in live mode the byte comparison covers it instead.
        results += [token_gates(), links(), anchors()]

    print("\n%s  (%d/%d gates)" % ("ALL PASS" if all(results) else "FAILED",
                                   sum(results), len(results)))
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
