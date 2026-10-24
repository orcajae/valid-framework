"""Gate for the Tier 2 sample report page, docs/sample-report.html.

The page shows a demonstration review for a fictional fund on synthetic data
(examples/t2_sample). It is not a study, and its numbers must never mix with
the published study numbers. Every check is exact.

  s1  wording: the service-copy banned list of gate_landing.py, its marketing
      list, study and trading words, the study verdict sentences, retired
      values, prices, em-dash; verdict words ("within the bound", "adequate",
      "weak power", ...) only inside bound verdicts or the rule list of
      section 4 (data-rules), so a descriptive output cannot be given a verdict
  s2  numbers: after the listed design phrases of the pre-registration are
      removed (ALLOWED_PHRASES), every digit left must sit in a bound element
      (data-v), a fingerprint (code.hash) or a section number (.rnum, which
      must run 1, 2, 3 ...); number words only in listed phrases
  s3  every data-v element shows exactly the value its path gives in
      examples/t2_sample/results/decision.json (or the file hashes and the
      pre-registration commit); every result is shown at least once outside
      any hidden element; in a table, a bound cell sits in the row and column
      its path names; a verdict carries the class of its outcome
  s4  the published files agree: sha256 of PREREGISTRATION.md, run.py and
      rows.csv equal decision.json, and decision.json is recomputed from
      rows.csv (counts, Wilson intervals, verdicts, descriptive outputs)
  s5  required wording, visible without any action: the sample line in a
      banner, in every section and in the footer; the nature of a review; the
      disclosure and the fee terms exactly as on the service page
  s6  no button-style call to action; mail links only in the footer
  s7  local links and in-page anchors resolve; the service page links here;
      neither stylesheet hides anything outside print rules

selftest() runs first on built-in examples; if it fails the script stops with
exit status 2.

usage: python3 -I -B scripts/gate_sample_report.py [--live [BASE_URL]]
"""
import csv
import hashlib
import json
import math
import os
import re
import sys
from html.parser import HTMLParser

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gate_landing as gl  # noqa: E402

ROOT = os.path.dirname(HERE)
PAGE = os.path.join(ROOT, "docs", "sample-report.html")
INDEX = os.path.join(ROOT, "docs", "index.html")
CSS = [os.path.join(ROOT, "docs", n) for n in ("style.css", "sample-report.css")]
SAMPLE = os.path.join(ROOT, "examples", "t2_sample")
RESULTS = os.path.join(SAMPLE, "results")

SAMPLE_LINE = gl.SAMPLE_LINE
ARMS = ("N0", "N1", "P1", "P2")
GATES = ("G1", "G2", "G3", "G4", "Gall")
SIZE_MAX, POWER_MIN, REPS = 15, 100, 200       # PREREGISTRATION.md section 4

# Fee terms and disclosure, word for word as on the service page.
REQUIRED = [gl.REQUIRED_SERVICE[0], gl.REQUIRED_SERVICE[5], gl.REQUIRED_SERVICE[6],
            gl.REQUIRED_SERVICE[7], "Fees are agreed in writing before any result exists.",
            "You can correct facts, not conclusions."]

SAMPLE_BANNED = (r"stud(?:y|ies)|research finding\w*|our research|evidence shows"
                 r"|buy\w*|sell\w*|does not work|doesn't work|useless|industry|most funds"
                 r"|everyone|always|never fails|proves?|proven")
# the verdict sentences of the published studies (E15, E17) do not belong here
STUDY_SENTENCES = (r"|".join(re.escape(x) for x in (gl.REQUIRED_SERVICE[3], gl.REQUIRED_SERVICE[4]))
                   + r"|low PBO\b.*\b(?:does not|doesn't|cannot) show"
                   + r"|thresholds? (?:are|is) not (?:a )?test levels?")
VERDICT_WORDS = (r"within (?:the )?bounds?|exceeds? (?:the )?bounds?|adequate\w*"
                 r"|weak power|power is weak|size is|sufficient\w*|acceptabl\w*"
                 r"|conservative\w*|small enough|(?:meets|fails) the (?:size|power) rule"
                 r"|safe|reliabl\w*|good power|low power|strong power")

# Design figures as the pre-registration states them, the run date, the
# cross-references and the reproduction notes. Removed before the digits are
# read; a bare number anywhere else fails.
ALLOWED_PHRASES = [
    "at most 15 of 200", "at most 15 of the 200", "at least 100 of 200",
    "at least 100 of the 200",                                        # bounds [4]
    "on 200 series in each of four groups", "out of 200", "of 200",  # group size [3]
    "200 series per group", "j from 0 to 199",                        # [3]
    "Trailing 5, 20 and 60-day return sums",                          # [1]
    "refitted every 100 days", "the earliest 400 usable days",        # [1]
    "About 536 test days per series",                                 # [1]
    "close to 0.5",                                                   # band [1]
    "18 bp for a round trip", "The fund's own cost is 18 bp",         # [1]
    "PBO < 0.20", "PBO below 0.20", "p < 0.05", "AUC > 0.55", "ratio > 0",  # [2]
    "configuration > 0.55", "configuration > 0",                      # [2]
    "16 contiguous blocks, all 12,870 splits", "shuffled 100 times",  # [2]
    "95% Wilson interval",                                            # [4]
    "daily volatility 0.02, ARCH 0.08, GARCH 0.90",                   # [3]
    "(5 degrees of freedom)", "over 1,000 trading days after a burn-in of 250 days",
    "persistence 0.9", "daily drift of 0.0005",                       # [3]
    "about 0.04 (weaker) or 0.08 (stronger)",                         # b [3]
    "at least 20 days",                                               # rotation [5]
    "0 bp", "9 bp", "18 bp", "36 bp",                                 # cost rows [5]
    "C 0.1", "C 1.0", "band 0.0", "band 0.02",                        # configurations [1]
    "GARCH(1,1)", "AR(1)",                                            # model names
    "24 October 2026",                                                # run date
    # time of the private pre-registration commit 5d7e0b1 (author and committer
    # date 2026-10-24 21:04:49 +0900, valid-journal); the run start is bound
    # from results/lock.json
    "at 21:04:49 KST",
    "section 1", "section 7", "section 9",                            # cross-references
    "about 20 seconds", "Python 3.13, numpy 2.5.3, pandas 3.0.5 and scipy 1.18.1",
    "--workers 2", "jwim1101", "t2_sample", "sha256", "shasum -a 256", "python3", "L2",
    "G1 to G4", "G1, G2, G3 and G4", "G1", "G2", "G3", "G4", "T1 review", "T1", "T2", "Tier 2",
]
NUMBER_WORD_PHRASES = [
    "four pass-or-fail", "four checks", "all four", "four groups", "two feature sets",
    "two strengths", "two trading bands", "Both strengths", "the eight configurations",
    "eight configurations", "Eight:", "best of the eight", "five-day", "next five",
    "two workers", "twelve VALID", "one daily", "one asset",
    "One pipeline, one grid, one label horizon and one cost rule",
    "one generator", "one of the fund's own features", "more than one way",
    "below zero", "above zero", "the reported one", "the published one", "third party",
    "one possible cause", "at least half", "seven seconds later",
]
ARM_HEADERS = {"No edge": "N1", "No edge, drift kept": "N0",
               "Weaker planted edge": "P1", "Stronger planted edge": "P2"}
VERDICT_ARM = {"size_N1": "N1", "size_N0_drift_kept": "N0", "power_P2": "P2"}
ROW_GATE = {"All four": "Gall", "Rotated labels": "G2c"}
HASH_ROWS = {"PREREGISTRATION.md": "sha:prereg", "run.py": "sha:runpy",
             "results/rows.csv": "sha:rows"}

GOOD_TAG = "rsec"


class Page(HTMLParser):
    """Visible text and bound values, with where each one sits."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.texts = []           # dicts: text, path, hash, hidden, rules, rnum
        self.bound = []           # dicts: path, text, hidden, cls, cell
        self.sections = 0
        self.rnums = []
        self.tags = []
        self.banner = []
        self.footer_tag = False
        self.btns = 0
        self.mailto_outside = 0
        self.hrefs = []
        self.ids = []
        self.errors = []
        self._stack = []
        self._skip = 0
        self._table = None        # {"heads": [...], "in_head": bool}
        self._row = None          # {"th": str, "col": int}
        self._cell = None         # {"text": str, "head": bool}

    def _up(self, key):
        for e in reversed(self._stack):
            if e[key] is not None and e[key] is not False:
                return e[key]
        return None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ("script", "style"):
            self._skip += 1
        if a.get("id"):
            self.ids.append(a["id"])
        classes = set((a.get("class") or "").split())
        style = re.sub(r"\s+", "", (a.get("style") or "").lower())
        hidden = bool(tag == "template" or "hidden" in a
                      or (a.get("aria-hidden") or "").lower() == "true"
                      or (tag == "details" and "open" not in a)
                      or re.search(r"display:none|visibility:hidden|opacity:0*\.?0+(?![.\d]*[1-9])"
                                   r"|font-size:0(?![.\d]*[1-9])", style)
                      or classes & {"sr-only", "visually-hidden"})
        if tag == "table":
            self._table = {"heads": [], "in_head": False}
        if tag == "thead" and self._table:
            self._table["in_head"] = True
        if tag == "tr":
            self._row = {"th": "", "col": -1}
        if tag in ("th", "td") and self._row is not None:
            self._row["col"] += 1
            self._cell = {"text": "", "head": tag == "th"}
        bidx = None
        if "data-v" in a:
            if self._up("bidx") is not None:
                self.errors.append("nested data-v %r" % a["data-v"])
            cell = None
            if self._table is not None and self._row is not None and not self._table["in_head"]:
                cell = (self._table, self._row, self._row["col"])
            self.bound.append({"path": a["data-v"], "text": "", "cls": classes,
                               "hidden": bool(hidden or self._up("hidden")), "cell": cell})
            bidx = len(self.bound) - 1
        kind = "banner" if "data-sample-banner" in a else ("tag" if "sample-tag" in classes else None)
        if tag == "section" and GOOD_TAG in classes:
            self.sections += 1
        if tag in ("a", "button") and "btn" in classes:
            self.btns += 1
        if tag == "a" and a.get("href"):
            self.hrefs.append(a["href"])
            if a["href"].startswith("mailto:") and not any(e["tag"] == "footer" for e in self._stack):
                self.mailto_outside += 1
        if tag not in gl.VOID_TAGS:
            self._stack.append({"tag": tag, "bidx": bidx, "hash": "hash" in classes and tag == "code",
                                "hidden": hidden, "kind": kind, "rules": "data-rules" in a,
                                "rnum": "rnum" in classes, "buf": []})
        if tag == "meta" and a.get("name") == "viewport":
            return
        for k, v in attrs:
            if k in gl.VISIBLE_ATTRS and v:
                self._text(v)

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1
        if tag in ("th", "td") and self._cell is not None:
            t = gl.norm(self._cell["text"])
            if self._table is not None and self._table["in_head"]:
                self._table["heads"].append(t)
            elif self._cell["head"] and self._row is not None:
                self._row["th"] = t
            self._cell = None
        if tag == "thead" and self._table:
            self._table["in_head"] = False
        if tag == "table":
            self._table = None
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i]["tag"] == tag:
                done = self._stack[i:]
                del self._stack[i:]
                for e in done:
                    if e["kind"] == "tag" and not e["hidden"]:
                        self.tags.append(gl.norm(" ".join(e["buf"])))
                    if e["rnum"]:
                        self.rnums.append(gl.norm(" ".join(e["buf"])))
                break

    def _text(self, t):
        hidden = bool(self._up("hidden"))
        b = self._up("bidx")
        self.texts.append({"text": t, "path": None if b is None else self.bound[b]["path"],
                           "hash": bool(self._up("hash")), "hidden": hidden,
                           "rules": bool(self._up("rules")), "rnum": bool(self._up("rnum"))})
        if b is not None:
            self.bound[b]["text"] += t
        if self._cell is not None:
            self._cell["text"] += " " + t
        for e in self._stack:
            if e["kind"] or e["rnum"]:
                e["buf"].append(t)
        kinds = [e["kind"] for e in self._stack if e["kind"]]
        if "banner" in kinds and not hidden:
            self.banner.append(t)
        if "tag" in kinds and not hidden and any(e["tag"] == "footer" for e in self._stack):
            self.footer_tag = True

    def handle_data(self, data):
        if not self._skip and data.strip():
            self._text(data)


def parse(doc):
    p = Page()
    p.feed(doc)
    p.close()
    return p


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ---------------------------------------------------------------- expected values
def fmt_cfg(c):
    fs, cc, band = c.split("|")
    return "%s · C %s · band %s" % (fs, cc.split("=")[1], band.split("=")[1])


def load_expected():
    """Path -> displayed text, from the published files."""
    dec = json.load(open(os.path.join(RESULTS, "decision.json")))
    lock = json.load(open(os.path.join(RESULTS, "lock.json")))
    from datetime import datetime, timedelta, timezone
    start = datetime.fromisoformat(lock["start"]).astimezone(timezone(timedelta(hours=9)))
    out = {"commit": lock["prereg_commit"], "run_start": start.strftime("%H:%M:%S KST"),
           "sha:prereg": dec["prereg_sha256"],
           "sha:runpy": dec["run_py_sha256"], "sha:rows": dec["rows_sha256"]}
    for g, arms in dec["gates"].items():
        for a, v in arms.items():
            out["gates.%s.%s.k" % (g, a)] = str(v["k"])
            out["gates.%s.%s.pct" % (g, a)] = "%.1f%%" % (100 * v["rate"])
            lo, hi = v["wilson95"]
            out["gates.%s.%s.ci" % (g, a)] = "%.1f–%.1f%%" % (100 * lo, 100 * hi)
    for g, vs in dec["verdicts"].items():
        for k, v in vs.items():
            out["verdicts.%s.%s" % (g, k)] = v
    d = dec["descriptive"]
    for col, arms in d["G4_by_cost"].items():
        for a, v in arms.items():
            out["cost.%s.%s" % (col, a)] = str(v)
    for a, v in d["mean_PBO"].items():
        out["pbo.%s" % a] = "%.3f" % v
    for a, v in d["mean_AUC_selected"].items():
        out["auc.%s" % a] = "%.3f" % v
    for a, cs in d["selected_config_counts"].items():
        for c, v in cs.items():
            out["sel.%s.%s" % (a, c)] = str(v)
    return out


def required_paths(exp):
    return sorted(exp)


def path_place(path):
    """(row key, arm) a bound table cell must sit at."""
    f = path.split(".")
    if f[0] == "gates":
        return f[1], f[2]
    if f[0] == "verdicts":
        return f[1], VERDICT_ARM[f[2]]
    if f[0] == "cost":
        return f[1], f[2]
    if f[0] in ("pbo", "auc"):
        return f[0], f[1]
    if f[0] == "sel":
        return "sel." + ".".join(f[2:]), f[1]
    if path.startswith("sha:"):
        return path, None
    return None


def row_key(th):
    m = re.match(r"(G[1-4]) ·", th)
    if m:
        return m.group(1)
    if th in ROW_GATE:
        return ROW_GATE[th]
    m = re.fullmatch(r"(\d+) bp", th)
    if m:
        return "G4_cost_%sbp" % m.group(1)
    if th == "PBO":
        return "pbo"
    if th == "AUC of the reported configuration":
        return "auc"
    if th in HASH_ROWS:
        return HASH_ROWS[th]
    m = re.fullmatch(r"([AB]) · C ([\d.]+) · band ([\d.]+)", th)
    if m:
        return "sel.%s|C=%s|band=%s" % m.groups()
    return None


def col_arm(heads, col):
    if col < len(heads) and heads[col] in ARM_HEADERS:
        return ARM_HEADERS[heads[col]]
    if col < len(heads) and heads[col] == "Verdict" and col >= 1:
        return ARM_HEADERS.get(heads[col - 1])
    return None


# ---------------------------------------------------------------- gates
def g_words(p):
    items = [("sample-report.html", gl.strip_sample_line(t["text"])) for t in p.texts
             if gl.strip_sample_line(t["text"])]
    free = [("sample-report.html", gl.norm(t["text"])) for t in p.texts
            if gl.norm(t["text"]) and not t["rules"]
            and not (t["path"] or "").startswith("verdicts.")]
    return all([
        gl.scan(items, gl.SERVICE_BANNED, "s1a banned in sample copy"),
        gl.scan(items, gl.BANNED, "s1b banned marketing terms"),
        gl.scan(items, SAMPLE_BANNED, "s1c study and trading words"),
        _phrases(items, STUDY_SENTENCES, "s1d study verdict sentences"),
        gl.scan(items, gl.RETIRED, "s1e retired values", flags=0),
        gl.scan(items, gl.PRICES, "s1f no prices"),
        gl.scan(free, VERDICT_WORDS, "s1g verdict words outside verdicts and rules"),
        _emdash(items),
    ])


def _phrases(items, pattern, name):
    hits = [s for _, s in items if re.search(pattern, s)]
    print("\n[%s]" % name)
    for s in hits:
        print("  FAIL %s" % s[:150])
    if not hits:
        print("  PASS  0 hits")
    return not hits


def _emdash(items):
    print("\n[s1h] no em-dash")
    hits = [s for _, s in items if "—" in s]
    for s in hits:
        print("  FAIL %s" % s[:150])
    if not hits:
        print("  PASS  0 hits")
    return not hits


def g_numbers(p):
    print("\n[s2] numbers are bound values, fingerprints, section numbers or listed design phrases")
    bad = []
    for t in p.texts:
        if t["path"] is not None or t["hash"] or t["rnum"]:
            continue
        s = gl.norm(t["text"])
        rest = s
        for ph in sorted(ALLOWED_PHRASES, key=len, reverse=True):
            rest = rest.replace(ph, " ")
        for tok in re.findall(r"\d", rest):
            bad.append((tok, s))
        words = s
        for ph in NUMBER_WORD_PHRASES:
            words = re.sub(re.escape(ph), " ", words, flags=re.I)
        for m in gl.NUMBER_WORDS.finditer(words):
            bad.append((m.group(0), s))
    want = [str(i) for i in range(1, len(p.rnums) + 1)]
    if p.rnums != want:
        bad.append(("section numbers %s" % p.rnums, "want %s" % want))
    seen = set()
    for tok, s in bad:
        if (tok, s) not in seen:
            seen.add((tok, s))
            print("  FAIL %r in: %s" % (tok, s[:150]))
    if not bad:
        print("  PASS  %d section numbers in order" % len(p.rnums))
    return not bad


def g_bound(p, exp):
    print("\n[s3] bound values equal the published outputs, sit in their cells, all shown")
    ok = True
    for e in p.errors:
        print("  FAIL %s" % e)
        ok = False
    shown = set()
    for b in p.bound:
        want = exp.get(b["path"])
        got = gl.norm(b["text"])
        if want is None or got != want:
            print("  FAIL %-34s got %r want %r" % (b["path"], got, want))
            ok = False
        if not b["hidden"]:
            shown.add(b["path"])
        if b["path"].startswith("verdicts."):
            cls = "v-bad" if ("exceeds" in got or "weak" in got) else "v-ok"
            if cls not in b["cls"] or ({"v-bad", "v-ok"} - {cls}) & b["cls"]:
                print("  FAIL %-34s class %s (want %s)" % (b["path"], sorted(b["cls"]), cls))
                ok = False
        if b["cell"] is not None:
            table, row, col = b["cell"]
            place = path_place(b["path"])
            have = (row_key(row["th"]), col_arm(table["heads"], col))
            good = place is not None and have[0] == place[0] and (place[1] is None or have[1] == place[1])
            if not good:
                head = table["heads"][col] if col < len(table["heads"]) else col
                print("  FAIL %-34s placed at row %r, column %r" % (b["path"], row["th"], head))
                ok = False
    for t in p.texts:
        if t["hash"] and t["path"] is None:
            print("  FAIL hash text not bound: %s" % t["text"][:80])
            ok = False
    missing = [k for k in required_paths(exp) if k not in shown]
    for k in missing:
        print("  FAIL not shown (or only hidden): %s" % k)
    ok = ok and not missing
    if ok:
        print("  PASS  %d bound elements, %d of %d results shown" % (len(p.bound), len(shown), len(exp)))
    return ok


def wilson(k, n, z=1.96):
    q = k / n
    d = 1 + z * z / n
    c = (q + z * z / (2 * n)) / d
    h = z * math.sqrt(q * (1 - q) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def g_files():
    print("\n[s4] published files agree")
    dec = json.load(open(os.path.join(RESULTS, "decision.json")))
    ok = True

    def check(label, good):
        nonlocal ok
        ok = ok and good
        print("  %s %s" % ("PASS" if good else "FAIL", label))

    check("PREREGISTRATION.md sha256", sha256(os.path.join(SAMPLE, "PREREGISTRATION.md")) == dec["prereg_sha256"])
    check("run.py sha256", sha256(os.path.join(SAMPLE, "run.py")) == dec["run_py_sha256"])
    check("rows.csv sha256", sha256(os.path.join(RESULTS, "rows.csv")) == dec["rows_sha256"])
    for f in ("lock.json", "run_log.json"):
        j = json.load(open(os.path.join(RESULTS, f)))
        check("%s hashes" % f, j["prereg_sha256"] == dec["prereg_sha256"]
              and j["run_py_sha256"] == dec["run_py_sha256"])
    log = json.load(open(os.path.join(RESULTS, "run_log.json")))
    check("run_log rows sha256 and count", log["rows_sha256"] == dec["rows_sha256"] and log["rows"] == 4 * REPS)
    rows = list(csv.DictReader(open(os.path.join(RESULTS, "rows.csv"), newline="")))
    check("rows: %d = 4 arms x %d" % (len(rows), REPS), len(rows) == 4 * REPS
          and all(sum(r["arm"] == a for r in rows) == REPS for a in ARMS))
    good = True
    for g in GATES + ("G2c",):
        for a in ARMS:
            k = sum(int(r[g]) for r in rows if r["arm"] == a)
            v = dec["gates"][g][a]
            lo, hi = wilson(k, REPS)
            good = good and v["k"] == k and v["n"] == REPS and abs(v["rate"] - k / REPS) < 1e-12 \
                and abs(v["wilson95"][0] - lo) < 1e-12 and abs(v["wilson95"][1] - hi) < 1e-12
    check("counts and Wilson intervals recomputed from rows.csv", good)
    good = True
    for g in GATES:
        n1, n0, p2 = (dec["gates"][g][a]["k"] for a in ("N1", "N0", "P2"))
        sv = lambda k: "size within bound" if k <= SIZE_MAX else "size exceeds bound"
        want = {"size_N1": sv(n1), "size_N0_drift_kept": sv(n0),
                "power_P2": "adequate power" if p2 >= POWER_MIN else "weak power"}
        good = good and dec["verdicts"][g] == want
    check("verdicts recomputed with bounds %d and %d of %d" % (SIZE_MAX, POWER_MIN, REPS), good)
    run_py = open(os.path.join(SAMPLE, "run.py"), encoding="utf-8").read()
    check("bounds match run.py", "SIZE_MAX, POWER_MIN = %d, %d" % (SIZE_MAX, POWER_MIN) in run_py
          and "REPS = %d" % REPS in run_py)
    d = dec["descriptive"]
    good = True
    for col, arms in d["G4_by_cost"].items():
        for a in ARMS:
            good = good and arms[a] == sum(int(r[col]) for r in rows if r["arm"] == a)
    for a in ARMS:
        sub = [r for r in rows if r["arm"] == a]
        pbo = sum(int(r["pbo_k"]) / int(r["pbo_K"]) for r in sub) / len(sub)
        auc = sum(float(r["auc_sel"]) for r in sub) / len(sub)
        good = good and abs(d["mean_PBO"][a] - pbo) < 1e-9 and abs(d["mean_AUC_selected"][a] - auc) < 1e-9
        cnt = {}
        for r in sub:
            cnt[r["sel_config"]] = cnt.get(r["sel_config"], 0) + 1
        good = good and d["selected_config_counts"][a] == cnt
    check("descriptive outputs recomputed from rows.csv", good)
    return ok


def g_required(p):
    print("\n[s5] required wording, visible")
    ok = True
    shown = " ".join(gl.norm(t["text"]) for t in p.texts if not t["hidden"])

    def check(label, good):
        nonlocal ok
        ok = ok and good
        print("  %s %s" % ("PASS" if good else "FAIL", label))

    check("banner carries the sample line", gl.norm(" ".join(p.banner)) == SAMPLE_LINE)
    check("sample line in every section (%d sections, %d tags)" % (p.sections, len(p.tags)),
          p.sections >= 1 and len(p.tags) == p.sections + 1 and all(t == SAMPLE_LINE for t in p.tags))
    check("sample line in the footer", p.footer_tag)
    for ph in REQUIRED:
        check(ph, ph in shown)
    return ok


def g_cta(p):
    print("\n[s6] calls to action: %d buttons, %d mail links outside the footer"
          % (p.btns, p.mailto_outside))
    good = p.btns == 0 and p.mailto_outside == 0
    print("  %s" % ("PASS" if good else "FAIL"))
    return good


def css_hides(text):
    """Hiding declarations outside @media print blocks."""
    out, i = [], 0
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    while True:
        m = re.search(r"@media\s+print\s*\{", text[i:])
        if not m:
            out.append(text[i:])
            break
        out.append(text[i:i + m.start()])
        j, depth = i + m.end(), 1
        while depth and j < len(text):
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        i = j
    rest = re.sub(r"\s+", "", "".join(out).lower())
    return re.findall(r"[^{}]*\{[^}]*(?:display:none|visibility:hidden|opacity:0*\.?0+(?![.\d]*[1-9])"
                      r"|font-size:0(?![.\d]*[1-9]))[^}]*\}", rest)


def g_links(p, css_texts=None):
    print("\n[s7] links resolve; stylesheets hide nothing")
    ok = True
    for u in p.hrefs:
        if u.startswith("#"):
            good = u[1:] in p.ids
        elif re.match(r"^(https?:|mailto:|//)", u):
            continue
        else:
            path, _, frag = u.partition("#")
            good = os.path.exists(os.path.join(ROOT, "docs", path))
            if good and frag and path.endswith(".html"):
                good = ('id="%s"' % frag) in open(os.path.join(ROOT, "docs", path), encoding="utf-8").read()
        ok = ok and good
        print("  %s %s" % ("PASS" if good else "FAIL", u))
    dup = sorted({i for i in p.ids if p.ids.count(i) > 1})
    for i in dup:
        print("  FAIL duplicate id %r" % i)
    if css_texts is None:
        css_texts = {os.path.basename(c): open(c, encoding="utf-8").read() for c in CSS}
        good = 'href="sample-report.html"' in open(INDEX, encoding="utf-8").read()
        print("  %s index.html links to sample-report.html" % ("PASS" if good else "FAIL"))
        ok = ok and good
    for name, text in css_texts.items():
        hits = css_hides(text)
        for h in hits:
            print("  FAIL %s hides: %s" % (name, h[:120]))
        ok = ok and not hits
    return ok and not dup


# ---------------------------------------------------------------- selftest
def _good():
    tag = '<p class="sample-tag">%s</p>' % SAMPLE_LINE
    req = " ".join(REQUIRED)
    return (
        "<html><head><title>Sample report</title></head><body>"
        '<div class="sample-bar" data-sample-banner>%s</div>' % SAMPLE_LINE
        + '<section class="rsec" id="s1"><h2><span class="rnum">1</span> Scope</h2>' + tag
        + "<p>%s</p>" % req
        + '<p>A check passes at most 15 of 200 series; <span data-v="gates.G1.N1.k">29</span> of 200 did.</p>'
        + '<ul data-rules><li>Size within the bound if it passes at most 15 of 200.</li></ul>'
        + '<table><thead><tr><th>Check</th><th>No edge</th><th>Verdict</th><th>Stronger planted edge</th></tr></thead>'
        + '<tbody><tr><th>G1 · PBO &lt; 0.20</th><td><span data-v="gates.G1.N1.k">29</span></td>'
        + '<td><span data-v="verdicts.G1.size_N1" class="v-bad">size exceeds bound</span></td>'
        + '<td><span data-v="gates.G1.P2.k">34</span></td></tr></tbody></table>'
        + '<p>Commit <code class="hash"><span data-v="commit">abc</span></code>.</p>'
        + '<p><a href="#s1">Top</a></p></section>'
        + '<footer>' + tag + '<a href="mailto:x@y.z">Mail</a></footer></body></html>')


GOOD_CSS = {"x.css": ".a { color: red; }\n@media print { .toc { display: none; } }"}


def _bad():
    g = _good()
    sub = lambda a, b: g.replace(a, b, 1)
    plain = "A check passes at most 15 of 200 series;"
    k29 = '<span data-v="gates.G1.N1.k">29</span> of 200 did.'
    swap = (g.replace('<td><span data-v="gates.G1.N1.k">29</span></td>', "@@A@@", 1)
            .replace('<td><span data-v="gates.G1.P2.k">34</span></td>', '<td><span data-v="gates.G1.N1.k">29</span></td>', 1)
            .replace("@@A@@", '<td><span data-v="gates.G1.P2.k">34</span></td>', 1))
    return [
        ({"s1"}, sub(plain, "The first check passes at most 15 of 200 series;")),
        ({"s1"}, sub(plain, "Our study: a check passes at most 15 of 200 series;")),
        ({"s1"}, sub(plain, "A verified check passes at most 15 of 200 series;")),
        ({"s1"}, sub(plain, "A check passes at most 15 of 200 series, pricing on request;")),
        ({"s1"}, sub(plain, "A check, at most 15 of 200 series — so;")),
        ({"s1"}, sub(plain, "A check with returns passes at most 15 of 200 series;")),
        ({"s1"}, sub(plain, "Sell a check that passes at most 15 of 200 series;")),
        ({"s1"}, sub(plain, "PBO does not work; a check passes at most 15 of 200 series;")),
        ({"s1"}, sub(plain, "PBO thresholds are not test levels. A check passes at most 15 of 200 series;")),
        ({"s1"}, sub(k29, k29 + " It stays within the bound.")),
        ({"s1"}, sub(k29, k29 + " Its power is adequate.")),
        ({"s1"}, sub(k29, k29 + " That is conservative.")),
        ({"s1"}, sub(k29, k29 + " It meets the size rule.")),
        ({"s1"}, sub(plain, "A low PBO by itself does not show an edge; a check passes at most 15 of 200 series;")),
        ({"s1", "s5"}, g.replace("illustrative only. Findings, no assurance.</div>",
                                 "illustrative only here. Findings, no assurance.</div>", 1)),
        ({"s1", "s2"}, sub(plain, "A check passes at most 15 of 200 series at $500;")),
        ({"s2"}, sub(plain, "27.0% of checks pass at most 15 of 200 series;")),
        ({"s2"}, sub(plain, "A check passes 100 of 200 series;")),
        ({"s2"}, sub(plain, "A check passes at most 15 of 200 series, twice;")),
        ({"s2"}, sub(k29, "29 of 200 did.")),
        ({"s2"}, sub('<span class="rnum">1</span>', '<span class="rnum">3</span>')),
        ({"s3"}, sub('<span data-v="gates.G1.N1.k">29</span> of', '<span data-v="gates.G1.N1.k">30</span> of')),
        ({"s3"}, sub('<span data-v="gates.G1.N1.k">29</span> of', '<span data-v="gates.G9.N1.k">29</span> of')),
        ({"s3"}, sub('<span data-v="commit">abc</span>', "abc")),
        ({"s3"}, sub('<span data-v="gates.G1.N1.k">29</span> of',
                     '<span data-v="gates.G1.N1.k">2<b data-v="commit">abc</b>9</span> of')),
        ({"s3"}, swap),
        ({"s3"}, sub('class="v-bad">size exceeds', 'class="v-ok">size exceeds')),
        ({"s3"}, sub('<td><span data-v="gates.G1.P2.k">34</span></td>',
                     '<td hidden><span data-v="gates.G1.P2.k">34</span></td>')),
        ({"s5"}, sub('<div class="sample-bar" data-sample-banner>%s</div>' % SAMPLE_LINE, "")),
        ({"s5"}, sub('<p class="sample-tag">%s</p>' % SAMPLE_LINE, "")),
        ({"s5"}, g.replace('<footer><p class="sample-tag">%s</p>' % SAMPLE_LINE, "<footer>", 1)),
        ({"s5"}, sub('<p class="sample-tag">%s</p>' % SAMPLE_LINE,
                     '<p class="sample-tag" hidden>%s</p>' % SAMPLE_LINE)),
        ({"s5"}, sub(REQUIRED[3], "")),
        ({"s5"}, sub(REQUIRED[4], "Fees are fixed.")),
        ({"s6"}, sub('<p><a href="#s1">Top</a></p>', '<p><a class="btn" href="#s1">Top</a></p>')),
        ({"s6"}, sub('<p><a href="#s1">Top</a></p>', '<p><a href="mailto:x@y.z">Mail</a></p>')),
        ({"s7"}, sub('<a href="#s1">', '<a href="#nowhere">')),
    ]


def _verdicts(doc, exp, css=None):
    import contextlib
    import io
    p = parse(doc)
    with contextlib.redirect_stdout(io.StringIO()):
        r = {"s1": g_words(p), "s2": g_numbers(p), "s3": g_bound(p, exp),
             "s5": g_required(p), "s6": g_cta(p), "s7": g_links(p, css or GOOD_CSS)}
    return {k for k, v in r.items() if not v}


def selftest():
    global required_paths
    exp = {"gates.G1.N1.k": "29", "gates.G1.P2.k": "34", "commit": "abc",
           "verdicts.G1.size_N1": "size exceeds bound"}
    print("\n[selftest] sample-report gates against built-in examples")
    keep = required_paths
    required_paths = lambda e: sorted(e)
    ok = True
    try:
        fired = _verdicts(_good(), exp)
        if fired:
            print("  FAIL clean example rejected by %s" % sorted(fired))
            ok = False
        cases = [(w, d, None) for w, d in _bad()] + [
            ({"s7"}, _good(), {"x.css": ".sample-tag, .v-bad { display: none }"}),
            ({"s7"}, _good(), {"x.css": ".tile { opacity: 0 }"}),
            ({"s7"}, _good(), {"x.css": ".sample-tag { opacity: .0 }"}),
        ]
        for want, doc, css in cases:
            if doc == _good() and css is None:
                print("  FAIL a bad example equals the clean one")
                ok = False
                continue
            got = _verdicts(doc, exp, css)
            good = got == want
            ok = ok and good
            print("  %s want %-8s got %s" % ("PASS" if good else "FAIL", ",".join(sorted(want)),
                                             ",".join(sorted(got)) or "-"))
    finally:
        required_paths = keep
    return ok


def live(base):
    """The deployed page and stylesheet must equal the working tree, byte for byte."""
    import urllib.request
    print("\n[s8] deployed bytes == working tree (%s)" % base)
    ok = True
    for name in ("sample-report.html", "sample-report.css"):
        local = open(os.path.join(ROOT, "docs", name), "rb").read()
        with urllib.request.urlopen("%s/%s" % (base.rstrip("/"), name), timeout=30) as r:
            remote = r.read()
        good = local == remote
        ok = ok and good
        print("  %s %-20s local %d B / live %d B" % ("PASS" if good else "FAIL", name, len(local), len(remote)))
    return ok


def main(argv):
    if not selftest():
        print("\nSELFTEST FAILED: the sample-report gates cannot be trusted; stop.")
        return 2
    p = parse(open(PAGE, encoding="utf-8").read())
    raw = [("sample-report.html:%d" % i, gl.norm(line))
           for i, line in enumerate(open(PAGE, encoding="utf-8"), 1) if line.strip()]
    results = [
        g_words(p),
        gl.scan(raw, gl.RETIRED, "s1i retired values, raw source", flags=0),
        g_numbers(p),
        g_bound(p, load_expected()),
        g_files(),
        g_required(p),
        g_cta(p),
        g_links(p),
    ]
    if len(argv) > 1 and argv[1] == "--live":
        results.append(live(argv[2] if len(argv) > 2 else "https://orcajae.github.io/valid-framework"))
    print("\n%s  (%d/%d gates)" % ("ALL PASS" if all(results) else "FAILED", sum(results), len(results)))
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
