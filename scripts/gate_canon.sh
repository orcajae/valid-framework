#!/usr/bin/env bash
#
# gate_canon.sh — three independent checks over the canonical figures.
#
#   CHECK A  canonical.json still matches a fresh derivation      exit 2
#   CHECK B  no retired token survives in tracked content         exit 3
#   CHECK C  every medium carrying a figure is accounted for      exit 4
#
# Each check owns its exit code so a caller can tell them apart. `all` runs
# every check and returns the first failing code, after running all three —
# a B failure must not hide a C failure.
#
# This script holds no patterns and no figures of its own. Everything it matches
# comes from forbidden.json, and everything it compares comes from canonical.json.
# That is deliberate: the previous gate carried its regexes as literals, which
# made the scanner match itself and forced a file whitelist to paper over it.
# With the patterns held as data the whitelist is one file — forbidden.json —
# and CHECK B prints it rather than applying it quietly.
#
# usage:
#   scripts/gate_canon.sh [a|b|c|all]
#   scripts/gate_canon.sh attest "<medium>" "<verified_by>"
#
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WARN_LOG="$ROOT/gate_warn.log"
MODE="${1:-all}"

hr() { printf '%s\n' "------------------------------------------------------------"; }

# ------------------------------------------------------------------ CHECK A
check_a() {
  hr; echo "CHECK A — canonical figures match a fresh derivation"; hr
  python3 - <<'PY'
import json, subprocess, sys
from pathlib import Path

ROOT = Path.cwd()
committed_path = ROOT / "canonical.json"
if not committed_path.exists():
    print("FAIL canonical.json is missing"); sys.exit(2)

proc = subprocess.run([sys.executable, "scripts/derive_canonical.py", "--stdout"],
                      capture_output=True, text=True)
if proc.returncode != 0:
    print("FAIL derive_canonical.py could not run:")
    print(proc.stderr.strip()[-2000:])
    sys.exit(2)

fresh = json.loads(proc.stdout)
committed = json.loads(committed_path.read_text(encoding="utf-8"))

failed = False
for section in ("figures", "manual", "unavailable_sources"):
    a, b = committed.get(section, {}), fresh.get(section, {})
    drift = sorted(set(a) ^ set(b)) + sorted(k for k in set(a) & set(b) if a[k] != b[k])
    if drift:
        failed = True
        print(f"FAIL {section}: {len(drift)} key(s) drifted")
        for k in drift:
            print(f"     - {k}")
            print(f"         committed : {json.dumps(a.get(k, '<absent>'), ensure_ascii=False)}")
            print(f"         derived   : {json.dumps(b.get(k, '<absent>'), ensure_ascii=False)}")
    else:
        print(f"PASS {section}: {len(a)} key(s) identical")

# 판정 1: audit/ is the canonical coding sheet; results/reference/ keeps a mirror
# for REPRODUCE paths. Two copies are tolerated only while they are the same file.
# Compare the FRESHLY hashed values, not the committed ones: two committed
# copies always agree with themselves, so checking those would pass while the
# files on disk had already diverged.
figs = fresh.get("figures", {})
src = figs.get("audit_csv_md5", {}).get("value")
mirror = figs.get("audit_csv_mirror_md5", {}).get("value")
if src is None or mirror is None:
    failed = True
    print("FAIL audit csv md5 keys absent from canonical.json")
elif src != mirror:
    failed = True
    print("FAIL audit coding sheet and its mirror diverged")
    print(f"     audit/literature_audit_80.csv             {src}")
    print(f"     results/reference/literature_audit_80.csv {mirror}")
else:
    print(f"PASS audit coding sheet == mirror  ({src})")

sys.exit(2 if failed else 0)
PY
  return $?
}

# ------------------------------------------------------------------ CHECK B
check_b() {
  hr; echo "CHECK B — retired tokens in tracked content"; hr
  python3 - "$WARN_LOG" <<'PY'
import json, re, subprocess, sys
from pathlib import Path

warn_log = Path(sys.argv[1])
rules = json.loads(Path("forbidden.json").read_text(encoding="utf-8"))

whitelist = set(rules["whitelist"])
print("whitelist (scanned files excluded, printed rather than applied quietly):")
for w in sorted(whitelist):
    print(f"  - {w}")
groups = rules["file_groups"]
print("file groups (scope of the landing-only rules):")
for name, members in sorted(groups.items()):
    print(f"  {name}: {', '.join(members)}")
exempt = [(r["id"], f, r.get("except_reason", ""))
          for r in rules["hard_fail"] + rules["hard_fail_landing"]
          for f in r.get("except_files", [])]
if exempt:
    print("exemptions applied (printed, never silent):")
    for rid, f, why in exempt:
        print(f"  {rid} not enforced in {f} — {why}")
prefixes = rules.get("except_prefixes", [])
for pref in prefixes:
    print(f"  hard_fail not enforced under {pref['prefix']} — {pref['reason']}")

# Paths and filenames are not content. `variants_340.csv` is a file reference,
# not an assertion that the corpus holds 340 variants, and GATE 1 measured that
# such references dominate the raw hit count.
PATHISH = re.compile(
    r"[\w./\\-]*\.(?:csv|py|json|md|pdf|png|jpe?g|js|html?|css|ya?ml|toml|cff|txt"
    r"|ipynb|lock|woff2?|ttf|otf|sh|tex|bib|gitkeep|nojekyll)\b")

def strip_paths(line):
    return PATHISH.sub(" ", line)

files = subprocess.run(["git", "ls-files"], capture_output=True, text=True,
                       check=True).stdout.split()

def is_text(p):
    try:
        chunk = Path(p).read_bytes()[:8192]
        return b"\0" not in chunk and bool(chunk.decode("utf-8"))
    except Exception:
        return False

targets = [f for f in files if f not in whitelist and is_text(f)]

def compile_rule(r):
    return re.compile(r["regex"], re.IGNORECASE if r.get("ignorecase", True) else 0)

def scan(rule_list, paths, label):
    rx_cache = [(r, compile_rule(r)) for r in rule_list]
    hits = []
    for path in paths:
        try:
            lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        for ln, raw in enumerate(lines, 1):
            line = strip_paths(raw)[:4000]
            for rule, rx in rx_cache:
                if path in rule.get("except_files", ()):
                    continue
                if label == "hard" and any(path.startswith(p["prefix"])
                                           for p in prefixes):
                    continue
                m = rx.search(line)
                if not m:
                    continue
                near = rule.get("unless_near")
                if near:
                    w = rule.get("unless_window", 60)
                    ctx = line[max(0, m.start() - w):m.start()]
                    if re.search(near, ctx, re.IGNORECASE):
                        continue
                hits.append((rule["id"], path, ln, re.sub(r"\s+", " ", line.strip())[:140]))
    return hits

hard = scan(rules["hard_fail"], targets, "hard")
for rule in rules["hard_fail_landing"]:
    scoped = set(groups[rule["scope"]])
    hard += scan([rule], [t for t in targets if t in scoped], "landing")

print()
if hard:
    print(f"FAIL B-1 hard tokens: {len(hard)} hit(s)")
    for rid, path, ln, ctx in hard:
        print(f"  [{rid}] {path}:{ln}: {ctx}")
else:
    print(f"PASS B-1 hard tokens: 0 hits across {len(targets)} tracked text files")

warn = scan(rules["warn"], targets, "warn")
by_rule = {}
for rid, path, ln, ctx in warn:
    by_rule.setdefault(rid, []).append((path, ln, ctx))
notes = {r["id"]: r.get("known_false_positive") for r in rules["warn"]}

with warn_log.open("w", encoding="utf-8") as fh:
    fh.write("# gate_canon.sh CHECK B-2 — context-dependent tokens.\n")
    fh.write("# These do NOT fail the gate. A human reads this file and judges.\n")
    fh.write("# Promoting any of them to a hard FAIL would red the gate on live\n")
    fh.write("# canonical values; see the known_false_positive notes below.\n\n")
    for rid in sorted(by_rule):
        fh.write(f"## {rid}  ({len(by_rule[rid])} hit(s))\n")
        if notes.get(rid):
            fh.write(f"   known false positive: {notes[rid]}\n")
        for path, ln, ctx in by_rule[rid]:
            fh.write(f"   {path}:{ln}: {ctx}\n")
        fh.write("\n")

print(f"WARN B-2 context tokens: {len(warn)} hit(s) across {len(by_rule)} rule(s) "
      f"-> {warn_log.name} (does not fail the gate)")
for rid in sorted(by_rule):
    tag = "  [known FP]" if notes.get(rid) else ""
    print(f"  {rid}: {len(by_rule[rid])}{tag}")

sys.exit(3 if hard else 0)
PY
  return $?
}

# ------------------------------------------------------------------ CHECK C
check_c() {
  hr; echo "CHECK C — cascade coverage"; hr
  python3 - <<'PY'
import datetime, json, re, sys
from pathlib import Path

# Why this check exists, and why it is allowed to fail on things CI cannot see:
#
# Six media carry these figures and only three of them are files in this repo.
# SSRN, LinkedIn, the printed poster, the talk script and the assistant memory
# cannot be read by CI at all. A gate that only checked what it could reach
# would report green while half the cascade sat stale -- which is the failure
# this whole exercise exists to stop. So repo media are verified mechanically,
# and off-repo media carry a human attestation that EXPIRES: any attestation
# older than canonical.json's generated_at fails the check. Absence of evidence
# is recorded as an explicit expiry, never as a pass.
#
# 322-denominator set (n_variants_ex_cost / pct_negative_ex_cost /
# n_above_bench_ex_cost): as of 2026-08 these three print only in the
# camera-ready PDF, and its content is sealed by the camera_ready_md5 key in
# CHECK A -- no separate pdftotext routine here, that would be a second check of
# the same bytes. They are listed on the cascade so a future change to the
# corpus propagates to them; DENOMINATORS.md (9월 항목) is where they get spelled
# out for a reader.

canonical = json.loads(Path("canonical.json").read_text(encoding="utf-8"))
manifest = json.loads(Path("cascade_manifest.json").read_text(encoding="utf-8"))
figures = canonical["figures"]
manual = canonical["manual"]
stamp = canonical["generated_at"]


def parse(ts):
    return datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))


def lookup(key):
    if key in figures:
        return figures[key]
    if key in manual:
        return manual[key]
    return None


def as_regex(text):
    """Whitespace-tolerant match: '9 / 12' also matches '9/12'."""
    return re.compile(r"\s*".join(re.escape(p) for p in text.split()))


failed, notices = False, []
canon_at = parse(stamp)
print(f"canonical.json generated_at = {stamp}\n")

for entry in manifest["media"]:
    medium, kind = entry["medium"], entry["kind"]

    if kind == "absent":
        notices.append(f"{medium}: {entry.get('note', 'not present in this repo')}")
        print(f"NOTE {medium:<34} {entry.get('note', '')}")
        continue

    if kind == "repo":
        path = Path(entry["path"])
        if not path.exists():
            failed = True
            print(f"FAIL {medium:<34} file not found: {path}")
            continue
        body = path.read_text(encoding="utf-8", errors="replace")
        missing = []
        for key in entry["keys"]:
            spec = lookup(key)
            if spec is None:
                missing.append(f"{key} (no such canonical key)")
                continue
            wanted = entry.get("match_override", {}).get(key, spec["print_precision"])
            if not as_regex(wanted).search(body):
                missing.append(f"{key} -> {wanted!r}")
        if missing:
            failed = True
            print(f"FAIL {medium:<34} {len(missing)} figure(s) absent or stale")
            for m in missing:
                print(f"       - {m}")
        else:
            print(f"PASS {medium:<34} {len(entry['keys'])} figure(s) current")
        continue

    # kind == "external"
    lv, cga = entry.get("last_verified_at"), entry.get("canonical_generated_at")
    who = entry.get("verified_by")
    if not lv or not who:
        failed = True
        print(f"FAIL {medium:<34} never attested against this canonical.json")
        print(f"       run: scripts/gate_canon.sh attest {medium!r} \"<your name>\"")
        continue
    if cga != stamp:
        failed = True
        print(f"FAIL {medium:<34} attested against {cga}, canonical is {stamp}")
        continue
    if parse(lv) < canon_at:
        failed = True
        print(f"FAIL {medium:<34} attestation {lv} predates {stamp}")
        continue
    print(f"PASS {medium:<34} attested {lv} by {who}")

if notices:
    print("\nnotices (recorded, not passed silently):")
    for n in notices:
        print(f"  - {n}")

sys.exit(4 if failed else 0)
PY
  return $?
}

# ------------------------------------------------------------------ attest
attest() {
  python3 - "$1" "$2" <<'PY'
import datetime, json, sys
from pathlib import Path

medium, who = sys.argv[1], sys.argv[2]
canonical = json.loads(Path("canonical.json").read_text(encoding="utf-8"))
mpath = Path("cascade_manifest.json")
manifest = json.loads(mpath.read_text(encoding="utf-8"))

now = (datetime.datetime.now(datetime.timezone.utc)
       .replace(microsecond=0).isoformat().replace("+00:00", "Z"))
for entry in manifest["media"]:
    if entry["medium"] == medium:
        if entry["kind"] != "external":
            sys.exit(f"{medium} is kind={entry['kind']}; only external media are attested")
        entry["last_verified_at"] = now
        entry["verified_by"] = who
        entry["canonical_generated_at"] = canonical["generated_at"]
        mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
        print(f"attested {medium}: {now} by {who} "
              f"against canonical {canonical['generated_at']}")
        sys.exit(0)
sys.exit(f"no medium named {medium!r} in cascade_manifest.json")
PY
  return $?
}

# -------------------------------------------------------------------- driver
case "$MODE" in
  a) check_a; exit $? ;;
  b) check_b; exit $? ;;
  c) check_c; exit $? ;;
  attest)
    [ $# -eq 3 ] || { echo "usage: $0 attest \"<medium>\" \"<verified_by>\"" >&2; exit 64; }
    attest "$2" "$3"; exit $? ;;
  all)
    check_a; A=$?
    check_b; B=$?
    check_c; C=$?
    hr
    printf 'CHECK A %s (exit %d)\nCHECK B %s (exit %d)\nCHECK C %s (exit %d)\n' \
      "$([ $A -eq 0 ] && echo PASS || echo FAIL)" "$A" \
      "$([ $B -eq 0 ] && echo PASS || echo FAIL)" "$B" \
      "$([ $C -eq 0 ] && echo PASS || echo FAIL)" "$C"
    for code in $A $B $C; do [ "$code" -ne 0 ] && exit "$code"; done
    exit 0 ;;
  *)
    echo "usage: $0 [a|b|c|all] | $0 attest \"<medium>\" \"<verified_by>\"" >&2
    exit 64 ;;
esac
