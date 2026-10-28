"""Shared recorder for the R2.3 verification scripts.  Every row has a KIND:
  RECOMPUTED  a value is computed in the script from a shipped data file (CSV/JSON) and compared with the expected value
  RERUN       a shipped script is executed again (in a temporary directory) and its output compared with the shipped output
  DEFINITION  an arithmetic/definitional identity evaluated in the script (no data file involved)
  TEXT        only the presence of a string in the built .tex is checked (no recomputation)
A TEXT row whose string was itself produced by a RECOMPUTED value in the same script is marked TEXT* (value-derived)."""
import os, sys

KINDS = ("RERUN", "RECOMPUTED", "DEFINITION", "TEXT*", "TEXT")

class Rec:
    def __init__(self, title):
        self.title, self.rows = title, []

    def add(self, kind, name, ok, detail=""):
        assert kind in KINDS, kind
        self.rows.append((kind, name, bool(ok), str(detail)))
        return bool(ok)

    def eq(self, kind, name, got, want, detail=""):
        return self.add(kind, name, got == want, "%s%s" % (("got %r != want %r; " % (got, want)) if got != want else "", detail))

    def text(self, name, needle, hay, derived=False):
        return self.add("TEXT*" if derived else "TEXT", name, needle in hay, needle[:90])

    def report(self, md_path=None):
        w = max(len(r[1]) for r in self.rows)
        print("== %s ==" % self.title)
        for k, n, ok, d in self.rows:
            print("%-10s %-*s | %-4s | %s" % (k, w, n, "OK" if ok else "FAIL", d[:100]))
        bad = sum(1 for r in self.rows if not r[2])
        cnt = {k: sum(1 for r in self.rows if r[0] == k) for k in KINDS}
        print("TOTAL %d  FAIL %d  | " % (len(self.rows), bad) + "  ".join("%s %d" % (k, cnt[k]) for k in KINDS if cnt[k]))
        if md_path:
            with open(md_path, "w", encoding="utf-8") as f:
                f.write("# %s\n\n| kind | check | result | detail |\n|---|---|---|---|\n" % self.title)
                for k, n, ok, d in self.rows:
                    f.write("| %s | %s | %s | %s |\n" % (k, n.replace("|", "/"), "OK" if ok else "**FAIL**", d.replace("|", "/")[:120]))
                f.write("\nTOTAL %d, FAIL %d (%s)\n" % (len(self.rows), bad, ", ".join("%s %d" % (k, cnt[k]) for k in KINDS if cnt[k])))
        return 1 if bad else 0

def tree_root(argv_index=1):
    """tree root = argument if given, else the parent of tools/"""
    if len(sys.argv) > argv_index and not sys.argv[argv_index].startswith("--"):
        return os.path.abspath(sys.argv[argv_index])
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
