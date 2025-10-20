# PPI_Statistics.py  —  Read-only diagnostic (print-only, relative paths)
# UPDATED: adds HGNC mapping (symbol/aliases & UniProt -> Entrez) so CORUM can match BioGRID (Entrez) nodes.

import os
import re
import pickle
from collections import Counter, defaultdict

import pandas as pd

# ---------------------------
# Normalization helpers
# ---------------------------
def load_mask_or_default(nodes, graphs_dir):
    """
    Try to load a mask of valid node indices (one index per line).
    If not found, default to 'all nodes valid'.
    Expected optional path: Graphs/ppi_valid_nodes.txt
    """
    candidate = os.path.join(graphs_dir, "ppi_valid_nodes.txt")
    if os.path.exists(candidate):
        valid = set()
        with open(candidate, "r") as f:
            for line in f:
                s = line.strip()
                if s:
                    try:
                        valid.add(int(s))
                    except ValueError:
                        pass
        print(f"[MASK] Loaded valid node mask from {candidate}  (|mask|={len(valid)})")
        return valid
    else:
        print("[MASK] No mask file found (Graphs/ppi_valid_nodes.txt). Using ALL nodes as valid.")
        return set(range(len(nodes)))

UNIPROT_RE = re.compile(r"^[A-NR-Z][0-9][A-Z0-9]{3}[0-9](?:-\d+)?$")  # UniProt with optional isoform

def norm_sym(s):
    if not isinstance(s, str): return None
    s = s.strip()
    return s.upper() if s else None

def norm_uid(u):
    if not isinstance(u, str): return None
    u = u.strip()
    if not u: return None
    return u.split("-", 1)[0]  # drop isoform suffix e.g. P12345-2 -> P12345

def is_entrez_like(x):
    return isinstance(x, int) or (isinstance(x, str) and x.isdigit())

def is_uniprot_like(x):
    return isinstance(x, str) and bool(UNIPROT_RE.match(x.strip()))

def is_symbol_like(x):
    if not isinstance(x, str): return False
    s = x.strip()
    return bool(s) and len(s) <= 20 and s[0].isalpha()

POSSIBLE_SYMBOL_KEYS = ["symbol", "gene_symbol", "GENE_SYMBOL", "Symbol", "SYMBOL"]
POSSIBLE_UNIPROT_KEYS = ["uniprot", "uniprot_id", "UNIPROT", "UniProt", "UNIPROT_ID"]

# ---------------------------
# Path resolution (relative to this file)
# ---------------------------

def find_graphs_dir():
    here = os.path.dirname(__file__)
    candidates = [
        os.path.join(here, "Graphs"),
        os.path.join(here, "Data", "Graphs"),
        os.path.join(here, "..", "Graphs"),
        os.path.join(here, "..", "Data", "Graphs"),
    ]
    for p in candidates:
        if os.path.isdir(p):
            return os.path.abspath(p)
    raise FileNotFoundError("Could not find 'Graphs' directory next to PPI_Statistics.py.")

# ---------------------------
# PPI lookups & namespace summary
# ---------------------------

def summarize_node_namespace(nodes, node_attrs_lookup, sample_k=500):
    sample = list(nodes)[:sample_k]
    id_types = Counter()
    for n in sample:
        if is_entrez_like(n):
            id_types["entrez_like"] += 1
        elif is_uniprot_like(str(n)):
            id_types["uniprot_like"] += 1
        elif is_symbol_like(str(n)):
            id_types["symbol_like"] += 1
        else:
            id_types["other"] += 1

    attr_completeness = {}
    for key in POSSIBLE_SYMBOL_KEYS + POSSIBLE_UNIPROT_KEYS:
        present = 0
        non_empty = 0
        for n in sample:
            attrs = node_attrs_lookup.get(n, {})
            if key in attrs:
                present += 1
                v = attrs.get(key)
                if isinstance(v, str) and v.strip():
                    non_empty += 1
        attr_completeness[key] = (present, non_empty)

    guess = id_types.most_common(1)[0][0] if id_types else None
    return id_types, guess, attr_completeness

def build_ppi_lookups(G):
    nodes = list(G.nodes())
    node_id_to_idx = {n: i for i, n in enumerate(nodes)}
    node_attrs_lookup = {n: (G.nodes[n] if hasattr(G, "nodes") else {}) for n in nodes}

    by_symbol, by_uniprot = {}, {}
    by_nodeid_str = {str(n): node_id_to_idx[n] for n in nodes}

    for n in nodes:
        attrs = node_attrs_lookup.get(n, {})
        for k in POSSIBLE_SYMBOL_KEYS:
            s = norm_sym(attrs.get(k))
            if s:
                by_symbol.setdefault(s, node_id_to_idx[n])
        for k in POSSIBLE_UNIPROT_KEYS:
            u = norm_uid(attrs.get(k))
            if u:
                by_uniprot.setdefault(u, node_id_to_idx[n])
        if is_uniprot_like(str(n)):
            by_uniprot.setdefault(norm_uid(str(n)), node_id_to_idx[n])

    return nodes, node_id_to_idx, node_attrs_lookup, by_symbol, by_uniprot, by_nodeid_str

# ---------------------------
# CORUM loading & explode
# ---------------------------

def load_corum(path):
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.lower() for c in df.columns]
    if "organism" not in df.columns and "species" in df.columns:
        df = df.rename(columns={"species": "organism"})
    return df

def explode_corum_members(df):
    cols = set(df.columns)
    id_col = "complex_id" if "complex_id" in cols else None
    name_col = "complex_name" if "complex_name" in cols else None
    org_col = "organism" if "organism" in cols else None
    sym_col = "subunits_gene_name" if "subunits_gene_name" in cols else None
    uid_col = "subunits_uniprot_id" if "subunits_uniprot_id" in cols else None

    if not id_col or not (sym_col or uid_col):
        raise RuntimeError("CORUM missing required columns. Need 'complex_id' and "
                           "one of 'subunits_gene_name'/'subunits_uniprot_id'.")

    base_cols = [c for c in [id_col, name_col, org_col] if c]
    rec_sym, rec_uid = [], []

    for _, row in df.iterrows():
        base = {c: row.get(c) for c in base_cols}
        if sym_col:
            raw = str(row.get(sym_col) or "")
            for s in [x for x in raw.split(";") if x != ""]:
                s_norm = norm_sym(s)
                if s_norm:
                    r = dict(base); r["member_symbol"] = s_norm
                    rec_sym.append(r)
        if uid_col:
            raw = str(row.get(uid_col) or "")
            for u in [x for x in raw.split(";") if x != ""]:
                u_norm = norm_uid(u)
                if u_norm:
                    r = dict(base); r["member_uniprot"] = u_norm
                    rec_uid.append(r)

    df_sym = pd.DataFrame(rec_sym) if rec_sym else pd.DataFrame(columns=base_cols + ["member_symbol"])
    df_uid = pd.DataFrame(rec_uid) if rec_uid else pd.DataFrame(columns=base_cols + ["member_uniprot"])
    return df_sym, df_uid

# ---------------------------
# HGNC loading & mapping builders
# ---------------------------

def load_hgnc(path_hgnc_txt):
    # HGNC complete set is tab-delimited with many columns; we only need a few.
    df = pd.read_csv(path_hgnc_txt, sep="\t", dtype=str)
    # Normalize important columns
    for col in ["symbol", "alias_symbol", "entrez_id", "uniprot_ids"]:
        if col not in df.columns:
            df[col] = None
    df["symbol"] = df["symbol"].map(norm_sym)
    df["entrez_id"] = df["entrez_id"].fillna("").astype(str)
    df["uniprot_ids"] = df["uniprot_ids"].fillna("").astype(str)
    df["alias_symbol"] = df["alias_symbol"].fillna("").astype(str)
    return df

def build_hgnc_mappers(hgnc_df):
    """
    Returns:
      sym_to_entrez: dict[str -> set[str]] (official + aliases, uppercased)
      uid_to_entrez: dict[str -> set[str]] (UniProt accession -> set of Entrez)
      entrez_to_symbol: dict[str -> str] (nice-to-have for reporting)
    """
    sym_to_entrez = defaultdict(set)
    uid_to_entrez = defaultdict(set)
    entrez_to_symbol = {}

    for _, r in hgnc_df.iterrows():
        sym = r.get("symbol")
        entrez = r.get("entrez_id")
        if entrez and entrez != "nan" and entrez.strip():
            entrez_to_symbol.setdefault(entrez, sym or "")
            if sym:
                sym_to_entrez[sym].add(entrez)

            # aliases: comma-separated
            aliases = [norm_sym(a) for a in r.get("alias_symbol", "").split(",")]
            for a in aliases:
                if a:
                    sym_to_entrez[a].add(entrez)

            # UniProt IDs: comma-separated
            for u in r.get("uniprot_ids", "").split(","):
                u_norm = norm_uid(u)
                if u_norm:
                    uid_to_entrez[u_norm].add(entrez)

    return sym_to_entrez, uid_to_entrez, entrez_to_symbol

# ---------------------------
# Trial mapping (print-only) with HGNC bridge
# ---------------------------

def trial_map_print(df_sym, df_uid, by_symbol, by_uniprot, by_nodeid_str,
                    node_count, hgnc_sym2entrez=None, hgnc_uid2entrez=None):
    """
    Attempt a read-only mapping of CORUM members to PPI nodes and print diagnostics.
    We try, in order:
      A) Direct node attributes (symbol/uniprot) if present in PPI (often empty for BioGRID Entrez graphs)
      B) HGNC bridge: symbol/aliases -> Entrez, and UniProt -> Entrez, then match to PPI by node-id string
    """
    mapped = []     # entries: (method, complex_id, member_value, node_idx)
    unmatched = []  # entries: (member_type, member_value, complex_id)
    ambiguous = []  # entries: (member_type, member_value, complex_id, candidates_found)

    # --- Symbol mapping ---
    if df_sym is not None and not df_sym.empty:
        for _, r in df_sym.iterrows():
            member = r.get("member_symbol")
            cid = r.get("complex_id")
            # A1) via PPI node attributes (rare in your graph)
            hit = by_symbol.get(member)
            if hit is not None:
                mapped.append(("symbol_attr", cid, member, hit))
                continue
            # A2) direct node-id match (if PPI nodes are symbols; not your case)
            hit2 = by_nodeid_str.get(member)
            if hit2 is not None:
                mapped.append(("direct_nodeid_str", cid, member, hit2))
                continue
            # B) HGNC: symbol/aliases -> Entrez -> PPI node
            if hgnc_sym2entrez is not None:
                entrez_candidates = hgnc_sym2entrez.get(member, set())
                if entrez_candidates:
                    # Which of these are present in the PPI?
                    node_hits = [by_nodeid_str.get(e) for e in entrez_candidates if by_nodeid_str.get(e) is not None]
                    node_hits = [h for h in node_hits if h is not None]
                    if len(node_hits) == 1:
                        mapped.append(("hgnc_symbol_to_entrez", cid, member, node_hits[0]))
                    elif len(node_hits) > 1:
                        ambiguous.append(("symbol", member, cid, list(entrez_candidates)))
                    else:
                        unmatched.append(("symbol", member, cid))
                else:
                    unmatched.append(("symbol", member, cid))
            else:
                unmatched.append(("symbol", member, cid))

    # --- UniProt mapping ---
    if df_uid is not None and not df_uid.empty:
        for _, r in df_uid.iterrows():
            member = r.get("member_uniprot")
            cid = r.get("complex_id")
            # A1) via PPI node attributes (rare in your graph)
            hit = by_uniprot.get(member)
            if hit is not None:
                mapped.append(("uniprot_attr_or_nodeid", cid, member, hit))
                continue
            # A2) direct node-id match (if PPI nodes are UniProt; not your case)
            hit2 = by_nodeid_str.get(member)
            if hit2 is not None:
                mapped.append(("direct_nodeid_str", cid, member, hit2))
                continue
            # B) HGNC: UniProt -> Entrez -> PPI node
            if hgnc_uid2entrez is not None:
                entrez_candidates = hgnc_uid2entrez.get(member, set())
                if entrez_candidates:
                    node_hits = [by_nodeid_str.get(e) for e in entrez_candidates if by_nodeid_str.get(e) is not None]
                    node_hits = [h for h in node_hits if h is not None]
                    if len(node_hits) == 1:
                        mapped.append(("hgnc_uniprot_to_entrez", cid, member, node_hits[0]))
                    elif len(node_hits) > 1:
                        ambiguous.append(("uniprot", member, cid, list(entrez_candidates)))
                    else:
                        unmatched.append(("uniprot", member, cid))
                else:
                    unmatched.append(("uniprot", member, cid))
            else:
                unmatched.append(("uniprot", member, cid))

    # --- Summary prints ---
    print("\n=== Mapping Summary (trial + HGNC) ===")
    print(f"Total mapped members: {len(mapped)}")
    print(f"Total unmatched members: {len(unmatched)}")

    method_counts = Counter(m for m, _, _, _ in mapped)
    print("Mapped by method:", dict(method_counts))

    memtype_counts = Counter()
    for method, _, _, _ in mapped:
        if method in ("symbol_attr", "hgnc_symbol_to_entrez"):
            memtype_counts["symbol"] += 1
        elif method in ("uniprot_attr_or_nodeid", "hgnc_uniprot_to_entrez"):
            memtype_counts["uniprot"] += 1
        else:
            memtype_counts["fallback"] += 1
    print("Mapped by member type:", dict(memtype_counts))

    per_complex = Counter(cid for _, cid, _, _ in mapped)
    print("Complexes with ≥1 mapped member:", len(per_complex))
    for k in (1, 2, 3, 5):
        print(f"Complexes with ≥{k} mapped members:", sum(1 for v in per_complex.values() if v >= k))

    per_node = Counter(idx for _, _, _, idx in mapped)
    n_nodes_with_any = sum(1 for v in per_node.values() if v > 0)
    node_cov_pct = 100.0 * n_nodes_with_any / node_count if node_count else 0.0
    print(f"Node coverage: {node_cov_pct:.2f}% of PPI nodes would have ≥1 complex membership.")

    if ambiguous:
        print(f"\nAmbiguous mappings encountered: {len(ambiguous)} (symbol/uniprot -> multiple Entrez present in PPI)")
    else:
        print("\nAmbiguous mappings encountered: 0")

    # Top unmatched
    if unmatched:
        print("\nTop unmatched members (type, value -> count):")
        top_unmatched = Counter((t, v) for (t, v, _) in unmatched).most_common(10)
        for (t, v), cnt in top_unmatched:
            print(f"  ({t}) {v} -> {cnt}")
    else:
        print("\nUnmatched members: 0")

    complex_to_nodes = defaultdict(set)
    for _, cid, _, idx in mapped:
        complex_to_nodes[cid].add(idx)

    # Deduplicate complexes that collapse to identical node sets
    dedup = {}
    seen = set()
    for cid, s in complex_to_nodes.items():
        key = frozenset(s)
        if key in seen:
            continue
        seen.add(key)
        dedup[cid] = s
    complex_to_nodes = dedup

    return complex_to_nodes, mapped, unmatched, ambiguous
# ---------------------------
# Driver (print-only)
# ---------------------------

def run_ppi_corum_diagnostic():
    graphs_dir = find_graphs_dir()
    ppi_path = os.path.join(graphs_dir, "ppi_graph.gpickle")
    corum_path = os.path.join(graphs_dir, "corum_humanComplexes.txt")
    hgnc_path  = os.path.join(graphs_dir, "hgnc_complete_set.txt")  # <-- place HGNC file here

    print(f"[Paths] Graphs dir: {graphs_dir}")
    print(f"[Paths] PPI: {ppi_path}")
    print(f"[Paths] CORUM: {corum_path}")
    print(f"[Paths] HGNC: {hgnc_path}")

    # Load PPI
    with open(ppi_path, "rb") as f:
        G = pickle.load(f)
    nodes, edges = list(G.nodes()), list(G.edges())
    print(f"\n[PPI] Nodes: {len(nodes)}  Edges: {len(edges)}  Directed: {getattr(G, 'is_directed', lambda: False)()}")

    # Build lookups & summarize namespace
    nodes, node_id_to_idx, node_attrs_lookup, by_symbol, by_uniprot, by_nodeid_str = build_ppi_lookups(G)
    id_counts, primary_guess, attr_comp = summarize_node_namespace(nodes, node_attrs_lookup)

    print("\n[PPI] Node-ID namespace (sample):", dict(id_counts))
    print("[PPI] Primary node-ID guess:", primary_guess)
    print("[PPI] Attribute completeness in sample (present / non-empty):")
    any_present = False
    for k, (present, non_empty) in attr_comp.items():
        if present > 0:
            any_present = True
            print(f"  {k:16s}  present={present:4d}  non_empty={non_empty:4d}")
    if not any_present:
        print("  (no common ID attributes found in sample)")
    print(f"[PPI] Lookup sizes -> by_symbol: {len(by_symbol)} | by_uniprot: {len(by_uniprot)}")

    # Load CORUM
    corum = load_corum(corum_path)
    print(f"\n[CORUM] Columns: {list(corum.columns)}  Rows: {len(corum)}")

    # Explode members
    df_sym, df_uid = explode_corum_members(corum)
    print(f"[CORUM] Exploded symbol rows: {len(df_sym)} | uniprot rows: {len(df_uid)}")

    # Complex size by symbol/uniprot
    size_sym = (df_sym.groupby("complex_id")["member_symbol"].nunique() if not df_sym.empty else pd.Series(dtype=int))
    size_uid = (df_uid.groupby("complex_id")["member_uniprot"].nunique() if not df_uid.empty else pd.Series(dtype=int))
    def describe_series(name, s):
        if s.empty:
            print(f"[CORUM] {name}: (no data)")
        else:
            print(f"[CORUM] {name}: complexes={len(s)}  min={s.min()}  median={int(s.median())}  max={s.max()}")
    describe_series("size_by_symbol", size_sym)
    describe_series("size_by_uniprot", size_uid)

    # Load HGNC and build bridge
    if not os.path.exists(hgnc_path):
        print("\n[HGNC] File not found at the expected path. Please place 'hgnc_complete_set.txt' in Graphs/ and re-run.")
        hgnc_sym2entrez = None
        hgnc_uid2entrez = None
    else:
        hgnc_df = load_hgnc(hgnc_path)
        hgnc_sym2entrez, hgnc_uid2entrez, _ = build_hgnc_mappers(hgnc_df)
        print(f"[HGNC] Loaded: {len(hgnc_df)} rows  |  sym→Entrez keys: {len(hgnc_sym2entrez)}  |  UniProt→Entrez keys: {len(hgnc_uid2entrez)}")

    # Trial mapping (no mutation, print-only) — now with HGNC bridge
    complex_to_nodes, mapped, unmatched, ambiguous = trial_map_print(
        df_sym, df_uid,
        by_symbol, by_uniprot, by_nodeid_str,
        node_count=len(nodes),
        hgnc_sym2entrez=hgnc_sym2entrez,
        hgnc_uid2entrez=hgnc_uid2entrez
    )

    # ---------- NEW: Label stats (raw vs masked) ----------
    print("\n=== LABEL / COMMUNITY STATS ===")

    # Raw complex counts
    raw_sets = [s for s in complex_to_nodes.values()]
    raw_sizes = [len(s) for s in raw_sets]
    raw_nonempty = sum(1 for s in raw_sets if len(s) >= 1)
    raw_ge2 = sum(1 for s in raw_sets if len(s) >= 2)

    print(f"[LABEL] Unique complexes mapped (raw, >=1 member): {raw_nonempty}")
    print(f"[LABEL] Unique complexes mapped (raw, >=2 members): {raw_ge2}")

    # Load mask (or use all nodes)
    valid_idx = load_mask_or_default(nodes, graphs_dir)

    # Masked complexes
    # Masked complexes (ONMI-style: keep >0 after mask, then dedup)
    masked_sets = [frozenset(s & valid_idx) for s in raw_sets if len(s & valid_idx) > 0]
    masked_unique = len(set(masked_sets))
    masked_ge2_unique = sum(1 for s in set(masked_sets) if len(s) >= 2)

    print(f"[LABEL] Complexes surviving mask (unique, >0): {masked_unique}")
    print(f"[LABEL] Complexes surviving mask (unique, >=2): {masked_ge2_unique}")

    # Unique labeled node coverage (raw vs masked)
    raw_labeled_nodes = set().union(*raw_sets) if raw_sets else set()
    masked_labeled_nodes = set().union(*masked_sets) if masked_sets else set()
    print(f"[LABEL] Unique labeled nodes (raw): {len(raw_labeled_nodes)}")
    print(f"[LABEL] Unique labeled nodes (after mask): {len(masked_labeled_nodes)}")

    # Decision prompts
    print("\n=== DECISION PROMPTS ===")
    print("- If coverage improved substantially, we can adopt Entrez as canonical (BioGRID-native) and proceed to label nodes.")
    print("- Filter to 'Homo sapiens' complexes if needed (organism column).")
    print("- Consider setting a min complex size (e.g., ≥3) for stability.")
    print("- Keep isoform-drop policy (P12345-2 → P12345) unless you want isoform-specific handling.")

# Entry point
if __name__ == "__main__":
    run_ppi_corum_diagnostic()
