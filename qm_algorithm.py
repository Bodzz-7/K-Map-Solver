"""
qm_algorithm.py

Quine–McCluskey (Tabular) Method
--------------------------------

This module implements:
- Phase 1: Prime implicant generation
- Phase 2: Essential prime implicants + a simple greedy cover

- Take minterms and don't-cares as integer indices.
- Convert each index to a padded binary string (length = number of variables).
- Group terms by the number of '1' bits.
- Repeatedly compare adjacent groups and combine terms that differ by exactly one bit.
  The combined term replaces the differing bit with '-' (dash).
- Any term that can no longer be combined is a Prime Implicant.

The returned prime implicants are patterns like:
    '0-1-'  meaning A=0, B=don't care, C=1, D=don't care

Notes:
- For Phase 2, this uses essentials + a simple greedy heuristic (not Petrick's method).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple


def minterm_to_literal(minterm_int: int, num_vars: int) -> str:
    """
    Convert an integer minterm index into its *full* literal product term.

    Example (4 variables, A is MSB):
        0  -> A'B'C'D'
        13 -> AB'CD

    Notes:
    - This returns ALL variables (no omissions), unlike implicant patterns that may contain '-'.
    """
    if not isinstance(minterm_int, int):
        raise TypeError("minterm_int must be an integer.")
    if not isinstance(num_vars, int) or num_vars <= 0:
        raise ValueError("num_vars must be a positive integer.")

    max_index = (1 << num_vars) - 1
    if minterm_int < 0 or minterm_int > max_index:
        raise ValueError(f"minterm_int {minterm_int} out of range for {num_vars} variables (0..{max_index}).")

    bits = format(minterm_int, f"0{num_vars}b")
    var_names = QuineMcCluskey._variable_names(num_vars)
    out: List[str] = []
    for name, b in zip(var_names, bits):
        out.append(name if b == "1" else f"{name}'")
    return "".join(out)


@dataclass(frozen=True)
class _Term:
    """
    Internal representation of a term/implicant.

    - **pattern**:
        A string of length `num_variables` made of '0', '1', and '-'.
        '-' means "this variable does not matter" for this implicant.

    - **covered**:
        The set of original indices that this implicant covers.
        This isn't strictly required to *generate* prime implicants,
        but is very useful later when you build the prime implicant chart.
    """

    pattern: str
    covered: frozenset[int]


class QuineMcCluskey:
    """
    Quine–McCluskey algorithm:
    - Phase 1: generate prime implicants (grouping/combining).
    - Phase 2: select a minimum-ish cover (essentials + greedy remainder).
    """

    def __init__(self, num_variables: int) -> None:
        # Keep it simple but safe: the algorithm works for any positive integer,
        # but desktop K-map apps usually use 2..4 variables (sometimes more).
        if not isinstance(num_variables, int) or num_variables <= 0:
            raise ValueError("num_variables must be a positive integer.")
        self.num_variables = num_variables

    def get_prime_implicants(self, minterms: List[int], dont_cares: List[int]) -> List[str]:
        """
        Generate all Prime Implicants from minterms and don't-cares.

        Args:
            minterms: list of integers where F = 1
            dont_cares: list of integers where F = X

        Returns:
            A sorted list of prime implicant patterns (strings with 0/1/-).

        Implementation overview (highly commented):

        1) Combine minterms + dont_cares (both participate in combining).
        2) Convert each integer index to a padded binary string.
        3) Group these strings by their number of '1' bits.
        4) Repeat:
           - Compare only adjacent groups (k ones) with (k+1 ones).
           - If two patterns differ in exactly one concrete bit, combine them:
             Example: 0101 and 0111 -> 01-1
           - Mark the originals as "used" in this round.
           - Anything not "used" becomes a prime implicant.
           - Regroup the newly-combined patterns and continue.
        5) Stop when no new combinations are produced.
        """

        n = self.num_variables
        max_index = (1 << n) - 1

        # ---- Input sanitation ----
        # Convert to sets to remove duplicates; order isn't important for correctness.
        m_set = set(minterms or [])
        d_set = set(dont_cares or [])

        # Disallow overlap (a term cannot be both a minterm and a don't-care).
        overlap = m_set & d_set
        if overlap:
            raise ValueError(f"Overlap between minterms and dont_cares: {sorted(overlap)}")

        # Validate type + range. For n variables, valid indices are 0..(2^n - 1).
        for v in sorted(m_set | d_set):
            if not isinstance(v, int):
                raise TypeError("minterms and dont_cares must contain integers.")
            if v < 0 or v > max_index:
                raise ValueError(f"Index {v} is out of range for {n} variables (0..{max_index}).")

        # ---- Create initial terms (no dashes yet) ----
        # We include dont_cares because they help form larger implicants.
        initial_terms: List[_Term] = []
        for v in sorted(m_set | d_set):
            pattern = format(v, f"0{n}b")  # padded binary string
            initial_terms.append(_Term(pattern=pattern, covered=frozenset({v})))

        # ---- Group by ones count ----
        current_groups = self._group_by_ones(initial_terms)

        # We'll collect prime implicants from ALL rounds here.
        prime_implicants: Set[_Term] = set()

        # ---- Iterative combining ----
        while True:
            used_this_round: Set[_Term] = set()
            combined_next_round: Set[_Term] = set()

            # Adjacent-group comparison:
            # If group k has patterns with k ones, they can only combine with group k+1.
            keys = sorted(current_groups.keys())
            for idx in range(len(keys) - 1):
                k = keys[idx]
                k_next = keys[idx + 1]

                # Only compare adjacent counts; if there's a gap, there's no adjacency.
                if k_next != k + 1:
                    continue

                group_a = current_groups.get(k, [])
                group_b = current_groups.get(k_next, [])

                for t1 in group_a:
                    for t2 in group_b:
                        merged = self._try_combine(t1, t2)
                        if merged is None:
                            continue

                        # Mark originals as "used" (they are not prime at this level).
                        used_this_round.add(t1)
                        used_this_round.add(t2)

                        # Add merged term to the set for next iteration (deduplicated).
                        combined_next_round.add(merged)

            # Anything not used in this round cannot be combined further at this stage,
            # so it is a prime implicant.
            for group_terms in current_groups.values():
                for term in group_terms:
                    if term not in used_this_round:
                        prime_implicants.add(term)

            # If we produced no merged terms, we are finished.
            if not combined_next_round:
                break

            # Otherwise regroup merged terms for the next iteration.
            current_groups = self._group_by_ones(combined_next_round)

        # Return only the patterns, sorted for stable output.
        return sorted({t.pattern for t in prime_implicants})

    def get_prime_implicant_terms(self, minterms: List[int], dont_cares: List[int]) -> List[_Term]:
        """
        Same as `get_prime_implicants`, but returns `_Term` objects so callers can
        retain `covered` minterm-index provenance through combining.
        """
        n = self.num_variables
        max_index = (1 << n) - 1

        m_set = set(minterms or [])
        d_set = set(dont_cares or [])

        overlap = m_set & d_set
        if overlap:
            raise ValueError(f"Overlap between minterms and dont_cares: {sorted(overlap)}")

        for v in sorted(m_set | d_set):
            if not isinstance(v, int):
                raise TypeError("minterms and dont_cares must contain integers.")
            if v < 0 or v > max_index:
                raise ValueError(f"Index {v} is out of range for {n} variables (0..{max_index}).")

        initial_terms: List[_Term] = []
        for v in sorted(m_set | d_set):
            pattern = format(v, f"0{n}b")
            initial_terms.append(_Term(pattern=pattern, covered=frozenset({v})))

        current_groups = self._group_by_ones(initial_terms)
        prime_implicants: Set[_Term] = set()

        while True:
            used_this_round: Set[_Term] = set()
            combined_next_round: Set[_Term] = set()

            keys = sorted(current_groups.keys())
            for idx in range(len(keys) - 1):
                k = keys[idx]
                k_next = keys[idx + 1]
                if k_next != k + 1:
                    continue

                group_a = current_groups.get(k, [])
                group_b = current_groups.get(k_next, [])
                for t1 in group_a:
                    for t2 in group_b:
                        merged = self._try_combine(t1, t2)
                        if merged is None:
                            continue
                        used_this_round.add(t1)
                        used_this_round.add(t2)
                        combined_next_round.add(merged)

            for group_terms in current_groups.values():
                for term in group_terms:
                    if term not in used_this_round:
                        prime_implicants.add(term)

            if not combined_next_round:
                break
            current_groups = self._group_by_ones(combined_next_round)

        # Deterministic ordering for UI/derivations.
        return sorted(prime_implicants, key=lambda t: t.pattern)

    def get_minimum_expression(
        self, prime_implicants: List[str], minterms: List[int]
    ) -> Tuple[List[str], str, Dict[str, List[int]]]:
        """
        Phase 2: choose Essential Prime Implicants, then cover remaining minterms.

        Important:
        - This phase considers ONLY the original minterms. Don't-cares are NOT required
          to be covered (and typically must NOT force extra implicants).

        Args:
            prime_implicants: list of implicant patterns from phase 1 (e.g., ['0-1-', '11--'])
            minterms: list of minterm indices that must be covered (F=1)

        Returns:
            (selected_prime_implicants, expression_string, coverage_by_selected)

        coverage_by_selected maps each selected implicant pattern -> sorted list of
        original minterm indices (from the `minterms` argument) that it covers.

        Process (high level):
        - Build a coverage chart: rows = prime implicants, columns = minterms
        - Select essential prime implicants (EPIs): any minterm covered by exactly one PI
        - If minterms remain uncovered, use a simple greedy heuristic:
          repeatedly pick the PI that covers the most remaining uncovered minterms.
        - Convert selected patterns into a readable SOP Boolean expression.
        """

        n = self.num_variables
        max_index = (1 << n) - 1

        # ---- Sanitize inputs ----
        if not isinstance(prime_implicants, list):
            raise TypeError("prime_implicants must be a list of pattern strings.")
        if not isinstance(minterms, list):
            raise TypeError("minterms must be a list of integers.")

        m_set = set(minterms or [])
        for m in sorted(m_set):
            if not isinstance(m, int):
                raise TypeError("minterms must contain integers.")
            if m < 0 or m > max_index:
                raise ValueError(f"Minterm {m} out of range for {n} variables (0..{max_index}).")

        # Remove duplicates in prime_implicants and validate patterns.
        pis = sorted(set(prime_implicants))
        for p in pis:
            if not isinstance(p, str):
                raise TypeError("prime_implicants must contain strings only.")
            if len(p) != n:
                raise ValueError(f"Prime implicant '{p}' length must be {n}.")
            for ch in p:
                if ch not in ("0", "1", "-"):
                    raise ValueError(f"Prime implicant '{p}' contains invalid char '{ch}'.")

        # Edge cases:
        # - No minterms => function is always 0 (no implicants needed).
        if not m_set:
            return ([], "0", {})

        # - No prime implicants but have minterms => cannot cover (shouldn't happen if phase 1 ran)
        if not pis:
            return ([], "UNSAT (no prime implicants to cover minterms)", {})

        # ---- Build coverage chart ----
        # coverage_by_pi[pi] = set of minterms that pi covers
        # coverage_by_minterm[m] = list of pis that cover m
        coverage_by_pi: Dict[str, Set[int]] = {}
        coverage_by_minterm: Dict[int, List[str]] = {m: [] for m in sorted(m_set)}

        for pi in pis:
            covered_minterms = set()
            for m in m_set:
                if self._pattern_covers_minterm(pi, m):
                    covered_minterms.add(m)
                    coverage_by_minterm[m].append(pi)
            coverage_by_pi[pi] = covered_minterms

        # ---- Select Essential Prime Implicants (EPIs) ----
        selected: List[str] = []
        covered: Set[int] = set()

        # A minterm column is "essential" if exactly one PI covers it.
        # That PI must be included in ANY cover.
        essentials = set()
        for m, pi_list in coverage_by_minterm.items():
            if len(pi_list) == 1:
                essentials.add(pi_list[0])

        for epi in sorted(essentials):
            if epi not in selected:
                selected.append(epi)
            covered |= coverage_by_pi.get(epi, set())

        # ---- Cover remaining minterms (greedy) ----
        remaining = set(m_set) - covered
        if remaining:
            # Candidate PIs exclude ones already selected.
            candidates = [pi for pi in pis if pi not in set(selected)]

            # Greedy heuristic:
            # repeatedly pick the PI that covers the largest number of *remaining* minterms.
            while remaining:
                best_pi = None
                best_gain = set()

                for pi in candidates:
                    gain = coverage_by_pi[pi] & remaining
                    if len(gain) > len(best_gain):
                        best_pi = pi
                        best_gain = gain

                if best_pi is None or not best_gain:
                    # Cannot cover further (should not occur if phase 1 produced valid PIs).
                    break

                selected.append(best_pi)
                remaining -= best_gain
                candidates = [pi for pi in candidates if pi != best_pi]

        # ---- Convert patterns to readable Boolean expression (SOP) ----
        expression = self.patterns_to_expression(selected)

        coverage_by_selected: Dict[str, List[int]] = {}
        for pi in selected:
            cov = sorted(coverage_by_pi.get(pi, set()) & m_set)
            coverage_by_selected[pi] = cov

        return (selected, expression, coverage_by_selected)

    def sop_to_pos(self, sop_expression: str) -> str:
        """
        Convert an SOP string into a POS string by applying De Morgan's Laws.

        This is a *string manipulation* helper used as:
            POS(F)  = SOP(F') converted via De Morgan
            POS(F') = SOP(F)  converted via De Morgan

        Given:
            sop_expression = P1 + P2 + ... + Pk
        where each Pi is a product (AND) of literals, e.g.:
            Pi = "A B' C"

        De Morgan:
            (P1 + P2 + ... + Pk)' = (P1)' (P2)' ... (Pk)'
            and
            (A B' C)' = (A' + B + C')

        So we convert:
            "A B' + C"
        to:
            "(A' + B)(C')"

        Notes / edge cases:
        - "0" (false)  -> complement is "1" => POS "1"
        - "1" (true)   -> complement is "0" => POS "0"
        - If any SOP term is exactly "1", the SOP is true => complement is 0 => POS "0"
        """

        if sop_expression is None:
            raise TypeError("sop_expression cannot be None.")

        s = " ".join(str(sop_expression).strip().split())
        if s == "":
            return "1"

        # Handle constants
        if s == "0":
            return "1"
        if s == "1":
            return "0"

        # Split SOP terms by '+'
        raw_terms = [t.strip() for t in s.split("+") if t.strip()]
        if not raw_terms:
            return "1"

        pos_factors: List[str] = []
        for term in raw_terms:
            # Normalize whitespace inside each product term
            term_norm = " ".join(term.split())
            if term_norm == "0":
                # (0)' = 1, which does not change an AND-chain; skip it.
                continue
            if term_norm == "1":
                # (1)' = 0, and AND-ing a 0 makes the whole POS 0.
                return "0"

            literals = [lit for lit in term_norm.split(" ") if lit]
            if not literals:
                continue

            # Invert every literal:
            # - A  -> A'
            # - A' -> A
            inverted: List[str] = []
            for lit in literals:
                if lit.endswith("'"):
                    inverted.append(lit[:-1])
                else:
                    inverted.append(lit + "'")

            # Product term negation becomes a sum (OR) of inverted literals.
            pos_factors.append("(" + " + ".join(inverted) + ")")

        if not pos_factors:
            # Everything collapsed to neutral terms.
            return "1"

        # ORs became ANDs => multiply factors (concatenate parentheses).
        return "".join(pos_factors)

    @staticmethod
    def generate_demorgans_steps(sop_expression: str, target_name: str) -> List[str]:
        """
        Generate step-by-step Boolean algebra derivation for converting an SOP expression
        into a POS expression using De Morgan's Laws.

        Args:
            sop_expression: SOP string like "A B' + C"
            target_name: name like "F" or "F'"

        Returns:
            List of formatted lines (strings).
        """
        if sop_expression is None:
            raise TypeError("sop_expression cannot be None.")
        if target_name is None:
            raise TypeError("target_name cannot be None.")

        target = str(target_name).strip() or "Target"
        s = " ".join(str(sop_expression).strip().split())

        steps: List[str] = []

        # Step 1: starting point
        steps.append(f"Step 1: {target} = ({s})'")

        # Normalize constants early for clarity
        if s == "":
            steps.append(f"Step 2: {target} = (1)'")
            steps.append(f"Step 3: {target} = 0")
            return steps
        if s == "0":
            steps.append(f"Step 2: {target} = (0)'")
            steps.append(f"Step 3: {target} = 1")
            return steps
        if s == "1":
            steps.append(f"Step 2: {target} = (1)'")
            steps.append(f"Step 3: {target} = 0")
            return steps

        # Split SOP terms by '+'
        raw_terms = [t.strip() for t in s.split("+") if t.strip()]
        if not raw_terms:
            steps.append(f"Step 2: {target} = (1)'")
            steps.append(f"Step 3: {target} = 0")
            return steps

        # Step 2: Break the line, change the sign
        # (P1 + P2 + ...)' = (P1)' * (P2)' * ...
        step2_factors = [f"({t})'" for t in raw_terms]
        steps.append(f"Step 2: {target} = " + " * ".join(step2_factors))

        # Step 3: Final internal inversion
        # (A B' C)' = (A' + B + C')
        pos_factors: List[str] = []
        for term in raw_terms:
            term_norm = " ".join(term.split())

            if term_norm == "0":
                # (0)' = 1 => neutral in multiplication
                pos_factors.append("(1)")
                continue
            if term_norm == "1":
                # (1)' = 0 => whole product becomes 0
                pos_factors = ["(0)"]
                break

            literals = [lit for lit in term_norm.split(" ") if lit]
            inverted: List[str] = []
            for lit in literals:
                if lit.endswith("'"):
                    inverted.append(lit[:-1])
                else:
                    inverted.append(lit + "'")

            if not inverted:
                pos_factors.append("(1)")
            else:
                pos_factors.append("(" + " + ".join(inverted) + ")")

        steps.append(f"Step 3: {target} = " + " * ".join(pos_factors))
        return steps

    def get_all_expressions(
        self, minterms: List[int], maxterms: List[int], dont_cares: List[int]
    ) -> Dict[str, object]:
        """
        Convenience wrapper to generate SOP and POS forms for both F and F'.

        Inputs:
        - minterms: indices where F = 1
        - maxterms: indices where F = 0  (i.e., where F' = 1)
        - dont_cares: indices where F = X (ignored for coverage in phase 2)

        Output dict (strings + metadata for derivations/teaching UI):
        - f_sop        : minimized SOP for F
        - f_pos        : POS for F (derived from minimized SOP for F')
        - f_prime_sop  : minimized SOP for F'
        - f_prime_pos  : POS for F' (derived from minimized SOP for F)
        """

        # ---- Solve F from minterms (1s) ----
        pis_f = self.get_prime_implicants(minterms, dont_cares)
        selected_f, f_sop, f_selected_coverage = self.get_minimum_expression(pis_f, minterms)

        # ---- Solve F' from maxterms (0s of F) ----
        pis_fp = self.get_prime_implicants(maxterms, dont_cares)
        selected_fp, f_prime_sop, fp_selected_coverage = self.get_minimum_expression(pis_fp, maxterms)

        # ---- POS conversion via De Morgan (teaching feature) ----
        f_pos = self.sop_to_pos(f_prime_sop)
        f_prime_pos = self.sop_to_pos(f_sop)

        return {
            "f_sop": f_sop,
            "f_pos": f_pos,
            "f_prime_sop": f_prime_sop,
            "f_prime_pos": f_prime_pos,
            # Extra metadata for derivations / teaching UI:
            # expose "all prime implicants" (phase 1) vs "selected implicants" (phase 2)
            "num_variables": self.num_variables,
            "f_all_prime_implicants": pis_f,
            "f_selected_prime_implicants": selected_f,
            "f_prime_all_prime_implicants": pis_fp,
            "f_prime_selected_prime_implicants": selected_fp,
            # For textbook-style derivations:
            "f_active_minterms": sorted(set(minterms or [])),
            "f_prime_active_minterms": sorted(set(maxterms or [])),
            "f_selected_pi_coverage": f_selected_coverage,
            "f_prime_selected_pi_coverage": fp_selected_coverage,
        }

    def patterns_to_expression(self, patterns: List[str]) -> str:
        """
        Convert implicant patterns to a readable SOP Boolean expression.

        Example:
            patterns = ['1-0', '0-1']
            -> "A C' + A' C"

        Rules:
        - '1'  => variable (e.g., A)
        - '0'  => complemented variable (e.g., A')
        - '-'  => omitted (don't care)
        - pattern of all '-' => "1" (tautology term)
        - empty list => "0"
        """

        if not patterns:
            return "0"

        var_names = self._variable_names(self.num_variables)
        terms: List[str] = []

        for pat in patterns:
            literals: List[str] = []
            for name, ch in zip(var_names, pat):
                if ch == "-":
                    continue
                if ch == "1":
                    literals.append(name)
                elif ch == "0":
                    literals.append(f"{name}'")
                else:
                    raise ValueError(f"Invalid pattern character '{ch}' in '{pat}'.")

            # If everything was '-', the implicant is always true.
            if not literals:
                terms.append("1")
            else:
                terms.append(" ".join(literals))

        # OR between product terms.
        return " + ".join(terms)

    def pattern_to_literal_string(self, pattern: str) -> str:
        """
        Convert a dash-pattern like '00-' into a compact Boolean literal term like "A'B'".
        '-' variables are omitted.
        """
        if pattern is None:
            raise TypeError("pattern cannot be None.")
        if len(pattern) != self.num_variables:
            raise ValueError(f"Pattern '{pattern}' length must be {self.num_variables}.")

        var_names = self._variable_names(self.num_variables)
        out: List[str] = []
        for name, ch in zip(var_names, pattern):
            if ch == "-":
                continue
            if ch == "1":
                out.append(name)
            elif ch == "0":
                out.append(f"{name}'")
            else:
                raise ValueError(f"Invalid pattern character '{ch}' in '{pattern}'.")
        return "".join(out) if out else "1"

    @staticmethod
    def _variable_names(n: int) -> List[str]:
        """
        Return variable names A, B, C, ... for the given n.
        """

        base = [chr(ord("A") + i) for i in range(26)]
        if n <= len(base):
            return base[:n]
        # If someone uses more than 26 variables, fall back to A0, A1... (unlikely for K-map apps).
        return base + [f"A{i}" for i in range(n - len(base))]

    def _pattern_covers_minterm(self, pattern: str, minterm: int) -> bool:
        """
        Check whether a prime implicant pattern covers a given minterm index.

        Coverage rule:
        - Convert minterm to padded binary string (same length as pattern)
        - For each position:
          - '-' matches both 0 and 1 (always ok)
          - '0' must match '0'
          - '1' must match '1'
        """

        bits = format(minterm, f"0{self.num_variables}b")
        for pch, bch in zip(pattern, bits):
            if pch == "-":
                continue
            if pch != bch:
                return False
        return True

    @staticmethod
    def _group_by_ones(terms: Iterable[_Term]) -> Dict[int, List[_Term]]:
        """
        Group terms by the number of literal '1' bits in their pattern.

        Important detail:
        - We ignore '-' when counting ones.
          '-' does not mean 0 or 1 specifically; it's "either".
          Standard Q-M grouping counts only actual '1' characters.
        """

        groups: Dict[int, List[_Term]] = {}
        for term in terms:
            ones = term.pattern.count("1")
            groups.setdefault(ones, []).append(term)

        # Sort each group for deterministic behavior/debugging.
        for k in groups:
            groups[k].sort(key=lambda t: t.pattern)
        return groups

    @staticmethod
    def _try_combine(t1: _Term, t2: _Term) -> _Term | None:
        """
        Try to combine two terms.

        Two terms are combinable if:
        - They differ in exactly ONE position, and
        - At that position they are '0' vs '1' (a single concrete bit flip), and
        - All other positions match exactly, including '-' positions.

        Why the strict '-' rule?
        - If one pattern has '-' and the other has '0'/'1' in a position, they are
          not differing by a single *concrete* bit; combining would be incorrect.

        Example (valid):
            0101
            0111   -> differ only at bit index 2 => 01-1

        Example (invalid):
            0-01
            0101   -> differs at '-' vs '1' (not a single concrete bit flip) => cannot combine
        """

        p1 = t1.pattern
        p2 = t2.pattern
        if len(p1) != len(p2):
            return None

        diff_index = -1

        for i, (a, b) in enumerate(zip(p1, p2)):
            if a == b:
                continue

            # If either is '-', they're not compatible for a single-bit combine.
            if a == "-" or b == "-":
                return None

            # Must be a literal 0/1 flip.
            if {a, b} != {"0", "1"}:
                return None

            # Only one differing position allowed.
            if diff_index != -1:
                return None
            diff_index = i

        # If diff_index was never set, patterns are identical -> no combining.
        if diff_index == -1:
            return None

        merged = list(p1)
        merged[diff_index] = "-"
        merged_pattern = "".join(merged)

        # Union of covered indices.
        return _Term(pattern=merged_pattern, covered=t1.covered | t2.covered)


def get_all_derivations(f_sop: str, f_prime_sop: str) -> str:
    """
    Return a single formatted multi-line string describing how SOP and POS are derived.

    - F (SOP) is derived from grouping the 1s.
    - F prime (SOP) is derived from grouping the 0s.
    - F (POS) comes from applying De Morgan to SOP(F').
    - F' (POS) comes from applying De Morgan to SOP(F).
    """
    lines: List[str] = []
    lines.append("F (SOP) is derived from grouping the 1s.")
    lines.append("F prime (SOP) is derived from grouping the 0s.")
    lines.append("")

    lines.append("Derivation of F (POS) from SOP(F') using De Morgan's Laws:")
    lines.extend(QuineMcCluskey.generate_demorgans_steps(f_prime_sop, "F"))
    lines.append("")

    lines.append("Derivation of F prime (POS) from SOP(F) using De Morgan's Laws:")
    lines.extend(QuineMcCluskey.generate_demorgans_steps(f_sop, "F'"))

    return "\n".join(lines)


def get_all_derivations_from_solution(expressions: Dict[str, object]) -> str:
    """
    Expanded derivations for the UI:
    - SOP simplification: sum(ALL prime implicants) -> minimized SOP
      by eliminating redundant implicants (coverage/consensus reasoning).
    - Then De Morgan steps: SOP(F') -> POS(F) and SOP(F) -> POS(F').
    """
    if expressions is None:
        raise TypeError("expressions cannot be None.")

    n_obj = expressions.get("num_variables")
    if not isinstance(n_obj, int) or n_obj <= 0:
        raise ValueError("expressions must include a positive integer 'num_variables'.")

    qm = QuineMcCluskey(int(n_obj))

    def _as_str_list(key: str) -> List[str]:
        v = expressions.get(key)
        if v is None:
            return []
        if not isinstance(v, list) or any(not isinstance(x, str) for x in v):
            raise TypeError(f"expressions['{key}'] must be a list[str].")
        return list(v)

    f_all = _as_str_list("f_all_prime_implicants")
    f_sel = _as_str_list("f_selected_prime_implicants")
    fp_all = _as_str_list("f_prime_all_prime_implicants")
    fp_sel = _as_str_list("f_prime_selected_prime_implicants")

    f_sop = str(expressions.get("f_sop", "")).strip()
    f_prime_sop = str(expressions.get("f_prime_sop", "")).strip()

    lines: List[str] = []
    lines.append("F (SOP) is derived from grouping the 1s.")
    lines.append("F prime (SOP) is derived from grouping the 0s.")
    lines.append("")

    def _sop_textbook_block(
        target_label: str,
        active_minterms: List[int],
        selected_pis: List[str],
        selected_coverage: Dict[str, List[int]],
        minimized_sop: str,
    ) -> List[str]:
        block: List[str] = []

        block.append(f"{target_label} SOP Derivation (Raw Minterms → Prime Implicants → Minimized SOP):")

        # Step 1: Unsimplified Expression (full literal form of all active minterms)
        unsimplified_terms = [minterm_to_literal(m, qm.num_variables) for m in sorted(set(active_minterms or []))]
        block.append(
            f"Step 1: {target_label} (Unsimplified Expression) = " + (" + ".join(unsimplified_terms) if unsimplified_terms else "0")
        )

        # Step 2: Grouping Breakdown (selected implicants + which raw minterms formed them)
        block.append("Step 2: Grouping Breakdown:")
        if not selected_pis:
            block.append("  (no prime implicants selected)")
        else:
            for pi in selected_pis:
                covered = selected_coverage.get(pi, [])
                covered_literals = [minterm_to_literal(m, qm.num_variables) for m in covered]
                left = " + ".join(covered_literals) if covered_literals else "(none)"
                right = qm.pattern_to_literal_string(pi)
                block.append(f"  {left} ➔ Simplifies to {right}")

        # Step 3: Final Minimized SOP
        block.append(f"Step 3: {target_label} (Final Minimized SOP) = {minimized_sop}")
        return block

    f_active = expressions.get("f_active_minterms")
    f_cov = expressions.get("f_selected_pi_coverage")
    if not isinstance(f_active, list) or any(not isinstance(x, int) for x in f_active):
        f_active = []
    if not isinstance(f_cov, dict) or any(not isinstance(k, str) or not isinstance(v, list) for k, v in f_cov.items()):
        f_cov = {}

    # F SOP derivation (textbook-style)
    lines.extend(_sop_textbook_block("F", list(f_active), f_sel, dict(f_cov), f_sop))
    lines.append("")

    # F POS via De Morgan from SOP(F')
    lines.append("Derivation of F (POS) from SOP(F') using De Morgan's Laws:")
    lines.extend(QuineMcCluskey.generate_demorgans_steps(f_prime_sop, "F"))
    lines.append("")

    fp_active = expressions.get("f_prime_active_minterms")
    fp_cov = expressions.get("f_prime_selected_pi_coverage")
    if not isinstance(fp_active, list) or any(not isinstance(x, int) for x in fp_active):
        fp_active = []
    if not isinstance(fp_cov, dict) or any(not isinstance(k, str) or not isinstance(v, list) for k, v in fp_cov.items()):
        fp_cov = {}

    # F' SOP derivation (textbook-style)
    lines.extend(_sop_textbook_block("F'", list(fp_active), fp_sel, dict(fp_cov), f_prime_sop))
    lines.append("")

    # F' POS via De Morgan from SOP(F)
    lines.append("Derivation of F prime (POS) from SOP(F) using De Morgan's Laws:")
    lines.extend(QuineMcCluskey.generate_demorgans_steps(f_sop, "F'"))

    return "\n".join(lines)

