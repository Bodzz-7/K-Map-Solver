import tkinter as tk
from tkinter import ttk

from kmap_visuals import KMapCanvas
from qm_algorithm import QuineMcCluskey, get_all_derivations, get_all_derivations_from_solution


class KMapMinimizerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("K-Map Logic Minimizer")
        self.minsize(1100, 640)

        self._setup_style()

        self.var_count = tk.IntVar(value=4)
        self.results_vars = {
            "f_sop": tk.StringVar(value="—"),
            "f_pos": tk.StringVar(value="—"),
            "f_prime_sop": tk.StringVar(value="—"),
            "f_prime_pos": tk.StringVar(value="—"),
        }

        # Centralized truth-table/K-map state:
        # each index holds '0', '1', or 'X' (don't care).
        self.cell_states: list[str] = []

        # Mapping between K-map (row, col) and truth table indices (minterm indices)
        self.kmap_rc_to_index: dict[tuple[int, int], int] = {}
        self.index_to_kmap_rc: dict[int, tuple[int, int]] = {}

        # Truth table widgets
        self.f_buttons: list[ttk.Button] = []

        # Latest solver outputs (used for Algebra Steps window)
        self.last_expressions: dict[str, object] | None = None

        root = ttk.Frame(self, padding=14, style="App.TFrame")
        root.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        root.rowconfigure(2, weight=0)

        self._build_top_controls(root)
        self._build_main_area(root)
        self._build_results_frame(root)

        self._on_var_count_changed(initial=True)

    # -----------------------
    # UI construction
    # -----------------------

    def _setup_style(self) -> None:
        self.configure(bg="#0f172a")
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("App.TFrame", background="#0f172a")
        style.configure("Card.TLabelframe", background="#0f172a", foreground="#e5e7eb")
        style.configure("Card.TLabelframe.Label", background="#0f172a", foreground="#e5e7eb")
        style.configure("Top.TFrame", background="#0f172a")
        style.configure("Top.TLabel", background="#0f172a", foreground="#e5e7eb", font=("Segoe UI", 11))

        style.configure(
            "Primary.TButton",
            font=("Segoe UI", 11, "bold"),
            padding=(14, 10),
            foreground="#0b1220",
            background="#60a5fa",
        )
        style.map("Primary.TButton", background=[("active", "#93c5fd"), ("pressed", "#3b82f6")])

        style.configure(
            "FCell.TButton",
            font=("Segoe UI", 11, "bold"),
            padding=(10, 8),
        )

        style.configure(
            "Header.TLabel",
            background="#111827",
            foreground="#e5e7eb",
            font=("Segoe UI", 10, "bold"),
            padding=(10, 8),
            anchor="center",
        )
        style.configure(
            "Cell.TLabel",
            background="#0b1220",
            foreground="#e5e7eb",
            font=("Consolas", 10),
            padding=(10, 8),
            anchor="center",
        )
        style.configure("Results.TLabelframe", background="#0f172a", foreground="#e5e7eb")
        style.configure("Results.TLabelframe.Label", background="#0f172a", foreground="#e5e7eb")
        style.configure("ResultsKey.TLabel", background="#0f172a", foreground="#cbd5e1", font=("Segoe UI", 11))
        style.configure("ResultsVal.TLabel", background="#0f172a", foreground="#e5e7eb", font=("Segoe UI", 12, "bold"))

    def _build_top_controls(self, parent: ttk.Frame) -> None:
        top = ttk.Frame(parent, style="Top.TFrame")
        top.grid(row=0, column=0, sticky="ew", padx=2, pady=(0, 12))
        top.columnconfigure(3, weight=1)

        ttk.Label(top, text="K-Map Logic Minimizer", style="Top.TLabel", font=("Segoe UI", 14, "bold")).grid(
            row=0, column=0, sticky="w", padx=(0, 18)
        )
        ttk.Label(top, text="Variables", style="Top.TLabel").grid(row=0, column=1, sticky="w", padx=(0, 8))

        self.var_combo = ttk.Combobox(
            top,
            values=(2, 3, 4),
            width=6,
            state="readonly",
            textvariable=self.var_count,
            font=("Segoe UI", 11),
        )
        self.var_combo.grid(row=0, column=2, sticky="w", padx=(0, 14))
        self.var_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_var_count_changed())

        ttk.Button(top, text="Solve", style="Primary.TButton", command=self.solve).grid(row=0, column=4, sticky="e")

    def _build_main_area(self, parent: ttk.Frame) -> None:
        main = ttk.Frame(parent, style="App.TFrame")
        main.grid(row=1, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        # Left: Truth Table (scrollable)
        left_border = tk.Frame(main, bg="#333333")
        left_border.grid(row=0, column=0, sticky="nsew", padx=(6, 10), pady=6)
        left_border.columnconfigure(0, weight=1)
        left_border.rowconfigure(0, weight=1)

        left = ttk.Labelframe(left_border, text="Truth Table", padding=10, style="Card.TLabelframe")
        left.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)

        self.tt_canvas = tk.Canvas(left, bg="#0f172a", highlightthickness=0)
        self.tt_canvas.grid(row=0, column=0, sticky="nsew")
        tt_vbar = ttk.Scrollbar(left, orient="vertical", command=self.tt_canvas.yview)
        tt_vbar.grid(row=0, column=1, sticky="ns")
        self.tt_canvas.configure(yscrollcommand=tt_vbar.set)

        self.table_frame = ttk.Frame(self.tt_canvas, style="App.TFrame")
        self._table_window = self.tt_canvas.create_window((0, 0), window=self.table_frame, anchor="nw")
        self.table_frame.bind("<Configure>", self._on_tt_configure)
        self.tt_canvas.bind("<Configure>", self._on_tt_canvas_configure)

        # Right: K-Map
        right_border = tk.Frame(main, bg="#333333")
        right_border.grid(row=0, column=1, sticky="nsew", padx=(10, 6), pady=6)
        right_border.columnconfigure(0, weight=1)
        right_border.rowconfigure(0, weight=1)

        right = ttk.Labelframe(right_border, text="K-Map", padding=10, style="Card.TLabelframe")
        right.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        self.kmap = KMapCanvas(right, self.var_count.get())
        self.kmap.grid(row=0, column=0, sticky="nsew")
        self.kmap.set_cell_click_callback(self._on_kmap_cell_clicked)

    def _build_results_frame(self, parent: ttk.Frame) -> None:
        results = ttk.Labelframe(parent, text="Results", padding=10, style="Results.TLabelframe")
        results.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        results.columnconfigure(1, weight=1)

        rows = [
            ("F (SOP):", "f_sop"),
            ("F (POS):", "f_pos"),
            ("F' (SOP):", "f_prime_sop"),
            ("F' (POS):", "f_prime_pos"),
        ]

        for r, (label, key) in enumerate(rows):
            ttk.Label(results, text=label, style="ResultsKey.TLabel").grid(row=r, column=0, sticky="w", padx=(2, 10))
            ttk.Label(results, textvariable=self.results_vars[key], style="ResultsVal.TLabel").grid(
                row=r, column=1, sticky="ew"
            )

        actions = ttk.Frame(results)
        actions.grid(row=len(rows), column=0, columnspan=2, sticky="e", pady=(10, 0), padx=2)

        self.algebra_button = ttk.Button(
            results,
            text="View Algebra Steps",
            command=self.show_algebra_steps,
            state="disabled",
        )
        self.algebra_button.grid(in_=actions, row=0, column=0, sticky="e")

    # -----------------------
    # Truth table behavior
    # -----------------------

    def _on_tt_configure(self, _event: tk.Event) -> None:
        self.tt_canvas.configure(scrollregion=self.tt_canvas.bbox("all"))

    def _on_tt_canvas_configure(self, event: tk.Event) -> None:
        self.tt_canvas.itemconfigure(self._table_window, width=event.width)

    def _on_var_count_changed(self, initial: bool = False) -> None:
        n = int(self.var_count.get())
        self._configure_kmap_index_mapping(n)

        # Reset state when variable count changes (or on first load).
        self.cell_states = ["0"] * (2**n)

        self._regenerate_truth_table()
        self.kmap.draw_grid(n)
        self.kmap.set_cell_click_callback(self._on_kmap_cell_clicked)
        self._sync_all_to_kmap()
        self._clear_results()
        if not initial:
            self.tt_canvas.yview_moveto(0.0)

    def _clear_results(self) -> None:
        for v in self.results_vars.values():
            v.set("—")
        self.last_expressions = None
        if hasattr(self, "algebra_button"):
            self.algebra_button.configure(state="disabled")

    def _configure_kmap_index_mapping(self, n: int) -> None:
        """
        Use the exact mappings provided by the user:

        2 Vars:
            {(0,0):0,(0,1):1,(1,0):2,(1,1):3}
        3 Vars:
            {(0,0):0,(0,1):1,(0,2):3,(0,3):2,(1,0):4,(1,1):5,(1,2):7,(1,3):6}
        4 Vars:
            {(0,0):0,(0,1):1,(0,2):3,(0,3):2,(1,0):4,(1,1):5,(1,2):7,(1,3):6,
             (2,0):12,(2,1):13,(2,2):15,(2,3):14,(3,0):8,(3,1):9,(3,2):11,(3,3):10}
        """

        if n == 2:
            self.kmap_rc_to_index = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
        elif n == 3:
            self.kmap_rc_to_index = {
                (0, 0): 0,
                (0, 1): 1,
                (0, 2): 3,
                (0, 3): 2,
                (1, 0): 4,
                (1, 1): 5,
                (1, 2): 7,
                (1, 3): 6,
            }
        elif n == 4:
            self.kmap_rc_to_index = {
                (0, 0): 0,
                (0, 1): 1,
                (0, 2): 3,
                (0, 3): 2,
                (1, 0): 4,
                (1, 1): 5,
                (1, 2): 7,
                (1, 3): 6,
                (2, 0): 12,
                (2, 1): 13,
                (2, 2): 15,
                (2, 3): 14,
                (3, 0): 8,
                (3, 1): 9,
                (3, 2): 11,
                (3, 3): 10,
            }
        else:
            self.kmap_rc_to_index = {}

        self.index_to_kmap_rc = {idx: rc for rc, idx in self.kmap_rc_to_index.items()}

    def _clear_truth_table(self) -> None:
        for child in self.table_frame.winfo_children():
            child.destroy()
        self.f_buttons.clear()

    def _regenerate_truth_table(self) -> None:
        self._clear_truth_table()

        n = int(self.var_count.get())
        var_names = ["A", "B", "C", "D"][:n]
        headers = [*var_names, "F"]

        for c, name in enumerate(headers):
            ttk.Label(self.table_frame, text=name, style="Header.TLabel").grid(
                row=0, column=c, sticky="nsew", padx=1, pady=(0, 2)
            )
            self.table_frame.columnconfigure(c, weight=1)

        row_count = 2**n

        for r in range(row_count):
            bits = self._row_to_bits(r, n)
            for c in range(n):
                ttk.Label(self.table_frame, text=str(bits[c]), style="Cell.TLabel").grid(
                    row=r + 1, column=c, sticky="nsew", padx=1, pady=1
                )

            btn = ttk.Button(
                self.table_frame,
                text=self.cell_states[r] if r < len(self.cell_states) else "0",
                style="FCell.TButton",
                command=lambda idx=r: self._cycle_state(idx),
                width=4,
            )
            btn.grid(row=r + 1, column=n, sticky="nsew", padx=1, pady=1)
            self.f_buttons.append(btn)

        self._sync_all_to_kmap()

    @staticmethod
    def _row_to_bits(row_index: int, n: int) -> list[int]:
        return [int(x) for x in format(row_index, f"0{n}b")]

    def _cycle_state(self, index: int) -> None:
        current = self.cell_states[index]
        nxt = {"0": "1", "1": "X", "X": "0"}[current]
        self.update_state(index, nxt)

    def update_state(self, index: int, new_value: str) -> None:
        """
        Single sync point for BOTH Truth Table and K-map clicks.

        Updates:
        - self.cell_states[index]
        - Truth Table button text at that index
        - K-map button text at corresponding (row, col)
        """

        if new_value not in ("0", "1", "X"):
            return
        if index < 0 or index >= len(self.cell_states):
            return

        self.cell_states[index] = new_value

        # Update truth table button
        if 0 <= index < len(self.f_buttons):
            self.f_buttons[index].configure(text=new_value)

        # Update kmap cell button
        rc = self.index_to_kmap_rc.get(index)
        if rc is not None:
            self.kmap.set_cell_value(rc[0], rc[1], new_value)

    def _sync_all_to_kmap(self) -> None:
        for idx, val in enumerate(self.cell_states):
            rc = self.index_to_kmap_rc.get(idx)
            if rc is not None:
                self.kmap.set_cell_value(rc[0], rc[1], val)

    def _on_kmap_cell_clicked(self, row: int, col: int) -> None:
        idx = self.kmap_rc_to_index.get((row, col))
        if idx is None:
            return
        self._cycle_state(idx)

    # -----------------------
    # Solve wiring
    # -----------------------

    def solve(self) -> None:
        n = int(self.var_count.get())

        minterms: list[int] = []
        maxterms: list[int] = []
        dont_cares: list[int] = []

        for i, v in enumerate(self.cell_states):
            if v == "1":
                minterms.append(i)
            elif v == "0":
                maxterms.append(i)
            elif v == "X":
                dont_cares.append(i)

        qm = QuineMcCluskey(n)
        expressions = qm.get_all_expressions(minterms=minterms, maxterms=maxterms, dont_cares=dont_cares)
        self.last_expressions = expressions

        self.results_vars["f_sop"].set(expressions["f_sop"])
        self.results_vars["f_pos"].set(expressions["f_pos"])
        self.results_vars["f_prime_sop"].set(expressions["f_prime_sop"])
        self.results_vars["f_prime_pos"].set(expressions["f_prime_pos"])

        if hasattr(self, "algebra_button"):
            self.algebra_button.configure(state="normal")

        # K-map values are already bound via update_state; still ensure sync.
        self._sync_all_to_kmap()

    def show_algebra_steps(self) -> None:
        if self.last_expressions is not None:
            derivation_text = get_all_derivations_from_solution(self.last_expressions)
        else:
            # Fallback (shouldn't happen if button is enabled only after solve)
            f_sop = self.results_vars["f_sop"].get()
            f_prime_sop = self.results_vars["f_prime_sop"].get()
            derivation_text = get_all_derivations(f_sop, f_prime_sop)

        win = tk.Toplevel(self)
        win.title("Boolean Algebra Derivations")
        win.geometry("800x600")
        win.minsize(800, 600)

        container = ttk.Frame(win, padding=10)
        container.grid(row=0, column=0, sticky="nsew")
        win.columnconfigure(0, weight=1)
        win.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

        scrollbar = ttk.Scrollbar(container, orient="vertical")
        scrollbar.grid(row=0, column=1, sticky="ns")

        text = tk.Text(
            container,
            wrap=tk.WORD,
            font=("Consolas", 11),
            yscrollcommand=scrollbar.set,
            padx=12,
            pady=12,
        )
        text.grid(row=0, column=0, sticky="nsew")
        scrollbar.configure(command=text.yview)

        text.tag_configure("body", spacing1=2, spacing2=2, spacing3=6)
        text.insert("1.0", derivation_text if derivation_text else "(no derivation available)", ("body",))
        text.configure(state="disabled")


def main() -> None:
    app = KMapMinimizerApp()
    app.mainloop()


if __name__ == "__main__":
    main()

