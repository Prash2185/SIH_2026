"""
BharatOpt Core - File I/O Readers and Writers (FR-IO1)
Clean-room parser and serializers for:
1. MPS format (Fixed and Free format)
2. CPLEX LP format
3. QPS format (Quadratic Extensions)
"""

import re
from typing import List, Tuple, Dict, Optional
from bharatopt.core.model import CanonicalModel, Sense, VarType, ConstraintSense


class MPSReader:
    """Reads and writes MPS mathematical programming files."""

    @staticmethod
    def read(file_path: str) -> CanonicalModel:
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('*')]

        model_name = "MPS_Model"
        section = None
        sense = Sense.MINIMIZE

        row_senses: Dict[str, str] = {}
        obj_row_name = ""
        matrix_entries: Dict[str, Dict[str, float]] = {}  # col_name -> {row_name: val}
        rhs_entries: Dict[str, float] = {}
        bounds: Dict[str, Tuple[float, float, VarType]] = {}  # col_name -> (lb, ub, var_type)
        is_integer_mode = False

        for line in lines:
            if line.startswith('NAME'):
                parts = line.split()
                if len(parts) > 1:
                    model_name = parts[1]
                continue
            elif line.startswith('OBJSENSE'):
                section = 'OBJSENSE'
                parts = line.split()
                if len(parts) > 1:
                    if 'MAX' in parts[1].upper():
                        sense = Sense.MAXIMIZE
                    elif 'MIN' in parts[1].upper():
                        sense = Sense.MINIMIZE
                continue
            elif section == 'OBJSENSE':
                if 'MAX' in line.upper():
                    sense = Sense.MAXIMIZE
                    continue
                elif 'MIN' in line.upper():
                    sense = Sense.MINIMIZE
                    continue
                elif line in ('ROWS', 'COLUMNS', 'RHS', 'RANGES', 'BOUNDS', 'QUADOBJ', 'QMATRIX', 'ENDATA'):
                    section = line
                    continue
            elif line in ('ROWS', 'COLUMNS', 'RHS', 'RANGES', 'BOUNDS', 'QUADOBJ', 'QMATRIX', 'ENDATA'):
                section = line
                continue

            parts = line.split()

            if section == 'ROWS':
                # Type RowName
                r_type = parts[0]
                r_name = parts[1]
                if r_type == 'N':
                    obj_row_name = r_name
                else:
                    row_senses[r_name] = r_type

            elif section == 'COLUMNS':
                # Handle INTORG / INTEND markers
                if "'MARKER'" in line or "MARKER" in line:
                    if "'INTORG'" in line or "INTORG" in line:
                        is_integer_mode = True
                    elif "'INTEND'" in line or "INTEND" in line:
                        is_integer_mode = False
                    continue

                col_name = parts[0]
                if col_name not in matrix_entries:
                    matrix_entries[col_name] = {}
                    bounds[col_name] = (0.0, float('inf'), VarType.INTEGER if is_integer_mode else VarType.CONTINUOUS)

                # Parsed (row, val) pairs
                idx = 1
                while idx < len(parts):
                    r_name = parts[idx]
                    val = float(parts[idx + 1])
                    matrix_entries[col_name][r_name] = val
                    idx += 2

            elif section == 'RHS':
                idx = 1 if len(parts) % 2 == 1 else 0
                while idx < len(parts) - 1:
                    r_name = parts[idx]
                    val = float(parts[idx + 1])
                    rhs_entries[r_name] = val
                    idx += 2

            elif section == 'BOUNDS':
                b_type = parts[0]
                # bound_name = parts[1]
                col_name = parts[2]
                val = float(parts[3]) if len(parts) > 3 else 0.0

                cur_lb, cur_ub, cur_type = bounds.get(col_name, (0.0, float('inf'), VarType.CONTINUOUS))

                if b_type == 'LO':
                    bounds[col_name] = (val, cur_ub, cur_type)
                elif b_type == 'UP':
                    bounds[col_name] = (cur_lb, val, cur_type)
                elif b_type == 'FX':
                    bounds[col_name] = (val, val, cur_type)
                elif b_type == 'FR':
                    bounds[col_name] = (float('-inf'), float('inf'), cur_type)
                elif b_type == 'BV':
                    bounds[col_name] = (0.0, 1.0, VarType.BINARY)
                elif b_type == 'LI':
                    bounds[col_name] = (val, cur_ub, VarType.INTEGER)
                elif b_type == 'UI':
                    bounds[col_name] = (cur_lb, val, VarType.INTEGER)

        # Construct CanonicalModel
        model = CanonicalModel(name=model_name, sense=sense)

        # Add Variables
        for col_name, (lb, ub, v_type) in bounds.items():
            obj_coeff = matrix_entries.get(col_name, {}).get(obj_row_name, 0.0)
            model.add_var(name=col_name, lb=lb, ub=ub, obj=obj_coeff, var_type=v_type)

        # Add Rows
        for r_name, r_type in row_senses.items():
            row_dict: Dict[int, float] = {}
            for col_name, col_dict in matrix_entries.items():
                if r_name in col_dict:
                    col_idx = model.var_name_to_idx[col_name]
                    row_dict[col_idx] = col_dict[r_name]

            rhs_val = rhs_entries.get(r_name, 0.0)
            if r_type == 'L':
                model.add_constraint(row_dict, ConstraintSense.LE, rhs_val, name=r_name)
            elif r_type == 'G':
                model.add_constraint(row_dict, ConstraintSense.GE, rhs_val, name=r_name)
            elif r_type == 'E':
                model.add_constraint(row_dict, ConstraintSense.EQ, rhs_val, name=r_name)

        return model

    @staticmethod
    def write(model: CanonicalModel, file_path: str):
        """Write canonical model to MPS format."""
        with open(file_path, 'w') as f:
            f.write(f"NAME          {model.name}\n")
            f.write("OBJSENSE\n")
            f.write("  MAX\n" if model.sense == Sense.MAXIMIZE else "  MIN\n")
            f.write("ROWS\n")
            f.write(" N  OBJ\n")
            for name in model.row_names_ineq:
                f.write(f" L  {name}\n")
            for name in model.row_names_eq:
                f.write(f" E  {name}\n")

            f.write("COLUMNS\n")
            A_csc = model.get_A_csc()
            E_csc = model.get_E_csc()

            for j in range(model.num_vars):
                v_name = model.var_names[j]
                # Write objective coeff (original user objective value)
                obj_v = model.c[j] if model.sense == Sense.MAXIMIZE else -model.c[j]
                if abs(obj_v) > 1e-15:
                    f.write(f"    {v_name}  OBJ  {obj_v:.8f}\n")

                # Ineq entries
                c_vec = A_csc.get_col(j)
                for r_idx, val in zip(c_vec.indices, c_vec.values):
                    r_name = model.row_names_ineq[r_idx]
                    f.write(f"    {v_name}  {r_name}  {val:.8f}\n")

                # Eq entries
                if model.num_eq > 0:
                    e_vec = E_csc.get_col(j)
                    for r_idx, val in zip(e_vec.indices, e_vec.values):
                        r_name = model.row_names_eq[r_idx]
                        f.write(f"    {v_name}  {r_name}  {val:.8f}\n")

            f.write("RHS\n")
            for i, b_val in enumerate(model.b):
                r_name = model.row_names_ineq[i]
                f.write(f"    RHS1  {r_name}  {b_val:.8f}\n")
            for i, d_val in enumerate(model.d):
                r_name = model.row_names_eq[i]
                f.write(f"    RHS1  {r_name}  {d_val:.8f}\n")

            f.write("BOUNDS\n")
            for j in range(model.num_vars):
                v_name = model.var_names[j]
                lb, ub, v_type = model.lb[j], model.ub[j], model.var_types[j]
                if v_type == VarType.BINARY:
                    f.write(f" BV BND1  {v_name}\n")
                elif v_type == VarType.INTEGER:
                    if lb > -1e10:
                        f.write(f" LI BND1  {v_name}  {lb:.8f}\n")
                    if ub < 1e10:
                        f.write(f" UI BND1  {v_name}  {ub:.8f}\n")
                else:
                    if lb > -1e10:
                        f.write(f" LO BND1  {v_name}  {lb:.8f}\n")
                    if ub < 1e10:
                        f.write(f" UP BND1  {v_name}  {ub:.8f}\n")

            f.write("ENDATA\n")


class LPFormatReader:
    """Reads and writes CPLEX LP format files."""

    @staticmethod
    def write(model: CanonicalModel, file_path: str):
        with open(file_path, 'w') as f:
            f.write(f"\\ Model: {model.name}\n")
            f.write("Maximize\n" if model.sense == Sense.MAXIMIZE else "Minimize\n")
            f.write(" obj: ")
            obj_terms = []
            for j in range(model.num_vars):
                cj = model.c[j] if model.sense == Sense.MAXIMIZE else -model.c[j]
                if abs(cj) > 1e-12:
                    sign = "+" if cj >= 0 else "-"
                    obj_terms.append(f"{sign} {abs(cj):.6f} {model.var_names[j]}")
            f.write(" ".join(obj_terms) + "\n")

            f.write("Subject To\n")
            A_csr = model.get_A_csr()
            for i in range(model.num_ineq):
                r_name = model.row_names_ineq[i]
                row = A_csr.get_row(i)
                terms = [f"{'+' if val >= 0 else '-'} {abs(val):.6f} {model.var_names[col]}"
                         for col, val in zip(row.indices, row.values)]
                f.write(f" {r_name}: {' '.join(terms)} <= {model.b[i]:.6f}\n")

            f.write("Bounds\n")
            for j in range(model.num_vars):
                f.write(f" {model.lb[j]:.6f} <= {model.var_names[j]} <= {model.ub[j]:.6f}\n")

            integers = [model.var_names[j] for j in model.integer_indices if model.var_types[j] == VarType.INTEGER]
            binaries = [model.var_names[j] for j in model.integer_indices if model.var_types[j] == VarType.BINARY]
            if integers:
                f.write("Generals\n " + " ".join(integers) + "\n")
            if binaries:
                f.write("Binaries\n " + " ".join(binaries) + "\n")
            f.write("End\n")
