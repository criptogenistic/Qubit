# Qubit

Qubit is a lightweight quantum programming language prototype and runtime for experimenting with qubit operations, measurement, and hybrid classical/quantum control.

This repository contains:
- a small QPL interpreter in Python
- a minimal runtime for single- and two-qubit gates
- example quantum programs

## Features

- qreg declarations
- gate application: H, X, Y, Z, Rx, Ry, Rz, CNOT
- measurement into classical bits
- conditional branching with `if`
- simple simulator backend using a state-vector model

## Quick start

1. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Run an example:

```bash
python qpl.py examples/hello.qpl
```

## Example language

```qpl
program bell_state {
    qreg q[2]
    H(q[0])
    CNOT(q[0], q[1])

    bit a = measure(q[0])
    bit b = measure(q[1])

    print("Bell state result:", a, b)
}
```

This produces a Bell state, then measures both qubits and prints the result.

## Minimal grammar

```ebnf
program       := "program" ident "{" stmt* "}"
stmt          := qreg_decl | gate_call | measure_stmt | if_stmt | print_stmt
qreg_decl     := "qreg" ident "[" int "]"
measure_stmt  := "bit" ident "=" "measure" "(" qubit_ref ")"
if_stmt       := "if" "(" ident "==" int ")" "{" stmt* "}"
gate_call     := gate_name "(" qubit_ref ["," qubit_ref] ")"
qubit_ref     := ident "[" int "]"
```

## Repository layout

```text
.
├── README.md
├── qpl.py
├── requirements.txt
└── examples/
    └── hello.qpl
```

## Notes

This project is intentionally small and educational. It is designed as a foundation for a richer quantum DSL and compiler pipeline.
