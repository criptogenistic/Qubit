#!/usr/bin/env python3

import math
import random
import sys

import numpy as np


class Token:
    def __init__(self, kind, value, pos):
        self.kind = kind
        self.value = value
        self.pos = pos


class QPLTokenizer:
    def __init__(self, source):
        self.source = source
        self.index = 0

    def tokenize(self):
        tokens = []
        while self.index < len(self.source):
            ch = self.source[self.index]
            if ch.isspace():
                self.index += 1
                continue

            if ch in "{}[](),;":
                tokens.append(Token(ch, ch, self.index))
                self.index += 1
                continue

            if ch == '=':
                if self.source.startswith("==", self.index):
                    tokens.append(Token("==", "==", self.index))
                    self.index += 2
                else:
                    tokens.append(Token("=", "=", self.index))
                    self.index += 1
                continue

            if ch == '"':
                tokens.append(self._read_string())
                continue

            if ch.isdigit() or (ch == '.' and self.index + 1 < len(self.source) and self.source[self.index + 1].isdigit()):
                tokens.append(self._read_number())
                continue

            if ch.isalpha() or ch == '_':
                tokens.append(self._read_identifier())
                continue

            raise ValueError(f"Unexpected character '{ch}' at position {self.index}")

        tokens.append(Token("EOF", None, self.index))
        return tokens

    def _read_string(self):
        start = self.index
        self.index += 1
        chars = []
        while self.index < len(self.source):
            ch = self.source[self.index]
            if ch == '\\':
                self.index += 1
                if self.index >= len(self.source):
                    raise ValueError("Unterminated escape sequence")
                escape_char = self.source[self.index]
                mapping = {'n': '\n', 't': '\t', '"': '"', '\\': '\\'}
                chars.append(mapping.get(escape_char, escape_char))
                self.index += 1
                continue
            if ch == '"':
                self.index += 1
                return Token("STRING", ''.join(chars), start)
            chars.append(ch)
            self.index += 1
        raise ValueError("Unterminated string literal")

    def _read_number(self):
        start = self.index
        saw_dot = False
        while self.index < len(self.source):
            ch = self.source[self.index]
            if ch.isdigit():
                self.index += 1
                continue
            if ch == '.' and not saw_dot:
                saw_dot = True
                self.index += 1
                continue
            break
        text = self.source[start:self.index]
        value = float(text) if '.' in text else int(text)
        return Token("NUMBER", value, start)

    def _read_identifier(self):
        start = self.index
        while self.index < len(self.source):
            ch = self.source[self.index]
            if ch.isalnum() or ch == '_':
                self.index += 1
            else:
                break
        text = self.source[start:self.index]
        return Token("IDENT", text, start)


class QuantumRuntime:
    def __init__(self):
        self.state = np.array([1.0], dtype=complex)
        self.registers = {}
        self.bits = {}
        self.n_qubits = 0

    def qreg(self, name, size):
        if size <= 0:
            raise ValueError("qreg size must be positive")
        if name in self.registers:
            raise ValueError(f"Register '{name}' already exists")

        new_qubits = list(range(self.n_qubits, self.n_qubits + size))
        self.registers[name] = new_qubits

        zero_state = np.zeros(2 ** size, dtype=complex)
        zero_state[0] = 1.0
        self.state = np.kron(self.state, zero_state)
        self.n_qubits += size

    def resolve_qubit(self, qubit_ref):
        reg_name, index = qubit_ref
        if reg_name not in self.registers:
            raise ValueError(f"Unknown qubit register '{reg_name}'")
        if index < 0 or index >= len(self.registers[reg_name]):
            raise ValueError(f"Qubit index {index} out of range for register '{reg_name}'")
        return self.registers[reg_name][index]

    def apply_gate(self, gate_name, refs):
        if gate_name == "H":
            matrix = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
            self._apply_single_qubit(matrix, refs[0])
            return

        if gate_name == "X":
            matrix = np.array([[0, 1], [1, 0]], dtype=complex)
            self._apply_single_qubit(matrix, refs[0])
            return

        if gate_name == "Y":
            matrix = np.array([[0, -1j], [1j, 0]], dtype=complex)
            self._apply_single_qubit(matrix, refs[0])
            return

        if gate_name == "Z":
            matrix = np.array([[1, 0], [0, -1]], dtype=complex)
            self._apply_single_qubit(matrix, refs[0])
            return

        if gate_name == "S":
            matrix = np.array([[1, 0], [0, 1j]], dtype=complex)
            self._apply_single_qubit(matrix, refs[0])
            return

        if gate_name == "T":
            matrix = np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=complex)
            self._apply_single_qubit(matrix, refs[0])
            return

        if gate_name == "Rx":
            theta = refs[1]
            c = math.cos(theta / 2)
            s = -1j * math.sin(theta / 2)
            matrix = np.array([[c, s], [s, c]], dtype=complex)
            self._apply_single_qubit(matrix, refs[0])
            return

        if gate_name == "Ry":
            theta = refs[1]
            c = math.cos(theta / 2)
            s = math.sin(theta / 2)
            matrix = np.array([[c, -s], [s, c]], dtype=complex)
            self._apply_single_qubit(matrix, refs[0])
            return

        if gate_name == "Rz":
            theta = refs[1]
            matrix = np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex)
            self._apply_single_qubit(matrix, refs[0])
            return

        if gate_name in {"CNOT", "CZ", "SWAP"}:
            if len(refs) != 2:
                raise ValueError(f"{gate_name} expects exactly two qubits")
            q0 = self.resolve_qubit(refs[0])
            q1 = self.resolve_qubit(refs[1])
            if gate_name == "CNOT":
                matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
            elif gate_name == "CZ":
                matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=complex)
            else:
                matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
            self._apply_two_qubit_matrix(matrix, q0, q1)
            return

        if gate_name == "Toffoli":
            if len(refs) != 3:
                raise ValueError("Toffoli expects exactly three qubits")
            q0 = self.resolve_qubit(refs[0])
            q1 = self.resolve_qubit(refs[1])
            q2 = self.resolve_qubit(refs[2])
            matrix = np.zeros((8, 8), dtype=complex)
            for i in range(8):
                matrix[i, i] = 1
            # flip basis states where q0=q1=1 and q2=0, to q2=1 and vice versa
            matrix[6, 6] = 0
            matrix[7, 7] = 0
            matrix[7, 6] = 1
            matrix[6, 7] = 1
            self._apply_three_qubit_matrix(matrix, q0, q1, q2)
            return

        raise ValueError(f"Unsupported gate '{gate_name}'")

    def _apply_single_qubit(self, matrix, qubit_ref):
        target = self.resolve_qubit(qubit_ref)
        dim = len(self.state)
        n = int(math.log2(dim))
        new_state = np.zeros_like(self.state)

        for basis in range(dim):
            amp = self.state[basis]
            if abs(amp) < 1e-12:
                continue
            bit = (basis >> target) & 1
            for out_bit in range(2):
                out_basis = basis
                mask = 1 << target
                if out_bit == 0:
                    out_basis &= ~mask
                else:
                    out_basis |= mask
                new_state[out_basis] += matrix[out_bit, bit] * amp

        self.state = new_state

    def _apply_two_qubit_matrix(self, matrix, q0, q1):
        dim = len(self.state)
        n = int(math.log2(dim))
        new_state = np.zeros_like(self.state)

        for basis in range(dim):
            amp = self.state[basis]
            if abs(amp) < 1e-12:
                continue
            bits = [(basis >> i) & 1 for i in range(n)]
            current = bits[q0] * 2 + bits[q1]
            for out in range(4):
                out_basis = basis
                for bit_index, target_q in enumerate([q0, q1]):
                    mask = 1 << target_q
                    bit = (out >> (1 - bit_index)) & 1
                    if bit == 0:
                        out_basis &= ~mask
                    else:
                        out_basis |= mask
                new_state[out_basis] += matrix[out, current] * amp

        self.state = new_state

    def _apply_three_qubit_matrix(self, matrix, q0, q1, q2):
        dim = len(self.state)
        n = int(math.log2(dim))
        new_state = np.zeros_like(self.state)

        for basis in range(dim):
            amp = self.state[basis]
            if abs(amp) < 1e-12:
                continue
            bits = [(basis >> i) & 1 for i in range(n)]
            current = bits[q0] * 4 + bits[q1] * 2 + bits[q2]
            for out in range(8):
                out_basis = basis
                targets = [q0, q1, q2]
                for bit_index, target_q in enumerate(targets):
                    mask = 1 << target_q
                    bit = (out >> (2 - bit_index)) & 1
                    if bit == 0:
                        out_basis &= ~mask
                    else:
                        out_basis |= mask
                new_state[out_basis] += matrix[out, current] * amp

        self.state = new_state

    def measure(self, qubit_ref):
        target = self.resolve_qubit(qubit_ref)
        dim = len(self.state)
        zero_prob = 0.0
        one_prob = 0.0

        for basis in range(dim):
            amp = self.state[basis]
            if ((basis >> target) & 1) == 0:
                zero_prob += abs(amp) ** 2
            else:
                one_prob += abs(amp) ** 2

        outcome = 0 if random.random() < zero_prob / max(zero_prob + one_prob, 1e-12) else 1
        new_state = np.zeros_like(self.state)
        for basis in range(dim):
            if ((basis >> target) & 1) == outcome:
                new_state[basis] = self.state[basis]

        norm = np.linalg.norm(new_state)
        if norm > 1e-12:
            new_state /= norm
        self.state = new_state
        return outcome

    def describe_state(self):
        return np.array2string(self.state, precision=3, separator=", ")


class Interpreter:
    def __init__(self):
        self.tokens = []
        self.index = 0
        self.runtime = QuantumRuntime()

    def run(self, source):
        tokenizer = QPLTokenizer(source)
        self.tokens = tokenizer.tokenize()
        self.index = 0
        while self.current().kind != "EOF":
            self.parse_statement()

    def current(self):
        return self.tokens[self.index]

    def peek(self, offset=1):
        return self.tokens[self.index + offset]

    def match(self, kind):
        if self.current().kind == kind:
            self.index += 1
            return True
        return False

    def match_keyword(self, value):
        if self.current().kind == "IDENT" and self.current().value == value:
            self.index += 1
            return True
        return False

    def expect(self, kind):
        token = self.current()
        if token.kind != kind:
            raise ValueError(f"Expected '{kind}' at position {token.pos}, got '{token.kind}'")
        self.index += 1
        return token

    def expect_keyword(self, keyword):
        token = self.current()
        if token.kind != "IDENT" or token.value != keyword:
            raise ValueError(f"Expected keyword '{keyword}' at position {token.pos}")
        self.index += 1
        return token

    def expect_ident(self):
        token = self.current()
        if token.kind != "IDENT":
            raise ValueError(f"Expected identifier at position {token.pos}")
        self.index += 1
        return token.value

    def expect_number(self):
        token = self.current()
        if token.kind != "NUMBER":
            raise ValueError(f"Expected number at position {token.pos}")
        self.index += 1
        return token.value

    def parse_program(self):
        self.expect_keyword("program")
        self.expect_ident()
        self.expect("{")
        while not self.match("}"):
            if self.current().kind == "EOF":
                raise ValueError("Unterminated program block")
            self.parse_statement()

    def parse_statement(self):
        if self.match_keyword("program"):
            self.parse_program()
            return

        if self.match_keyword("qreg"):
            name = self.expect_ident()
            self.expect("[")
            size = int(self.expect_number())
            self.expect("]")
            self.runtime.qreg(name, size)
            self.match(";")
            return

        if self.match_keyword("bit"):
            name = self.expect_ident()
            self.expect("=")
            self.expect_keyword("measure")
            self.expect("(")
            ref = self.parse_qubit_ref()
            self.expect(")")
            self.runtime.bits[name] = self.runtime.measure(ref)
            self.match(";")
            return

        if self.match_keyword("print"):
            self.expect("(")
            args = []
            while self.current().kind != ")":
                if self.current().kind == "STRING":
                    args.append(self.current().value)
                    self.index += 1
                elif self.current().kind == "NUMBER":
                    args.append(self.current().value)
                    self.index += 1
                elif self.current().kind == "IDENT":
                    ref_name = self.current().value
                    self.index += 1
                    args.append(self.runtime.bits.get(ref_name, ref_name))
                else:
                    raise ValueError(f"Unsupported print argument: {self.current().kind}")
                if self.match(","):
                    continue
                break
            self.expect(")")
            self.match(";")
            print(*args)
            return

        gate_name = self.current().value if self.current().kind == "IDENT" else None
        if gate_name in {"H", "X", "Y", "Z", "S", "T", "CNOT", "CZ", "SWAP", "Toffoli", "Rx", "Ry", "Rz"}:
            self.index += 1
            self.expect("(")
            refs = []
            if gate_name in {"Rx", "Ry", "Rz"}:
                arg = self.parse_qubit_ref()
                refs.append(arg)
                self.expect(",")
                angle = self.expect_number()
                refs.append(float(angle))
            else:
                refs.append(self.parse_qubit_ref())
                while self.match(","):
                    refs.append(self.parse_qubit_ref())
            self.expect(")")
            self.runtime.apply_gate(gate_name, refs)
            self.match(";")
            return

        if self.match("{"):
            while not self.match("}"):
                self.parse_statement()
            return

        if self.match_keyword("if"):
            self.expect("(")
            lhs = self.parse_value()
            self.expect("==")
            rhs = self.parse_value()
            self.expect(")")
            self.expect("{")
            if lhs == rhs:
                while not self.match("}"):
                    self.parse_statement()
            else:
                self._skip_block()
            return

        raise ValueError(f"Unsupported statement starting with '{self.current().kind}' at position {self.current().pos}")

    def parse_value(self):
        if self.current().kind == "NUMBER":
            return self.expect_number()
        if self.current().kind == "IDENT":
            name = self.expect_ident()
            return self.runtime.bits.get(name, name)
        raise ValueError(f"Unsupported expression token '{self.current().kind}' at position {self.current().pos}")

    def parse_qubit_ref(self):
        reg_name = self.expect_ident()
        self.expect("[")
        index = int(self.expect_number())
        self.expect("]")
        return (reg_name, index)

    def _skip_block(self):
        # Skips until the matching closing brace for an if/else block.
        depth = 1
        while depth > 0:
            if self.current().kind == "EOF":
                raise ValueError("Unterminated block")
            if self.current().kind == "{":
                depth += 1
                self.index += 1
            elif self.current().kind == "}":
                depth -= 1
                self.index += 1
            else:
                self.index += 1


def run_file(path):
    with open(path, "r", encoding="utf-8") as handle:
        source = handle.read()
    interpreter = Interpreter()
    interpreter.run(source)
    return interpreter.runtime


def main():
    if len(sys.argv) != 2:
        print("Usage: python qpl.py <program.qpl>")
        return 1

    runtime = run_file(sys.argv[1])
    print(runtime.describe_state())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
