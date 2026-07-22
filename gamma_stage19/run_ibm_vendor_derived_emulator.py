#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
import math
import platform
from collections import defaultdict
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeTorino

OUT = Path('gamma_stage19/results')
OUT.mkdir(parents=True, exist_ok=True)
SHOTS = 4096
SEED = 190722

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
I4 = np.eye(4, dtype=complex)
FBS = np.array([
    [1, 0, 0, 0],
    [0, -1 / math.sqrt(2), 1 / math.sqrt(2), 0],
    [0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0],
    [0, 0, 0, -1],
], dtype=complex)


def pauli_xy_rotation(phi: float) -> np.ndarray:
    p = np.kron(X, Y)
    return math.cos(phi) * I4 - 1j * math.sin(phi) * p


def add_codeword(c: QuantumCircuit, qs: list[int], twin: str, branch: int, partner: bool) -> None:
    c.h(qs[0]); c.cx(qs[0], qs[1]); c.h(qs[2]); c.cx(qs[2], qs[3])
    if branch == 1:
        c.z(qs[0]); c.z(qs[2])
    phi = math.pi / 4 if twin == 'singular' and branch == 0 else (0.0 if twin == 'singular' else math.pi / 8)
    if abs(phi) > 0:
        c.append(UnitaryGate(pauli_xy_rotation(phi), label='RXY'), [qs[1], qs[3]])
    if partner:
        for q in qs:
            c.y(q)


def add_even_parity_twist(c: QuantumCircuit, qs: list[int]) -> None:
    for j in (1, 2, 3):
        c.cx(qs[j], qs[0])
    c.x(qs[0])
    for j in (1, 2, 3):
        c.cp(-math.pi * j / 4, qs[0], qs[j])
    c.x(qs[0])
    for j in (3, 2, 1):
        c.cx(qs[j], qs[0])


def add_periodic_fermionic_fft(c: QuantumCircuit, qs: list[int]) -> None:
    box = UnitaryGate(FBS, label='FBS')
    c.append(box, [qs[0], qs[2]])
    c.append(box, [qs[1], qs[3]])
    c.s(qs[3])
    c.append(box, [qs[0], qs[1]])
    c.append(box, [qs[2], qs[3]])


def build_local_block() -> QuantumCircuit:
    c = QuantumCircuit(4)
    add_even_parity_twist(c, [0, 1, 2, 3])
    add_periodic_fermionic_fft(c, [0, 1, 2, 3])
    return c


def right_cycle_matrix() -> np.ndarray:
    mat = np.zeros((16, 16), dtype=complex)
    for x in range(16):
        bits = [(x >> j) & 1 for j in range(4)]
        out = [bits[3], bits[0], bits[1], bits[2]]
        y = sum(out[j] << j for j in range(4))
        mat[y, x] = 1
    return mat


def local_labels() -> tuple[np.ndarray, float]:
    u = Operator(build_local_block()).data
    d = u @ right_cycle_matrix() @ u.conj().T
    off = float(np.linalg.norm(d - np.diag(np.diag(d))))
    roots = np.array([1, 1j, -1, -1j])
    labels = np.array([int(np.argmin(np.abs(roots - v))) for v in np.diag(d)], dtype=np.int8)
    return labels, off


def build_circuit(twin: str, branches: tuple[int, int, int, int]) -> QuantumCircuit:
    c = QuantumCircuit(16, 16, name=f'spectral_{twin}_{"".join(map(str, branches))}')
    for r in range(4):
        add_codeword(c, [4 * r + j for j in range(4)], twin, branches[r], partner=(r % 2 == 1))
    for site in range(4):
        qs = [site, 4 + site, 8 + site, 12 + site]
        add_even_parity_twist(c, qs)
        add_periodic_fermionic_fft(c, qs)
    c.measure(range(16), range(16))
    return c


def estimator_from_bitstring(bitstring: str, labels: np.ndarray) -> float:
    s = bitstring.replace(' ', '')[::-1]
    bits = [int(ch) for ch in s]
    total = 0
    for site in range(4):
        idx = 0
        for r in range(4):
            idx |= bits[4 * r + site] << r
        total = (total + int(labels[idx])) % 4
    return float(math.cos(math.pi * total / 2))


def summarize_counts(counts_list, labels):
    strata = []
    for counts in counts_list:
        n = sum(counts.values())
        mean = sum(estimator_from_bitstring(k, labels) * v for k, v in counts.items()) / n
        second = sum((estimator_from_bitstring(k, labels) ** 2) * v for k, v in counts.items()) / n
        strata.append({'shots': n, 'mean': mean, 'variance': max(0.0, second - mean * mean)})
    mean = float(np.mean([x['mean'] for x in strata]))
    strat_var = float(np.mean([x['variance'] for x in strata]))
    return mean, strat_var, strata


def main() -> None:
    labels, diagonalization_offdiag = local_labels()
    backend = FakeTorino()
    noise_model = NoiseModel.from_backend(backend)
    simulator = AerSimulator(
        noise_model=noise_model,
        basis_gates=noise_model.basis_gates,
        coupling_map=backend.coupling_map,
        seed_simulator=SEED,
    )

    circuits = []
    meta = []
    for twin in ('singular', 'regular'):
        for b in range(16):
            branches = tuple((b >> r) & 1 for r in range(4))
            circuits.append(build_circuit(twin, branches))
            meta.append((twin, branches))

    compiled = transpile(
        circuits,
        backend=backend,
        optimization_level=3,
        seed_transpiler=SEED,
        initial_layout=list(range(16)),
    )
    job = simulator.run(compiled, shots=SHOTS)
    result = job.result()

    grouped = defaultdict(list)
    resources = []
    for i, circ in enumerate(compiled):
        grouped[meta[i][0]].append(result.get_counts(i))
        resources.append({
            'name': circ.name,
            'twin': meta[i][0],
            'branches': list(meta[i][1]),
            'depth': int(circ.depth()),
            'size': int(circ.size()),
            'num_nonlocal_gates': int(circ.num_nonlocal_gates()),
            'op_counts': {str(k): int(v) for k, v in circ.count_ops().items()},
        })

    summaries = {}
    for twin in ('singular', 'regular'):
        mean, var, strata = summarize_counts(grouped[twin], labels)
        summaries[twin] = {'mean': mean, 'stratified_variance': var, 'strata': strata}

    gap = summaries['singular']['mean'] - summaries['regular']['mean']
    variance_of_gap = (summaries['singular']['stratified_variance'] + summaries['regular']['stratified_variance']) / SHOTS
    stderr = math.sqrt(variance_of_gap)
    z = gap / stderr if stderr > 0 else None
    visibility = gap / (1 / 32)
    quarter_gap = 1 / 128

    payload = {
        'status': 'IBM_VENDOR_DERIVED_FAKE_BACKEND_EMULATION_COMPLETE',
        'qualification': 'This is a local Qiskit Aer emulation using the IBM FakeTorino calibrated backend snapshot and derived noise model. It is vendor-derived but not an authenticated cloud vendor emulator run.',
        'backend': backend.name,
        'shots_per_stratum': SHOTS,
        'strata_per_twin': 16,
        'total_shots': SHOTS * 32,
        'seed': SEED,
        'versions': {
            'python': platform.python_version(),
        },
        'local_diagonalization_offdiag_norm': diagonalization_offdiag,
        'ideal': {'singular': 1 / 16, 'regular': 1 / 32, 'gap': 1 / 32},
        'observed': {
            'singular': summaries['singular']['mean'],
            'regular': summaries['regular']['mean'],
            'gap': gap,
            'stderr_gap': stderr,
            'z_score': z,
            'visibility': visibility,
            'quarter_ideal_gap': quarter_gap,
            'passes_visibility_0_6': bool(visibility > 0.6),
            'passes_positive_5sigma': bool(z is not None and z > 5),
        },
        'resources': {
            'mean_depth': float(np.mean([x['depth'] for x in resources])),
            'max_depth': max(x['depth'] for x in resources),
            'mean_nonlocal_gates': float(np.mean([x['num_nonlocal_gates'] for x in resources])),
            'max_nonlocal_gates': max(x['num_nonlocal_gates'] for x in resources),
        },
        'per_circuit_resources': resources,
    }
    (OUT / 'ibm_fake_torino_result.json').write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
