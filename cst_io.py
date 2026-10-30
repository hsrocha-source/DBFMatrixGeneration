# -*- coding: utf-8 -*-
"""Readers for the CST exports used by the DBF pipeline: farfield sources (.ffs), Touchstone
S-parameters (.sNp) and the port map (cst/port_map.csv).

Inspect a file before a full run:
    python cst_io.py path/to/farfield_f1.3_port71.ffs
    python cst_io.py path/to/model.s160p
"""
import csv
import re
import sys
from dataclasses import dataclass

import numpy as np

ETA0 = 376.730313668
_trapz = getattr(np, 'trapezoid', None) or np.trapz     # numpy 1.x has only trapz


# ============================================================================ farfield sources
@dataclass
class FarfieldSource:
    """One CST farfield source file. Fields are r*E [V] on a regular (theta, phi) grid in the
    file's local frame (x_axis, z_axis), phase-referenced to `position`."""
    path: str
    position: np.ndarray        # (3,) phase reference
    x_axis: np.ndarray          # (3,) local frame
    z_axis: np.ndarray
    freqs: np.ndarray           # (n_f,) Hz
    p_rad: np.ndarray           # (n_f,) W, as written in the header
    p_acc: np.ndarray
    p_stim: np.ndarray
    theta: np.ndarray           # (n_theta,) rad, ascending
    phi: np.ndarray             # (n_phi,) rad, ascending, one full period without the 2*pi duplicate
    e_theta: np.ndarray         # (n_f, n_theta, n_phi) complex
    e_phi: np.ndarray

    @property
    def port(self):
        return port_from_filename(self.path)

    def freq_index(self, f_hz, tol_hz=1e3):
        i = int(np.argmin(np.abs(self.freqs - f_hz)))
        if abs(self.freqs[i] - f_hz) > tol_hz:
            raise KeyError(f"{self.path}: no frequency within {tol_hz:g} Hz of {f_hz:g} Hz")
        return i

    def integrated_power(self, fi):
        """(1 / 2 eta) * closed integral of |r E|^2 over the sphere, i.e. radiated power if the
        file holds peak phasors. NaN if the grid does not cover theta = 0 ... 180 deg."""
        if self.theta[0] > 1e-6 or abs(self.theta[-1] - np.pi) > 1e-6:
            return np.nan
        p = np.abs(self.e_theta[fi])**2 + np.abs(self.e_phi[fi])**2
        dphi = 2 * np.pi / len(self.phi)
        return _trapz(p.sum(axis=1) * dphi * np.sin(self.theta), self.theta) / (2 * ETA0)


def port_from_filename(path):
    m = re.search(r'port[_ ]?(\d+)', str(path), re.I) or re.search(r'\[(\d+)\]', str(path))
    return int(m.group(1)) if m else None


def _floats(line):
    return [float(t) for t in line.replace(',', ' ').split()]


def _to_hz(f):
    # CST writes Hz; tolerate files written in GHz or MHz
    return f * 1e9 if f < 1e3 else (f * 1e6 if f < 1e6 else f)


def read_ffs(path, header_only=False):
    """Parse a CST farfield source. header_only: stop before the field data (frequencies,
    powers, position and axes only; empty grid)."""
    with open(path, errors='replace') as fh:
        lines = [ln.strip() for ln in fh]

    def next_data(i):
        j = i + 1
        while j < len(lines) and (not lines[j] or lines[j].startswith('//')):
            j += 1
        if j >= len(lines):
            raise ValueError(f"{path}: unexpected end of file after line {i + 1}")
        return j

    position, z_axis, x_axis = np.zeros(3), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
    n_f, powers, blocks = 1, None, []
    i = 0
    while i < len(lines):
        s = lines[i].lower()
        if s.startswith('//'):
            if '#frequencies' in s:
                i = next_data(i)
                n_f = int(float(lines[i].split()[0]))
            elif 'position' in s:
                i = next_data(i)
                position = np.array(_floats(lines[i])[:3])
            elif 'zaxis' in s:
                i = next_data(i)
                z_axis = np.array(_floats(lines[i])[:3])
            elif 'xaxis' in s:
                i = next_data(i)
                x_axis = np.array(_floats(lines[i])[:3])
            elif 'power' in s and 'frequency' in s:
                vals = []
                while len(vals) < 4 * n_f:
                    i = next_data(i)
                    vals += _floats(lines[i])
                powers = np.array(vals[:4 * n_f]).reshape(n_f, 4)
            elif 'phi samples' in s and 'theta samples' in s:
                if header_only:
                    break
                i = next_data(i)
                n_phi, n_th = (int(float(t)) for t in lines[i].split()[:2])
                k = next_data(i)
                rows = np.array(' '.join(lines[k:k + n_phi * n_th]).replace(',', ' ').split(), dtype=float)
                if rows.size != 6 * n_phi * n_th:
                    raise ValueError(f"{path}: expected {n_phi * n_th} rows of 6 values after line {i + 1}")
                blocks.append(rows.reshape(-1, 6))
                i = k + n_phi * n_th - 1
        i += 1

    if powers is not None and header_only:
        empty = np.zeros((n_f, 0, 0), complex)
        return FarfieldSource(path=str(path), position=position, x_axis=x_axis, z_axis=z_axis,
                              freqs=np.array([_to_hz(f) for f in powers[:, 3]]),
                              p_rad=powers[:, 0], p_acc=powers[:, 1], p_stim=powers[:, 2],
                              theta=np.zeros(0), phi=np.zeros(0), e_theta=empty, e_phi=empty)
    if powers is None or not blocks:
        raise ValueError(f"{path}: no power block or no field data found - not a CST farfield source?")
    if len(blocks) != n_f:
        raise ValueError(f"{path}: header says {n_f} frequencies, found {len(blocks)} data blocks")

    phi_deg = np.unique(np.round(blocks[0][:, 0], 6))
    th_deg = np.unique(np.round(blocks[0][:, 1], 6))
    for name, g in (('phi', phi_deg), ('theta', th_deg)):
        if len(g) > 2 and np.ptp(np.diff(g)) > 1e-4:
            raise ValueError(f"{path}: {name} grid is not uniform")
    e_th = np.zeros((n_f, len(th_deg), len(phi_deg)), complex)
    e_ph = np.zeros_like(e_th)
    for b, rows in enumerate(blocks):
        ip = np.searchsorted(phi_deg, np.round(rows[:, 0], 6))
        it = np.searchsorted(th_deg, np.round(rows[:, 1], 6))
        e_th[b, it, ip] = rows[:, 2] + 1j * rows[:, 3]
        e_ph[b, it, ip] = rows[:, 4] + 1j * rows[:, 5]

    if len(phi_deg) > 1 and abs(phi_deg[-1] - phi_deg[0] - 360.0) < 1e-6:     # drop the 360 deg duplicate
        phi_deg, e_th, e_ph = phi_deg[:-1], e_th[..., :-1], e_ph[..., :-1]
    dphi = phi_deg[1] - phi_deg[0] if len(phi_deg) > 1 else 360.0
    if abs(len(phi_deg) * dphi - 360.0) > 1e-4:
        raise ValueError(f"{path}: phi grid does not cover a full 360 deg period")

    return FarfieldSource(path=str(path), position=position, x_axis=x_axis, z_axis=z_axis,
                          freqs=np.array([_to_hz(f) for f in powers[:, 3]]),
                          p_rad=powers[:, 0], p_acc=powers[:, 1], p_stim=powers[:, 2],
                          theta=np.radians(th_deg), phi=np.radians(phi_deg), e_theta=e_th, e_phi=e_ph)


# ============================================================================ Touchstone
@dataclass
class Touchstone:
    path: str
    freqs: np.ndarray       # (n_f,) Hz
    s: np.ndarray           # (n_f, n, n) complex, s[f, m, n] = S_{m+1, n+1} (m receives, n excited)
    z0: float

    def at(self, f_hz):
        """S-matrix at the nearest frequency point, and that point's offset [Hz]."""
        i = int(np.argmin(np.abs(self.freqs - f_hz)))
        return self.s[i], self.freqs[i] - f_hz


def read_touchstone(path):
    m = re.search(r'\.s(\d+)p$', str(path), re.I)
    n_ports = int(m.group(1)) if m else None
    unit, fmt, z0 = 1e9, 'MA', 50.0
    tokens = []
    with open(path, errors='replace') as fh:
        for raw in fh:
            line = raw.split('!', 1)[0].strip()
            if not line:
                continue
            if line.startswith('#'):
                t = line[1:].upper().split()
                for j, w in enumerate(t):
                    if w in ('HZ', 'KHZ', 'MHZ', 'GHZ'):
                        unit = {'HZ': 1.0, 'KHZ': 1e3, 'MHZ': 1e6, 'GHZ': 1e9}[w]
                    elif w in ('MA', 'DB', 'RI'):
                        fmt = w
                    elif w == 'R' and j + 1 < len(t):
                        z0 = float(t[j + 1])
                    elif w in ('Y', 'Z', 'H', 'G'):
                        raise ValueError(f"{path}: only S-parameter files are supported")
                continue
            if line.startswith('['):                      # Touchstone 2 keywords
                key = line.lower()
                if key.startswith('[number of ports]'):
                    n_ports = int(line.split(']')[1])
                elif key.startswith('[matrix format]') and 'full' not in key:
                    raise ValueError(f"{path}: only [Matrix Format] Full is supported")
                continue
            tokens.append(line)
    if n_ports is None:
        raise ValueError(f"{path}: cannot tell the number of ports (use a .sNp file name)")
    v = np.array(' '.join(tokens).split(), dtype=float)
    block = 1 + 2 * n_ports**2
    if v.size % block:
        raise ValueError(f"{path}: {v.size} values is not a multiple of {block} (1 + 2 * {n_ports}^2)")
    v = v.reshape(-1, block)
    a, b = v[:, 1::2], v[:, 2::2]
    if fmt == 'RI':
        s = a + 1j * b
    elif fmt == 'MA':
        s = a * np.exp(1j * np.radians(b))
    else:
        s = 10**(a / 20) * np.exp(1j * np.radians(b))
    s = s.reshape(-1, n_ports, n_ports)
    if n_ports == 2:                                  # Touchstone 1 2-port order: S11 S21 S12 S22
        s = s.transpose(0, 2, 1)
    return Touchstone(path=str(path), freqs=v[:, 0] * unit, s=s, z0=z0)


# ============================================================================ port map
def read_port_map(path):
    """Rows of cst/port_map.csv as dicts with int fields and the `eep_class` string."""
    with open(path, newline='') as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ('element', 'port_H', 'port_V', 'col', 'row', 'x_index', 'y_index'):
            r[k] = int(r[k])
        r['eep_class'] = r['eep_class'].strip()
    return rows


# ============================================================================ inspection
def _inspect(path):
    if re.search(r'\.s\d+p$', path, re.I):
        ts = read_touchstone(path)
        n = ts.s.shape[1]
        print(f"{path}: {n} ports, {len(ts.freqs)} frequencies "
              f"{ts.freqs[0] / 1e6:.3f} ... {ts.freqs[-1] / 1e6:.3f} MHz, Z0 = {ts.z0:g} ohm")
        s0, _ = ts.at(ts.freqs[len(ts.freqs) // 2])
        col_pow = np.sum(np.abs(s0)**2, axis=0)
        print(f"  at {ts.freqs[len(ts.freqs) // 2] / 1e6:.3f} MHz: non-zero columns {int(np.sum(col_pow > 0))} / {n}, "
              f"max column sum |S|^2 = {col_pow.max():.3f}")
        return
    src = read_ffs(path)
    print(f"{path}: port {src.port}, {len(src.freqs)} frequency(ies)")
    print(f"  position {src.position}, x_axis {src.x_axis}, z_axis {src.z_axis}")
    print(f"  grid: {len(src.theta)} theta ({np.degrees(src.theta[0]):g} ... {np.degrees(src.theta[-1]):g} deg), "
          f"{len(src.phi)} phi (step {np.degrees(src.phi[1] - src.phi[0]):g} deg)")
    for fi, f in enumerate(src.freqs):
        p_int = src.integrated_power(fi)
        g = 2 * np.pi * (np.abs(src.e_theta[fi])**2 + np.abs(src.e_phi[fi])**2).max() / (ETA0 * src.p_stim[fi])
        print(f"  {f / 1e6:9.3f} MHz | P_rad {src.p_rad[fi]:.4g}  P_acc {src.p_acc[fi]:.4g}  P_stim {src.p_stim[fi]:.4g} W"
              f" | integral/P_rad = {p_int / src.p_rad[fi]:.3f} (1: peak phasors, 0.5: RMS)"
              f" | max realized gain {10 * np.log10(g):.2f} dBi (if peak)")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        _inspect(p)
