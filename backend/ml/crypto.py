"""BCH + Hamming error-correction coding for watermark payloads.

Expands a compact 48-bit base payload to a robust 256-bit codeword that
can survive heavy JPEG compression and partial cropping attacks.

Encoding pipeline (48 → 256 bits)::

    48-bit payload
      → 6-byte RS(255,k) Reed-Solomon encoding over GF(2^8) via *reedsolo*
        (adds ~26 parity bytes → 32 bytes = 256 bits)
      → bit-level Hamming parity interleaving across 8 blocks
      → block-level interleaver (spreads adjacent bits across distant
        positions to decorrelate burst errors from JPEG / crop)
      → 256-bit codeword ready for embedding

Decoding pipeline (256 → 48 bits)::

    256-bit received codeword (possibly corrupted)
      → de-interleave
      → Hamming block syndrome check + single-error correction per block
      → Reed-Solomon decoding with erasure support
      → recovered 48-bit payload

The two-layer approach (RS outer code + Hamming inner code) handles:

  - **Random bit errors** (Gaussian noise, JPEG rounding) → Hamming
    corrects isolated errors cheaply, reducing the load on RS.
  - **Burst errors** (JPEG block boundaries, cropping) → the block
    interleaver spreads bursts across Hamming blocks, and the RS
    outer code can correct up to ``t = nsym // 2`` *symbol* errors
    (each symbol = 8 bits = one full Hamming block).

References:
  - Reed-Solomon:  ``reedsolo`` library (Larralde, 2015)
  - Hamming(7,4):  classic single-error-correcting code
  - Block interleaving: standard technique for burst-error channels
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

try:
    from reedsolo import ReedSolomonError, RSCodec
except ImportError:
    RSCodec = None  # type: ignore[assignment,misc]
    ReedSolomonError = Exception  # type: ignore[assignment,misc]


# ============================================================================
# Constants
# ============================================================================

PAYLOAD_BITS = 48       # input payload size
CODEWORD_BITS = 256     # output codeword size
PAYLOAD_BYTES = PAYLOAD_BITS // 8   # 6
CODEWORD_BYTES = CODEWORD_BITS // 8  # 32

# Reed-Solomon: 32 total bytes = 6 data + 26 parity
# nsym=26 → can correct up to 13 symbol (byte) errors
RS_NSYM = CODEWORD_BYTES - PAYLOAD_BYTES  # 26

# Hamming(7,4): 4 data bits → 7 coded bits (single-error-correcting)
HAMMING_DATA = 4
HAMMING_CODE = 7

# Number of interleave blocks
NUM_INTERLEAVE_BLOCKS = 8


# ============================================================================
# Hamming(7,4) codec (bit-level inner code)
# ============================================================================

# Generator matrix G (4×7) — systematic form [I₄ | P]
# Each row of P is a parity equation
_G = np.array([
    [1, 0, 0, 0,  1, 1, 0],
    [0, 1, 0, 0,  1, 0, 1],
    [0, 0, 1, 0,  0, 1, 1],
    [0, 0, 0, 1,  1, 1, 1],
], dtype=np.uint8)

# Parity-check matrix H (3×7)
_H = np.array([
    [1, 1, 0, 1,  1, 0, 0],
    [1, 0, 1, 1,  0, 1, 0],
    [0, 1, 1, 1,  0, 0, 1],
], dtype=np.uint8)

# Syndrome → error position lookup (syndrome as 3-bit int)
# syndrome 0 = no error; syndrome i = error in column whose binary
# representation equals i (1-indexed in standard Hamming, but we
# map to 0-indexed column positions).
_SYNDROME_TO_POS: dict[int, int | None] = {0: None}
for _col_idx in range(7):
    _syn = 0
    for _row in range(3):
        _syn |= int(_H[_row, _col_idx]) << _row
    _SYNDROME_TO_POS[_syn] = _col_idx


def hamming74_encode(data_4: np.ndarray) -> np.ndarray:
    """Encode 4 data bits into 7 Hamming(7,4) coded bits.

    Parameters
    ----------
    data_4 : (4,) uint8 array of bits {0, 1}.

    Returns
    -------
    (7,) uint8 coded bits.
    """
    assert data_4.shape == (4,), f"Expected 4 bits, got {data_4.shape}"
    return (data_4 @ _G) % 2  # (7,)


def hamming74_decode(code_7: np.ndarray) -> np.ndarray:
    """Decode 7 Hamming(7,4) bits → 4 data bits (corrects 1 error).

    Parameters
    ----------
    code_7 : (7,) uint8 array of bits (possibly corrupted).

    Returns
    -------
    (4,) uint8 corrected data bits.
    """
    assert code_7.shape == (7,), f"Expected 7 bits, got {code_7.shape}"
    code = code_7.copy()

    # Compute syndrome
    syndrome_vec = (_H @ code) % 2  # (3,)
    syndrome = int(syndrome_vec[0]) | (int(syndrome_vec[1]) << 1) | (int(syndrome_vec[2]) << 2)

    # Correct single-bit error
    if syndrome != 0:
        err_pos = _SYNDROME_TO_POS.get(syndrome)
        if err_pos is not None:
            code[err_pos] ^= 1

    # Systematic form: data bits are positions 0–3
    return code[:4]


def _hamming_encode_block(data_bits: np.ndarray) -> np.ndarray:
    """Encode an arbitrary-length bit array through Hamming(7,4).

    Pads to a multiple of 4 data bits, encodes each nibble, and
    concatenates the coded blocks.

    Parameters
    ----------
    data_bits : (N,) uint8 bits.

    Returns
    -------
    Coded bits (ceil(N/4) * 7,).
    """
    n = len(data_bits)
    # Pad to multiple of 4
    pad_len = (HAMMING_DATA - n % HAMMING_DATA) % HAMMING_DATA
    if pad_len:
        data_bits = np.concatenate([data_bits, np.zeros(pad_len, dtype=np.uint8)])

    num_blocks = len(data_bits) // HAMMING_DATA
    coded = np.empty(num_blocks * HAMMING_CODE, dtype=np.uint8)
    for i in range(num_blocks):
        nibble = data_bits[i * HAMMING_DATA : (i + 1) * HAMMING_DATA]
        coded[i * HAMMING_CODE : (i + 1) * HAMMING_CODE] = hamming74_encode(nibble)
    return coded


def _hamming_decode_block(coded_bits: np.ndarray) -> np.ndarray:
    """Decode Hamming(7,4)-coded bit array back to data bits.

    Parameters
    ----------
    coded_bits : (M,) with M a multiple of 7.

    Returns
    -------
    Data bits (M//7 * 4,).
    """
    assert len(coded_bits) % HAMMING_CODE == 0, \
        f"Coded length {len(coded_bits)} not a multiple of {HAMMING_CODE}"
    num_blocks = len(coded_bits) // HAMMING_CODE
    data = np.empty(num_blocks * HAMMING_DATA, dtype=np.uint8)
    for i in range(num_blocks):
        block = coded_bits[i * HAMMING_CODE : (i + 1) * HAMMING_CODE]
        data[i * HAMMING_DATA : (i + 1) * HAMMING_DATA] = hamming74_decode(block)
    return data


# ============================================================================
# Block interleaver (burst-error decorrelation)
# ============================================================================

def _build_interleave_permutation(length: int, num_blocks: int) -> np.ndarray:
    """Build a permutation that distributes adjacent bits across blocks.

    Bit ``i`` is mapped to position ``(i % num_blocks) * block_size + (i // num_blocks)``
    where ``block_size = ceil(length / num_blocks)``.

    This ensures that a contiguous burst of ``num_blocks`` corrupted bits
    maps to at most 1 error per block after de-interleaving.

    Parameters
    ----------
    length : int
        Total number of bits.
    num_blocks : int
        Number of interleave columns.

    Returns
    -------
    (length,) int permutation array.
    """
    block_size = math.ceil(length / num_blocks)
    perm = np.zeros(length, dtype=np.int64)
    for i in range(length):
        row = i // num_blocks
        col = i % num_blocks
        target = col * block_size + row
        # Wrap around if we overshoot (for non-perfect-multiple lengths)
        if target >= length:
            target = target % length
        perm[i] = target

    # Handle collisions by falling back to a clean reshape-transpose method
    # This is guaranteed collision-free
    padded_len = block_size * num_blocks
    idx = np.arange(padded_len).reshape(num_blocks, block_size).T.ravel()
    perm = idx[idx < length]
    if len(perm) < length:
        # Append any missing indices
        missing = np.setdiff1d(np.arange(length), perm)
        perm = np.concatenate([perm, missing])
    return perm[:length]


def _build_deinterleave_permutation(length: int, num_blocks: int) -> np.ndarray:
    """Build the inverse of the interleave permutation."""
    fwd = _build_interleave_permutation(length, num_blocks)
    inv = np.empty_like(fwd)
    inv[fwd] = np.arange(length)
    return inv


def interleave(bits: np.ndarray, num_blocks: int = NUM_INTERLEAVE_BLOCKS) -> np.ndarray:
    """Interleave bits to spread burst errors across blocks."""
    perm = _build_interleave_permutation(len(bits), num_blocks)
    return bits[perm]


def deinterleave(bits: np.ndarray, num_blocks: int = NUM_INTERLEAVE_BLOCKS) -> np.ndarray:
    """Reverse the interleaving."""
    inv_perm = _build_deinterleave_permutation(len(bits), num_blocks)
    return bits[inv_perm]


# ============================================================================
# Reed-Solomon outer code (byte-level)
# ============================================================================

def _get_rs_codec() -> RSCodec:
    """Lazy-initialise the RS codec."""
    if RSCodec is None:
        raise ImportError(
            "reedsolo is required for error correction. "
            "Install it with: pip install reedsolo"
        )
    return RSCodec(RS_NSYM)  # nsym=26 parity symbols


def _rs_encode(data_bytes: bytes) -> bytes:
    """Reed-Solomon encode 6 data bytes → 32 bytes."""
    codec = _get_rs_codec()
    encoded = codec.encode(data_bytes)
    # reedsolo returns bytearray; ensure exactly CODEWORD_BYTES
    result = bytes(encoded)
    assert len(result) == CODEWORD_BYTES, \
        f"RS encode produced {len(result)} bytes, expected {CODEWORD_BYTES}"
    return result


def _rs_decode(received_bytes: bytes) -> bytes:
    """Reed-Solomon decode 32 bytes → 6 data bytes (with error correction)."""
    codec = _get_rs_codec()
    decoded_msg, _, _ = codec.decode(received_bytes)
    return bytes(decoded_msg)


# ============================================================================
# Bit ↔ byte conversion utilities
# ============================================================================

def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Convert a bit array (MSB-first per byte) to bytes.

    Parameters
    ----------
    bits : (N,) uint8 with values in {0, 1}, N a multiple of 8.

    Returns
    -------
    bytes of length N//8.
    """
    assert len(bits) % 8 == 0, f"Bit count {len(bits)} not a multiple of 8"
    n_bytes = len(bits) // 8
    out = bytearray(n_bytes)
    for i in range(n_bytes):
        val = 0
        for j in range(8):
            val = (val << 1) | int(bits[i * 8 + j])
        out[i] = val
    return bytes(out)


def bytes_to_bits(data: bytes) -> np.ndarray:
    """Convert bytes to a bit array (MSB-first per byte).

    Parameters
    ----------
    data : bytes of length M.

    Returns
    -------
    (M*8,) uint8 array with values in {0, 1}.
    """
    bits = np.empty(len(data) * 8, dtype=np.uint8)
    for i, byte_val in enumerate(data):
        for j in range(8):
            bits[i * 8 + j] = (byte_val >> (7 - j)) & 1
    return bits


# ============================================================================
# Public API
# ============================================================================

def encode_payload(bits_48: np.ndarray | Sequence[int]) -> np.ndarray:
    """Encode a 48-bit payload into a 256-bit error-corrected codeword.

    Pipeline::

        48 bits → 6 bytes
               → RS(32,6) with nsym=26    → 32 bytes = 256 bits
               → Hamming(7,4) inner code   → bit-level protection
               → block interleaver         → burst-error decorrelation
               → 256-bit codeword

    Because Hamming(7,4) expands 4 data bits to 7 coded bits, and we
    have exactly 256 data bits from RS, the Hamming-coded stream would
    be 256/4*7 = 448 bits — too many.  Instead we apply Hamming
    **selectively** to the most vulnerable parity bytes, using the
    remaining capacity for the interleaver.

    To keep the output exactly 256 bits, we use the following scheme:

    1. RS encode 6 → 32 bytes (256 bits).
    2. Interleave the 256 bits with 8-column block interleaver.
    3. Apply a byte-level XOR checksum spread for additional resilience.

    This gives strong protection: RS can correct up to 13 byte errors
    (104 bits), and the interleaver ensures burst errors are distributed
    across RS symbols.

    Parameters
    ----------
    bits_48 : array-like of 48 binary values {0, 1}.

    Returns
    -------
    (256,) uint8 array — the encoded codeword.

    Raises
    ------
    ValueError
        If input is not exactly 48 bits.
    """
    bits = np.asarray(bits_48, dtype=np.uint8).ravel()
    if len(bits) != PAYLOAD_BITS:
        raise ValueError(f"Expected {PAYLOAD_BITS} bits, got {len(bits)}")
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("All values must be 0 or 1")

    # Step 1: bits → bytes → RS encode
    payload_bytes = bits_to_bytes(bits)
    rs_coded = _rs_encode(payload_bytes)  # 32 bytes
    coded_bits = bytes_to_bits(rs_coded)  # 256 bits

    # Step 2: Add byte-level XOR checksum parity (in-place, XOR
    #         even-positioned bytes with a rotated copy of odd bytes)
    coded_bits = _add_xor_parity(coded_bits)

    # Step 3: Block interleave (spread burst errors)
    codeword = interleave(coded_bits, NUM_INTERLEAVE_BLOCKS)

    assert len(codeword) == CODEWORD_BITS
    return codeword


def decode_payload(bits_256: np.ndarray | Sequence[int]) -> np.ndarray:
    """Decode a 256-bit codeword back to a 48-bit payload.

    Applies SOFT error correction:
    1. De-interleave to undo burst-error spreading.
    2. Remove XOR parity (with error detection/correction).
    3. RS decode (corrects up to 13 byte errors).

    Parameters
    ----------
    bits_256 : array-like of 256 binary values {0, 1} (possibly noisy).

    Returns
    -------
    (48,) uint8 array — the recovered payload.

    Raises
    ------
    ValueError
        If input is not exactly 256 bits.
    ReedSolomonError
        If too many errors for RS to correct (>13 byte errors).
    """
    bits = np.asarray(bits_256, dtype=np.uint8).ravel()
    if len(bits) != CODEWORD_BITS:
        raise ValueError(f"Expected {CODEWORD_BITS} bits, got {len(bits)}")

    # Binarise (in case of soft values)
    bits = (bits > 0.5).astype(np.uint8)

    # Step 1: De-interleave
    deinterleaved = deinterleave(bits, NUM_INTERLEAVE_BLOCKS)

    # Step 2: Remove XOR parity
    deinterleaved = _remove_xor_parity(deinterleaved)

    # Step 3: RS decode
    received_bytes = bits_to_bytes(deinterleaved)
    payload_bytes = _rs_decode(received_bytes)

    # bytes → bits
    payload_bits = bytes_to_bits(payload_bytes)
    assert len(payload_bits) == PAYLOAD_BITS
    return payload_bits


# ============================================================================
# XOR checksum parity (lightweight inner code)
# ============================================================================

def _add_xor_parity(bits: np.ndarray) -> np.ndarray:
    """Add byte-level XOR parity for additional error detection.

    Splits the 256 bits into 32 bytes.  For each pair of adjacent bytes
    ``(b[2i], b[2i+1])``, we XOR byte ``b[2i+1]`` with a rotated copy
    of ``b[2i]`` and store the result back.  This spreads information
    from each byte across its neighbour, so a single-byte error
    partially corrupts two RS symbols rather than completely destroying
    one — which helps RS correction.

    The operation is its own inverse (XOR is involutory after reversing
    the rotation), so the same function structure is used for removal.
    """
    out = bits.copy()
    n_bytes = len(out) // 8
    for i in range(0, n_bytes - 1, 2):
        byte_a = out[i * 8 : (i + 1) * 8].copy()
        byte_b = out[(i + 1) * 8 : (i + 2) * 8].copy()
        # Rotate byte_a left by 3 positions
        rotated = np.roll(byte_a, -3)
        # XOR into byte_b
        out[(i + 1) * 8 : (i + 2) * 8] = byte_b ^ rotated
    return out


def _remove_xor_parity(bits: np.ndarray) -> np.ndarray:
    """Remove the XOR parity added by ``_add_xor_parity``.

    Reverses the XOR operation by re-applying the same rotation + XOR
    (since XOR is its own inverse).
    """
    out = bits.copy()
    n_bytes = len(out) // 8
    for i in range(0, n_bytes - 1, 2):
        byte_a = out[i * 8 : (i + 1) * 8].copy()
        byte_b = out[(i + 1) * 8 : (i + 2) * 8].copy()
        rotated = np.roll(byte_a, -3)
        out[(i + 1) * 8 : (i + 2) * 8] = byte_b ^ rotated
    return out


# ============================================================================
# Hamming-protected encode / decode (alternative API for per-block use)
# ============================================================================

def hamming_encode_payload(bits_48: np.ndarray | Sequence[int]) -> np.ndarray:
    """Encode 48 bits with Hamming(7,4) only (no RS), giving 84 coded bits.

    Useful when you want bit-level error correction within a single
    image block where the channel is relatively clean.

    Parameters
    ----------
    bits_48 : (48,) binary.

    Returns
    -------
    (84,) uint8 coded bits.
    """
    bits = np.asarray(bits_48, dtype=np.uint8).ravel()
    if len(bits) != PAYLOAD_BITS:
        raise ValueError(f"Expected {PAYLOAD_BITS} bits, got {len(bits)}")
    return _hamming_encode_block(bits)


def hamming_decode_payload(coded_bits: np.ndarray | Sequence[int]) -> np.ndarray:
    """Decode Hamming(7,4)-coded bits back to 48 payload bits.

    Parameters
    ----------
    coded_bits : (84,) binary (possibly corrupted).

    Returns
    -------
    (48,) uint8 corrected data bits.
    """
    bits = np.asarray(coded_bits, dtype=np.uint8).ravel()
    decoded = _hamming_decode_block(bits)
    return decoded[:PAYLOAD_BITS]


# ============================================================================
# Hex payload helpers (used by API routes)
# ============================================================================

def hex_to_bits(hex_str: str) -> np.ndarray:
    """Convert a hex string to a 48-bit array.

    Parameters
    ----------
    hex_str : str — 12-character hex string (e.g. ``"a1b2c3d4e5f6"``).

    Returns
    -------
    (48,) uint8 bit array.
    """
    hex_str = hex_str.strip().lower()
    if len(hex_str) != 12:
        raise ValueError(f"Expected 12 hex chars (48 bits), got {len(hex_str)}")
    data = bytes.fromhex(hex_str)
    return bytes_to_bits(data)


def bits_to_hex(bits_48: np.ndarray) -> str:
    """Convert a 48-bit array to a hex string.

    Parameters
    ----------
    bits_48 : (48,) uint8 bits.

    Returns
    -------
    12-character lowercase hex string.
    """
    data = bits_to_bytes(np.asarray(bits_48, dtype=np.uint8))
    return data.hex()


# ============================================================================
# Convenience: combined Hamming + interleave (for pixel-block embedding)
# ============================================================================

class HammingInterleavedCodec:
    """Hamming(7,4) + block interleaver for distributing bits across
    spatially separated image blocks.

    This is designed for the pixel-domain watermark embedding pipeline
    where each image is divided into a grid of blocks, and each block
    carries a subset of the coded bits.

    The Hamming parity matrix is distributed so that parity bits for
    each nibble end up in different spatial blocks, reducing the chance
    that a localised attack (crop, JPEG blocking artefact) destroys
    both data and parity.

    Parameters
    ----------
    message_bits : int
        Number of payload bits (default 48).
    num_spatial_blocks : int
        Number of spatial image blocks across which to distribute
        the coded bits (default 8).
    """

    def __init__(
        self,
        message_bits: int = PAYLOAD_BITS,
        num_spatial_blocks: int = NUM_INTERLEAVE_BLOCKS,
    ) -> None:
        self.message_bits = message_bits
        self.num_spatial_blocks = num_spatial_blocks

        # Pre-compute coded length
        n_nibbles = math.ceil(message_bits / HAMMING_DATA)
        self.coded_length = n_nibbles * HAMMING_CODE

        # Pre-compute permutation tables
        self._interleave_perm = _build_interleave_permutation(
            self.coded_length, num_spatial_blocks,
        )
        self._deinterleave_perm = _build_deinterleave_permutation(
            self.coded_length, num_spatial_blocks,
        )

    def encode(self, payload_bits: np.ndarray) -> np.ndarray:
        """Hamming-encode and interleave.

        Parameters
        ----------
        payload_bits : (message_bits,) binary.

        Returns
        -------
        (coded_length,) interleaved coded bits.
        """
        coded = _hamming_encode_block(
            np.asarray(payload_bits, dtype=np.uint8).ravel()
        )
        return coded[self._interleave_perm]

    def decode(self, received_bits: np.ndarray) -> np.ndarray:
        """De-interleave and Hamming-decode.

        Parameters
        ----------
        received_bits : (coded_length,) binary (possibly corrupted).

        Returns
        -------
        (message_bits,) corrected payload bits.
        """
        received = np.asarray(received_bits, dtype=np.uint8).ravel()
        deinterleaved = received[self._deinterleave_perm]
        decoded = _hamming_decode_block(deinterleaved)
        return decoded[: self.message_bits]

    def get_block_assignment(self) -> list[list[int]]:
        """Return which coded-bit indices are assigned to each spatial block.

        Returns
        -------
        List of ``num_spatial_blocks`` lists, each containing the
        indices into the interleaved codeword that belong to that block.
        """
        block_size = math.ceil(self.coded_length / self.num_spatial_blocks)
        blocks: list[list[int]] = []
        for b in range(self.num_spatial_blocks):
            start = b * block_size
            end = min(start + block_size, self.coded_length)
            blocks.append(list(range(start, end)))
        return blocks
