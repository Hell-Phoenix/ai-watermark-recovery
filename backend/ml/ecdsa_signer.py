"""ECDSA signing and verification for watermark payload binding.

Implements **secp256k1 ECDSA** signing of SHA-256 hashes derived from
DINOv2 perceptual embeddings.  The compact 64-byte signature (r‖s) is
embedded as part of the watermark payload, binding the watermark to the
visual content of the image.

Workflow:
  1. ``generate_keypair()`` → ECDSA private + public key on secp256k1
  2. ``sign_embedding(embedding, private_key)`` → 64-byte signature
     - SHA-256( L2-normalised embedding bytes ) → 32-byte digest
     - ECDSA-sign digest with private key → (r, s) integers
     - Serialise as ``r.to_bytes(32) + s.to_bytes(32)`` → 64 bytes
  3. ``verify_embedding(embedding, signature, public_key)`` → bool
  4. Helpers for PEM serialisation and hex conversion.

The 64-byte signature can be split into 512 bits and included in the
watermark payload for later extraction and verification.

References:
  - secp256k1: Certicom Standard for Efficient Cryptography (SEC 2)
  - MetaSeal: content-dependent cryptographic watermark binding
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, utils

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ECDSAConfig:
    """Configuration for ECDSA signing.

    Parameters
    ----------
    curve_name : str
        Elliptic curve identifier (default ``"secp256k1"``).
    hash_algorithm : str
        Hash algorithm for signing (default ``"sha256"``).
    signature_bytes : int
        Expected compact signature length (default 64 = 32 + 32).
    """

    curve_name: str = "secp256k1"
    hash_algorithm: str = "sha256"
    signature_bytes: int = 64


# ---------------------------------------------------------------------------
# Curve mapping
# ---------------------------------------------------------------------------

_CURVES = {
    "secp256k1": ec.SECP256K1(),
    "secp256r1": ec.SECP256R1(),   # P-256, NIST standard
    "secp384r1": ec.SECP384R1(),
}


# ---------------------------------------------------------------------------
# Key management
# ---------------------------------------------------------------------------

class ECDSAKeyPair:
    """ECDSA key pair on a given elliptic curve.

    Supports key generation, PEM import/export, and raw byte access.

    Parameters
    ----------
    config : ECDSAConfig | None
        Configuration; uses defaults (secp256k1) if None.
    private_key : ec.EllipticCurvePrivateKey | None
        Pre-existing private key.  If None, a fresh key pair is generated.
    """

    def __init__(
        self,
        config: ECDSAConfig | None = None,
        private_key: ec.EllipticCurvePrivateKey | None = None,
    ) -> None:
        self.cfg = config or ECDSAConfig()
        curve = _CURVES.get(self.cfg.curve_name)
        if curve is None:
            raise ValueError(
                f"Unsupported curve: {self.cfg.curve_name}. "
                f"Supported: {list(_CURVES.keys())}"
            )

        if private_key is not None:
            self._private_key = private_key
        else:
            self._private_key = ec.generate_private_key(curve, default_backend())

        self._public_key = self._private_key.public_key()

    # -- Properties --

    @property
    def private_key(self) -> ec.EllipticCurvePrivateKey:
        return self._private_key

    @property
    def public_key(self) -> ec.EllipticCurvePublicKey:
        return self._public_key

    # -- PEM serialisation --

    def private_pem(self) -> bytes:
        """Export private key as PEM-encoded bytes."""
        return self._private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

    def public_pem(self) -> bytes:
        """Export public key as PEM-encoded bytes."""
        return self._public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

    @classmethod
    def from_private_pem(
        cls,
        pem_data: bytes,
        config: ECDSAConfig | None = None,
    ) -> ECDSAKeyPair:
        """Load key pair from PEM-encoded private key."""
        private_key = serialization.load_pem_private_key(
            pem_data, password=None, backend=default_backend(),
        )
        if not isinstance(private_key, ec.EllipticCurvePrivateKey):
            raise TypeError("PEM does not contain an EC private key")
        return cls(config=config, private_key=private_key)

    @classmethod
    def from_public_pem(
        cls,
        pem_data: bytes,
        config: ECDSAConfig | None = None,
    ) -> ec.EllipticCurvePublicKey:
        """Load a public key from PEM bytes (for verification only)."""
        public_key = serialization.load_pem_public_key(
            pem_data, backend=default_backend(),
        )
        if not isinstance(public_key, ec.EllipticCurvePublicKey):
            raise TypeError("PEM does not contain an EC public key")
        return public_key


# ---------------------------------------------------------------------------
# Embedding → digest
# ---------------------------------------------------------------------------

def _embedding_to_digest(embedding: torch.Tensor) -> bytes:
    """Compute SHA-256 digest of a perceptual hash embedding.

    The embedding is L2-normalised, converted to float32 bytes, and
    hashed with SHA-256 to produce a 32-byte deterministic digest.

    Parameters
    ----------
    embedding : (D,) or (1, D) — perceptual hash vector (should be
        L2-normalised, but will be normalised here for safety).

    Returns
    -------
    32-byte SHA-256 digest.
    """
    emb = embedding.detach().cpu().float()
    if emb.dim() > 1:
        emb = emb.squeeze(0)
    # L2-normalise for consistency (use F.normalize for numerical
    # stability so that scale-invariance holds at byte level).
    emb = F.normalize(emb.unsqueeze(0), p=2, dim=-1).squeeze(0)
    # Round to 4 decimal places so that minor floating-point
    # differences (e.g. emb vs. 3*emb after normalisation) produce
    # identical digests.
    emb = torch.round(emb * 1e4) / 1e4
    raw_bytes = emb.numpy().tobytes()
    return hashlib.sha256(raw_bytes).digest()


# ---------------------------------------------------------------------------
# Signing
# ---------------------------------------------------------------------------

class ECDSASigner:
    """Sign SHA-256 hashes of perceptual embeddings with ECDSA.

    Parameters
    ----------
    key_pair : ECDSAKeyPair
        Key pair containing the private key for signing.
    """

    def __init__(self, key_pair: ECDSAKeyPair) -> None:
        self.key_pair = key_pair

    def sign_digest(self, digest: bytes) -> bytes:
        """Sign a raw 32-byte digest and return compact 64-byte signature.

        The DER-encoded ECDSA signature is decoded to (r, s) integers,
        then serialised as ``r_bytes(32) || s_bytes(32)`` = 64 bytes.

        Parameters
        ----------
        digest : bytes — 32-byte SHA-256 digest

        Returns
        -------
        64-byte compact signature (big-endian r‖s).
        """
        # Sign using the Prehashed scheme (digest already computed)
        der_sig = self.key_pair.private_key.sign(
            digest,
            ec.ECDSA(utils.Prehashed(hashes.SHA256())),
        )
        # Decode DER → (r, s)
        r, s = utils.decode_dss_signature(der_sig)
        # Serialise to fixed 64 bytes
        r_bytes = r.to_bytes(32, byteorder="big")
        s_bytes = s.to_bytes(32, byteorder="big")
        return r_bytes + s_bytes

    def sign_embedding(self, embedding: torch.Tensor) -> bytes:
        """Sign a perceptual hash embedding.

        Computes SHA-256 of the embedding, then ECDSA-signs the digest.

        Parameters
        ----------
        embedding : (D,) or (1, D) — perceptual hash vector

        Returns
        -------
        64-byte compact signature.
        """
        digest = _embedding_to_digest(embedding)
        return self.sign_digest(digest)

    def sign_embedding_to_bits(self, embedding: torch.Tensor) -> torch.Tensor:
        """Sign embedding and return signature as binary tensor.

        Returns
        -------
        (512,) — binary {0, 1} float tensor representing the 64-byte
        signature as 512 bits (MSB first per byte).
        """
        sig_bytes = self.sign_embedding(embedding)
        return signature_to_bits(sig_bytes)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

class ECDSAVerifier:
    """Verify ECDSA signatures against perceptual embeddings.

    Parameters
    ----------
    public_key : ec.EllipticCurvePublicKey
        The signer's public key for verification.
    """

    def __init__(self, public_key: ec.EllipticCurvePublicKey) -> None:
        self.public_key = public_key

    def verify_digest(self, digest: bytes, signature: bytes) -> bool:
        """Verify a compact 64-byte signature against a raw digest.

        Parameters
        ----------
        digest : 32-byte SHA-256 digest
        signature : 64-byte compact signature (r‖s)

        Returns
        -------
        True if signature is valid, False otherwise.
        """
        if len(signature) != 64:
            return False
        r = int.from_bytes(signature[:32], byteorder="big")
        s = int.from_bytes(signature[32:], byteorder="big")
        der_sig = utils.encode_dss_signature(r, s)
        try:
            self.public_key.verify(
                der_sig,
                digest,
                ec.ECDSA(utils.Prehashed(hashes.SHA256())),
            )
            return True
        except InvalidSignature:
            return False

    def verify_embedding(
        self,
        embedding: torch.Tensor,
        signature: bytes,
    ) -> bool:
        """Verify signature against a perceptual hash embedding.

        Parameters
        ----------
        embedding : (D,) or (1, D) — recomputed perceptual hash
        signature : 64-byte compact signature

        Returns
        -------
        True if the signature was created from the same embedding content.
        """
        digest = _embedding_to_digest(embedding)
        return self.verify_digest(digest, signature)

    def verify_embedding_bits(
        self,
        embedding: torch.Tensor,
        signature_bits: torch.Tensor,
    ) -> bool:
        """Verify a signature represented as a binary tensor.

        Parameters
        ----------
        embedding : (D,) or (1, D)
        signature_bits : (512,) — binary {0, 1}

        Returns
        -------
        True if valid.
        """
        sig_bytes = bits_to_signature(signature_bits)
        return self.verify_embedding(embedding, sig_bytes)


# ---------------------------------------------------------------------------
# Bit ↔ byte conversion utilities
# ---------------------------------------------------------------------------

def signature_to_bits(signature: bytes) -> torch.Tensor:
    """Convert a 64-byte signature to a 512-element binary tensor.

    Parameters
    ----------
    signature : 64 bytes

    Returns
    -------
    (512,) — float tensor of {0, 1}, MSB first per byte.
    """
    bits: list[int] = []
    for byte_val in signature:
        for bit_pos in range(7, -1, -1):
            bits.append((byte_val >> bit_pos) & 1)
    return torch.tensor(bits, dtype=torch.float32)


def bits_to_signature(bits: torch.Tensor) -> bytes:
    """Convert a 512-element binary tensor back to a 64-byte signature.

    Parameters
    ----------
    bits : (512,) — binary {0, 1} (rounds to nearest integer)

    Returns
    -------
    64-byte signature.
    """
    bit_list = bits.detach().cpu().round().long().tolist()
    if len(bit_list) != 512:
        raise ValueError(f"Expected 512 bits, got {len(bit_list)}")
    byte_vals: list[int] = []
    for i in range(0, 512, 8):
        byte_val = 0
        for j in range(8):
            byte_val = (byte_val << 1) | int(bit_list[i + j])
        byte_vals.append(byte_val)
    return bytes(byte_vals)


def signature_to_hex(signature: bytes) -> str:
    """Convert 64-byte signature to hex string."""
    return signature.hex()


def hex_to_signature(hex_str: str) -> bytes:
    """Convert hex string to 64-byte signature."""
    return bytes.fromhex(hex_str)
