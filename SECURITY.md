# Security Policy

## Reporting a Vulnerability

The SecureID project prioritizes biometric data privacy and edge security. If you discover a vulnerability or security issue within this offline biometric pipeline, please report it responsibly.

### Disclosure Process
1. **Do not create public GitHub issues** for security vulnerabilities.
2. Email your findings directly to the repository maintainer (`swaymrawat862@gmail.com`).
3. Include:
   - Description of the vulnerability and potential attack vector.
   - Reproduction steps or proof-of-concept payload.
   - Estimated impact on biometric templates or authentication decisions.

## Privacy & Offline Guarantees

- **Zero Network Egress**: The biometric pipeline must run completely air-gapped without telemetry, cloud APIs, or outbound network calls.
- **Template Non-Reversibility**: Only compact histogram/feature representations (LBPH) are persisted. Raw camera frames are immediately discarded from volatile memory upon template extraction.
- **Side-Channel Protection**: Feature comparison and chi-square distance calculations avoid early-exit timing leaks where feasible.
