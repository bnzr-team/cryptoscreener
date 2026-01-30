# Trading/VOL Harvesting v2 — CHANGELOG

## Unreleased
### Added
- Initial v2 docs pack (SSOT templates): index, scope/boundary, decisions, spec, state, changelog
- `03_CONTRACTS.md`: Full v2 data contracts SSOT with 6 contracts (OrderIntent, OrderAck, FillEvent, PositionSnapshot, SessionState, RiskBreachEvent), global invariants, JSON examples, roundtrip test plan
- `src/cryptoscreener/trading/contracts/`: Pydantic v2 implementation of all 6 contracts
  - Decimal types for all monetary fields (no float precision loss)
  - Strict enums for status/side/type fields
  - `extra='forbid'` on all models
  - `dedupe_key` property on each contract per SSOT spec
- `tests/fixtures/trading_contracts/`: 11 fixture files with manifest and SHA256 checksums
- `tests/trading/`: 50 tests (roundtrip, validation, dedupe)

### Changed
- N/A

### Fixed
- N/A
