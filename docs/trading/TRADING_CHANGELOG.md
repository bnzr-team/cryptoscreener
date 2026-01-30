# Trading/VOL Harvesting v2 — CHANGELOG

## Unreleased
### Added
- **DEC-043:** Policy Library docs (SSOT)
  - `04_STRATEGY_CATALOG.md`: 6 strategy modes (grid, skew, unwind, toxic_avoid, pause, kill)
  - `05_ML_POLICY_LIBRARY.md`: 20 policy rules (POL-001 to POL-020) with config-first design
  - `06_RISK_COST_MODEL.md`: Fee, slippage, latency, inventory, kill switch parameters
  - `07_SIM_EXPECTATIONS.md`: KPI expectations per fixture, policy regression matrix
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
