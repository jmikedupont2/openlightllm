# OpenLightLLM — Manifesto & Roadmap

## What

OpenLightLLM is a community fork of litellm with all proprietary code removed and verifiable performance witnesses for every LLM request.

**litellm routes requests to 100+ LLM providers. We make that routing provable.**

## Why

1. **Enterprise lock-in**: 136 proprietary files, license gates on features
2. **Nag spam**: Star-begging, PostHog analytics, telemetry in production
3. **No proof of service**: "200 OK" is a claim, not a proof

## Done (v1.82.3-stable.patch.2)

- Remove `enterprise/` — 136 files, LICENSE → MIT-only
- Strip 949 license check lines from 83 files
- Remove nag prompts
- Add petals decentralized provider
- Add flake.nix + CI
- Rust migration plan (CRQ-003)

## Next: zkPerf Witness Integration

Every LLM request produces a cryptographic witness proving:
- Request was sent to the claimed provider
- Latency is real (hardware perf counters)
- Token count is verified independently
- Response is content-addressed

### Architecture

```
Request → FRACTRAN Router (19 providers = 19 primes)
       → zkPerf Witness (cycles, cache, latency)
       → DA51 Shard (orbifold coordinate, CBOR)
       → Merkle anchor (Solana/IPFS)
```

### Roadmap

| Phase | Status |
|-------|--------|
| 1. Clean fork | ✅ Done |
| 2. FRACTRAN router | ✅ Bred, Rust skeleton |
| 3. zkPerf witness | Next |
| 4. DA51 shards | Next |
| 5. Rust core | 8 weeks (CRQ-003) |

## Links

- Fork: https://github.com/jmikedupont2/openlightllm
- zkPerf: https://github.com/meta-introspector/zkperf
- kagenti: https://github.com/meta-introspector/kagenti

## License

MIT. All of it. No exceptions.
