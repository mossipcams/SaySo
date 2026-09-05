# Changelog

## [0.6.0](https://github.com/mossipcams/SaySo/compare/0.5.3...0.6.0) (2026-09-05)


### Features

* **training:** add apostrophe-safe parser and corrective curriculum ([9626d1f](https://github.com/mossipcams/SaySo/commit/9626d1ffdcb42072cd2958f11e7abfa9903e0f40))
* **training:** add first Base SFT quality eval and 10k generator ([aa0f367](https://github.com/mossipcams/SaySo/commit/aa0f367e46492415648a4d8ce8a2a876ff958ad4))
* **training:** expand Assist catalog and add device-type tier schema v2 ([d955c30](https://github.com/mossipcams/SaySo/commit/d955c30341c79bd8d158b8eebc81810e15abe7ea))
* **training:** expand synthetic generator with capability registry ([606b3a2](https://github.com/mossipcams/SaySo/commit/606b3a29c1a42c2093fd656e0d6bc4e6f27baefb))
* **training:** pin synthetic generation to schema v2 tool catalog ([55ee03c](https://github.com/mossipcams/SaySo/commit/55ee03c43194dcb0e629b47280f9ce9014a50b53))


### Bug Fixes

* **training:** score no-call mismatches and aggregate eval metrics correctly ([d643146](https://github.com/mossipcams/SaySo/commit/d643146cf63396387755b489c51e91985a01db12))
* **training:** stop calling JSON parse schema-valid and skip inference errors ([922d4d0](https://github.com/mossipcams/SaySo/commit/922d4d0bc9f4c3313cbe24d6516eb2deb276348e))
* **training:** validate schema from args and use category rate denominators ([d0838ab](https://github.com/mossipcams/SaySo/commit/d0838abbed084371d4af30a82d7234f25bd49895))


### Documentation

* **training:** collapse plans onto Base rsLoRA ([e304d45](https://github.com/mossipcams/SaySo/commit/e304d45641960d1683cbb894cdbc46d2b7553b5e))

## [0.5.3](https://github.com/mossipcams/SaySo/compare/0.5.2...0.5.3) (2026-09-03)


### Bug Fixes

* flush wake preroll into STT and record HA tool error class ([401f42c](https://github.com/mossipcams/SaySo/commit/401f42cbe2cb7e2e97027cec898031877487df78))

## [0.5.2](https://github.com/mossipcams/SaySo/compare/0.5.1...0.5.2) (2026-09-03)


### Bug Fixes

* keep live voice ack and Assist tool compile working ([a2e05e5](https://github.com/mossipcams/SaySo/commit/a2e05e5c6fb1efc08d6c86e4a9a3e7cab5467196))


### Documentation

* **training:** align plan with LFM ([6d2137f](https://github.com/mossipcams/SaySo/commit/6d2137f2cb5493ae6dec69e472a862096406a276))

## [0.5.1](https://github.com/mossipcams/SaySo/compare/0.5.0...0.5.1) (2026-09-03)


### Bug Fixes

* **sayso:** declare voluptuous-openapi so the conversation agent can load ([2ad2f68](https://github.com/mossipcams/SaySo/commit/2ad2f68ecccaba9014e1c6cd5148365330c9cd28))

## [0.5.0](https://github.com/mossipcams/SaySo/compare/0.4.0...0.5.0) (2026-09-03)


### Features

* **satellite:** add trained Sayso ONNX wake model ([d10377f](https://github.com/mossipcams/SaySo/commit/d10377f7677985fcc3700aed5037d121655ec948))
* **satellite:** detect wake on LVA processed PCM ([3a05d34](https://github.com/mossipcams/SaySo/commit/3a05d3421039120ab9479f18ae35917505ffa51b))
* **training:** lock v1 tool schema and add label-first generation ([f8fdf82](https://github.com/mossipcams/SaySo/commit/f8fdf82fa0d07ff4341a903ba80ca5ea27f9d130))


### Documentation

* align architecture and agent guidance ([ca594a3](https://github.com/mossipcams/SaySo/commit/ca594a346f7960d16415d6f6850db84454dc5fb4))

## [0.4.0](https://github.com/mossipcams/SaySo/compare/0.3.0...0.4.0) (2026-09-03)


### Features

* **satellite:** add Linux Voice Assistant overlay with wake hop fix ([7aa7e28](https://github.com/mossipcams/SaySo/commit/7aa7e28c1a0dc10cc9577651748a01c1953e87f5))


### Bug Fixes

* accept unprefixed model tool names against HA 2026.9 prefixes ([40fd013](https://github.com/mossipcams/SaySo/commit/40fd01343564cc54bbe8948d657d6586742ae000))
* **satellite:** restore wake and TTS voice path ([46e7e28](https://github.com/mossipcams/SaySo/commit/46e7e28c1fcf85d06772ae67f333054064129f0d))
* treat GetLiveContext misses as completed tool results ([2c1a915](https://github.com/mossipcams/SaySo/commit/2c1a915819921b0020cbed1c30bccfb27f2a0e57))


### Documentation

* rewrite AGENTS.md to match the current conversation-agent topology ([579e6dd](https://github.com/mossipcams/SaySo/commit/579e6ddf0bbf7103637d725a9be262f98ab97456))

## [0.3.0](https://github.com/mossipcams/SaySo/compare/0.2.0...0.3.0) (2026-09-02)


### Bug Fixes

* match Release Please tags to existing 0.x releases ([44f43a1](https://github.com/mossipcams/SaySo/commit/44f43a1a5aa969b85f3eda3b0251bbf25ba8fdf1))

## [0.2.0](https://github.com/mossipcams/SaySo/compare/0.1.3...0.2.0) (2026-09-02)


### Features

* Native Home Assistant conversation-agent rewrite on the ajax/full-rebuild branch
