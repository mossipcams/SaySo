# Changelog

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
