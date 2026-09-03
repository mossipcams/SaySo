SaySo overlay for OHF-Voice/linux-voice-assistant.

Upstream LVA lives in linux-voice-assistant/ and must stay mergeable.
Custom code is only under sayso/ plus patches/0001-sayso-stable-device-name.patch.

Operational commands (also /usr/local/bin/sayso-satellite):

  sayso-satellite start|stop|restart|status
  sayso-satellite logs
  sayso-satellite validate
  sayso-satellite devices
  sayso-satellite test-mic
  sayso-satellite test-speaker
  sayso-satellite test-wake-word

Copy `models/sayso.onnx` to `/opt/sayso-satellite/models/sayso.onnx` before start.
