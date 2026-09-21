"""Canonical chat-template rendering for parent-eval protocol v3.

Why this exists
---------------
Protocol v2 rendered each parent's prompts through that parent's OWN
``chat_template``. The C/D tournaments (2026-09-08) proved the gate's
sibling lesson at the template layer: C and D tokenize identically to
parent A on a public-domain probe (17,232-token exact match), yet their
re-serialized tokenizers carry a *different* chat template, so under v2
the prompt text itself would differ across parents — a comparability
break no tokenizer gate can see. Protocol v3 pins ONE canonical
rendering for every parent.

Canonical source: the official Qwen/Qwen3.8-27B template (the model
family's own contract with its weights — not any derivative
checkpoint's re-serialization). The full template is embedded below as
base64 (JSON-escaping a Jinja template invites quote-transposition
bugs; base64 is byte-exact and trivially checkable).

Contract (fail closed)
----------------------
- ``render_canonical()`` is the ONLY sanctioned renderer for v3
  tournament prompts; workers must call it instead of
  ``tokenizer.apply_chat_template``.
- ``verify_canonical_template`` recomputes the digest of the embedded
  template and refuses to run if the module's bytes were altered —
  provenance is checked at load, not trusted.
- The template's digest is part of the protocol-v3 fingerprint
  (``ParentEvalSpec.canonical_template_sha256``), so swapping it is a
  visible protocol change, never a silent edit.

This module is content-free: no suite text, no protected material —
just the template rendering machinery.
"""

from __future__ import annotations

import base64
import hashlib
from typing import Any

# Official Qwen/Qwen3.8-27B chat template (parent A's tokenizer at the
# program-pinned revision), sha256 = c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041.
_TEMPLATE_B64 = (
    "eyUtIHNldCBpbWFnZV9jb3VudCA9IG5hbWVzcGFjZSh2YWx1ZT0wKSAlfQp7JS0gc2V0IHZpZGVv"
    "X2NvdW50ID0gbmFtZXNwYWNlKHZhbHVlPTApICV9CnslLSBtYWNybyByZW5kZXJfY29udGVudChj"
    "b250ZW50LCBkb192aXNpb25fY291bnQsIGlzX3N5c3RlbV9jb250ZW50PWZhbHNlKSAlfQogICAg"
    "eyUtIGlmIGNvbnRlbnQgaXMgc3RyaW5nICV9CiAgICAgICAge3stIGNvbnRlbnQgfX0KICAgIHsl"
    "LSBlbGlmIGNvbnRlbnQgaXMgaXRlcmFibGUgYW5kIGNvbnRlbnQgaXMgbm90IG1hcHBpbmcgJX0K"
    "ICAgICAgICB7JS0gZm9yIGl0ZW0gaW4gY29udGVudCAlfQogICAgICAgICAgICB7JS0gaWYgJ2lt"
    "YWdlJyBpbiBpdGVtIG9yICdpbWFnZV91cmwnIGluIGl0ZW0gb3IgaXRlbS50eXBlID09ICdpbWFn"
    "ZScgJX0KICAgICAgICAgICAgICAgIHslLSBpZiBpc19zeXN0ZW1fY29udGVudCAlfQogICAgICAg"
    "ICAgICAgICAgICAgIHt7LSByYWlzZV9leGNlcHRpb24oJ1N5c3RlbSBtZXNzYWdlIGNhbm5vdCBj"
    "b250YWluIGltYWdlcy4nKSB9fQogICAgICAgICAgICAgICAgeyUtIGVuZGlmICV9CiAgICAgICAg"
    "ICAgICAgICB7JS0gaWYgZG9fdmlzaW9uX2NvdW50ICV9CiAgICAgICAgICAgICAgICAgICAgeyUt"
    "IHNldCBpbWFnZV9jb3VudC52YWx1ZSA9IGltYWdlX2NvdW50LnZhbHVlICsgMSAlfQogICAgICAg"
    "ICAgICAgICAgeyUtIGVuZGlmICV9CiAgICAgICAgICAgICAgICB7JS0gaWYgYWRkX3Zpc2lvbl9p"
    "ZCAlfQogICAgICAgICAgICAgICAgICAgIHt7LSAnUGljdHVyZSAnIH4gaW1hZ2VfY291bnQudmFs"
    "dWUgfiAnOiAnIH19CiAgICAgICAgICAgICAgICB7JS0gZW5kaWYgJX0KICAgICAgICAgICAgICAg"
    "IHt7LSAnPHx2aXNpb25fc3RhcnR8Pjx8aW1hZ2VfcGFkfD48fHZpc2lvbl9lbmR8PicgfX0KICAg"
    "ICAgICAgICAgeyUtIGVsaWYgJ3ZpZGVvJyBpbiBpdGVtIG9yIGl0ZW0udHlwZSA9PSAndmlkZW8n"
    "ICV9CiAgICAgICAgICAgICAgICB7JS0gaWYgaXNfc3lzdGVtX2NvbnRlbnQgJX0KICAgICAgICAg"
    "ICAgICAgICAgICB7ey0gcmFpc2VfZXhjZXB0aW9uKCdTeXN0ZW0gbWVzc2FnZSBjYW5ub3QgY29u"
    "dGFpbiB2aWRlb3MuJykgfX0KICAgICAgICAgICAgICAgIHslLSBlbmRpZiAlfQogICAgICAgICAg"
    "ICAgICAgeyUtIGlmIGRvX3Zpc2lvbl9jb3VudCAlfQogICAgICAgICAgICAgICAgICAgIHslLSBz"
    "ZXQgdmlkZW9fY291bnQudmFsdWUgPSB2aWRlb19jb3VudC52YWx1ZSArIDEgJX0KICAgICAgICAg"
    "ICAgICAgIHslLSBlbmRpZiAlfQogICAgICAgICAgICAgICAgeyUtIGlmIGFkZF92aXNpb25faWQg"
    "JX0KICAgICAgICAgICAgICAgICAgICB7ey0gJ1ZpZGVvICcgfiB2aWRlb19jb3VudC52YWx1ZSB+"
    "ICc6ICcgfX0KICAgICAgICAgICAgICAgIHslLSBlbmRpZiAlfQogICAgICAgICAgICAgICAge3st"
    "ICc8fHZpc2lvbl9zdGFydHw+PHx2aWRlb19wYWR8Pjx8dmlzaW9uX2VuZHw+JyB9fQogICAgICAg"
    "ICAgICB7JS0gZWxpZiAndGV4dCcgaW4gaXRlbSAlfQogICAgICAgICAgICAgICAge3stIGl0ZW0u"
    "dGV4dCB9fQogICAgICAgICAgICB7JS0gZWxzZSAlfQogICAgICAgICAgICAgICAge3stIHJhaXNl"
    "X2V4Y2VwdGlvbignVW5leHBlY3RlZCBpdGVtIHR5cGUgaW4gY29udGVudC4nKSB9fQogICAgICAg"
    "ICAgICB7JS0gZW5kaWYgJX0KICAgICAgICB7JS0gZW5kZm9yICV9CiAgICB7JS0gZWxpZiBjb250"
    "ZW50IGlzIG5vbmUgb3IgY29udGVudCBpcyB1bmRlZmluZWQgJX0KICAgICAgICB7ey0gJycgfX0K"
    "ICAgIHslLSBlbHNlICV9CiAgICAgICAge3stIHJhaXNlX2V4Y2VwdGlvbignVW5leHBlY3RlZCBj"
    "b250ZW50IHR5cGUuJykgfX0KICAgIHslLSBlbmRpZiAlfQp7JS0gZW5kbWFjcm8gJX0KeyUtIGlm"
    "IG5vdCBtZXNzYWdlcyAlfQogICAge3stIHJhaXNlX2V4Y2VwdGlvbignTm8gbWVzc2FnZXMgcHJv"
    "dmlkZWQuJykgfX0KeyUtIGVuZGlmICV9CnslLSBzZXQgcmVhc29uaW5nX2luc3RydWN0aW9ucyA9"
    "ICcnICV9CnslLSBpZiBlbmFibGVfdGhpbmtpbmcgaXMgdW5kZWZpbmVkIG9yIGVuYWJsZV90aGlu"
    "a2luZyBpcyB0cnVlICV9CiAgICB7JS0gc2V0IHJlc29sdmVkX3JlYXNvbmluZ19lZmZvcnQgPSBy"
    "ZWFzb25pbmdfZWZmb3J0fGRlZmF1bHQoJ3hoaWdoJykgJX0KICAgIHslLSBpZiByZXNvbHZlZF9y"
    "ZWFzb25pbmdfZWZmb3J0IG5vdCBpbiAoJ3hoaWdoJywgJ21lZGl1bScsICdsb3cnKSAlfQogICAg"
    "ICAgIHt7LSByYWlzZV9leGNlcHRpb24oJ1VuZXhwZWN0ZWQgcmVhc29uaW5nIGVmZm9ydCAnIH4g"
    "cmVhc29uaW5nX2VmZm9ydCB+ICcuIFN1cHBvcnRlZCB0eXBlcyBhcmUgeGhpZ2ggKGRlZmF1bHQp"
    "LCBtZWRpdW0sIGFuZCBsb3cuJykgfX0KICAgIHslLSBlbmRpZiAlfQogICAgeyUtIGlmIHJlc29s"
    "dmVkX3JlYXNvbmluZ19lZmZvcnQgPT0gJ3hoaWdoJyAlfQogICAgICAgIHslLSBzZXQgcmVhc29u"
    "aW5nX2luc3RydWN0aW9ucyA9ICdSZWFzb25pbmcgZWZmb3J0IGlzIHNldCB0byB4aGlnaC4gUGxl"
    "YXNlIHRoaW5rIGNhcmVmdWxseSB0aHJvdWdoIHRoZSB0YXNrLCB2YWxpZGF0ZSBrZXkgYXNzdW1w"
    "dGlvbnMsIGNvbnNpZGVyIHBsYXVzaWJsZSBhbHRlcm5hdGl2ZXMsIGFuZCBwcmlvcml0aXplIGNv"
    "cnJlY3RuZXNzLCBjb25zaXN0ZW5jeSwgYW5kIGNsYXJpdHkgaW4gdGhlIGZpbmFsIGFuc3dlci4n"
    "ICV9CiAgICB7JS0gZWxpZiByZXNvbHZlZF9yZWFzb25pbmdfZWZmb3J0ID09ICdsb3cnICV9CiAg"
    "ICAgICAgeyUtIHNldCByZWFzb25pbmdfaW5zdHJ1Y3Rpb25zID0gJ1JlYXNvbmluZyBlZmZvcnQg"
    "aXMgc2V0IHRvIGxvdy4gS2VlcCB5b3VyIHRoaW5raW5nIGJyaWVmIGFuZCBmb2N1c2VkLCBtb3Zp"
    "bmcgZGlyZWN0bHkgdG8gdGhlIGNvbmNsdXNpb24gd2l0aG91dCB1bm5lY2Vzc2FyeSBlbGFib3Jh"
    "dGlvbi4nICV9CiAgICB7JS0gZW5kaWYgJX0KeyUtIGVuZGlmICV9CnslLSBpZiB0b29scyBhbmQg"
    "dG9vbHMgaXMgaXRlcmFibGUgYW5kIHRvb2xzIGlzIG5vdCBtYXBwaW5nICV9CiAgICB7ey0gJzx8"
    "aW1fc3RhcnR8PnN5c3RlbVxuJyB9fQogICAgeyUtIGlmIHJlYXNvbmluZ19pbnN0cnVjdGlvbnMg"
    "JX0KICAgICAgICB7ey0gcmVhc29uaW5nX2luc3RydWN0aW9ucyArICdcblxuJyB9fQogICAgeyUt"
    "IGVuZGlmICV9CiAgICB7ey0gIiMgVG9vbHNcblxuWW91IGhhdmUgYWNjZXNzIHRvIHRoZSBmb2xs"
    "b3dpbmcgZnVuY3Rpb25zOlxuXG48dG9vbHM+IiB9fQogICAgeyUtIGZvciB0b29sIGluIHRvb2xz"
    "ICV9CiAgICAgICAge3stICJcbiIgfX0KICAgICAgICB7ey0gdG9vbCB8IHRvanNvbiB9fQogICAg"
    "eyUtIGVuZGZvciAlfQogICAge3stICJcbjwvdG9vbHM+IiB9fQogICAge3stICdcblxuSWYgeW91"
    "IGNob29zZSB0byBjYWxsIGEgZnVuY3Rpb24gT05MWSByZXBseSBpbiB0aGUgZm9sbG93aW5nIGZv"
    "cm1hdCB3aXRoIE5PIHN1ZmZpeDpcblxuPHRvb2xfY2FsbD5cbjxmdW5jdGlvbj1leGFtcGxlX2Z1"
    "bmN0aW9uX25hbWU+XG48cGFyYW1ldGVyPWV4YW1wbGVfcGFyYW1ldGVyXzE+XG52YWx1ZV8xXG48"
    "L3BhcmFtZXRlcj5cbjxwYXJhbWV0ZXI9ZXhhbXBsZV9wYXJhbWV0ZXJfMj5cblRoaXMgaXMgdGhl"
    "IHZhbHVlIGZvciB0aGUgc2Vjb25kIHBhcmFtZXRlclxudGhhdCBjYW4gc3BhblxubXVsdGlwbGUg"
    "bGluZXNcbjwvcGFyYW1ldGVyPlxuPC9mdW5jdGlvbj5cbjwvdG9vbF9jYWxsPlxuXG48SU1QT1JU"
    "QU5UPlxuUmVtaW5kZXI6XG4tIEZ1bmN0aW9uIGNhbGxzIE1VU1QgZm9sbG93IHRoZSBzcGVjaWZp"
    "ZWQgZm9ybWF0OiBhbiBpbm5lciA8ZnVuY3Rpb249Li4uPjwvZnVuY3Rpb24+IGJsb2NrIG11c3Qg"
    "YmUgbmVzdGVkIHdpdGhpbiA8dG9vbF9jYWxsPjwvdG9vbF9jYWxsPiBYTUwgdGFnc1xuLSBSZXF1"
    "aXJlZCBwYXJhbWV0ZXJzIE1VU1QgYmUgc3BlY2lmaWVkXG4tIFlvdSBtYXkgcHJvdmlkZSBvcHRp"
    "b25hbCByZWFzb25pbmcgZm9yIHlvdXIgZnVuY3Rpb24gY2FsbCBpbiBuYXR1cmFsIGxhbmd1YWdl"
    "IEJFRk9SRSB0aGUgZnVuY3Rpb24gY2FsbCwgYnV0IE5PVCBhZnRlclxuLSBJZiB0aGVyZSBpcyBu"
    "byBmdW5jdGlvbiBjYWxsIGF2YWlsYWJsZSwgYW5zd2VyIHRoZSBxdWVzdGlvbiBsaWtlIG5vcm1h"
    "bCB3aXRoIHlvdXIgY3VycmVudCBrbm93bGVkZ2UgYW5kIGRvIG5vdCB0ZWxsIHRoZSB1c2VyIGFi"
    "b3V0IGZ1bmN0aW9uIGNhbGxzXG48L0lNUE9SVEFOVD4nIH19CiAgICB7JS0gaWYgbWVzc2FnZXNb"
    "MF0ucm9sZSA9PSAnc3lzdGVtJyAlfQogICAgICAgIHslLSBzZXQgY29udGVudCA9IHJlbmRlcl9j"
    "b250ZW50KG1lc3NhZ2VzWzBdLmNvbnRlbnQsIGZhbHNlLCB0cnVlKXx0cmltICV9CiAgICAgICAg"
    "eyUtIGlmIGNvbnRlbnQgJX0KICAgICAgICAgICAge3stICdcblxuJyArIGNvbnRlbnQgfX0KICAg"
    "ICAgICB7JS0gZW5kaWYgJX0KICAgIHslLSBlbmRpZiAlfQogICAge3stICc8fGltX2VuZHw+XG4n"
    "IH19CnslLSBlbHNlICV9CiAgICB7JS0gaWYgbWVzc2FnZXNbMF0ucm9sZSA9PSAnc3lzdGVtJyAl"
    "fQogICAgICAgIHslLSBzZXQgY29udGVudCA9IHJlbmRlcl9jb250ZW50KG1lc3NhZ2VzWzBdLmNv"
    "bnRlbnQsIGZhbHNlLCB0cnVlKXx0cmltICV9CiAgICAgICAgeyUtIGlmIGNvbnRlbnQgJX0KICAg"
    "ICAgICAgICAge3stICc8fGltX3N0YXJ0fD5zeXN0ZW1cbicgKyAocmVhc29uaW5nX2luc3RydWN0"
    "aW9ucyArICdcblxuJyBpZiByZWFzb25pbmdfaW5zdHJ1Y3Rpb25zIGVsc2UgJycpICArIGNvbnRl"
    "bnQgKyAnPHxpbV9lbmR8PlxuJyB9fQogICAgICAgIHslLSBlbGlmIHJlYXNvbmluZ19pbnN0cnVj"
    "dGlvbnMgJX0KICAgICAgICAgICAge3stICc8fGltX3N0YXJ0fD5zeXN0ZW1cbicgKyByZWFzb25p"
    "bmdfaW5zdHJ1Y3Rpb25zICsgJzx8aW1fZW5kfD5cbicgfX0KICAgICAgICB7JS0gZW5kaWYgJX0K"
    "ICAgIHslLSBlbGlmIHJlYXNvbmluZ19pbnN0cnVjdGlvbnMgJX0KICAgICAgICB7ey0gJzx8aW1f"
    "c3RhcnR8PnN5c3RlbVxuJyArIHJlYXNvbmluZ19pbnN0cnVjdGlvbnMgKyAnPHxpbV9lbmR8Plxu"
    "JyB9fQogICAgeyUtIGVuZGlmICV9CnslLSBlbmRpZiAlfQp7JS0gc2V0IG5zID0gbmFtZXNwYWNl"
    "KG11bHRpX3N0ZXBfdG9vbD10cnVlLCBsYXN0X3F1ZXJ5X2luZGV4PW1lc3NhZ2VzfGxlbmd0aCAt"
    "IDEpICV9CnslLSBmb3IgbWVzc2FnZSBpbiBtZXNzYWdlc1s6Oi0xXSAlfQogICAgeyUtIHNldCBp"
    "bmRleCA9IChtZXNzYWdlc3xsZW5ndGggLSAxKSAtIGxvb3AuaW5kZXgwICV9CiAgICB7JS0gaWYg"
    "bnMubXVsdGlfc3RlcF90b29sIGFuZCBtZXNzYWdlLnJvbGUgPT0gInVzZXIiICV9CiAgICAgICAg"
    "eyUtIHNldCBjb250ZW50ID0gcmVuZGVyX2NvbnRlbnQobWVzc2FnZS5jb250ZW50LCBmYWxzZSl8"
    "dHJpbSAlfQogICAgICAgIHslLSBpZiBub3QoY29udGVudC5zdGFydHN3aXRoKCc8dG9vbF9yZXNw"
    "b25zZT4nKSBhbmQgY29udGVudC5lbmRzd2l0aCgnPC90b29sX3Jlc3BvbnNlPicpKSAlfQogICAg"
    "ICAgICAgICB7JS0gc2V0IG5zLm11bHRpX3N0ZXBfdG9vbCA9IGZhbHNlICV9CiAgICAgICAgICAg"
    "IHslLSBzZXQgbnMubGFzdF9xdWVyeV9pbmRleCA9IGluZGV4ICV9CiAgICAgICAgeyUtIGVuZGlm"
    "ICV9CiAgICB7JS0gZW5kaWYgJX0KeyUtIGVuZGZvciAlfQp7JS0gaWYgbnMubXVsdGlfc3RlcF90"
    "b29sICV9CiAgICB7ey0gcmFpc2VfZXhjZXB0aW9uKCdObyB1c2VyIHF1ZXJ5IGZvdW5kIGluIG1l"
    "c3NhZ2VzLicpIH19CnslLSBlbmRpZiAlfQp7JS0gZm9yIG1lc3NhZ2UgaW4gbWVzc2FnZXMgJX0K"
    "ICAgIHslLSBzZXQgY29udGVudCA9IHJlbmRlcl9jb250ZW50KG1lc3NhZ2UuY29udGVudCwgdHJ1"
    "ZSl8dHJpbSAlfQogICAgeyUtIGlmIG1lc3NhZ2Uucm9sZSA9PSAic3lzdGVtIiAlfQogICAgICAg"
    "IHslLSBpZiBub3QgbG9vcC5maXJzdCAlfQogICAgICAgICAgICB7ey0gcmFpc2VfZXhjZXB0aW9u"
    "KCdTeXN0ZW0gbWVzc2FnZSBtdXN0IGJlIGF0IHRoZSBiZWdpbm5pbmcuJykgfX0KICAgICAgICB7"
    "JS0gZW5kaWYgJX0KICAgIHslLSBlbGlmIG1lc3NhZ2Uucm9sZSA9PSAidXNlciIgJX0KICAgICAg"
    "ICB7ey0gJzx8aW1fc3RhcnR8PicgKyBtZXNzYWdlLnJvbGUgKyAnXG4nICsgY29udGVudCArICc8"
    "fGltX2VuZHw+JyArICdcbicgfX0KICAgIHslLSBlbGlmIG1lc3NhZ2Uucm9sZSA9PSAiYXNzaXN0"
    "YW50IiAlfQogICAgICAgIHslLSBzZXQgcmVhc29uaW5nX2NvbnRlbnQgPSAnJyAlfQogICAgICAg"
    "IHslLSBpZiBtZXNzYWdlLnJlYXNvbmluZ19jb250ZW50IGlzIHN0cmluZyAlfQogICAgICAgICAg"
    "ICB7JS0gc2V0IHJlYXNvbmluZ19jb250ZW50ID0gbWVzc2FnZS5yZWFzb25pbmdfY29udGVudCAl"
    "fQogICAgICAgIHslLSBlbmRpZiAlfQogICAgICAgIHslLSBzZXQgcmVhc29uaW5nX2NvbnRlbnQg"
    "PSByZWFzb25pbmdfY29udGVudHx0cmltICV9CiAgICAgICAgeyUtIGlmIHByZXNlcnZlX3RoaW5r"
    "aW5nIGlzIHVuZGVmaW5lZCBvciBwcmVzZXJ2ZV90aGlua2luZyBpcyB0cnVlIG9yIGxvb3AuaW5k"
    "ZXgwID4gbnMubGFzdF9xdWVyeV9pbmRleCAlfQogICAgICAgICAgICB7ey0gJzx8aW1fc3RhcnR8"
    "PicgKyBtZXNzYWdlLnJvbGUgKyAnXG48dGhpbms+XG4nICsgcmVhc29uaW5nX2NvbnRlbnQgKyAn"
    "XG48L3RoaW5rPlxuXG4nICsgY29udGVudCB9fQogICAgICAgIHslLSBlbHNlICV9CiAgICAgICAg"
    "ICAgIHt7LSAnPHxpbV9zdGFydHw+JyArIG1lc3NhZ2Uucm9sZSArICdcbicgKyBjb250ZW50IH19"
    "CiAgICAgICAgeyUtIGVuZGlmICV9CiAgICAgICAgeyUtIGlmIG1lc3NhZ2UudG9vbF9jYWxscyBh"
    "bmQgbWVzc2FnZS50b29sX2NhbGxzIGlzIGl0ZXJhYmxlIGFuZCBtZXNzYWdlLnRvb2xfY2FsbHMg"
    "aXMgbm90IG1hcHBpbmcgJX0KICAgICAgICAgICAgeyUtIGZvciB0b29sX2NhbGwgaW4gbWVzc2Fn"
    "ZS50b29sX2NhbGxzICV9CiAgICAgICAgICAgICAgICB7JS0gaWYgdG9vbF9jYWxsLmZ1bmN0aW9u"
    "IGlzIGRlZmluZWQgJX0KICAgICAgICAgICAgICAgICAgICB7JS0gc2V0IHRvb2xfY2FsbCA9IHRv"
    "b2xfY2FsbC5mdW5jdGlvbiAlfQogICAgICAgICAgICAgICAgeyUtIGVuZGlmICV9CiAgICAgICAg"
    "ICAgICAgICB7JS0gaWYgbG9vcC5maXJzdCAlfQogICAgICAgICAgICAgICAgICAgIHslLSBpZiBj"
    "b250ZW50fHRyaW0gJX0KICAgICAgICAgICAgICAgICAgICAgICAge3stICdcblxuPHRvb2xfY2Fs"
    "bD5cbjxmdW5jdGlvbj0nICsgdG9vbF9jYWxsLm5hbWUgKyAnPlxuJyB9fQogICAgICAgICAgICAg"
    "ICAgICAgIHslLSBlbHNlICV9CiAgICAgICAgICAgICAgICAgICAgICAgIHt7LSAnPHRvb2xfY2Fs"
    "bD5cbjxmdW5jdGlvbj0nICsgdG9vbF9jYWxsLm5hbWUgKyAnPlxuJyB9fQogICAgICAgICAgICAg"
    "ICAgICAgIHslLSBlbmRpZiAlfQogICAgICAgICAgICAgICAgeyUtIGVsc2UgJX0KICAgICAgICAg"
    "ICAgICAgICAgICB7ey0gJ1xuPHRvb2xfY2FsbD5cbjxmdW5jdGlvbj0nICsgdG9vbF9jYWxsLm5h"
    "bWUgKyAnPlxuJyB9fQogICAgICAgICAgICAgICAgeyUtIGVuZGlmICV9CiAgICAgICAgICAgICAg"
    "ICB7JS0gaWYgdG9vbF9jYWxsLmFyZ3VtZW50cyBpcyBkZWZpbmVkIGFuZCB0b29sX2NhbGwuYXJn"
    "dW1lbnRzICE9ICcnICV9CiAgICAgICAgICAgICAgICAgICAgeyUtIGZvciBhcmdzX25hbWUsIGFy"
    "Z3NfdmFsdWUgaW4gdG9vbF9jYWxsLmFyZ3VtZW50c3xpdGVtcyAlfQogICAgICAgICAgICAgICAg"
    "ICAgICAgICB7ey0gJzxwYXJhbWV0ZXI9JyArIGFyZ3NfbmFtZSArICc+XG4nIH19CiAgICAgICAg"
    "ICAgICAgICAgICAgICAgIHslLSBzZXQgYXJnc192YWx1ZSA9IGFyZ3NfdmFsdWUgfCBzdHJpbmcg"
    "aWYgYXJnc192YWx1ZSBpcyBzdHJpbmcgZWxzZSBhcmdzX3ZhbHVlIHwgdG9qc29uIHwgc2FmZSAl"
    "fQogICAgICAgICAgICAgICAgICAgICAgICB7ey0gYXJnc192YWx1ZSB9fQogICAgICAgICAgICAg"
    "ICAgICAgICAgICB7ey0gJ1xuPC9wYXJhbWV0ZXI+XG4nIH19CiAgICAgICAgICAgICAgICAgICAg"
    "eyUtIGVuZGZvciAlfQogICAgICAgICAgICAgICAgeyUtIGVuZGlmICV9CiAgICAgICAgICAgICAg"
    "ICB7ey0gJzwvZnVuY3Rpb24+XG48L3Rvb2xfY2FsbD4nIH19CiAgICAgICAgICAgIHslLSBlbmRm"
    "b3IgJX0KICAgICAgICB7JS0gZW5kaWYgJX0KICAgICAgICB7ey0gJzx8aW1fZW5kfD5cbicgfX0K"
    "ICAgIHslLSBlbGlmIG1lc3NhZ2Uucm9sZSA9PSAidG9vbCIgJX0KICAgICAgICB7JS0gaWYgbG9v"
    "cC5wcmV2aXRlbSBhbmQgbG9vcC5wcmV2aXRlbS5yb2xlICE9ICJ0b29sIiAlfQogICAgICAgICAg"
    "ICB7ey0gJzx8aW1fc3RhcnR8PnVzZXInIH19CiAgICAgICAgeyUtIGVuZGlmICV9CiAgICAgICAg"
    "e3stICdcbjx0b29sX3Jlc3BvbnNlPlxuJyB9fQogICAgICAgIHt7LSBjb250ZW50IH19CiAgICAg"
    "ICAge3stICdcbjwvdG9vbF9yZXNwb25zZT4nIH19CiAgICAgICAgeyUtIGlmIG5vdCBsb29wLmxh"
    "c3QgYW5kIGxvb3AubmV4dGl0ZW0ucm9sZSAhPSAidG9vbCIgJX0KICAgICAgICAgICAge3stICc8"
    "fGltX2VuZHw+XG4nIH19CiAgICAgICAgeyUtIGVsaWYgbG9vcC5sYXN0ICV9CiAgICAgICAgICAg"
    "IHt7LSAnPHxpbV9lbmR8PlxuJyB9fQogICAgICAgIHslLSBlbmRpZiAlfQogICAgeyUtIGVsc2Ug"
    "JX0KICAgICAgICB7ey0gcmFpc2VfZXhjZXB0aW9uKCdVbmV4cGVjdGVkIG1lc3NhZ2Ugcm9sZS4n"
    "KSB9fQogICAgeyUtIGVuZGlmICV9CnslLSBlbmRmb3IgJX0KeyUtIGlmIGFkZF9nZW5lcmF0aW9u"
    "X3Byb21wdCAlfQogICAge3stICc8fGltX3N0YXJ0fD5hc3Npc3RhbnRcbicgfX0KICAgIHslLSBp"
    "ZiBlbmFibGVfdGhpbmtpbmcgaXMgZGVmaW5lZCBhbmQgZW5hYmxlX3RoaW5raW5nIGlzIGZhbHNl"
    "ICV9CiAgICAgICAge3stICc8dGhpbms+XG5cbjwvdGhpbms+XG5cbicgfX0KICAgIHslLSBlbHNl"
    "ICV9CiAgICAgICAge3stICc8dGhpbms+XG4nIH19CiAgICB7JS0gZW5kaWYgJX0KeyUtIGVuZGlm"
    "ICV9"
)

_EXPECTED_SHA256 = "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"


def _template_bytes() -> bytes:
    return base64.b64decode("".join(_TEMPLATE_B64))


def canonical_template_sha256() -> str:
    """Digest of the embedded canonical template (protocol fingerprint)."""
    return hashlib.sha256(_template_bytes()).hexdigest()


def verify_canonical_template() -> str:
    """Fail closed unless the embedded template matches its pinned digest.

    Returns the template source text on success. Any edit to this
    module's template block (even whitespace) changes the digest and
    stops the tournament — canonical means byte-identical, or it means
    nothing.
    """
    text = _template_bytes().decode("utf-8")
    actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if actual != _EXPECTED_SHA256:
        raise RuntimeError(
            "canonical chat template failed its pinned digest check: "
            f"expected {_EXPECTED_SHA256}, got {actual}. The embedded "
            "template was altered; a template change is a protocol "
            "change and must go through a new protocol version, not an "
            "in-place edit."
        )
    return text


def render_canonical(tokenizer: Any, prompt: str) -> str:
    """Render one user prompt through the canonical template.

    Uses the pinned canonical template (verified at call time) with the
    tokenizer that will encode it — the template is protocol identity,
    the tokenizer only supplies special-token spellings. Fail closed if
    the embedded template fails verification.
    """
    template = verify_canonical_template()
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        chat_template=template,
    )
    if not isinstance(rendered, str) or not rendered:
        raise RuntimeError("canonical template rendering produced no text")
    return rendered
