# CircuitJS/Falstad local

Runtime: `backend/vendor/circuitjs/circuitjs.html`.

The web assets were extracted from the official Falstad offline distribution,
https://www.falstad.com/circuit/offline/circuitjs1-win.zip . Electron binaries
are not included or installed. All compiled simulator JavaScript is unmodified.
The entry HTML removes root-relative manifest and service-worker references
that are unavailable in the offline web archive.

Copyright Paul Falstad and Iain Sharp. GPL version 2 or later, with the license
in `backend/vendor/circuitjs/COPYING.txt` and `UPSTREAM_COPYING` here.

`upstream-source.zip` includes the complete upstream source snapshot preceding
the official web bundle timestamp. `PROVENANCE.json` identifies its immutable
commit and both download hashes. The official prebuilt archive does not expose
its commit identifier, so a byte-identical reproducible build is not claimed.
Build instructions and upstream license are included in that source archive.

Official API: https://www.falstad.com/circuit/doc/js-interface.html
Repository: https://github.com/pfalstad/circuitjs1

Circuit Scanner serves the simulator under its own localhost origin and sends
the confirmed circuit using `CircuitJS1.importCircuit`. A small empty `cct`
startup deck prevents a stock example from replacing the API import and keeps
URLs short. No circuit upload to Falstad's website is needed.

Conversion is in `backend/circuitjs_export.py`. It documents native element
post orders and generic-model approximations. The corresponding digital layout
is shared by SVG and KiCad in `backend/digital_twin.py`. Confirmed logical nets,
not geometric proximity, determine electrical connections.
