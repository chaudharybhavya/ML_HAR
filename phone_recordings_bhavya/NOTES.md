# Phone recordings: person "bhavya" (14 Nov 2026)

- Phone: vivo V2439 (Android 16), app phyphox, accelerometer *with* g + gyroscope, 50 Hz.
- Position: waist, side of the hip, landscape, screen facing outward. Same position for all recordings.
- Orientation check (standing): gravity x = +1.00 g, y = 0.01, z = 0.17 (UCI standing: 1.02, -0.12, 0.10), so no axis_map needed.
- Files in `data/phone/bhavya/` follow the layout of 09_phone_pipeline.ipynb (Step 4): `<activity>_acc.csv` + `<activity>_gyro.csv`.
  - Not trimmed: the loader cuts the first and last 5 s itself (after filtering).
  - About 43 windows per activity (lying about 55) after the loader's trims.
- One file per activity (the loader reads one per activity). For the stairs, the longer recording was used:
  - upstairs = 15-31-14 (67 s); includes a short landing/turn around 35-40 s and one 0.2 s timestamp gap (the loader interpolates it).
  - downstairs = 15-29-35 (67 s).
  - The extra stair recordings (upstairs 15-27-59, 34 s; downstairs 15-25-59, 63 s) are kept in `original_xlsx/` but not used.
- Expected confusions (from simple summaries): walking vs upstairs (same bounce, differ by forward tilt and rhythm); sitting vs standing (differ only by tilt, z 0.34 vs 0.17).
- Lying: gravity mostly on y (0.99) rather than split y/z as in UCI (0.79, 0.57), because the phone sat more on the side of the hip.
