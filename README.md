# Colour-to-pH analysis

Estimates pH from the colour of the anthocyanin indicator images.

## Folder layout
    ph_project/
      ph_pipeline.py        the script (feature extraction, models, tests, charts)
      requirements.txt
      PH Cropped/
        pH_filenames.xlsx   labels (columns: Filename, pH)
        cropped/            612 images, 51 pH levels (3.0 to 8.0), 12 per level

## Run
    pip install -r requirements.txt
    python ph_pipeline.py

## Output (written to this folder)
- features.csv       9 colour numbers per image plus pH and replicate
- results.csv        error of each model under test A and test B
- results_chart.png  colour vs pH, predicted vs actual, error by pH region

## Notes
- A = random 5-fold cross-validation. B = train on replicates 1-9, test on 10-12.
- If A is much better than B, near-duplicate images are probably leaking between train and test.
- Runs on CPU in about a minute. No GPU needed.
