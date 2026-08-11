# 🔥 ForestGuard AI — Forest Fire Detection

An attractive Flask web application for classifying forest images as **Fire** or **No Fire** using a trained **Keras / TensorFlow EfficientNetB2** model.

## Model performance

Evaluated on 600 held-out test images:

| Metric            |  Score |
| ----------------- | -----: |
| Accuracy          | 96.83% |
| Fire Precision    | 95.18% |
| Fire Recall       | 98.67% |
| Fire F1           | 96.89% |
| No Fire Precision | 98.62% |
| No Fire Recall    | 95.00% |
| ROC-AUC           | 99.60% |

The optimized decision threshold is **0.70**. The model's sigmoid output represents the probability of **No Fire** because the training class mapping was:

- `0` → `fire`
- `1` → `no_fire`

## Project structure

```text
forest_fire_classifier/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── models/
│   ├── .gitkeep
│   └── forest_fire_deployment.keras   # put your model here
├── templates/
│   └── index.html
├── static/
│   └── css/
│       └── style.css
└── uploads/
    └── .gitkeep
```

## 1. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use:

```powershell
.\.venv\Scripts\activate.bat
```

## 2. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> TensorFlow is included because the application loads the `.keras` model locally. The listed versions are aligned with the environment used for this project.

## 3. Add the trained model

Copy:

```text
models\forest_fire_deployment.keras
```

into:

```text
models/
```

So the final path is:

```text
models\forest_fire_deployment.keras
```

## 4. Run the application

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## 5. Optional: run with Waitress on Windows

For a more production-like local server:

```powershell
waitress-serve --host=127.0.0.1 --port=5000 app:app
```

Then open:

```text
http://127.0.0.1:5000
```

## How prediction works

1. User uploads JPG/JPEG/PNG/WEBP.
2. Flask validates the image.
3. PIL converts it to RGB and resizes it to `260 × 260`.
4. The Keras EfficientNetB2 model predicts the sigmoid probability.
5. Because class `1` is `no_fire`, the app calculates:

```text
fire_probability = 1 - no_fire_probability
```

6. With the optimized threshold:

```text
no_fire_probability >= 0.70 → No Fire
otherwise → Fire
```

## Important model note

The `.keras` model is intentionally ignored by `.gitignore` because model files can be large. For GitHub, use **Git LFS** or a model-release/storage service rather than committing a large binary directly.

## Security notes

- Uploaded filenames are sanitized.
- Uploaded files are limited to 10 MB.
- Only common image extensions are accepted.
- The model is loaded once when the Flask process starts.
- Do not expose the development server directly to the public internet.
- Change `FLASK_SECRET_KEY` when deploying.

## License

Add your preferred project license before publishing.
