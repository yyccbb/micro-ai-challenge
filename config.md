### 📊 Model Evaluation Results

| Model               | Hidden Size | Test Loss | MSE    | MAE    | R² Score |
|---------------------|-------------|-----------|--------|--------|----------|
| baseline-1 (hidden) | 128         | 0.0372    | 0.0372 | 0.1238 | -0.2327  |
| baseline-1 (hidden) | 256         | 0.0283    | 0.0283 | 0.1247 | 0.0604   |
| baseline-2 (pooled) | 128         | 0.0277    | 0.0277 | 0.1172 | 0.0812   |
| baseline-2 (pooled) | 256         | 0.0254    | 0.0255 | 0.1135 | 0.1552   |

Observations:
1. Pooled results are better then non pooled results.
2. Hidden size of 256 is better than 128.
3. Setting patience to 20 will run more epochs leading to better results.
4. Reducing lr to 1e-4 and increasing min_delta do not yield better results.

Todo:
1. Examine early stopping, some trainings might early stop prematurely.
2. Suspect model strucure might influence results more significantly than hyper parameter search, so might change some of the model strucutre.