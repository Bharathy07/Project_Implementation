# Deployment

## Local run

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app/paper_results.py
```

Open `http://localhost:8501`. The alternate five-case view is available with
`python -m streamlit run app/counterfactual_cases.py`.

## Streamlit Community Cloud

1. Push the Git repository root (the directory containing this project folder) to
   the configured GitHub repository.
2. Create a new Streamlit Community Cloud app from the repository and select the
   `master` branch.
3. Set **Main file path** to `Project_Implementation/app/paper_results.py`, then
   deploy.

The repository-root `requirements.txt` delegates to this project's dependency
list so Streamlit Cloud installs PyYAML and the rest of the required packages.
