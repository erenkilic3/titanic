# Titanic Passenger Insights

An interactive Streamlit dashboard for exploring passenger survival patterns in
the Titanic dataset and demonstrating common data-preparation techniques.

## Run the app

Install dependencies and start the dashboard:

```bash
uv sync
uv run streamlit run app.py
```

By default, the app loads the Titanic CSV from the Seaborn data repository. To
use a downloaded copy instead, save it as `titanic.csv` beside
`app.py`.

## Explore and prepare the data

Use the sidebar to filter passengers by class, sex, and age group. The interface
is organized into Overview, Explore, Preprocessing, and Data Explorer tabs.
Overview reports the selected passenger count, survivors, survival rate, and
average fare; charts in Overview and Explore respond to the filters.

Data Explorer shows observed dimensions, column names and types, the first five
rows, and missing-value counts. Preprocessing shows before-and-after imputation
counts and previews in four expandable sections. The data table supports Raw,
Cleaned, and Transformed views, column selection, filtered row counts, and a
sample-record preview.

- Median imputation for age and most-frequent-value imputation for embarkation
- Min-Max normalization of age and fare while retaining their original columns
- Age discretization using the original ages
- Dummy coding of sex and embarkation, without dummy-coding survival
