from pathlib import Path
from urllib.error import URLError

import pandas as pd
import streamlit as st


DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
REQUIRED_COLUMNS = ["survived", "pclass", "sex", "age", "fare", "embarked"]
AGE_BINS = [0, 12, 18, 35, 60, float("inf")]
AGE_LABELS = ["Child", "Teen", "Young Adult", "Adult", "Older Adult"]

st.set_page_config(page_title="Titanic Passenger Insights", page_icon="🚢", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg, #f7f8fc 0%, #edf3f7 100%); }
    h1 { color: #203864; }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #087f8c; border-bottom-color: #087f8c;
    }
    [data-testid="stMetric"] {
        background: white; border: 1px solid #dce5ed; border-radius: 12px;
        padding: 16px 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading Titanic passenger data...")
def load_data() -> tuple[pd.DataFrame, str]:
    local_file = Path(__file__).with_name("titanic.csv")
    if local_file.exists():
        return pd.read_csv(local_file), f"Local file: {local_file.name}"
    return pd.read_csv(DATA_URL), "Seaborn Titanic dataset (online)"


def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")

    prepared = data.loc[:, REQUIRED_COLUMNS].copy()
    prepared["age_group"] = pd.cut(
        prepared["age"], bins=AGE_BINS, labels=AGE_LABELS, include_lowest=True
    ).cat.add_categories("Unknown").fillna("Unknown")
    prepared["age"] = prepared["age"].fillna(prepared["age"].median())
    embarked_mode = prepared["embarked"].mode()
    if embarked_mode.empty:
        raise ValueError("Cannot impute embarked: the column has no non-missing values.")
    prepared["embarked"] = prepared["embarked"].fillna(embarked_mode.iloc[0])
    prepared["age_norm"] = min_max_normalize(prepared["age"])
    prepared["fare_norm"] = min_max_normalize(prepared["fare"])
    return prepared


def min_max_normalize(series: pd.Series) -> pd.Series:
    minimum = series.min()
    span = series.max() - minimum
    if span == 0:
        return pd.Series(0.0, index=series.index)
    return (series - minimum) / span


st.title("🚢 Titanic Passenger Insights")
st.caption("Explore survival patterns across passenger groups.")

try:
    raw_data, data_source = load_data()
    passenger_data = prepare_data(raw_data)
except (OSError, ValueError, pd.errors.ParserError, URLError) as error:
    st.error(f"Could not load or prepare the Titanic dataset: {error}")
    st.stop()

if passenger_data.empty:
    st.error("The Titanic dataset contains no passenger records.")
    st.stop()

st.caption(f"Data source: {data_source}")

with st.sidebar:
    st.header("Passenger filters")
    selected_classes = st.multiselect(
        "Passenger class",
        options=sorted(passenger_data["pclass"].dropna().unique().tolist()),
        default=sorted(passenger_data["pclass"].dropna().unique().tolist()),
        format_func=lambda value: f"Class {value}",
    )
    selected_sexes = st.multiselect(
        "Sex",
        options=sorted(passenger_data["sex"].dropna().unique().tolist()),
        default=sorted(passenger_data["sex"].dropna().unique().tolist()),
    )
    age_groups = passenger_data["age_group"].cat.categories.tolist()
    selected_age_groups = st.multiselect(
        "Age group",
        options=age_groups,
        default=age_groups,
    )

filtered_data = passenger_data[
    passenger_data["pclass"].isin(selected_classes)
    & passenger_data["sex"].isin(selected_sexes)
    & passenger_data["age_group"].isin(selected_age_groups)
].copy()

overview_tab, explore_tab, preprocessing_tab, data_tab = st.tabs(
    ["Overview", "Explore", "Preprocessing", "Data Explorer"]
)

with overview_tab:
    st.subheader("At a glance")
    if filtered_data.empty:
        st.info("No passengers match the selected filters.")
    else:
        survival_rate = filtered_data["survived"].mean() * 100
        metric_columns = st.columns(4)
        metric_columns[0].metric("Passengers in selection", f"{len(filtered_data):,}")
        metric_columns[1].metric("Survivors", f"{int(filtered_data['survived'].sum()):,}")
        metric_columns[2].metric("Survival rate", f"{survival_rate:.1f}%")
        metric_columns[3].metric("Average fare", f"${filtered_data['fare'].mean():.2f}")

        class_chart, outcome_chart = st.columns(2)
        with class_chart:
            st.caption("Survival rate by passenger class")
            survival_by_class = (
                filtered_data.groupby("pclass", observed=True)["survived"]
                .mean()
                .mul(100)
                .rename("Survival rate (%)")
                .rename_axis("Passenger class")
                .reset_index()
            )
            survival_by_class["Passenger class"] = survival_by_class[
                "Passenger class"
            ].map(lambda value: f"Class {value}")
            st.bar_chart(
                survival_by_class,
                x="Passenger class",
                y="Survival rate (%)",
                color="Passenger class",
                x_label="Passenger class",
                y_label="Survival rate (%)",
            )
        with outcome_chart:
            st.caption("Survival counts")
            outcome_counts = (
                filtered_data.assign(
                    outcome=filtered_data["survived"].map(
                        {0: "Did not survive", 1: "Survived"}
                    )
                )
                .groupby("outcome", observed=True)
                .size()
                .rename("Passengers")
                .rename_axis("Outcome")
                .reset_index()
            )
            st.bar_chart(
                outcome_counts,
                x="Outcome",
                y="Passengers",
                color="Outcome",
                x_label="Outcome",
                y_label="Passenger count",
            )

    sex_rates = passenger_data.groupby("sex", observed=True)["survived"].mean().mul(100)
    if len(sex_rates) > 1:
        higher_rate_sex = sex_rates.idxmax()
        lower_rate_sex = sex_rates.idxmin()
        st.info(
            f"**Finding:** In this dataset, {higher_rate_sex} passengers had a higher "
            f"observed survival rate ({sex_rates[higher_rate_sex]:.1f}%) than "
            f"{lower_rate_sex} passengers ({sex_rates[lower_rate_sex]:.1f}%). "
            "This is an association, not evidence of causation."
        )

with explore_tab:
    st.subheader("Compare passenger groups")
    if filtered_data.empty:
        st.info("No passengers match the selected filters.")
    else:
        class_chart, sex_chart = st.columns(2)
        with class_chart:
            st.caption("Survival rate by passenger class")
            survival_by_class = (
                filtered_data.groupby("pclass", observed=True)["survived"]
                .mean()
                .mul(100)
                .rename("Survival rate (%)")
                .rename_axis("Passenger class")
                .reset_index()
            )
            survival_by_class["Passenger class"] = survival_by_class[
                "Passenger class"
            ].map(lambda value: f"Class {value}")
            st.bar_chart(
                survival_by_class,
                x="Passenger class",
                y="Survival rate (%)",
                color="Passenger class",
                x_label="Passenger class",
                y_label="Survival rate (%)",
            )
        with sex_chart:
            st.caption("Survival rate by sex")
            survival_by_sex = (
                filtered_data.groupby("sex", observed=True)["survived"]
                .mean()
                .mul(100)
                .rename("Survival rate (%)")
                .rename_axis("Sex")
                .reset_index()
            )
            st.bar_chart(
                survival_by_sex,
                x="Sex",
                y="Survival rate (%)",
                color="Sex",
                x_label="Sex",
                y_label="Survival rate (%)",
            )

        distribution_chart, fare_chart = st.columns(2)
        with distribution_chart:
            st.caption("Age distribution (10-year bins)")
            age_bin_edges = [0, 10, 20, 30, 40, 50, 60, 70, 80, float("inf")]
            age_distribution = (
                pd.cut(
                    filtered_data["age"],
                    bins=age_bin_edges,
                    include_lowest=True,
                    right=False,
                )
                .value_counts(sort=False)
                .rename("Passengers")
                .to_frame()
            )
            age_distribution = age_distribution.rename_axis("Age range").reset_index()
            age_distribution["Age range"] = age_distribution["Age range"].astype(str)
            st.bar_chart(
                age_distribution,
                x="Age range",
                y="Passengers",
                x_label="Age range (years)",
                y_label="Passenger count",
            )
        with fare_chart:
            st.caption("Average fare by passenger class")
            average_fare_by_class = (
                filtered_data.groupby("pclass", observed=True)["fare"]
                .mean()
                .rename("Average fare")
                .rename_axis("Passenger class")
                .reset_index()
            )
            average_fare_by_class["Passenger class"] = average_fare_by_class[
                "Passenger class"
            ].map(lambda value: f"Class {value}")
            st.bar_chart(
                average_fare_by_class,
                x="Passenger class",
                y="Average fare",
                color="Passenger class",
                x_label="Passenger class",
                y_label="Average fare",
            )

with preprocessing_tab:
    st.subheader("Prepare the data")
    with st.expander("1. Missing values", expanded=True):
        st.write(
            "Age is filled with its median and embarked with its most frequent "
            "category; deck is excluded because it has many missing values."
        )
        imputation_counts = pd.DataFrame(
            {
                "Missing before": raw_data[["age", "embarked"]].isna().sum(),
                "Missing after": passenger_data[["age", "embarked"]].isna().sum(),
            }
        )
        st.dataframe(imputation_counts, width="stretch")

    with st.expander("2. Min-Max normalization"):
        st.write(
            "Min-Max normalization scales age and fare to the 0–1 range while "
            "keeping each original column unchanged."
        )
        st.dataframe(
            passenger_data.loc[:, ["age", "age_norm", "fare", "fare_norm"]].head(20),
            width="stretch",
        )

    with st.expander("3. Age discretization"):
        st.write(
            "Age groups are created from original ages: 0–12 Child, 13–18 Teen, "
            "19–35 Young Adult, 36–60 Adult, and 61+ Older Adult."
        )
        st.dataframe(
            passenger_data.loc[:, ["age", "age_group"]].head(20),
            width="stretch",
        )

    with st.expander("4. Dummy coding"):
        st.write(
            "One reference category is omitted for each field to reduce redundant "
            "columns (female and embarkation port C in this dataset). Survived is "
            "not dummy-coded and dummy columns are not normalized."
        )
        before, after = st.columns(2)
        with before:
            st.markdown("**Before**")
            st.dataframe(
                passenger_data.loc[:, ["sex", "embarked"]].head(5),
                width="stretch",
            )
        with after:
            st.markdown("**After**")
            dummy_coded = pd.get_dummies(
                passenger_data.loc[:, ["sex", "embarked"]],
                columns=["sex", "embarked"],
                drop_first=True,
                dtype=int,
            )
            st.dataframe(dummy_coded.head(5), width="stretch")

with data_tab:
    st.subheader("Dataset inspection")
    st.write(f"**{raw_data.shape[0]:,} rows × {raw_data.shape[1]} columns**")
    inspection_tabs = st.tabs(["Column names and types", "First five rows", "Missing values"])
    with inspection_tabs[0]:
        column_info = pd.DataFrame(
            {"Column": raw_data.columns, "Data type": raw_data.dtypes.astype(str).values}
        )
        st.dataframe(column_info, width="stretch", hide_index=True)
    with inspection_tabs[1]:
        st.dataframe(raw_data.iloc[:5], width="stretch", hide_index=True)
    with inspection_tabs[2]:
        st.dataframe(
            raw_data.isna().sum().rename("Missing values").to_frame(),
            width="stretch",
        )

    st.divider()
    st.subheader("Filtered passenger records")
    data_view = st.radio(
        "Select view",
        ["Raw", "Cleaned", "Transformed"],
        horizontal=True,
    )
    transformed_data = pd.get_dummies(
        passenger_data,
        columns=["sex", "embarked"],
        drop_first=True,
        dtype=int,
    )
    view_data = {
        "Raw": raw_data.loc[filtered_data.index],
        "Cleaned": passenger_data.loc[filtered_data.index],
        "Transformed": transformed_data.loc[filtered_data.index],
    }[data_view]
    if data_view == "Raw":
        default_columns = [column for column in REQUIRED_COLUMNS if column in view_data]
    elif data_view == "Cleaned":
        default_columns = REQUIRED_COLUMNS + ["age_group"]
    else:
        default_columns = [
            column
            for column in [
                "survived",
                "pclass",
                "age",
                "age_group",
                "age_norm",
                "fare",
                "fare_norm",
                "sex_male",
                "embarked_Q",
                "embarked_S",
            ]
            if column in view_data
        ]
    selected_columns = st.multiselect(
        "Choose columns to display",
        options=view_data.columns.tolist(),
        default=default_columns,
        key=f"data_columns_{data_view.lower()}",
    )
    st.metric("Rows after filters", f"{len(view_data):,}")
    if selected_columns:
        with st.expander("Inspect sample records"):
            st.dataframe(
                view_data.loc[:, selected_columns].head(5),
                width="stretch",
                hide_index=True,
            )
        st.dataframe(
            view_data.loc[:, selected_columns],
            width="stretch",
            height=420,
            hide_index=True,
        )
    else:
        st.info("Select at least one column to display records.")
