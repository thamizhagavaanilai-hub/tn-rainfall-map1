import io
import re
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import geopandas as gpd
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
from shapely.geometry import Point


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Tamil Nadu Rainfall Map",
    page_icon="🌧️",
    layout="wide",
)

st.title("🌧️ Tamil Nadu Rainfall Map")
st.caption("TN-SMART Rain Gauge Data • Date-wise Accumulation • IDW Interpolation")


# ============================================================
# SETTINGS
# ============================================================
TN_SMART_URLS = [
    "https://tnsmart-beta.rimes.int/index.php/MIS/Rainfall/raingauge_stations/",
    "https://beta-tnsmart.rimes.int/index.php/MIS/Rainfall/raingauge_stations/",
]

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0 Safari/537.36"
)

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": TN_SMART_URLS[0],
}

DEFAULT_START = date.today() - timedelta(days=22)
DEFAULT_END = date.today()


# ============================================================
# SESSION
# ============================================================
@st.cache_resource
def get_session():
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def clean_number(value):
    """Convert strings such as '12.4', '12 mm', '--' to float."""
    if pd.isna(value):
        return np.nan

    text = str(value).strip()
    if not text:
        return np.nan

    text = text.replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)

    if not match:
        return np.nan

    try:
        return float(match.group())
    except Exception:
        return np.nan


def normalize_column_name(name):
    return (
        str(name)
        .strip()
        .lower()
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("(", "")
        .replace(")", "")
        .replace(".", "")
        .replace("_", " ")
    )


def find_column(columns, keywords):
    normalized = {
        col: normalize_column_name(col)
        for col in columns
    }

    # Exact-ish matching first
    for col, text in normalized.items():
        if all(k in text for k in keywords):
            return col

    # Any matching keyword
    for col, text in normalized.items():
        if any(k in text for k in keywords):
            return col

    return None


def find_rainfall_table(html):
    """
    Find the TN-SMART table containing station, latitude,
    longitude and rainfall values.
    """
    try:
        tables = pd.read_html(io.StringIO(html))
    except Exception:
        return None

    for df in tables:
        if df.empty:
            continue

        # Flatten MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                " ".join([str(x) for x in col if str(x) != "nan"]).strip()
                for col in df.columns
            ]

        columns = list(df.columns)
        normalized = [normalize_column_name(c) for c in columns]

        lat_col = None
        lon_col = None
        rain_col = None
        station_col = None

        for c, n in zip(columns, normalized):
            if lat_col is None and ("latitude" in n or n == "lat"):
                lat_col = c

            if lon_col is None and (
                "longitude" in n or "long" in n or n == "lon"
            ):
                lon_col = c

            if rain_col is None and (
                "rainfall" in n
                or "rainfall recorded" in n
                or "rain" in n
                or "recorded rain" in n
            ):
                rain_col = c

            if station_col is None and (
                "name of the station" in n
                or "station" in n
                or "rain gauge station" in n
            ):
                station_col = c

        if lat_col and lon_col and rain_col:
            out = pd.DataFrame()

            out["station"] = (
                df[station_col].astype(str)
                if station_col
                else "Unknown Station"
            )

            out["latitude"] = df[lat_col].apply(clean_number)
            out["longitude"] = df[lon_col].apply(clean_number)
            out["rainfall_mm"] = df[rain_col].apply(clean_number)

            # Optional district extraction from the district/taluk/village column
            district_col = None
            for c, n in zip(columns, normalized):
                if (
                    "district" in n
                    and "taluk" in n
                    and "village" in n
                ):
                    district_col = c
                    break

            if district_col:
                out["district_info"] = df[district_col].astype(str)
            else:
                out["district_info"] = ""

            out = out.replace([np.inf, -np.inf], np.nan)
            out = out.dropna(
                subset=["latitude", "longitude", "rainfall_mm"]
            )

            # Tamil Nadu approximate geographic extent
            out = out[
                out["latitude"].between(7.5, 14.0)
                & out["longitude"].between(74.5, 80.5)
            ]

            # Rainfall cannot be negative
            out.loc[out["rainfall_mm"] < 0, "rainfall_mm"] = 0

            if len(out) >= 2:
                return out.reset_index(drop=True)

    return None


def response_contains_requested_date(html, requested_date):
    """
    TN-SMART sometimes returns a page successfully but ignores the
    requested date. Do not silently use today's rainfall for a
    historical date.
    """
    iso = requested_date.strftime("%Y-%m-%d")
    dmy = requested_date.strftime("%d-%m-%Y")
    readable = requested_date.strftime("%d-%b-%Y")
    readable2 = requested_date.strftime("%d-%B-%Y")

    text = re.sub(r"\s+", " ", html).lower()

    return (
        iso.lower() in text
        or dmy.lower() in text
        or readable.lower() in text
        or readable2.lower() in text
    )


def post_for_date(session, url, requested_date):
    """
    Try the common TN-SMART date parameter formats.
    """
    iso = requested_date.strftime("%Y-%m-%d")
    dmy = requested_date.strftime("%d-%m-%Y")
    slash_dmy = requested_date.strftime("%d/%m/%Y")

    payloads = [
        {"date_on": iso, "search_submit": "View Data"},
        {"date_on": dmy, "search_submit": "View Data"},
        {"date_on": slash_dmy, "search_submit": "View Data"},
        {"date_on": iso, "search_submit": "Search"},
        {"date_on": dmy, "search_submit": "Search"},
        {"date_on": iso},
        {"date_on": dmy},
        {"date": iso, "search_submit": "View Data"},
        {"date": dmy, "search_submit": "View Data"},
    ]

    for payload in payloads:
        try:
            response = session.post(
                url,
                data=payload,
                headers={"Referer": url},
                timeout=60,
                allow_redirects=True,
            )

            if response.status_code != 200:
                continue

            html = response.text

            if not response_contains_requested_date(
                html, requested_date
            ):
                continue

            df = find_rainfall_table(html)

            if df is not None and len(df) >= 2:
                return df, response.url

        except requests.RequestException:
            continue

    return None, None


@st.cache_data(ttl=1800, show_spinner=False)
def download_daily_rainfall(requested_date):
    """
    Download one day's statewide TN-SMART rain gauge data.

    The function:
      1. Opens TN-SMART first to establish cookies/session.
      2. Tries the date form using multiple date formats.
      3. Validates that the returned page actually contains the
         requested date.
      4. Parses latitude, longitude and rainfall.
    """
    session = get_session()

    last_error = ""

    for base_url in TN_SMART_URLS:
        try:
            initial = session.get(
                base_url,
                timeout=60,
                allow_redirects=True,
            )

            if initial.status_code != 200:
                last_error = (
                    f"HTTP {initial.status_code} from {base_url}"
                )
                continue

            # Try POST-based date selection.
            df, final_url = post_for_date(
                session,
                initial.url or base_url,
                requested_date,
            )

            if df is not None:
                df["date"] = requested_date.isoformat()
                return df

            # A few deployments use a direct GET parameter.
            iso = requested_date.strftime("%Y-%m-%d")
            dmy = requested_date.strftime("%d-%m-%Y")

            get_urls = [
                f"{base_url}?date_on={iso}",
                f"{base_url}?date_on={dmy}",
                f"{base_url}?date={iso}",
                f"{base_url}?date={dmy}",
            ]

            for url in get_urls:
                try:
                    response = session.get(
                        url,
                        timeout=60,
                        allow_redirects=True,
                    )

                    if response.status_code != 200:
                        continue

                    if not response_contains_requested_date(
                        response.text, requested_date
                    ):
                        continue

                    df = find_rainfall_table(response.text)

                    if df is not None:
                        df["date"] = requested_date.isoformat()
                        return df

                except requests.RequestException:
                    continue

        except requests.RequestException as exc:
            last_error = str(exc)
            continue

    raise RuntimeError(
        f"TN-SMART did not return rainfall data for "
        f"{requested_date.isoformat()}. {last_error}"
    )


# ============================================================
# DOWNLOAD DATE RANGE
# ============================================================
def download_date_range(start_date, end_date, delay_seconds=0.25):
    all_days = []
    failed_days = []

    current = start_date

    progress = st.progress(0)
    status = st.empty()

    total_days = (end_date - start_date).days + 1

    for i in range(total_days):
        current = start_date + timedelta(days=i)

        status.info(
            f"Downloading {current.strftime('%d-%m-%Y')} "
            f"({i + 1}/{total_days})..."
        )

        try:
            df = download_daily_rainfall(current)

            if df is not None and not df.empty:
                all_days.append(df)

        except Exception as exc:
            failed_days.append(
                {
                    "date": current.isoformat(),
                    "error": str(exc),
                }
            )

        progress.progress((i + 1) / total_days)

        if delay_seconds:
            time.sleep(delay_seconds)

    progress.empty()
    status.empty()

    if not all_days:
        raise RuntimeError(
            "No rainfall data was downloaded for the selected "
            "date range."
        )

    combined = pd.concat(
        all_days,
        ignore_index=True,
    )

    return combined, failed_days


# ============================================================
# ACCUMULATE RAINFALL
# ============================================================
def accumulate_station_rainfall(daily_df):
    """
    Sum rainfall by station/coordinates over the selected period.
    """
    df = daily_df.copy()

    df["station_key"] = (
        df["station"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Average coordinates for stations if tiny coordinate changes occur.
    grouped = (
        df.groupby(
            ["station_key"],
            as_index=False,
        )
        .agg(
            station=("station", "first"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            rainfall_mm=("rainfall_mm", "sum"),
            observations=("date", "count"),
        )
    )

    grouped = grouped.dropna(
        subset=["latitude", "longitude", "rainfall_mm"]
    )

    return grouped.reset_index(drop=True)


# ============================================================
# SHAPEFILE
# ============================================================
def load_tamil_nadu_shapefile(uploaded_file=None):
    """
    Supports:
      - Streamlit uploaded ZIP containing .shp/.shx/.dbf/.prj
      - Local data/tn_districts.shp
    """
    if uploaded_file is not None:
        import tempfile
        import os

        temp_dir = tempfile.mkdtemp(prefix="tn_shape_")
        zip_path = os.path.join(
            temp_dir,
            "shape.zip",
        )

        with open(zip_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(temp_dir)

        shp_files = list(
            Path(temp_dir).rglob("*.shp")
        )

        if not shp_files:
            raise FileNotFoundError(
                "No .shp file was found inside the ZIP."
            )

        return gpd.read_file(shp_files[0])

    local_shp = Path("data") / "tn_districts.shp"

    if local_shp.exists():
        return gpd.read_file(local_shp)

    return None


# ============================================================
# IDW INTERPOLATION
# ============================================================
def idw_interpolation(
    points_df,
    polygon,
    grid_size=500,
    power=2.0,
    k_neighbors=20,
):
    """
    IDW interpolation inside the Tamil Nadu polygon.
    """
    if len(points_df) < 3:
        raise ValueError(
            "At least 3 rainfall stations are required for IDW."
        )

    minx, miny, maxx, maxy = polygon.bounds

    # Prevent excessively large memory usage on Streamlit Cloud.
    grid_size = int(np.clip(grid_size, 100, 800))

    x = np.linspace(minx, maxx, grid_size)
    y = np.linspace(miny, maxy, grid_size)

    xx, yy = np.meshgrid(x, y)

    grid_points = np.column_stack(
        [xx.ravel(), yy.ravel()]
    )

    station_xy = points_df[
        ["longitude", "latitude"]
    ].to_numpy(dtype=float)

    station_values = points_df[
        "rainfall_mm"
    ].to_numpy(dtype=float)

    tree = cKDTree(station_xy)

    k = min(k_neighbors, len(station_xy))

    distances, indices = tree.query(
        grid_points,
        k=k,
    )

    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    distances = np.maximum(distances, 1e-12)

    weights = 1.0 / np.power(
        distances,
        power,
    )

    values = station_values[indices]

    interpolated = (
        np.sum(weights * values, axis=1)
        / np.sum(weights, axis=1)
    )

    # Mask outside polygon.
    inside = np.array(
        [
            polygon.contains(Point(xy))
            or polygon.touches(Point(xy))
            for xy in grid_points
        ],
        dtype=bool,
    )

    interpolated[~inside] = np.nan

    return (
        xx,
        yy,
        interpolated.reshape(xx.shape),
    )


# ============================================================
# PLOT
# ============================================================
def create_rainfall_map(
    stations,
    districts,
    start_date,
    end_date,
    grid_size,
    power,
    k_neighbors,
):
    """
    Create a professional Tamil Nadu rainfall map.
    """
    if districts.crs is None:
        districts = districts.set_crs("EPSG:4326")

    districts = districts.to_crs("EPSG:4326")

    # Merge all district polygons for interpolation mask.
    tn_polygon = districts.geometry.union_all()

    xx, yy, zz = idw_interpolation(
        stations,
        tn_polygon,
        grid_size=grid_size,
        power=power,
        k_neighbors=k_neighbors,
    )

    fig, ax = plt.subplots(
        figsize=(10, 12),
        dpi=180,
    )

    # Rainfall classes in mm.
    levels = [
        0,
        5,
        10,
        15,
        25,
        50,
        75,
        100,
        150,
        200,
        300,
        500,
    ]

    contour = ax.contourf(
        xx,
        yy,
        zz,
        levels=levels,
        cmap="Blues",
        extend="max",
        alpha=0.82,
    )

    # District boundaries
    districts.boundary.plot(
        ax=ax,
        linewidth=0.55,
        color="black",
        alpha=0.75,
    )

    # Station points
    ax.scatter(
        stations["longitude"],
        stations["latitude"],
        s=8,
        c="black",
        alpha=0.65,
        linewidths=0,
        zorder=5,
    )

    # Title
    if start_date == end_date:
        title = (
            f"TAMIL NADU RAINFALL\n"
            f"{start_date.strftime('%d %B %Y')}"
        )
    else:
        title = (
            f"TAMIL NADU RAINFALL\n"
            f"{start_date.strftime('%d %b %Y')} – "
            f"{end_date.strftime('%d %b %Y')}"
        )

    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold",
        pad=14,
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    cbar = fig.colorbar(
        contour,
        ax=ax,
        shrink=0.72,
        pad=0.03,
    )

    cbar.set_label(
        "Accumulated Rainfall (mm)",
        fontsize=10,
    )

    ax.set_aspect("equal")

    ax.text(
        0.01,
        0.01,
        "Source: TN-SMART / RIMES\n"
        "Interpolation: IDW",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            alpha=0.8,
            edgecolor="gray",
        ),
    )

    fig.tight_layout()

    return fig


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("⚙️ Map Settings")

    start_date = st.date_input(
        "Start date",
        value=DEFAULT_START,
    )

    end_date = st.date_input(
        "End date",
        value=DEFAULT_END,
    )

    grid_size = st.slider(
        "IDW grid resolution",
        min_value=200,
        max_value=700,
        value=500,
        step=50,
        help="Higher values give a smoother map but require more processing.",
    )

    power = st.slider(
        "IDW power",
        min_value=1.0,
        max_value=4.0,
        value=2.0,
        step=0.5,
    )

    k_neighbors = st.slider(
        "IDW neighbouring stations",
        min_value=5,
        max_value=40,
        value=20,
        step=5,
    )

    uploaded_shape = st.file_uploader(
        "Optional: upload Tamil Nadu district shapefile ZIP",
        type=["zip"],
        help=(
            "ZIP must contain .shp, .shx, .dbf and preferably .prj"
        ),
    )

    run_button = st.button(
        "🌧️ Generate Rainfall Map",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# VALIDATION
# ============================================================
if start_date > end_date:
    st.error("Start date cannot be after end date.")
    st.stop()


# ============================================================
# LOAD SHAPEFILE
# ============================================================
try:
    districts = load_tamil_nadu_shapefile(
        uploaded_shape
    )
except Exception as exc:
    st.error(f"Shapefile error: {exc}")
    st.stop()

if districts is None:
    st.warning(
        "Tamil Nadu district shapefile was not found.\n\n"
        "Put these files inside the GitHub project's `data` folder:\n"
        "`tn_districts.shp`, `tn_districts.shx`, "
        "`tn_districts.dbf`, `tn_districts.prj`\n\n"
        "Or upload a ZIP using the sidebar."
    )
    st.stop()


# ============================================================
# RUN
# ============================================================
if run_button:
    try:
        with st.spinner(
            "Downloading rainfall data from TN-SMART..."
        ):
            daily_df, failed_days = download_date_range(
                start_date,
                end_date,
            )

        if failed_days:
            st.warning(
                f"{len(failed_days)} day(s) could not be downloaded. "
                "Only successfully downloaded days are included."
            )

            with st.expander("Show failed dates"):
                st.dataframe(
                    pd.DataFrame(failed_days),
                    use_container_width=True,
                )

        st.success(
            f"Downloaded {len(daily_df):,} station-day records."
        )

        accumulated = accumulate_station_rainfall(
            daily_df
        )

        if accumulated.empty:
            st.error(
                "No valid station rainfall records were found."
            )
            st.stop()

        # Summary
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Stations",
                f"{len(accumulated):,}",
            )

        with col2:
            st.metric(
                "Total rainfall records",
                f"{len(daily_df):,}",
            )

        with col3:
            st.metric(
                "Maximum station rainfall",
                f"{accumulated['rainfall_mm'].max():.1f} mm",
            )

        with col4:
            st.metric(
                "Average station rainfall",
                f"{accumulated['rainfall_mm'].mean():.1f} mm",
            )

        # Map
        with st.spinner("Creating IDW rainfall map..."):
            fig = create_rainfall_map(
                accumulated,
                districts,
                start_date,
                end_date,
                grid_size,
                power,
                k_neighbors,
            )

        st.pyplot(
            fig,
            use_container_width=True,
        )

        # PNG download
        png_buffer = io.BytesIO()

        fig.savefig(
            png_buffer,
            format="png",
            dpi=300,
            bbox_inches="tight",
        )

        png_buffer.seek(0)

        st.download_button(
            label="⬇️ Download Rainfall Map PNG",
            data=png_buffer,
            file_name=(
                f"TN_Rainfall_"
                f"{start_date.strftime('%Y%m%d')}_"
                f"{end_date.strftime('%Y%m%d')}.png"
            ),
            mime="image/png",
        )

        # CSV download
        csv_data = accumulated.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇️ Download Station Rainfall CSV",
            data=csv_data,
            file_name=(
                f"TN_Rainfall_Stations_"
                f"{start_date.strftime('%Y%m%d')}_"
                f"{end_date.strftime('%Y%m%d')}.csv"
            ),
            mime="text/csv",
        )

        # Station table
        st.subheader("📊 Station-wise Accumulated Rainfall")

        display_df = accumulated.copy()

        display_df["latitude"] = display_df[
            "latitude"
        ].round(5)

        display_df["longitude"] = display_df[
            "longitude"
        ].round(5)

        display_df["rainfall_mm"] = display_df[
            "rainfall_mm"
        ].round(1)

        display_df = display_df.sort_values(
            "rainfall_mm",
            ascending=False,
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            height=500,
        )

    except Exception as exc:
        st.error(
            "Rainfall processing failed."
        )

        st.code(
            str(exc),
            language="text",
        )

        st.markdown(
            """
### If the error says `TN-SMART did not return rainfall data`

This means the TN-SMART server did not provide the requested
date through the date-selection form.

The app deliberately does **not** substitute today's rainfall
for a historical date.

Try:
1. Select a recent date.
2. Run again.
3. If the error continues, open TN-SMART in a browser and check
   whether the selected date is available.
"""
        )

else:
    st.info(
        "Select the date range and click **Generate Rainfall Map**."
    )

    st.markdown(
        """
### 📌 Required GitHub structure

```text
Tamil-Nadu-Rainfall/
│
├── app.py
├── requirements.txt
├── README.md
│
└── data/
    ├── tn_districts.shp
    ├── tn_districts.shx
    ├── tn_districts.dbf
    └── tn_districts.prj
```

### Data workflow

**TN-SMART → Daily station rainfall → Date-range accumulation
→ IDW interpolation → Tamil Nadu district mask → Rainfall map**
"""
    )
