import pandas as pd
import numpy as np
import folium
import branca.colormap as cm
from src.config import FULL_FILE, OUTPUT_DIR


def compute_protection_gap(df):
    """
    Compute protection gap for each event.
    Uses insurance_penetration as proxy where insured loss unavailable.
    """
    df = df.copy()
    df["estimated_insured_loss"] = (
        df["damage_million_usd"] * df["insurance_penetration"]
    )
    df["protection_gap_pct"] = (
        (1 - df["insurance_penetration"]).clip(0, 1) * 100
    )
    return df


def build_protection_gap_map(df):
    """
    Build Folium choropleth showing protection gap by country.
    """
    # Drop rows with missing coordinates — cannot plot without them
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    df = df[df["latitude"].between(-90, 90)].copy()
    df = df[df["longitude"].between(-180, 180)].copy()
    print(f"Rows with valid coords: {len(df)}")

    # Aggregate by country
    country_agg = df.groupby(["iso", "country"]).agg(
        avg_damage    = ("damage_million_usd", "mean"),
        total_damage  = ("damage_million_usd", "sum"),
        avg_gap_pct   = ("protection_gap_pct", "mean"),
        n_events      = ("event_id", "count"),
        avg_lat       = ("latitude", "mean"),
        avg_lon       = ("longitude", "mean"),
        insurance_pen = ("insurance_penetration", "mean")
    ).reset_index()

    country_agg = country_agg.dropna(
        subset=["avg_lat", "avg_lon", "avg_gap_pct"]
    ).copy()
    print(f"Countries on map: {len(country_agg)}")

    country_agg["uninsured_exposure"] = (
        country_agg["total_damage"] * (1 - country_agg["insurance_pen"])
    )

    # Build Folium map
    m = folium.Map(
        location=[20, 0],
        zoom_start=2,
        tiles="CartoDB positron"
    )

    colormap = cm.LinearColormap(
        colors=["#fee8c8", "#fdbb84", "#e34a33"],
        vmin=country_agg["avg_gap_pct"].min(),
        vmax=country_agg["avg_gap_pct"].max(),
        caption="Average Protection Gap (%)"
    )

    for _, row in country_agg.iterrows():
        if pd.isna(row["avg_lat"]) or pd.isna(row["avg_lon"]):
            continue
        folium.CircleMarker(
            location=[row["avg_lat"], row["avg_lon"]],
            radius=max(4, min(20, row["n_events"] * 0.5)),
            color=colormap(row["avg_gap_pct"]),
            fill=True,
            fill_color=colormap(row["avg_gap_pct"]),
            fill_opacity=0.7,
            popup=folium.Popup(
                f"<b>{row['country']}</b><br>"
                f"Events: {row['n_events']}<br>"
                f"Avg loss: ${row['avg_damage']:.1f}M<br>"
                f"Protection gap: {row['avg_gap_pct']:.1f}%<br>"
                f"Uninsured exposure: ${row['uninsured_exposure']:.1f}M",
                max_width=220
            )
        ).add_to(m)

    colormap.add_to(m)

    map_path = OUTPUT_DIR / "protection_gap_map.html"
    m.save(str(map_path))
    print(f"Map saved: {map_path}")

    top10 = country_agg.nlargest(10, "uninsured_exposure")[
        ["country", "avg_damage", "avg_gap_pct",
         "uninsured_exposure", "n_events"]
    ]
    print("\nTop 10 Underinsured High-Risk Countries:")
    print(top10.to_string(index=False))

    return m, country_agg, top10