# 🌊 ReinsureSight
### Geospatial Flood Loss Intelligence for Reinsurance Underwriting

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

> **Live Demo:** [reinsuresight-mnesckoq5cse6p9row25dt.streamlit.app](https://reinsuresight-mnesckoq5cse6p9row25dt.streamlit.app/)  
> **Author:** Aishrica Dhiman | MSc Data Science, IIIT Lucknow

---

## What Is This Project?

ReinsureSight is a machine learning pipeline that estimates economic 
flood losses at a country level using satellite climate data and 
socioeconomic indicators — and wraps every prediction in a 
statistically valid uncertainty interval using conformal prediction.

It was built to address a specific problem in the reinsurance 
industry: how do you price flood risk in markets where no historical 
claims data exists?

The answer this project demonstrates is: use physical signals 
(precipitation, soil moisture, vegetation, temperature) and 
vulnerability indicators (GDP per capita, insurance penetration) 
that are available globally, regardless of insurance history.

---

## Why Do Insurance Companies Need This?

Reinsurers like Munich Re take on the catastrophic tail of primary 
insurers' loss distributions. To price this correctly, they need 
two things:

**1. A loss estimate** — how much damage will this flood cause?  
**2. A confidence range** — how certain are we in that estimate?

Standard actuarial models need decades of claims history to answer 
these questions. That data does not exist in South Asia, Southeast 
Asia, or Sub-Saharan Africa — regions that collectively experience 
the majority of global flood events.

In 2024 alone, global flood losses exceeded $100B. Less than 30% 
was insured. The remaining 70% — the protection gap — represents 
both an unmitigated humanitarian risk and an unmeasured commercial 
opportunity for reinsurers willing to enter these markets.

ReinsureSight demonstrates that loss estimation is possible in 
these markets using freely available data. Specifically it enables:

- **Treaty pricing** without historical claims data
- **Protection gap quantification** by country and region  
- **Uncertainty-aware predictions** that underwriters can actually 
  use for risk loading decisions
- **Portfolio accumulation analysis** across geographies

---

## Objectives

1. Build a global flood loss regression model trained on verified 
   economic damage records from EM-DAT (2000–2026)
2. Enrich each event with physical climate features extracted from 
   ERA5-Land satellite data via Google Earth Engine
3. Wrap predictions in distribution-free uncertainty intervals 
   using conformal prediction (MAPIE)
4. Quantify the insurance protection gap by country using modelled 
   losses and country-level insurance penetration rates
5. Deploy as an interactive dashboard accessible to non-technical 
   stakeholders

---

## Dataset

All data used in this project is publicly available.

### EM-DAT — International Disaster Database
- Source: emdat.be (Centre for Research on the Epidemiology of 
  Disasters, UCLouvain)
- Contains: global natural disaster records since 1960
- Used: flood events 2000–2026, filtered to records with verified 
  economic damage figures
- Records after filtering: **1,202 flood events** across 131 
  countries with confirmed damage data
- Full dataset (including events without damage): **4,271 events** 
  used for protection gap mapping

### ERA5-Land — Satellite Climate Reanalysis
- Source: ECMWF via Google Earth Engine (Copernicus Climate Data 
  Store)
- Extracted per event: precipitation (30-day and 7-day 
  antecedent), soil moisture, 2m temperature, NDVI from MODIS
- Coverage: 99.8% of the 1,202 training events successfully 
  enriched

### World Bank Open Data
- Indicator: NY.GDP.PCAP.CD (GDP per capita, current USD)
- Coverage: 258 countries, 2000–2024
- Merged by ISO country code and event year

### Insurance Penetration
- Manually compiled from Swiss Re Sigma reports for 27 major 
  countries; regional medians used for remaining countries

---

## Features Used in the Model

### Physical Hazard (from ERA5-Land via GEE)
| Feature | Description |
|---|---|
| `precip_30d_mm` | Total precipitation 30 days before flood start |
| `precip_7d_mm` | Precipitation in final 7 days before flood |
| `soil_moisture` | Volumetric soil water content day before event |
| `temp_2m_c` | Mean 2m air temperature during event |
| `ndvi_pre` | MODIS vegetation index 30 days before event |

### Flood Event Characteristics (from EM-DAT)
| Feature | Description |
|---|---|
| `duration_days` | Event length in days (start to end date) |
| `flood_subtype` | Flash / Riverine / Coastal / Unspecified |
| `season` | Monsoon / Pre-monsoon / Post-monsoon / Winter |

### Exposure and Vulnerability
| Feature | Description |
|---|---|
| `total_affected` | People affected by the event |
| `total_deaths` | Fatalities recorded |
| `gdp_per_capita` | GDP per capita of affected country (year-matched) |
| `insurance_penetration` | Estimated % of losses typically insured |

### Geography and Time
| Feature | Description |
|---|---|
| `latitude`, `longitude` | Event coordinates (geocoded where missing) |
| `region`, `subregion` | UN M.49 geographic classification |
| `year` | Event year (captures urbanisation trend) |
| `coord_is_centroid` | Flag: 1 if coordinates were imputed from centroid |

**Total features entering the model: 15**

---

---

## Results

### Model Performance (Validation Set: 2023–2024)

| Metric | Value |
|---|---|
| RMSE (log scale) | 1.5677 |
| MAPE | 282.63% |
| R² (log scale) | 0.4789 |
| Conformal coverage | 87.1% |
| Validation samples | 62 events |

**On MAPE:** A MAPE of 282% is high but expected for this problem. 
Catastrophe loss distributions are heavily right-skewed — a single 
event like the 2011 Thailand floods ($57B) creates irreducible 
percentage error. The model's primary value is in rank-ordering 
risk across geographies and quantifying the protection gap, not 
predicting exact dollar figures for individual events.

**On R²:** Explaining 48% of variance in global flood losses using 
only open-data climate and socioeconomic features — with no 
proprietary claims data — is a reasonable result for this problem.

**On conformal coverage:** The target was 90%. Achieved 87.1% on 
62 held-out validation events. The slight shortfall is consistent 
with the small test set size and the distribution shift between 
training (pre-2022) and validation (2023–2024) periods.

### Protection Gap Analysis

| Country | Avg Loss ($M) | Gap | Uninsured Exposure ($M) | Events |
|---|---|---|---|---|
| China | 1,959 | 92% | 290,190 | 225 |
| India | 1,505 | 96% | 101,202 | 205 |
| Thailand | 2,382 | 94% | 73,919 | 76 |
| Pakistan | 2,858 | 98% | 47,616 | 89 |
| USA | 1,270 | 46% | 45,577 | 123 |
| Brazil | 361 | 88% | 16,224 | 119 |
| Japan | 6,506 | 57% | 11,126 | 9 |
| Germany | 10,255 | 52% | 10,665 | 3 |
| Indonesia | 276 | 96% | 10,082 | 217 |
| Australia | 1,640 | 58% | 9,517 | 16 |

---

## Project Structure
```
ReinsureSight/
│
├── data/                     # Not in repo — add locally
│   ├── flood_model_final.csv
│   └── flood_full_pregee.csv
│
├── src/
│   ├── config.py             # Paths, constants, feature lists
│   ├── preprocess.py         # Cleaning, encoding, splits
│   ├── train.py              # LightGBM + Optuna
│   ├── evaluate.py           # Metrics + SHAP
│   ├── predict.py            # MAPIE conformal prediction
│   └── protection_gap.py     # Gap analysis + Folium map
│
├── app/
│   └── streamlit_app.py      # Streamlit dashboard
│
├── models/                   # Generated by main.py
│   ├── lgbm_model.pkl
│   ├── mapie_model.pkl
│   └── feature_cols.json
│
├── outputs/                  # Generated by main.py
│   ├── shap_summary.png
│   ├── protection_gap_map.html
│   └── metrics.json
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modelling.ipynb
│   └── 03_protection_gap.ipynb
│
├── main.py
├── requirements.txt
└── README.md
```

---

## Limitations

**Small validation set** — 62 events in 2023–2024 is a limited 
test set. Metrics should be interpreted directionally, not as 
definitive performance benchmarks.

**Country-level coordinates** — 32% of events were geocoded to 
country centroids rather than precise event locations. This 
reduces the accuracy of ERA5 feature extraction for those events.

**Insurance penetration proxy** — country-level penetration rates 
are averages. They do not capture within-country variation between 
urban and rural areas, or between asset classes.

**ERA5 temporal lag** — ERA5-Land data has a ~3 month lag. Events 
from late 2025 and 2026 were excluded from GEE feature extraction 
and imputed with regional medians.

**No building exposure data** — a production NatCat model would 
incorporate high-resolution building stock and asset value data. 
This model uses population and GDP as proxies.

---

## Future Improvements

- **Higher resolution exposure data** — integrate OpenStreetMap 
  building footprints and WorldPop gridded population to replace 
  country-level proxies
- **Multi-peril extension** — add tropical cyclone and earthquake 
  perils using the same pipeline architecture
- **Sentinel-1 SAR flood extent** — extract actual inundated area 
  in km² per event to replace the ERA5 precipitation proxy
- **River discharge features** — ERA5-Land river discharge band 
  extraction failed for this version; adding it would directly 
  capture streamflow dynamics
- **Temporal models** — replace LightGBM with an LSTM or 
  Temporal Fusion Transformer to capture multi-day flood 
  accumulation dynamics
- **Larger validation set** — as EM-DAT validates 2025–2026 
  records annually, the test set will grow and metrics will 
  become more reliable

---

## Conclusion

ReinsureSight demonstrates that meaningful flood loss estimation 
is achievable using freely available satellite and socioeconomic 
data — without proprietary claims history. The pipeline covers 
the full workflow from raw disaster records to uncertainty-aware 
predictions and interactive protection gap intelligence.

The core finding is that China, India, Thailand, Pakistan, and 
Indonesia collectively represent over $520B in uninsured flood 
exposure across the 2000–2026 observation period. These markets 
have the highest event frequency, the weakest insurance 
penetration, and — from a reinsurance perspective — the largest 
untapped premium opportunity.

The conformal prediction framework ensures that every loss 
estimate comes with a statistically valid range, making the 
output usable for underwriting decisions rather than just 
academic benchmarking.

---

## Tech Stack

| Category | Tools |
|---|---|
| Data processing | pandas, numpy, geopandas |
| Satellite data | Google Earth Engine, ERA5-Land, MODIS |
| Machine learning | LightGBM, scikit-learn |
| Hyperparameter tuning | Optuna |
| Uncertainty quantification | MAPIE (conformal prediction) |
| Explainability | SHAP |
| Visualisation | Folium, matplotlib, seaborn |
| Dashboard | Streamlit |
| Geocoding | Nominatim (geopy) |
| External APIs | World Bank Open Data, EM-DAT |

---

## License

MIT License — free to use with attribution.