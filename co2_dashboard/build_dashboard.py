"""Build a single-file interactive OWID CO2 dashboard with Pyecharts."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from jinja2 import Template
from pyecharts import options as opts
from pyecharts.charts import Bar, HeatMap, Line, Map, Pie, Sankey, Scatter, WordCloud
from pyecharts.commons.utils import JsCode


BG = "#08111f"
CARD = "rgba(13, 27, 42, 0.92)"
TEXT = "#eaf2ff"
MUTED = "#9fb3c8"
GRID = "rgba(159,179,200,.13)"
POPULATION_THRESHOLD = 5_000_000
FORECAST_START_YEAR = 2025
DEFAULT_FORECAST_END_YEAR = 2035
FORECAST_TRAIN_START_YEAR = 2005
MIN_FORECAST_POINTS = 5


@dataclass
class PreparedData:
    df: pd.DataFrame
    real_countries: pd.DataFrame
    annual_countries: pd.DataFrame
    world_df: pd.DataFrame
    world_annual: pd.DataFrame
    df_2024: pd.DataFrame
    df_2022: pd.DataFrame
    df_trade_2023: pd.DataFrame
    forecast_end_year: int


@dataclass
class Card:
    key: str
    title: str
    subtitle: str
    span: str
    content: str
    note: str = ""
    timeline: Optional[Tuple[int, int, int]] = None


MAP_NAME_ALIASES = {
    "Democratic Republic of Congo": "Democratic Republic of the Congo",
    "Congo": "Republic of the Congo",
    "Cote d'Ivoire": "Côte d'Ivoire",
    "Timor": "East Timor",
    "Czechia": "Czech Republic",
    "Eswatini": "Swaziland",
    "North Macedonia": "Macedonia",
    "Turkiye": "Turkey",
}


def safe_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def fmt_num(value: Any, digits: int = 2) -> str:
    number = safe_float(value)
    return "数据缺失" if number is None else f"{number:,.{digits}f}"


def fmt_million_tonnes(value: Any) -> str:
    return f"{fmt_num(value)} MtCO₂" if safe_float(value) is not None else "数据缺失"


def fmt_population(value: Any) -> str:
    number = safe_float(value)
    if number is None:
        return "数据缺失"
    if abs(number) >= 1_000_000_000:
        return f"{number / 1_000_000_000:.2f} billion"
    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:.2f} million"
    return f"{number:,.0f}"


def fmt_gdp(value: Any) -> str:
    number = safe_float(value)
    if number is None:
        return "数据缺失"
    if abs(number) >= 1_000_000_000_000:
        return f"${number / 1_000_000_000_000:.2f} trillion"
    return f"${number / 1_000_000_000:.2f} billion"


def js_object_literal(mapping: Dict[str, Any]) -> str:
    """Return a JsCode-safe object literal without double-quoted keys.

    Pyecharts escapes double quotes embedded in JsCode formatters, so keys use
    single-quoted JavaScript strings while numeric/list values remain JSON.
    """
    parts = []
    for key, value in mapping.items():
        escaped_key = str(key).replace("\\", "\\\\").replace("'", "\\'")
        parts.append(f"'{escaped_key}':{json.dumps(value, ensure_ascii=True, allow_nan=False)}")
    return "{" + ",".join(parts) + "}"


def load_data(path: Union[str, Path]) -> pd.DataFrame:
    data_path = Path(path).expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"找不到数据文件：{data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    required = {"country", "year", "iso_code"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"数据缺少基础字段：{', '.join(sorted(missing))}")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["year"].notna()].copy()
    df["year"] = df["year"].astype(int)
    return df


def forecast_series(
    frame: pd.DataFrame,
    value_col: str,
    future_years: Sequence[int],
    *,
    log_model: bool = False,
    allow_negative: bool = False,
    train_start: int = FORECAST_TRAIN_START_YEAR,
    min_points: int = MIN_FORECAST_POINTS,
) -> Dict[int, float]:
    """Forecast one annual series with a small, transparent regression model.

    This is intentionally simple: recent valid observations are fitted with a
    first-degree polynomial.  GDP and population use a log-linear variant,
    while emissions and climate-responsibility indicators use a linear trend.
    """
    if value_col not in frame.columns or not future_years:
        return {}
    sample = frame[["year", value_col]].copy()
    sample[value_col] = pd.to_numeric(sample[value_col], errors="coerce")
    sample = sample.dropna(subset=["year", value_col]).sort_values("year")
    if log_model:
        sample = sample[sample[value_col] > 0]
    recent = sample[sample["year"] >= train_start]
    if len(recent) < min_points:
        recent = sample.tail(max(min_points, 10))
    if len(recent) < min_points or recent[value_col].nunique(dropna=True) <= 1:
        return {}
    x = recent["year"].astype(float).to_numpy()
    y = recent[value_col].astype(float).to_numpy()
    try:
        if log_model:
            coef = np.polyfit(x, np.log(y), 1)
            raw = np.exp(np.polyval(coef, np.asarray(future_years, dtype=float)))
        else:
            coef = np.polyfit(x, y, 1)
            raw = np.polyval(coef, np.asarray(future_years, dtype=float))
    except (np.linalg.LinAlgError, FloatingPointError, ValueError):
        return {}

    max_hist = float(np.nanmax(y)) if len(y) else 0.0
    predictions: Dict[int, float] = {}
    for year, value in zip(future_years, raw):
        value = float(value)
        if not math.isfinite(value):
            continue
        if not allow_negative:
            value = max(0.0, value)
            if max_hist > 0:
                value = min(value, max_hist * (4.0 if log_model else 3.0))
        predictions[int(year)] = value
    return predictions


def build_country_forecasts(real: pd.DataFrame, forecast_end_year: int) -> pd.DataFrame:
    if forecast_end_year < FORECAST_START_YEAR or real.empty:
        result = real.copy()
        result["is_forecast"] = False
        return result
    future_years = list(range(FORECAST_START_YEAR, forecast_end_year + 1))
    rows: List[Dict[str, Any]] = []
    columns = list(real.columns)
    value_specs = {
        "co2": {"log_model": False, "allow_negative": False},
        "coal_co2": {"log_model": False, "allow_negative": False},
        "oil_co2": {"log_model": False, "allow_negative": False},
        "gas_co2": {"log_model": False, "allow_negative": False},
        "cement_co2": {"log_model": False, "allow_negative": False},
        "flaring_co2": {"log_model": False, "allow_negative": False},
        "population": {"log_model": True, "allow_negative": False},
        "gdp": {"log_model": True, "allow_negative": False},
        "trade_co2": {"log_model": False, "allow_negative": True},
        "temperature_change_from_ghg": {"log_model": False, "allow_negative": False},
    }
    for _, group in real.groupby("country", sort=False):
        group = group.sort_values("year")
        latest = group.iloc[-1]
        forecasts = {
            column: forecast_series(group, column, future_years, **spec)
            for column, spec in value_specs.items()
            if column in group.columns
        }
        if not forecasts or not any(forecasts.values()):
            continue
        last_cumulative = None
        if "cumulative_co2" in group.columns:
            cumulative_sample = pd.to_numeric(group["cumulative_co2"], errors="coerce").dropna()
            if not cumulative_sample.empty:
                last_cumulative = float(cumulative_sample.iloc[-1])
        running_cumulative = last_cumulative
        for year in future_years:
            record = {column: np.nan for column in columns}
            record["country"] = latest.get("country")
            record["iso_code"] = latest.get("iso_code")
            record["year"] = int(year)
            for column, predicted in forecasts.items():
                if year in predicted:
                    record[column] = predicted[year]
            co2 = safe_float(record.get("co2"))
            population = safe_float(record.get("population"))
            gdp = safe_float(record.get("gdp"))
            if co2 is not None and population is not None and population > 0:
                record["co2_per_capita"] = co2 * 1_000_000 / population
            if co2 is not None and gdp is not None and gdp > 0:
                record["co2_per_gdp"] = co2 * 1_000_000_000 / gdp
            if running_cumulative is not None and co2 is not None:
                running_cumulative += co2
                record["cumulative_co2"] = running_cumulative
            rows.append(record)
    future = pd.DataFrame(rows, columns=columns)
    if not future.empty:
        future["is_forecast"] = True
        if "co2" in future.columns:
            for year, group in future.groupby("year"):
                total = pd.to_numeric(group["co2"], errors="coerce").sum()
                if total > 0 and "share_global_co2" in future.columns:
                    future.loc[group.index, "share_global_co2"] = pd.to_numeric(group["co2"], errors="coerce") / total * 100
        if "cumulative_co2" in future.columns:
            for year, group in future.groupby("year"):
                total = pd.to_numeric(group["cumulative_co2"], errors="coerce").sum()
                if total > 0 and "share_global_cumulative_co2" in future.columns:
                    future.loc[group.index, "share_global_cumulative_co2"] = pd.to_numeric(group["cumulative_co2"], errors="coerce") / total * 100
        if "temperature_change_from_ghg" in future.columns and "share_of_temperature_change_from_ghg" in future.columns:
            for year, group in future.groupby("year"):
                total = pd.to_numeric(group["temperature_change_from_ghg"], errors="coerce").sum()
                if total > 0:
                    future.loc[group.index, "share_of_temperature_change_from_ghg"] = pd.to_numeric(group["temperature_change_from_ghg"], errors="coerce") / total * 100
    historical = real.copy()
    historical["is_forecast"] = False
    return pd.concat([historical, future], ignore_index=True, sort=False) if not future.empty else historical


def build_world_forecasts(world_df: pd.DataFrame, forecast_end_year: int) -> pd.DataFrame:
    world = world_df.copy()
    world["is_forecast"] = False
    if forecast_end_year < FORECAST_START_YEAR or world.empty:
        return world
    future_years = list(range(FORECAST_START_YEAR, forecast_end_year + 1))
    fields = ["co2", "coal_co2", "oil_co2", "gas_co2", "cement_co2", "flaring_co2", "co2_per_capita"]
    rows = []
    latest = world.sort_values("year").iloc[-1]
    forecasts = {
        field: forecast_series(world, field, future_years, log_model=False, allow_negative=False)
        for field in fields
        if field in world.columns
    }
    for year in future_years:
        record = {column: np.nan for column in world_df.columns}
        record["country"] = "World"
        record["iso_code"] = latest.get("iso_code")
        record["year"] = int(year)
        for field, predicted in forecasts.items():
            if year in predicted:
                record[field] = predicted[year]
        rows.append(record)
    future = pd.DataFrame(rows, columns=world_df.columns)
    if not future.empty:
        future["is_forecast"] = True
    return pd.concat([world, future], ignore_index=True, sort=False)


def prepare_data(df: pd.DataFrame, forecast_end_year: int = DEFAULT_FORECAST_END_YEAR) -> PreparedData:
    real = df[df["iso_code"].notna() & ~df["iso_code"].astype(str).str.startswith("OWID_")].copy()
    world = df[df["country"] == "World"].copy()
    forecast_end_year = max(2024, int(forecast_end_year))
    annual_countries = build_country_forecasts(real, forecast_end_year)
    world_annual = build_world_forecasts(world, forecast_end_year)
    df_2024 = real[(real["year"] == 2024) & real.get("co2", pd.Series(index=real.index, dtype=float)).notna()].copy()
    econ_fields = ["gdp", "co2", "population"]
    if all(column in real.columns for column in econ_fields):
        df_2022 = real[(real["year"] == 2022) & real[econ_fields].notna().all(axis=1)].copy()
    else:
        df_2022 = real.iloc[0:0].copy()
    if "trade_co2" in real.columns:
        df_trade = real[(real["year"] == 2023) & real["trade_co2"].notna()].copy()
    else:
        df_trade = real.iloc[0:0].copy()
    return PreparedData(df, real, annual_countries, world, world_annual, df_2024, df_2022, df_trade, forecast_end_year)


def missing_html(message: str) -> str:
    return (
        '<div class="missing"><div class="missing-icon">◌</div>'
        f'<strong>数据不足</strong><p>{message}</p></div>'
    )


def require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> Optional[str]:
    missing = [column for column in columns if column not in frame.columns]
    return f"字段缺失：{', '.join(missing)}" if missing else None


def base_axis_opts(name: str = "") -> opts.AxisOpts:
    return opts.AxisOpts(
        name=name,
        name_textstyle_opts=opts.TextStyleOpts(color=MUTED),
        axislabel_opts=opts.LabelOpts(color=MUTED),
        axisline_opts=opts.AxisLineOpts(linestyle_opts=opts.LineStyleOpts(color="#38506a")),
        splitline_opts=opts.SplitLineOpts(is_show=True, linestyle_opts=opts.LineStyleOpts(color=GRID)),
    )


def common_title(title: str) -> opts.TitleOpts:
    return opts.TitleOpts(
        title=title,
        pos_left="1%",
        title_textstyle_opts=opts.TextStyleOpts(color=TEXT, font_size=13, font_weight="normal"),
    )


def common_toolbox() -> opts.ToolboxOpts:
    return opts.ToolboxOpts(
        is_show=True,
        pos_right="1%",
        feature={
            "saveAsImage": {"title": "保存图片", "backgroundColor": BG},
            "restore": {"title": "还原"},
            "dataView": {"title": "数据视图", "readOnly": True},
        },
    )


def chart_html(chart: Any) -> str:
    return chart.render_embed()


def build_annual_payload(real_countries: pd.DataFrame) -> Dict[str, List[List[Any]]]:
    """Build one compact shared annual dataset for every dynamic chart.

    Array schema:
    country, map_name, co2, co2_per_capita, population, share_global_co2,
    coal, oil, gas, cement, flaring, cumulative, cumulative_share,
    temperature, temperature_share, gdp, co2_per_gdp, trade_co2, is_forecast.
    """
    fields = [
        "country", "year", "co2", "co2_per_capita", "population", "share_global_co2",
        "coal_co2", "oil_co2", "gas_co2", "cement_co2", "flaring_co2",
        "cumulative_co2", "share_global_cumulative_co2", "temperature_change_from_ghg",
        "share_of_temperature_change_from_ghg", "gdp", "co2_per_gdp", "trade_co2",
        "is_forecast",
    ]
    frame = real_countries.copy()
    for field in fields:
        if field not in frame.columns:
            frame[field] = False if field == "is_forecast" else np.nan
    payload: Dict[str, List[List[Any]]] = {}
    for year, group in frame[fields].groupby("year", sort=True):
        rows: List[List[Any]] = []
        for row in group.itertuples(index=False):
            country = str(row.country)
            rows.append([
                country,
                MAP_NAME_ALIASES.get(country, country),
                safe_float(row.co2),
                safe_float(row.co2_per_capita),
                safe_float(row.population),
                safe_float(row.share_global_co2),
                safe_float(row.coal_co2),
                safe_float(row.oil_co2),
                safe_float(row.gas_co2),
                safe_float(row.cement_co2),
                safe_float(row.flaring_co2),
                safe_float(row.cumulative_co2),
                safe_float(row.share_global_cumulative_co2),
                safe_float(row.temperature_change_from_ghg),
                safe_float(row.share_of_temperature_change_from_ghg),
                safe_float(row.gdp),
                safe_float(row.co2_per_gdp),
                safe_float(row.trade_co2),
                bool(row.is_forecast),
            ])
        payload[str(int(year))] = rows
    return payload


def create_world_map(df_2024: pd.DataFrame) -> str:
    error = require_columns(df_2024, ["country", "co2", "co2_per_capita", "population"])
    if error or df_2024.empty:
        return missing_html(error or "2024 年国家排放数据为空。")
    data = []
    for row in df_2024.itertuples(index=False):
        country = MAP_NAME_ALIASES.get(str(row.country), str(row.country))
        data.append((country, [safe_float(row.co2) or 0, safe_float(row.co2_per_capita), safe_float(row.population)]))
    vmax = float(df_2024["co2"].quantile(0.98))
    chart = Map(init_opts=opts.InitOpts(width="100%", height="560px", theme="dark", bg_color="transparent"))
    chart.add("2024 CO₂", data, maptype="world", is_map_symbol_show=False)
    chart.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    chart.set_global_opts(
        title_opts=common_title("2024 年国家 CO₂ 排放空间分布"),
        toolbox_opts=common_toolbox(),
        legend_opts=opts.LegendOpts(is_show=False),
        tooltip_opts=opts.TooltipOpts(formatter=JsCode("""
          function(p){var v=Array.isArray(p.value)?p.value:[p.value,null,null];
          function n(x,d){return (x==null||isNaN(x))?'数据缺失':Number(x).toLocaleString(undefined,{maximumFractionDigits:d});}
          function pop(x){if(x==null||isNaN(x))return '数据缺失';return x>=1e9?(x/1e9).toFixed(2)+' billion':(x/1e6).toFixed(2)+' million';}
          return '<b>'+p.name+'</b><br/>CO₂：'+n(v[0],2)+' MtCO₂<br/>人均：'+n(v[1],2)+' tCO₂/person<br/>人口：'+pop(v[2]);}
        """)),
        visualmap_opts=opts.VisualMapOpts(
            min_=0, max_=max(vmax, 1), dimension=0, is_calculable=True, orient="horizontal",
            pos_left="center", pos_bottom="1%", range_color=["#2166ac", "#67a9cf", "#f6c85f", "#ef8a62", "#b2182b"],
            textstyle_opts=opts.TextStyleOpts(color=MUTED),
        ),
    )
    return chart_html(chart)


def _bar_items(values: Sequence[float], colors: Sequence[str]) -> List[opts.BarItem]:
    count = max(len(values) - 1, 1)
    items = []
    for index, value in enumerate(values):
        color = colors[min(int(index / count * (len(colors) - 1)), len(colors) - 1)]
        items.append(opts.BarItem(name="", value=round(float(value), 4), itemstyle_opts=opts.ItemStyleOpts(color=color, border_radius=[0, 5, 5, 0])))
    return items


def create_top_emitters_bar(df_2024: pd.DataFrame) -> str:
    fields = ["country", "co2", "share_global_co2", "co2_per_capita"]
    error = require_columns(df_2024, fields)
    if error or df_2024.empty:
        return missing_html(error or "2024 年排放排名数据为空。")
    top = df_2024.dropna(subset=["co2"]).nlargest(20, "co2").sort_values("co2")
    countries, values = top["country"].tolist(), top["co2"].tolist()
    meta = {row.country: [safe_float(row.co2), safe_float(row.share_global_co2), safe_float(row.co2_per_capita)] for row in top.itertuples()}
    chart = Bar(init_opts=opts.InitOpts(width="100%", height="560px", theme="dark", bg_color="transparent"))
    chart.add_xaxis(countries).add_yaxis(
        "CO₂", _bar_items(values, ["#5b8db8", "#f6c85f", "#ef8a62", "#b2182b"]),
        label_opts=opts.LabelOpts(is_show=True, position="right", color=TEXT, formatter=JsCode("function(p){return p.dataIndex>=17?p.value.toLocaleString(undefined,{maximumFractionDigits:0}):'';}")),
    ).reversal_axis()
    chart.set_global_opts(
        title_opts=common_title("2024 年 CO₂ 排放前 20 国家"), toolbox_opts=common_toolbox(),
        legend_opts=opts.LegendOpts(is_show=False), xaxis_opts=base_axis_opts("MtCO₂"),
        yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(color=MUTED, font_size=11), axisline_opts=opts.AxisLineOpts(is_show=False)),
        tooltip_opts=opts.TooltipOpts(formatter=JsCode(f"""
          function(p){{var meta={js_object_literal(meta)}; var v=meta[p.name]||[]; function n(x,d){{return x==null?'数据缺失':Number(x).toLocaleString(undefined,{{maximumFractionDigits:d}});}}
          return '<b>'+p.name+'</b><br/>CO₂：'+n(v[0],2)+' MtCO₂<br/>全球占比：'+n(v[1],2)+'%<br/>人均排放：'+n(v[2],2)+' tCO₂/person';}}
        """)),
    )
    chart.options["grid"] = {"left": "27%", "right": "11%", "top": "10%", "bottom": "8%"}
    return chart_html(chart)


def create_historical_line(df: pd.DataFrame) -> str:
    error = require_columns(df, ["country", "year", "co2"])
    if error:
        return missing_html(error)
    entities = ["World", "China", "United States", "India", "Russia", "Japan", "Germany"]
    max_year = int(pd.to_numeric(df["year"], errors="coerce").max()) if not df.empty else 2024
    max_year = max(2024, max_year)
    years = list(range(1850, max_year + 1))
    palette = {"World": "#f6c85f", "China": "#ef4b5f", "United States": "#59a5ff", "India": "#ff9f43", "Russia": "#9b8afb", "Japan": "#65d6ad", "Germany": "#aab7c4"}
    chart = Line(init_opts=opts.InitOpts(width="100%", height="500px", theme="dark", bg_color="transparent"))
    chart.add_xaxis(years)
    added = 0
    added_colors = []
    for entity in entities:
        subset = df[(df["country"] == entity) & df["year"].between(1850, max_year)][["year", "co2"]].drop_duplicates("year")
        if subset.empty:
            continue
        lookup = subset.set_index("year")["co2"].to_dict()
        values = [safe_float(lookup.get(year)) if year <= 2024 else None for year in years]
        chart.add_yaxis(
            entity, values, is_symbol_show=False, is_connect_nones=True,
            linestyle_opts=opts.LineStyleOpts(width=3.5 if entity == "World" else 2, color=palette[entity], opacity=1 if entity in entities[:4] else .72),
            itemstyle_opts=opts.ItemStyleOpts(color=palette[entity]),
            label_opts=opts.LabelOpts(is_show=False),
        )
        # Numeric x labels make Pyecharts emit [year, value] pairs.  Because the
        # axis is categorical, ECharts needs a plain one-dimensional y series.
        chart.options["series"][-1]["data"] = values
        added_colors.append(palette[entity])
        added += 1
        if max_year >= FORECAST_START_YEAR and any(safe_float(lookup.get(year)) is not None for year in range(FORECAST_START_YEAR, max_year + 1)):
            forecast_values = [safe_float(lookup.get(year)) if year >= 2024 else None for year in years]
            chart.add_yaxis(
                f"{entity} 预测", forecast_values, is_symbol_show=False, is_connect_nones=True,
                linestyle_opts=opts.LineStyleOpts(width=2.4 if entity == "World" else 1.8, color=palette[entity], opacity=.6, type_="dashed"),
                itemstyle_opts=opts.ItemStyleOpts(color=palette[entity], opacity=.7),
                label_opts=opts.LabelOpts(is_show=False),
            )
            chart.options["series"][-1]["data"] = forecast_values
            added_colors.append(palette[entity])
    if not added:
        return missing_html("1850–2024 年趋势数据为空。")
    # ECharts legend markers read the series/item palette.  Supplying the same
    # ordered palette as lineStyle keeps every legend swatch and curve aligned.
    chart.set_colors(added_colors)
    chart.set_global_opts(
        title_opts=common_title(f"1850–{max_year} 年主要经济体 CO₂ 排放趋势（虚线为简单回归预测）"), toolbox_opts=common_toolbox(),
        legend_opts=opts.LegendOpts(pos_top="5%", textstyle_opts=opts.TextStyleOpts(color=MUTED)),
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross", value_formatter=JsCode("function(v){return v==null?'数据缺失':Number(v).toLocaleString(undefined,{maximumFractionDigits:2})+' MtCO₂';}")),
        xaxis_opts=base_axis_opts("年份"), yaxis_opts=base_axis_opts("MtCO₂"),
        datazoom_opts=[opts.DataZoomOpts(pos_bottom="2%", range_start=0, range_end=100, filter_mode="none")],
    )
    chart.options["grid"] = {"left": "7%", "right": "4%", "top": "14%", "bottom": "16%"}
    return chart_html(chart)


def create_source_area(world_df: pd.DataFrame) -> str:
    fields = ["year", "coal_co2", "oil_co2", "gas_co2", "cement_co2", "flaring_co2"]
    error = require_columns(world_df, fields)
    if error:
        return missing_html(error)
    max_year = int(pd.to_numeric(world_df["year"], errors="coerce").max()) if not world_df.empty else 2024
    max_year = max(2024, max_year)
    subset = world_df[world_df["year"].between(1850, max_year)].sort_values("year")
    if subset.empty:
        return missing_html("World 1850–2024 年来源数据为空。")
    sources = [("煤炭", "coal_co2", "#3b3433"), ("石油", "oil_co2", "#c46b3c"), ("天然气", "gas_co2", "#3f83c5"), ("水泥", "cement_co2", "#aab7c4"), ("火炬燃烧", "flaring_co2", "#cf3f4f")]
    chart = Line(init_opts=opts.InitOpts(width="100%", height="480px", theme="dark", bg_color="transparent"))
    chart.add_xaxis(subset["year"].astype(int).tolist())
    source_colors = []
    for label, field, color in sources:
        values = [0 if pd.isna(v) else round(float(v), 4) for v in subset[field]]
        chart.add_yaxis(
            label, values, stack="sources", is_symbol_show=False, is_smooth=True,
            areastyle_opts=opts.AreaStyleOpts(opacity=.72, color=color),
            linestyle_opts=opts.LineStyleOpts(width=1.2, color=color),
            itemstyle_opts=opts.ItemStyleOpts(color=color),
            label_opts=opts.LabelOpts(is_show=False),
        )
        chart.options["series"][-1]["data"] = values
        source_colors.append(color)
    # Keep the legend swatch, line boundary and stacked area on the same
    # semantic source palette instead of falling back to ECharts defaults.
    chart.set_colors(source_colors)
    if max_year >= FORECAST_START_YEAR and chart.options.get("series"):
        chart.options["series"][0]["markLine"] = {
            "symbol": "none",
            "silent": True,
            "lineStyle": {"color": "#f6c85f", "type": "dashed", "opacity": 0.65},
            "label": {"formatter": "预测起点", "color": TEXT},
            "data": [{"xAxis": FORECAST_START_YEAR}],
        }
    chart.set_global_opts(
        title_opts=common_title(f"全球化石燃料与工业来源排放结构（1850–{max_year}，2025 后为预测）"), toolbox_opts=common_toolbox(),
        legend_opts=opts.LegendOpts(pos_top="5%", textstyle_opts=opts.TextStyleOpts(color=MUTED)),
        tooltip_opts=opts.TooltipOpts(trigger="axis", value_formatter=JsCode("function(v){return Number(v).toLocaleString(undefined,{maximumFractionDigits:2})+' MtCO₂';}")),
        xaxis_opts=base_axis_opts("年份"), yaxis_opts=base_axis_opts("MtCO₂"),
        datazoom_opts=[opts.DataZoomOpts(pos_bottom="2%", range_start=0, range_end=100, filter_mode="none")],
    )
    chart.options["grid"] = {"left": "9%", "right": "4%", "top": "15%", "bottom": "17%"}
    return chart_html(chart)


def create_gdp_co2_scatter(df_2022: pd.DataFrame) -> str:
    fields = ["country", "gdp", "co2", "population", "co2_per_capita", "co2_per_gdp"]
    error = require_columns(df_2022, fields)
    if error:
        return missing_html(error)
    subset = df_2022.dropna(subset=fields).query("population >= @POPULATION_THRESHOLD and gdp > 0 and co2 > 0 and co2_per_gdp >= 0").copy()
    if subset.empty:
        return missing_html("2022 年满足人口阈值及完整字段条件的数据为空。")
    points = [[float(r.gdp), float(r.co2), float(r.population), float(r.co2_per_capita), float(r.co2_per_gdp), str(r.country)] for r in subset.itertuples()]
    vmin, vmax = float(subset["co2_per_gdp"].quantile(.03)), float(subset["co2_per_gdp"].quantile(.97))
    chart = Scatter(init_opts=opts.InitOpts(width="100%", height="510px", theme="dark", bg_color="transparent"))
    chart.add_xaxis([]).add_yaxis(
        "国家", points,
        symbol_size=JsCode("function(v){return Math.max(7,Math.min(48,Math.sqrt(v[2]/1000000)*2.15));}"),
        label_opts=opts.LabelOpts(
            is_show=True,
            formatter=JsCode("function(p){var n=p.value[5]; return ['China','United States','India','Japan','Germany','Russia','Brazil','Indonesia'].includes(n)?n:'';}"),
            color=TEXT, position="top", distance=7, font_size=12, font_weight="bold",
            background_color="rgba(8,17,31,.86)", border_color="rgba(234,242,255,.34)",
            border_width=1, border_radius=5, padding=[4, 7], text_border_color="transparent",
        ),
        itemstyle_opts=opts.ItemStyleOpts(opacity=.68, border_color="rgba(255,255,255,.55)", border_width=1),
    )
    # Keep the ECharts multi-dimensional [x, y, population, ...] layout intact;
    # Scatter.add_yaxis otherwise tries to pair it with categorical x-axis data.
    chart.options["series"][0]["data"] = points
    chart.options["series"][0]["labelLayout"] = {"hideOverlap": True, "moveOverlap": "shiftY"}
    chart.options["series"][0]["emphasis"] = {
        "focus": "series",
        "scale": True,
        "label": {
            "show": True,
            "formatter": JsCode("function(p){return p.value[5];}"),
            "position": "top",
            "distance": 8,
            "color": TEXT,
            "fontSize": 12,
            "fontWeight": "bold",
            "backgroundColor": "rgba(8,17,31,.94)",
            "borderColor": "rgba(246,200,95,.72)",
            "borderWidth": 1,
            "borderRadius": 5,
            "padding": [4, 7],
        },
    }
    chart.set_global_opts(
        title_opts=common_title("2022 年 GDP 与 CO₂ 排放关系（人口 ≥ 500 万）"), toolbox_opts=common_toolbox(),
        legend_opts=opts.LegendOpts(is_show=False),
        xaxis_opts=opts.AxisOpts(type_="log", name="GDP（美元，对数轴）", name_textstyle_opts=opts.TextStyleOpts(color=MUTED), axislabel_opts=opts.LabelOpts(color=MUTED, formatter=JsCode("function(v){return '$'+(v>=1e12?(v/1e12).toFixed(1)+'T':(v/1e9).toFixed(0)+'B');}")), splitline_opts=opts.SplitLineOpts(is_show=True, linestyle_opts=opts.LineStyleOpts(color=GRID))),
        yaxis_opts=opts.AxisOpts(type_="log", name="CO₂（Mt，对数轴）", name_textstyle_opts=opts.TextStyleOpts(color=MUTED), axislabel_opts=opts.LabelOpts(color=MUTED), splitline_opts=opts.SplitLineOpts(is_show=True, linestyle_opts=opts.LineStyleOpts(color=GRID))),
        tooltip_opts=opts.TooltipOpts(formatter=JsCode("""
          function(p){var v=p.value; function n(x,d){return Number(x).toLocaleString(undefined,{maximumFractionDigits:d});}
          var pop=v[2]>=1e9?(v[2]/1e9).toFixed(2)+' billion':(v[2]/1e6).toFixed(2)+' million';
          var g=v[0]>=1e12?'$'+(v[0]/1e12).toFixed(2)+' trillion':'$'+(v[0]/1e9).toFixed(2)+' billion';
          return '<b>'+v[5]+'</b><br/>GDP：'+g+'<br/>CO₂：'+n(v[1],2)+' MtCO₂<br/>人均：'+n(v[3],2)+' tCO₂/person<br/>单位 GDP：'+n(v[4],3)+' kgCO₂/$<br/>人口：'+pop;}
        """)),
        visualmap_opts=opts.VisualMapOpts(min_=vmin, max_=max(vmax, vmin + .001), dimension=4, is_calculable=True, orient="horizontal", pos_left="center", pos_bottom="1%", range_color=["#2ec4b6", "#f6c85f", "#ef4b5f"], range_text=["高碳强度", "低碳强度"], textstyle_opts=opts.TextStyleOpts(color=MUTED)),
        datazoom_opts=[
            opts.DataZoomOpts(type_="inside", xaxis_index=0, range_start=0, range_end=100),
            opts.DataZoomOpts(type_="inside", yaxis_index=0, range_start=0, range_end=100),
        ],
    )
    chart.options["grid"] = {"left": "8%", "right": "5%", "top": "15%", "bottom": "17%", "containLabel": True}
    return chart_html(chart)


def create_per_capita_bar(df_2024: pd.DataFrame) -> str:
    fields = ["country", "population", "co2", "co2_per_capita"]
    error = require_columns(df_2024, fields)
    if error:
        return missing_html(error)
    subset = df_2024.dropna(subset=fields).query("population >= @POPULATION_THRESHOLD").nlargest(20, "co2_per_capita").sort_values("co2_per_capita")
    if subset.empty:
        return missing_html("2024 年满足人口阈值的人均排放数据为空。")
    meta = {r.country: [safe_float(r.co2_per_capita), safe_float(r.population), safe_float(r.co2)] for r in subset.itertuples()}
    chart = Bar(init_opts=opts.InitOpts(width="100%", height="510px", theme="dark", bg_color="transparent"))
    chart.add_xaxis(subset["country"].tolist()).add_yaxis("人均 CO₂", _bar_items(subset["co2_per_capita"].tolist(), ["#67a9cf", "#f6c85f", "#ef8a62", "#b2182b"]), label_opts=opts.LabelOpts(is_show=False)).reversal_axis()
    chart.set_global_opts(
        title_opts=common_title("2024 年人均 CO₂ 前 20（人口 ≥ 500 万）"), toolbox_opts=common_toolbox(), legend_opts=opts.LegendOpts(is_show=False),
        xaxis_opts=base_axis_opts("tCO₂/person"), yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(color=MUTED, font_size=11), axisline_opts=opts.AxisLineOpts(is_show=False)),
        tooltip_opts=opts.TooltipOpts(formatter=JsCode(f"""
          function(p){{var m={js_object_literal(meta)}; var v=m[p.name]; var pop=v[1]>=1e9?(v[1]/1e9).toFixed(2)+' billion':(v[1]/1e6).toFixed(2)+' million';
          return '<b>'+p.name+'</b><br/>人均排放：'+v[0].toFixed(2)+' tCO₂/person<br/>人口：'+pop+'<br/>总排放：'+v[2].toLocaleString(undefined,{{maximumFractionDigits:2}})+' MtCO₂';}}
        """)),
    )
    chart.options["grid"] = {"left": "37%", "right": "7%", "top": "10%", "bottom": "8%"}
    return chart_html(chart)


def create_heatmap(real_countries: pd.DataFrame, df_2024: pd.DataFrame) -> str:
    error = require_columns(real_countries, ["country", "year", "co2"])
    if error or df_2024.empty:
        return missing_html(error or "无法确定 2024 年前 15 排放国家。")
    countries = df_2024.nlargest(15, "co2")["country"].tolist()[::-1]
    years = list(range(1990, 2025))
    subset = real_countries[real_countries["country"].isin(countries) & real_countries["year"].between(1990, 2024)]
    lookup = {(r.country, int(r.year)): safe_float(r.co2) for r in subset.itertuples()}
    values = [[xi, yi, lookup.get((country, year))] for yi, country in enumerate(countries) for xi, year in enumerate(years) if lookup.get((country, year)) is not None]
    if not values:
        return missing_html("1990–2024 年热力图数据为空。")
    vmax = float(subset["co2"].quantile(.98))
    chart = HeatMap(init_opts=opts.InitOpts(width="100%", height="540px", theme="dark", bg_color="transparent"))
    chart.add_xaxis(years).add_yaxis("CO₂", countries, values, label_opts=opts.LabelOpts(is_show=False))
    chart.set_global_opts(
        title_opts=common_title("1990–2024 年主要排放国家路径热力图"), toolbox_opts=common_toolbox(), legend_opts=opts.LegendOpts(is_show=False),
        xaxis_opts=opts.AxisOpts(type_="category", axislabel_opts=opts.LabelOpts(color=MUTED, interval=4, rotate=30), splitarea_opts=opts.SplitAreaOpts(is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=.03))),
        yaxis_opts=opts.AxisOpts(type_="category", axislabel_opts=opts.LabelOpts(color=MUTED), splitarea_opts=opts.SplitAreaOpts(is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=.03))),
        tooltip_opts=opts.TooltipOpts(formatter=JsCode("function(p){return '<b>'+p.name+'</b> · '+p.value[0]+'<br/>CO₂：'+Number(p.value[2]).toLocaleString(undefined,{maximumFractionDigits:2})+' MtCO₂';}")),
        visualmap_opts=opts.VisualMapOpts(min_=0, max_=max(vmax, 1), is_calculable=True, orient="horizontal", pos_left="center", pos_bottom="0", range_color=["#102a43", "#2166ac", "#f6c85f", "#ef8a62", "#b2182b"], textstyle_opts=opts.TextStyleOpts(color=MUTED)),
        datazoom_opts=[opts.DataZoomOpts(type_="inside", xaxis_index=0), opts.DataZoomOpts(pos_bottom="5%", xaxis_index=0)],
    )
    return chart_html(chart)


def create_sankey(df_2024: pd.DataFrame) -> str:
    fields = ["country", "co2", "coal_co2", "oil_co2", "gas_co2", "cement_co2", "flaring_co2"]
    error = require_columns(df_2024, fields)
    if error:
        return missing_html(error)
    top = df_2024.nlargest(10, "co2")
    sources = [("Coal", "coal_co2", "#423b39"), ("Oil", "oil_co2", "#d17a45"), ("Gas", "gas_co2", "#4c91cf"), ("Cement", "cement_co2", "#b7c1ca"), ("Flaring", "flaring_co2", "#cf3f4f")]
    nodes = [{"name": label, "itemStyle": {"color": color}} for label, _, color in sources]
    country_colors = ["#b2182b", "#d6604d", "#ef8a62", "#f6c85f", "#67a9cf", "#4393c3", "#2b7a78", "#6c7a89", "#8d6e63", "#7e57c2"]
    nodes += [{"name": r.country, "itemStyle": {"color": country_colors[i]}} for i, r in enumerate(top.itertuples())]
    links = []
    for row in top.itertuples():
        for label, field, _ in sources:
            value = safe_float(getattr(row, field))
            if value is not None and value > 0:
                links.append({"source": label, "target": row.country, "value": round(value, 4)})
    if not links:
        return missing_html("2024 年前 10 国家来源拆分没有可用的正值。")
    chart = Sankey(init_opts=opts.InitOpts(width="100%", height="480px", theme="dark", bg_color="transparent"))
    chart.add(
        "来源 → 国家", nodes=nodes, links=links, pos_left="2%", pos_right="12%", node_width=18, node_gap=10,
        linestyle_opt=opts.LineStyleOpts(opacity=.38, curve=.5, color="source"),
        label_opts=opts.LabelOpts(color=TEXT, font_size=10),
        tooltip_opts=opts.TooltipOpts(formatter=JsCode("function(p){if(p.dataType==='edge'){return '<b>'+p.data.source+' → '+p.data.target+'</b><br/>'+Number(p.data.value).toLocaleString(undefined,{maximumFractionDigits:2})+' MtCO₂';} return p.name;}")),
    )
    chart.set_global_opts(title_opts=common_title("2024 年主要排放国家：来源 → 国家"), toolbox_opts=common_toolbox(), legend_opts=opts.LegendOpts(is_show=False))
    return chart_html(chart)


def create_trade_bar(df_trade_2023: pd.DataFrame) -> str:
    error = require_columns(df_trade_2023, ["country", "trade_co2"])
    if error or df_trade_2023.empty:
        return missing_html(error or "2023 年 trade_co2 数据为空。")
    clean = df_trade_2023.dropna(subset=["trade_co2"]).sort_values("trade_co2")
    selected = pd.concat([clean.head(15), clean.tail(15)]).drop_duplicates("country").sort_values("trade_co2")
    items = [opts.BarItem(name="", value=float(v), itemstyle_opts=opts.ItemStyleOpts(color="#2ec4b6" if v >= 0 else "#ef4b5f", opacity=.88, border_radius=[4, 4, 4, 4])) for v in selected["trade_co2"]]
    chart = Bar(init_opts=opts.InitOpts(width="100%", height="650px", theme="dark", bg_color="transparent"))
    chart.add_xaxis(selected["country"].tolist()).add_yaxis("trade_co2", items, label_opts=opts.LabelOpts(is_show=False)).reversal_axis()
    chart.set_global_opts(
        title_opts=common_title("2023 年隐含碳贸易差异（净进口 / 净出口）"), toolbox_opts=common_toolbox(), legend_opts=opts.LegendOpts(is_show=False),
        xaxis_opts=base_axis_opts("MtCO₂（正值净进口，负值净出口）"), yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(color=MUTED, font_size=10), axisline_opts=opts.AxisLineOpts(is_show=False)),
        tooltip_opts=opts.TooltipOpts(formatter=JsCode("function(p){var v=Number(p.value); var meaning=v>=0?'消费端排放高于生产端：净进口隐含碳':'生产端排放高于消费端：净出口隐含碳'; return '<b>'+p.name+'</b><br/>'+v.toLocaleString(undefined,{maximumFractionDigits:2})+' MtCO₂<br/><span style=\"color:#9fb3c8\">'+meaning+'</span>'; }")),
    )
    chart.options["grid"] = {"left": "22%", "right": "10%", "top": "8%", "bottom": "8%"}
    return chart_html(chart)


def create_cumulative_rose(df_2024: pd.DataFrame) -> str:
    fields = ["country", "cumulative_co2", "share_global_cumulative_co2"]
    error = require_columns(df_2024, fields)
    if error:
        return missing_html(error)
    clean = df_2024.dropna(subset=["cumulative_co2"]).sort_values("cumulative_co2", ascending=False)
    if clean.empty:
        return missing_html("2024 年累计 CO₂ 数据为空。")
    top = clean.head(12)
    others = max(float(clean.iloc[12:]["cumulative_co2"].sum()), 0)
    data = [(r.country, round(float(r.cumulative_co2), 3)) for r in top.itertuples()]
    if others > 0:
        data.append(("Others", round(others, 3)))
    meta = {r.country: safe_float(r.share_global_cumulative_co2) for r in top.itertuples()}
    meta["Others"] = sum(v for v in [safe_float(x) for x in clean.iloc[12:]["share_global_cumulative_co2"]] if v is not None)
    chart = Pie(init_opts=opts.InitOpts(width="100%", height="650px", theme="dark", bg_color="transparent"))
    chart.add(
        "累计 CO₂", data, radius=["15%", "62%"], center=["50%", "43%"], rosetype="radius",
        label_opts=opts.LabelOpts(
            color=TEXT, position="outside", font_size=10,
            formatter=JsCode("function(p){return p.dataIndex<6?p.name:'';}"),
        ),
    )
    chart.set_colors(["#7f1d1d", "#9f3a32", "#b95d47", "#cf7a55", "#d99b68", "#e5bd82", "#a97155", "#8d6e63", "#6d7f91", "#58758a", "#496b7e", "#3c5e70", "#536271"])
    chart.set_global_opts(
        title_opts=common_title("截至 2024 年累计 CO₂：历史责任结构"), toolbox_opts=common_toolbox(),
        legend_opts=opts.LegendOpts(type_="scroll", orient="horizontal", pos_left="5%", pos_right="5%", pos_bottom="1%", textstyle_opts=opts.TextStyleOpts(color=MUTED, font_size=10)),
        tooltip_opts=opts.TooltipOpts(formatter=JsCode(f"""function(p){{var m={js_object_literal(meta)}; var s=m[p.name]; return '<b>'+p.name+'</b><br/>累计排放：'+Number(p.value).toLocaleString(undefined,{{maximumFractionDigits:2}})+' MtCO₂<br/>全球累计占比：'+(s==null?'数据缺失':Number(s).toFixed(2)+'%');}}""")),
    )
    return chart_html(chart)


def create_temperature_wordcloud(df_2024: pd.DataFrame) -> str:
    fields = ["country", "temperature_change_from_ghg", "share_of_temperature_change_from_ghg"]
    error = require_columns(df_2024, fields)
    if error:
        return missing_html(error)
    subset = df_2024.dropna(subset=fields).nlargest(40, "temperature_change_from_ghg")
    if len(subset) < 5:
        return missing_html(f"2024 年温度变化贡献有效国家仅 {len(subset)} 个，不足以形成可靠排名。")
    meta = {r.country: [safe_float(r.temperature_change_from_ghg), safe_float(r.share_of_temperature_change_from_ghg)] for r in subset.itertuples()}
    data = [(str(r.country), float(r.temperature_change_from_ghg)) for r in subset.itertuples()]
    chart = WordCloud(init_opts=opts.InitOpts(width="100%", height="540px", theme="dark", bg_color="transparent"))
    chart.add(
        "温度贡献", data_pair=data, word_size_range=[14, 78], shape="circle", rotate_step=90,
        textstyle_opts=opts.TextStyleOpts(font_family="Microsoft YaHei"),
    )
    chart.set_global_opts(
        title_opts=common_title("2024 年温室气体温度变化贡献词云"), toolbox_opts=common_toolbox(), legend_opts=opts.LegendOpts(is_show=False),
        tooltip_opts=opts.TooltipOpts(formatter=JsCode(f"""function(p){{var m={js_object_literal(meta)}; var v=m[p.name]; return '<b>'+p.name+'</b><br/>温度变化贡献：'+v[0].toFixed(4)+' °C<br/>全球占比：'+v[1].toFixed(2)+'%';}}""")),
    )
    return chart_html(chart)


def build_kpis(prepared: PreparedData) -> List[Dict[str, str]]:
    world_2024 = prepared.world_df[prepared.world_df["year"] == 2024]
    world_co2 = world_2024["co2"].iloc[0] if not world_2024.empty and "co2" in world_2024 else None
    world_pc = world_2024["co2_per_capita"].iloc[0] if not world_2024.empty and "co2_per_capita" in world_2024 else None
    top_name = "数据缺失"
    top_value = ""
    if not prepared.df_2024.empty:
        top = prepared.df_2024.loc[prepared.df_2024["co2"].idxmax()]
        top_name, top_value = str(top["country"]), fmt_million_tonnes(top["co2"])
    return [
        {"label": "2024 全球 CO₂ 排放", "value": fmt_million_tonnes(world_co2), "detail": "World 口径 · 当年总量", "tone": "red"},
        {"label": "2024 国家/地区记录", "value": f"{len(prepared.df_2024):,}", "detail": "排除 OWID_ 汇总实体", "tone": "blue"},
        {"label": "2024 排放最高国家", "value": top_name, "detail": top_value, "tone": "gold"},
        {"label": "2024 全球人均 CO₂", "value": f"{fmt_num(world_pc)} tCO₂/person" if safe_float(world_pc) is not None else "数据缺失", "detail": "World 口径 · 人均值", "tone": "green"},
    ]


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <title>全球碳排放与气候责任交互式数据看板（1750–2024）</title>
  <script>(function(){try{document.documentElement.dataset.theme=localStorage.getItem('co2-dashboard-theme')==='light'?'light':'dark'}catch(e){document.documentElement.dataset.theme='dark'}})();</script>
  <script src="https://assets.pyecharts.org/assets/v5/echarts.min.js"></script>
  <script src="https://assets.pyecharts.org/assets/v5/maps/world.js"></script>
  <script src="https://assets.pyecharts.org/assets/v5/echarts-wordcloud.min.js"></script>
  <style>
    :root{--bg:#08111f;--card:rgba(13,27,42,.92);--text:#eaf2ff;--muted:#9fb3c8;--line:rgba(103,169,207,.2)}
    :root[data-theme="light"]{--bg:#edf3f7;--card:rgba(255,255,255,.94);--text:#172b3d;--muted:#526a7e;--line:rgba(33,102,172,.2)}
    *{box-sizing:border-box} html{scroll-behavior:smooth} body{margin:0;background:radial-gradient(circle at 15% 0%,#102c49 0,transparent 32%),radial-gradient(circle at 88% 18%,rgba(178,24,43,.12) 0,transparent 26%),var(--bg);color:var(--text);font-family:Inter,"PingFang SC","Microsoft YaHei",sans-serif;min-height:100vh;transition:background .28s ease,color .22s ease}
    body:before{content:"";position:fixed;inset:0;pointer-events:none;background-image:linear-gradient(rgba(255,255,255,.018) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.018) 1px,transparent 1px);background-size:32px 32px;mask-image:linear-gradient(to bottom,black,transparent 70%)}
    .dashboard{max-width:1680px;margin:0 auto;padding:24px;position:relative}.hero{padding:34px 38px;margin-bottom:18px;border:1px solid var(--line);border-radius:18px;background:linear-gradient(125deg,rgba(18,45,72,.94),rgba(13,27,42,.9) 55%,rgba(74,22,32,.7));box-shadow:0 20px 60px rgba(0,0,0,.25);overflow:hidden;position:relative;transition:background .28s ease,border-color .22s ease,box-shadow .22s ease}.hero:after{content:"1750 — 2024";position:absolute;right:32px;top:62px;color:rgba(234,242,255,.06);font-size:64px;font-weight:800;letter-spacing:4px}.eyebrow{color:#67a9cf;text-transform:uppercase;letter-spacing:.18em;font-size:12px;font-weight:700}.hero h1{font-size:clamp(28px,4vw,48px);margin:10px 0 12px;max-width:1050px;line-height:1.15}.hero p{color:var(--muted);line-height:1.75;max-width:1080px;margin:0}.pill-row{display:flex;gap:8px;flex-wrap:wrap;margin-top:18px}.pill{font-size:12px;color:#bcd0e4;border:1px solid rgba(103,169,207,.3);background:rgba(8,17,31,.4);padding:7px 11px;border-radius:999px}.theme-toggle{position:absolute;z-index:3;right:24px;top:22px;display:inline-flex;align-items:center;gap:7px;border:1px solid rgba(103,169,207,.42);border-radius:999px;background:rgba(8,17,31,.66);color:#eaf2ff;padding:8px 12px;font:600 12px/1 Inter,"Microsoft YaHei",sans-serif;cursor:pointer;backdrop-filter:blur(8px);transition:transform .18s ease,background .2s ease,border-color .2s ease}.theme-toggle:hover{transform:translateY(-1px);background:rgba(22,64,95,.88)}.theme-toggle:focus-visible{outline:2px solid #f6c85f;outline-offset:3px}.theme-icon{font-size:15px;line-height:1}
    .kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:18px;margin-bottom:18px}.kpi{position:relative;overflow:hidden;background:var(--card);border:1px solid var(--line);border-radius:16px;padding:20px 22px;box-shadow:0 12px 35px rgba(0,0,0,.18)}.kpi:before{content:"";position:absolute;left:0;top:0;bottom:0;width:4px;background:var(--accent)}.kpi.red{--accent:#ef4b5f}.kpi.blue{--accent:#59a5ff}.kpi.gold{--accent:#f6c85f}.kpi.green{--accent:#2ec4b6}.kpi-label{font-size:13px;color:var(--muted)}.kpi-value{font-size:clamp(20px,2.2vw,30px);font-weight:750;margin:8px 0 5px;white-space:nowrap}.kpi-detail{font-size:12px;color:#71879d}
    .grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}.card{background:var(--card);border:1px solid var(--line);border-radius:16px;box-shadow:0 14px 38px rgba(0,0,0,.2);overflow:hidden;min-width:0}.card-wide{grid-column:span 3}.card-2{grid-column:span 2}.card-1{grid-column:span 1}.card-head{padding:19px 22px 0}.card-head h2{font-size:18px;margin:0 0 7px}.card-head p{font-size:13px;line-height:1.6;color:var(--muted);margin:0}.year-control{display:grid;grid-template-columns:34px 1fr 58px;gap:10px;align-items:center;margin:13px 22px 0;padding:9px 12px;border:1px solid rgba(103,169,207,.25);border-radius:10px;background:rgba(8,17,31,.46)}.year-control button{width:30px;height:28px;border:1px solid rgba(103,169,207,.35);border-radius:7px;background:#102a43;color:#eaf2ff;cursor:pointer}.year-control button:hover{background:#16405f}.year-control input{width:100%;accent-color:#f6c85f;cursor:pointer}.year-control output{font-weight:750;color:#f6c85f;text-align:right;font-variant-numeric:tabular-nums}.chart{padding:4px 8px 8px;min-height:300px}.chart>div{max-width:100%}.chart-note{margin:0 20px 18px;padding:10px 12px;border-left:3px solid #f6c85f;background:rgba(246,200,95,.06);color:#b7c7d8;font-size:12px;line-height:1.65}.missing{height:330px;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;color:var(--muted);padding:30px}.missing-icon{font-size:42px;color:#48657f}.missing strong{color:var(--text);font-size:18px}.missing p{max-width:560px}.notes{grid-column:span 3;padding:25px 28px}.notes h2{margin:0 0 15px;font-size:20px}.notes-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px 30px}.notes p{margin:0;color:var(--muted);font-size:13px;line-height:1.7}.notes b{color:#dce8f4}.footer{color:#61768b;text-align:center;padding:28px 0 6px;font-size:12px}
    :root[data-theme="light"] body{background:radial-gradient(circle at 13% 0%,rgba(103,169,207,.24) 0,transparent 31%),radial-gradient(circle at 88% 16%,rgba(239,138,98,.16) 0,transparent 27%),var(--bg)}
    :root[data-theme="light"] body:before{background-image:linear-gradient(rgba(33,102,172,.045) 1px,transparent 1px),linear-gradient(90deg,rgba(33,102,172,.045) 1px,transparent 1px)}
    :root[data-theme="light"] .hero{background:linear-gradient(125deg,rgba(255,255,255,.96),rgba(236,245,250,.96) 58%,rgba(252,235,229,.88));box-shadow:0 18px 50px rgba(34,67,90,.13)}
    :root[data-theme="light"] .hero:after{color:rgba(23,43,61,.055)}
    :root[data-theme="light"] .eyebrow{color:#2166ac}:root[data-theme="light"] .pill{color:#3d5d73;border-color:rgba(33,102,172,.24);background:rgba(255,255,255,.62)}
    :root[data-theme="light"] .theme-toggle{color:#17324a;background:rgba(255,255,255,.82);border-color:rgba(33,102,172,.3);box-shadow:0 5px 16px rgba(43,80,104,.11)}:root[data-theme="light"] .theme-toggle:hover{background:#fff}
    :root[data-theme="light"] .card,:root[data-theme="light"] .kpi{box-shadow:0 12px 34px rgba(35,69,94,.11)}:root[data-theme="light"] .kpi-detail{color:#6c8191}
    :root[data-theme="light"] .year-control{background:rgba(235,243,248,.9);border-color:rgba(33,102,172,.2)}:root[data-theme="light"] .year-control button{background:#dcebf4;color:#17324a;border-color:rgba(33,102,172,.28)}:root[data-theme="light"] .year-control button:hover{background:#c9dfec}
    :root[data-theme="light"] .chart-note{background:rgba(246,200,95,.16);color:#526477}:root[data-theme="light"] .notes b{color:#244f6d}:root[data-theme="light"] .footer{color:#6b8192}
    @media(max-width:1100px){.grid{grid-template-columns:1fr}.card-wide,.card-2,.card-1,.notes{grid-column:span 1}.kpis{grid-template-columns:repeat(2,1fr)}.hero:after{display:none}}
    @media(max-width:620px){.dashboard{padding:12px}.hero{padding:68px 20px 25px}.theme-toggle{top:18px;right:18px}.kpis{grid-template-columns:1fr}.notes-grid{grid-template-columns:1fr}.card-head{padding:17px 16px 0}.kpi-value{white-space:normal}}
  </style>
</head>
<body><main class="dashboard">
  <section class="hero"><button class="theme-toggle" type="button" aria-label="切换到亮色主题" title="切换亮色/暗色主题"><span class="theme-icon" aria-hidden="true">☀</span><span class="theme-label">亮色</span></button><div class="eyebrow">Global Carbon Emissions and Climate Responsibility Dashboard</div><h1>全球碳排放与气候责任交互式数据看板</h1><p>从工业革命以来的排放轨迹，到当代国家排放、经济规模、贸易转移与历史责任：本看板以 OWID CO₂ 数据构建一套可交互的多视角证据链。地图与国家排名排除 OWID 汇总实体，World 仅用于全球趋势与 KPI。</p><div class="pill-row"><span class="pill">动态年份 1750–2024</span><span class="pill">8 个逐年时间轴</span><span class="pill">GDP 截至 2022</span><span class="pill">隐含碳贸易截至 2023</span><span class="pill">人口阈值：500 万</span></div></section>
  <section class="kpis">{% for k in kpis %}<article class="kpi {{k.tone}}"><div class="kpi-label">{{k.label}}</div><div class="kpi-value">{{k.value}}</div><div class="kpi-detail">{{k.detail}}</div></article>{% endfor %}</section>
  <section class="grid">{% for c in cards %}<article class="card {{c.span}}" data-chart-key="{{c.key}}"><header class="card-head"><h2>{{c.title}}</h2><p>{{c.subtitle}}</p></header>{% if c.timeline %}<div class="year-control" data-key="{{c.key}}"><button type="button" aria-label="播放{{c.title}}年份动画">▶</button><input type="range" min="{{c.timeline[0]}}" max="{{c.timeline[1]}}" value="{{c.timeline[2]}}" step="1" aria-label="{{c.title}}年份"><output>{{c.timeline[2]}}</output></div>{% endif %}<div class="chart">{{c.content}}</div>{% if c.note %}<div class="chart-note">{{c.note}}</div>{% endif %}</article>{% endfor %}
    <section class="card notes"><h2>数据口径说明与使用约束</h2><div class="notes-grid">
      <p><b>预测口径：</b>2025 年后的排放、来源结构、人口、GDP、贸易隐含碳与温度贡献为基于历史序列的简单回归外推，仅用于课程展示和趋势探索，不代表正式气候情景或政策预测。</p>
      <p><b>国家范围：</b>`iso_code` 为空或以 `OWID_` 开头的世界、洲、收入组等汇总实体，不进入地图与国家排名；全球趋势保留 World。</p>
      <p><b>年份选择：</b>单年图表均可逐年拖动或播放；GDP 可用年份截至 2022 年，`trade_co2` 可用年份为 1990–2023 年，温度贡献为 1851–2024 年。</p>
      <p><b>缺失值：</b>能源来源分解存在统计覆盖差异，空值不等于零。面积图为保持序列连续将来源空值绘制为 0；桑基图直接过滤空值与非正值。</p>
      <p><b>人口阈值：</b>人均排名与 GDP–CO₂ 气泡图只纳入人口 ≥ 500 万的经济体，避免微型产油经济体主导尺度。</p>
      <p><b>解释边界：</b>`trade_co2` 正值代表消费端排放高于生产端（净进口隐含碳），负值代表生产端排放高于消费端（净出口隐含碳）。</p>
      <p><b>数据扩展：</b>本数据集不足以单独完成可再生能源转型分析；相关结论需要额外补充 OWID Energy 数据集。</p>
    </div></section>
  </section><div class="footer">Data: Our World in Data CO₂ dataset · Built with Pandas + Pyecharts · 交互提示：悬停查看详情，滚轮或滑块缩放，图例可筛选</div>
</main>
<script>
const ANNUAL={{annual_json}};
const IX={COUNTRY:0,MAP:1,CO2:2,PC:3,POP:4,SHARE:5,COAL:6,OIL:7,GAS:8,CEMENT:9,FLARING:10,CUM:11,CUMSHARE:12,TEMP:13,TEMPSHARE:14,GDP:15,CO2GDP:16,TRADE:17,FORECAST:18};
const WARM=['#5b8db8','#67a9cf','#f6c85f','#ef8a62','#b2182b'];
const ROSE=['#7f1d1d','#9f3a32','#b95d47','#cf7a55','#d99b68','#e5bd82','#a97155','#8d6e63','#6d7f91','#58758a','#496b7e','#3c5e70','#536271'];
const SOURCE=[['Coal',IX.COAL,'#423b39'],['Oil',IX.OIL,'#d17a45'],['Gas',IX.GAS,'#4c91cf'],['Cement',IX.CEMENT,'#b7c1ca'],['Flaring',IX.FLARING,'#cf3f4f']];
const THEME_BUTTON=document.querySelector('.theme-toggle');
function styledList(items,fn){const list=Array.isArray(items)?items:(items?[items]:[]);return list.map((item,index)=>fn(item||{},index))}
function applyChartTheme(instance,theme){
  if(!instance)return;
  const option=instance.getOption(),light=theme==='light';
  const text=light?'#172b3d':'#eaf2ff',muted=light?'#526a7e':'#9fb3c8';
  const axisLine=light?'rgba(54,91,118,.42)':'rgba(159,179,200,.42)';
  const splitLine=light?'rgba(54,91,118,.13)':'rgba(159,179,200,.12)';
  const tooltipBg=light?'rgba(255,255,255,.98)':'rgba(8,17,31,.97)';
  const tooltipBorder=light?'rgba(33,102,172,.28)':'rgba(103,169,207,.32)';
  const axisTheme=axis=>({
    ...axis,
    axisLabel:{...(axis.axisLabel||{}),color:muted},
    nameTextStyle:{...(axis.nameTextStyle||{}),color:muted},
    axisLine:{...(axis.axisLine||{}),lineStyle:{...((axis.axisLine&&axis.axisLine.lineStyle)||{}),color:axisLine}},
    splitLine:{...(axis.splitLine||{}),lineStyle:{...((axis.splitLine&&axis.splitLine.lineStyle)||{}),color:splitLine}}
  });
  const labelTheme=label=>({...label,color:text,textBorderColor:'transparent',textShadowColor:'transparent'});
  const scatterLabelTheme=label=>({...labelTheme(label),backgroundColor:light?'rgba(255,255,255,.94)':'rgba(8,17,31,.88)',borderColor:light?'rgba(33,102,172,.38)':'rgba(234,242,255,.34)',borderWidth:1,borderRadius:5,padding:[4,7],fontWeight:'bold'});
  const stateTheme=state=>({...state,label:labelTheme((state&&state.label)||{})});
  const seriesTheme=series=>{
    if(series.type==='wordCloud')return {};
    const patch={
      id:series.id,name:series.name,type:series.type,
      label:labelTheme(series.label||{}),
      edgeLabel:labelTheme(series.edgeLabel||{}),
      labelLine:{...(series.labelLine||{}),lineStyle:{...((series.labelLine&&series.labelLine.lineStyle)||{}),color:muted}},
      emphasis:stateTheme(series.emphasis||{}),
      blur:stateTheme(series.blur||{}),
      select:stateTheme(series.select||{})
    };
    if(series.type==='sankey'||series.type==='pie'){
      patch.data=(series.data||[]).map(item=>item&&typeof item==='object'&&!Array.isArray(item)?({...item,label:labelTheme(item.label||{})}):item);
    }
    if(series.type==='sankey'){
      patch.levels=(series.levels||[]).map(level=>({...level,label:labelTheme(level.label||{})}));
      patch.links=(series.links||series.edges||[]).map(link=>({...link,label:labelTheme(link.label||{})}));
    }
    if(series.type==='scatter'){
      patch.label=scatterLabelTheme(series.label||{});
      patch.emphasis={...(series.emphasis||{}),label:scatterLabelTheme((series.emphasis&&series.emphasis.label)||{})};
      patch.itemStyle={...(series.itemStyle||{}),borderColor:light?'rgba(23,43,61,.5)':'rgba(255,255,255,.55)'};
    }
    return patch;
  };
  instance.setOption({
    backgroundColor:'transparent',
    textStyle:{...(option.textStyle||{}),color:text},
    title:styledList(option.title,title=>({...title,textStyle:{...(title.textStyle||{}),color:text},subtextStyle:{...(title.subtextStyle||{}),color:muted}})),
    legend:styledList(option.legend,legend=>({...legend,textStyle:{...(legend.textStyle||{}),color:muted}})),
    xAxis:styledList(option.xAxis,axisTheme),
    yAxis:styledList(option.yAxis,axisTheme),
    tooltip:styledList(option.tooltip,tooltip=>({...tooltip,backgroundColor:tooltipBg,borderColor:tooltipBorder,textStyle:{...(tooltip.textStyle||{}),color:text},axisPointer:{...(tooltip.axisPointer||{}),lineStyle:{...((tooltip.axisPointer&&tooltip.axisPointer.lineStyle)||{}),color:axisLine},crossStyle:{...((tooltip.axisPointer&&tooltip.axisPointer.crossStyle)||{}),color:axisLine},label:{...((tooltip.axisPointer&&tooltip.axisPointer.label)||{}),color:light?'#fff':'#08111f',backgroundColor:light?'#35627f':'#f6c85f'}}})),
    toolbox:styledList(option.toolbox,toolbox=>({...toolbox,iconStyle:{...(toolbox.iconStyle||{}),borderColor:muted},emphasis:{...(toolbox.emphasis||{}),iconStyle:{...((toolbox.emphasis&&toolbox.emphasis.iconStyle)||{}),borderColor:light?'#2166ac':'#f6c85f'}}})),
    visualMap:styledList(option.visualMap,visual=>({...visual,textStyle:{...(visual.textStyle||{}),color:muted},borderColor:tooltipBorder})),
    dataZoom:styledList(option.dataZoom,zoom=>({...zoom,textStyle:{...(zoom.textStyle||{}),color:muted},borderColor:tooltipBorder,backgroundColor:light?'rgba(224,235,242,.72)':'rgba(8,17,31,.5)',fillerColor:light?'rgba(33,102,172,.22)':'rgba(103,169,207,.2)',handleStyle:{...(zoom.handleStyle||{}),color:light?'#fff':'#102a43',borderColor:light?'#2166ac':'#67a9cf'},dataBackground:{...(zoom.dataBackground||{}),lineStyle:{...((zoom.dataBackground&&zoom.dataBackground.lineStyle)||{}),color:muted},areaStyle:{...((zoom.dataBackground&&zoom.dataBackground.areaStyle)||{}),color:light?'rgba(82,106,126,.18)':'rgba(159,179,200,.14)'}}})),
    series:styledList(option.series,seriesTheme)
  });
}
function setDashboardTheme(theme,persist=true){
  const normalized=theme==='light'?'light':'dark';
  document.documentElement.dataset.theme=normalized;
  if(persist){try{localStorage.setItem('co2-dashboard-theme',normalized)}catch(e){}}
  if(THEME_BUTTON){
    const target=normalized==='dark'?'亮色':'暗色';
    THEME_BUTTON.querySelector('.theme-icon').textContent=normalized==='dark'?'☀':'☾';
    THEME_BUTTON.querySelector('.theme-label').textContent=target;
    THEME_BUTTON.setAttribute('aria-label','切换到'+target+'主题');
  }
  requestAnimationFrame(()=>document.querySelectorAll('[_echarts_instance_]').forEach(el=>applyChartTheme(echarts.getInstanceByDom(el),normalized)));
}
if(THEME_BUTTON)THEME_BUTTON.addEventListener('click',()=>setDashboardTheme(document.documentElement.dataset.theme==='light'?'dark':'light'));
function rows(y){return ANNUAL[String(y)]||[]}
function chart(key){const el=document.querySelector('[data-chart-key="'+key+'"] .chart>div[id]');if(!el)return null;return window['chart_'+el.id]||echarts.getInstanceByDom(el)}
function finite(v){return v!==null&&v!==undefined&&Number.isFinite(Number(v))}
function num(v,d=2){return finite(v)?Number(v).toLocaleString(undefined,{maximumFractionDigits:d}):'数据缺失'}
function pop(v){if(!finite(v))return '数据缺失';v=Number(v);return v>=1e9?(v/1e9).toFixed(2)+' billion':v>=1e6?(v/1e6).toFixed(2)+' million':v.toLocaleString()}
function dtype(r){return r&&r[IX.FORECAST]?'<br/><span style="color:#f6c85f">数据类型：模型预测</span>':'<br/><span style="color:'+themeMuted()+'">数据类型：历史观测</span>'}
function themeText(){return document.documentElement.dataset.theme==='light'?'#172b3d':'#eaf2ff'}
function themeMuted(){return document.documentElement.dataset.theme==='light'?'#526a7e':'#9fb3c8'}
function q98(a){if(!a.length)return 1;const b=a.slice().sort((x,y)=>x-y);return Math.max(1,b[Math.floor((b.length-1)*.98)])}
function colorAt(i,n,palette=WARM){return palette[Math.min(palette.length-1,Math.floor(i/Math.max(1,n-1)*(palette.length-1)))]}
function updateMap(year){const c=chart('map');if(!c)return;const data=rows(year).filter(r=>finite(r[IX.CO2])).map(r=>({name:r[IX.MAP],value:[r[IX.CO2],r[IX.PC],r[IX.POP]],country:r[IX.COUNTRY],meta:r,itemStyle:{opacity:r[IX.FORECAST]?0.78:1}}));c.setOption({title:{text:year+' 年国家 CO₂ 排放空间分布'+(data.some(d=>d.meta[IX.FORECAST])?'（预测）':'')},visualMap:{max:q98(data.map(d=>d.value[0]))},tooltip:{formatter:p=>{const v=p.data&&p.data.value||[],r=p.data&&p.data.meta;return '<b>'+((p.data&&p.data.country)||p.name)+'</b><br/>CO₂：'+num(v[0])+' MtCO₂<br/>人均：'+num(v[1])+' tCO₂/person<br/>人口：'+pop(v[2])+dtype(r)}},series:[{data}]})}
function updateTop(year){const c=chart('top');if(!c)return;const a=rows(year).filter(r=>finite(r[IX.CO2])).sort((a,b)=>b[IX.CO2]-a[IX.CO2]).slice(0,20).reverse();const data=a.map((r,i)=>({value:r[IX.CO2],meta:r,itemStyle:{color:colorAt(i,a.length),opacity:r[IX.FORECAST]?0.74:.92,borderRadius:[0,5,5,0]}}));c.setOption({title:{text:year+' 年 CO₂ 排放前 20 国家'+(a.some(r=>r[IX.FORECAST])?'（预测）':'')},yAxis:{data:a.map(r=>r[IX.COUNTRY])},tooltip:{formatter:p=>{const r=p.data.meta;return '<b>'+r[IX.COUNTRY]+'</b><br/>CO₂：'+num(r[IX.CO2])+' MtCO₂<br/>全球占比：'+num(r[IX.SHARE])+'%<br/>人均排放：'+num(r[IX.PC])+' tCO₂/person'+dtype(r)}},series:[{data,label:{formatter:p=>p.dataIndex>=Math.max(0,a.length-3)?num(p.value,0):''}}]})}
function updatePerCap(year){const c=chart('percap');if(!c)return;const a=rows(year).filter(r=>finite(r[IX.PC])&&finite(r[IX.POP])&&r[IX.POP]>=5000000).sort((a,b)=>b[IX.PC]-a[IX.PC]).slice(0,20).reverse();const data=a.map((r,i)=>({value:r[IX.PC],meta:r,itemStyle:{color:colorAt(i,a.length),opacity:r[IX.FORECAST]?0.74:.92,borderRadius:[0,5,5,0]}}));c.setOption({title:{text:year+' 年人均 CO₂ 前 20（人口 ≥ 500 万）'+(a.some(r=>r[IX.FORECAST])?'（预测）':'')},yAxis:{data:a.map(r=>r[IX.COUNTRY])},tooltip:{formatter:p=>{const r=p.data.meta;return '<b>'+r[IX.COUNTRY]+'</b><br/>人均排放：'+num(r[IX.PC])+' tCO₂/person<br/>人口：'+pop(r[IX.POP])+'<br/>总排放：'+num(r[IX.CO2])+' MtCO₂'+dtype(r)}},series:[{data}]})}
function updateSankey(year){const c=chart('sankey');if(!c)return;const top=rows(year).filter(r=>finite(r[IX.CO2])).sort((a,b)=>b[IX.CO2]-a[IX.CO2]).slice(0,10);const nodes=SOURCE.map(s=>({name:s[0],itemStyle:{color:s[2]},label:{color:themeText()}})).concat(top.map((r,i)=>({name:r[IX.COUNTRY],forecast:r[IX.FORECAST],itemStyle:{color:colorAt(i,top.length,WARM.slice().reverse()),opacity:r[IX.FORECAST]?0.78:1},label:{color:themeText()}})));const links=[];top.forEach(r=>SOURCE.forEach(s=>{if(finite(r[s[1]])&&r[s[1]]>0)links.push({source:s[0],target:r[IX.COUNTRY],value:r[s[1]],forecast:r[IX.FORECAST]})}));c.setOption({title:{text:year+' 年主要排放国家：来源 → 国家'+(top.some(r=>r[IX.FORECAST])?'（预测）':'')},tooltip:{formatter:p=>p.dataType==='edge'?'<b>'+p.data.source+' → '+p.data.target+'</b><br/>'+num(p.data.value)+' MtCO₂'+(p.data.forecast?'<br/><span style="color:#f6c85f">数据类型：模型预测</span>':'<br/><span style="color:'+themeMuted()+'">数据类型：历史观测</span>'):p.name},series:[{label:{color:themeText(),textBorderColor:'transparent'},edgeLabel:{color:themeText()},lineStyle:{opacity:top.some(r=>r[IX.FORECAST])?0.28:.38,curve:.5,color:'source'},data:nodes,links}]})}
function updateScatter(year){const c=chart('scatter');if(!c)return;const a=rows(year).filter(r=>finite(r[IX.GDP])&&r[IX.GDP]>0&&finite(r[IX.CO2])&&r[IX.CO2]>0&&finite(r[IX.POP])&&r[IX.POP]>=5000000&&finite(r[IX.PC])&&finite(r[IX.CO2GDP]));const data=a.map(r=>[r[IX.GDP],r[IX.CO2],r[IX.POP],r[IX.PC],r[IX.CO2GDP],r[IX.COUNTRY],r[IX.FORECAST]]);const intens=data.map(v=>v[4]).sort((a,b)=>a-b);c.setOption({title:{text:year+' 年 GDP 与 CO₂ 排放关系（人口 ≥ 500 万）'+(a.some(r=>r[IX.FORECAST])?'（预测）':'')},visualMap:{min:intens.length?intens[Math.floor(intens.length*.03)]:0,max:intens.length?intens[Math.floor(intens.length*.97)]:1},tooltip:{formatter:p=>{const v=p.value;const g=v[0]>=1e12?'$'+(v[0]/1e12).toFixed(2)+' trillion':'$'+(v[0]/1e9).toFixed(2)+' billion';return '<b>'+v[5]+'</b><br/>GDP：'+g+'<br/>CO₂：'+num(v[1])+' MtCO₂<br/>人均：'+num(v[3])+' tCO₂/person<br/>单位 GDP：'+num(v[4],3)+' kgCO₂/$<br/>人口：'+pop(v[2])+(v[6]?'<br/><span style="color:#f6c85f">数据类型：模型预测</span>':'<br/><span style="color:'+themeMuted()+'">数据类型：历史观测</span>')}},dataZoom:[{start:0,end:100},{start:0,end:100}],series:[{data,itemStyle:{opacity:a.some(r=>r[IX.FORECAST])?0.58:.68}}]})}
function updateTrade(year){const c=chart('trade');if(!c)return;const all=rows(year).filter(r=>finite(r[IX.TRADE])).sort((a,b)=>a[IX.TRADE]-b[IX.TRADE]);const picked=all.slice(0,15).concat(all.slice(-15)).filter((r,i,a)=>a.findIndex(x=>x[IX.COUNTRY]===r[IX.COUNTRY])===i).sort((a,b)=>a[IX.TRADE]-b[IX.TRADE]);const data=picked.map(r=>({value:r[IX.TRADE],meta:r,itemStyle:{color:r[IX.TRADE]>=0?'#2ec4b6':'#ef4b5f',opacity:r[IX.FORECAST]?0.72:.88,borderRadius:4}}));c.setOption({title:{text:year+' 年隐含碳贸易差异（净进口 / 净出口）'+(picked.some(r=>r[IX.FORECAST])?'（预测）':'')},yAxis:{data:picked.map(r=>r[IX.COUNTRY])},tooltip:{formatter:p=>{const v=p.data.value,r=p.data.meta;return '<b>'+p.name+'</b><br/>'+num(v)+' MtCO₂<br/><span style="color:var(--muted)">'+(v>=0?'消费端排放高于生产端：净进口隐含碳':'生产端排放高于消费端：净出口隐含碳')+'</span>'+dtype(r)}},series:[{data}]})}
function updateRose(year){const c=chart('rose');if(!c)return;const all=rows(year).filter(r=>finite(r[IX.CUM])).sort((a,b)=>b[IX.CUM]-a[IX.CUM]);const top=all.slice(0,12);const data=top.map((r,i)=>({name:r[IX.COUNTRY],value:r[IX.CUM],share:r[IX.CUMSHARE],forecast:r[IX.FORECAST],itemStyle:{color:ROSE[i],opacity:r[IX.FORECAST]?0.76:1},label:{color:themeText()}}));if(all.length>12)data.push({name:'Others',value:all.slice(12).reduce((s,r)=>s+r[IX.CUM],0),share:all.slice(12).reduce((s,r)=>s+(finite(r[IX.CUMSHARE])?r[IX.CUMSHARE]:0),0),forecast:all.slice(12).some(r=>r[IX.FORECAST]),itemStyle:{color:ROSE[12]},label:{color:themeText()}});c.setOption({title:{text:'截至 '+year+' 年累计 CO₂：历史责任结构'+(top.some(r=>r[IX.FORECAST])?'（预测）':'')},legend:{data:data.map(d=>d.name)},tooltip:{formatter:p=>'<b>'+p.name+'</b><br/>累计排放：'+num(p.value)+' MtCO₂<br/>全球累计占比：'+num(p.data.share)+'%'+(p.data.forecast?'<br/><span style="color:#f6c85f">数据类型：模型预测</span>':'<br/><span style="color:'+themeMuted()+'">数据类型：历史观测</span>')},series:[{label:{color:themeText(),textBorderColor:'transparent'},labelLine:{lineStyle:{color:themeMuted()}},data}]})}
function updateTemp(year){const c=chart('temp');if(!c)return;const a=rows(year).filter(r=>finite(r[IX.TEMP])).sort((a,b)=>b[IX.TEMP]-a[IX.TEMP]).slice(0,40);const data=a.map((r,i)=>({name:r[IX.COUNTRY],value:r[IX.TEMP],share:r[IX.TEMPSHARE],forecast:r[IX.FORECAST],textStyle:{color:colorAt(i,a.length,['#7f1d1d','#cf7a55','#f6c85f','#67a9cf','#2ec4b6'])}}));c.setOption({title:{text:year+' 年温室气体温度变化贡献词云'+(a.some(r=>r[IX.FORECAST])?'（预测）':'')},tooltip:{formatter:p=>'<b>'+p.name+'</b><br/>温度变化贡献：'+num(p.value,4)+' °C<br/>全球占比：'+num(p.data.share)+'%'+(p.data.forecast?'<br/><span style="color:#f6c85f">数据类型：模型预测</span>':'<br/><span style="color:'+themeMuted()+'">数据类型：历史观测</span>')},series:[{data}]})}
const UPDATERS={map:updateMap,top:updateTop,percap:updatePerCap,sankey:updateSankey,scatter:updateScatter,trade:updateTrade,rose:updateRose,temp:updateTemp};
const YEAR_CONTROLS=Array.from(document.querySelectorAll('.year-control'));
YEAR_CONTROLS.forEach(ctrl=>{const key=ctrl.dataset.key,input=ctrl.querySelector('input'),out=ctrl.querySelector('output'),button=ctrl.querySelector('button');let timer=null;const apply=()=>{out.value=input.value;out.textContent=input.value;if(UPDATERS[key])UPDATERS[key](Number(input.value))};input.addEventListener('input',apply);button.addEventListener('click',()=>{if(timer){clearInterval(timer);timer=null;button.textContent='▶';return}button.textContent='❚❚';timer=setInterval(()=>{let y=Number(input.value)+1;if(y>Number(input.max))y=Number(input.min);input.value=String(y);apply()},450)});});
window.addEventListener('load',()=>{YEAR_CONTROLS.forEach(ctrl=>ctrl.querySelector('input').dispatchEvent(new Event('input')));setDashboardTheme(document.documentElement.dataset.theme||'dark',false)});
window.addEventListener('resize',()=>{document.querySelectorAll('[_echarts_instance_]').forEach(el=>{const c=echarts.getInstanceByDom(el);if(c)c.resize()})});
</script></body></html>"""


def render_dashboard(prepared: PreparedData, output_path: Union[str, Path]) -> Path:
    forecast_end = prepared.forecast_end_year
    future_countries = prepared.annual_countries[prepared.annual_countries.get("is_forecast", False) == True]
    future_world = prepared.world_annual[prepared.world_annual.get("is_forecast", False) == True]
    trend_frame = pd.concat([prepared.df, future_countries, future_world], ignore_index=True, sort=False)
    cards = [
        Card("map", "全球 CO₂ 排放动态地图", f"拖动年份轴查看 1750–{forecast_end} 年真实国家排放、人口与人均排放；2025 年后为简单回归预测。", "card-2", create_world_map(prepared.df_2024), timeline=(1750, forecast_end, 2024)),
        Card("top", "年度排放总量前 20", f"按所选年份真实国家排放总量排序；2025–{forecast_end} 年为模型预测值。", "card-1", create_top_emitters_bar(prepared.df_2024), timeline=(1750, forecast_end, 2024)),
        Card("trend", "主要经济体长期排放轨迹", f"比较 World 与六个主要经济体自 1850 年以来排放增长、转折与分化；虚线延伸至 {forecast_end} 年。", "card-wide", create_historical_line(trend_frame)),
        Card("source", "全球排放来源结构", f"煤炭、石油、天然气、水泥与火炬燃烧共同塑造全球排放曲线；2025–{forecast_end} 年为简单回归预测。", "card-2", create_source_area(prepared.world_annual), "能源来源分解存在统计覆盖差异，空值不等于零；2025 年后为基于 World 历史序列的简单回归预测。"),
        Card("sankey", "来源流向主要排放国家", f"拖动年份轴查看来源类别、目标国家与排放规模如何变化；2025–{forecast_end} 年为预测。", "card-1", create_sankey(prepared.df_2024), "国家层面的来源拆分可能存在统计覆盖差异；预测连线仅用于趋势探索。", (1750, forecast_end, 2024)),
        Card("scatter", "经济规模、排放与碳强度", f"气泡横轴为 GDP、纵轴为 CO₂，大小表示人口，颜色表示单位 GDP 排放；GDP 最新历史口径到 2022 年，2025–{forecast_end} 年为预测。", "card-2", create_gdp_co2_scatter(prepared.df_2022), timeline=(1820, forecast_end, 2022)),
        Card("percap", "最低人口阈值下的人均排放", f"在人口 ≥ 500 万门槛下比较人均排放；2025–{forecast_end} 年由预测 CO₂ 与预测人口计算。", "card-1", create_per_capita_bar(prepared.df_2024), timeline=(1750, forecast_end, 2024)),
        Card("heat", "主要排放国家的时间路径", "颜色亮度揭示 1990–2024 年不同国家排放扩张、平台期与下降的时间差异。", "card-wide", create_heatmap(prepared.real_countries, prepared.df_2024)),
        Card("trade", "隐含碳贸易的两端", f"拖动 1990–{forecast_end} 年份轴；2024 年后为 trade_co₂ 简单回归预测。", "card-2", create_trade_bar(prepared.df_trade_2023), timeline=(1990, forecast_end, 2023)),
        Card("rose", "累计排放与历史责任", f"拖动年份轴查看累计 CO₂ 前 12 国家与 Others 的历史责任结构；2025–{forecast_end} 年由预测年排放递推。", "card-1", create_cumulative_rose(prepared.df_2024), timeline=(1750, forecast_end, 2024)),
        Card("temp", "温度变化贡献词云", f"词语大小编码各国温室气体温度变化贡献；2025–{forecast_end} 年为简单回归预测。", "card-wide", create_temperature_wordcloud(prepared.df_2024), timeline=(1851, forecast_end, 2024)),
    ]
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    annual_json = json.dumps(build_annual_payload(prepared.annual_countries), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    html = Template(HTML_TEMPLATE).render(kpis=build_kpis(prepared), cards=cards, annual_json=annual_json)
    output.write_text(html, encoding="utf-8")
    return output


def resolve_default_data() -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates = [Path.cwd() / "owid-co2-data.csv", script_dir / "owid-co2-data.csv", script_dir.parent / "owid-co2-data.csv"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成全球碳排放与气候责任交互式数据看板")
    parser.add_argument("--data", type=Path, default=None, help="OWID CSV 路径；默认自动查找 owid-co2-data.csv")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "output" / "co2_dashboard.html", help="输出 HTML 路径")
    parser.add_argument("--forecast-end", type=int, default=DEFAULT_FORECAST_END_YEAR, help="预测结束年份，默认 2035")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = args.data if args.data is not None else resolve_default_data()
    df = load_data(data_path)
    prepared = prepare_data(df, forecast_end_year=args.forecast_end)
    output = render_dashboard(prepared, args.output)
    try:
        display = output.relative_to(Path.cwd())
    except ValueError:
        display = output
    print(f"Dashboard generated: {display.as_posix()}")


if __name__ == "__main__":
    main()
