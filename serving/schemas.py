from pydantic import BaseModel, Field
from typing import Literal


WIND_DIRECTIONS = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
]

STATIONS = [
    "Aotizhongxin", "Changping", "Dingling", "Dongsi", "Guanyuan",
    "Gucheng", "Huairou", "Nongzhanguan", "Shunyi", "Tiantan",
    "Wanliu", "Wanshouxigong",
]


class MeteoInput(BaseModel):
    year: int = Field(..., example=2016, ge=2013, le=2017, description="Year")
    month: int = Field(..., example=12, ge=1, le=12, description="Month (1-12)")
    day: int = Field(..., example=15, ge=1, le=31, description="Day of month")
    hour: int = Field(..., example=8, ge=0, le=23, description="Hour of day (0-23)")
    TEMP: float = Field(..., example=-3.5, description="Temperature (Celsius)")
    PRES: float = Field(..., example=1024.0, description="Air pressure (hPa)")
    DEWP: float = Field(..., example=-12.0, description="Dew point (Celsius)")
    RAIN: float = Field(..., example=0.0, description="Precipitation (mm)")
    WSPM: float = Field(..., example=1.2, description="Wind speed (m/s)")
    wd: Literal[tuple(WIND_DIRECTIONS)] = Field(..., example="NW", description="Wind direction")
    station: Literal[tuple(STATIONS)] = Field(..., example="Dongsi", description="Monitoring station")

    model_config = {
        "json_schema_extra": {
            "example": {
                "year": 2016, "month": 12, "day": 15, "hour": 8,
                "TEMP": -3.5, "PRES": 1024.0, "DEWP": -12.0,
                "RAIN": 0.0, "WSPM": 1.2, "wd": "NW", "station": "Dongsi",
            }
        }
    }


class PM25Prediction(BaseModel):
    pm25: float = Field(..., description="Predicted PM2.5 concentration (ug/m3)")
    category: str = Field(..., description="WHO/EPA air-quality category")
    status: str = Field(default="success")


def pm25_to_category(pm25: float) -> str:
    if pm25 <= 12:
        return "Good"
    elif pm25 <= 35.4:
        return "Moderate"
    elif pm25 <= 55.4:
        return "Unhealthy for Sensitive Groups"
    elif pm25 <= 150.4:
        return "Unhealthy"
    elif pm25 <= 250.4:
        return "Very Unhealthy"
    return "Hazardous"
