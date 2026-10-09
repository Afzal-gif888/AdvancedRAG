from typing import Literal, Optional

from pydantic import BaseModel

Theme = Literal["dark", "light", "system"]


class UserSettings(BaseModel):
    theme: Theme = "dark"


class SettingsUpdate(BaseModel):
    model_config = {"extra": "forbid"}

    theme: Optional[Theme] = None
