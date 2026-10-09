import re

from pydantic import BaseModel, EmailStr, Field, field_validator

from app.schemas.user import UserResponse

PASSWORD_MIN, PASSWORD_MAX = 8, 128


def normalize_email(email: str) -> str:
    return email.strip().lower()


def check_password_strength(password: str) -> str:
    if len(password) < PASSWORD_MIN:
        raise ValueError(f"Password must be at least {PASSWORD_MIN} characters.")
    if len(password) > PASSWORD_MAX:
        raise ValueError(f"Password must be at most {PASSWORD_MAX} characters.")
    if not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        raise ValueError("Password must contain at least one letter and one number.")
    if password.strip() != password:
        raise ValueError("Password must not start or end with whitespace.")
    return password


class _EmailModel(BaseModel):
    email: EmailStr = Field(max_length=320)

    @field_validator("email", mode="after")
    @classmethod
    def _normalize(cls, v: str) -> str:
        return normalize_email(v)


class RegisterRequest(_EmailModel):
    password: str = Field(max_length=PASSWORD_MAX)
    full_name: str = Field(min_length=1, max_length=100)

    @field_validator("password")
    @classmethod
    def _password_strength(cls, v: str) -> str:
        return check_password_strength(v)

    @field_validator("full_name")
    @classmethod
    def _clean_name(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("Please enter your name.")
        return v


class LoginRequest(_EmailModel):
    # No strength check on login: legacy passwords must still be able to sign in.
    password: str = Field(min_length=1, max_length=PASSWORD_MAX)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class ForgotPasswordRequest(_EmailModel):
    pass


class ResetPasswordRequest(BaseModel):
    # Format is checked by the service so a malformed token gets the same INVALID_TOKEN answer.
    token: str = Field(max_length=512)
    new_password: str = Field(max_length=PASSWORD_MAX)

    @field_validator("new_password")
    @classmethod
    def _password_strength(cls, v: str) -> str:
        return check_password_strength(v)


class MessageOut(BaseModel):
    message: str
