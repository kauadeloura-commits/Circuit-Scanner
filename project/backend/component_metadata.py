"""Electrical identity and simulation metadata for recognized components.

The visual detector answers *what symbol is this?*.  This module keeps that
answer separate from the part identity written beside the symbol.  It also
provides conservative, common defaults when a semiconductor model was not
written, so later netlist/schematic stages never receive an empty model.
"""

from __future__ import annotations

import copy
import math
import re


MODEL_COMPONENT_CLASSES = {
    "diode",
    "led",
    "transistor_npn",
    "transistor_pnp",
    "mosfet_n",
    "mosfet_p",
    "opamp",
}

TRANSISTOR_CLASSES = {"transistor_npn", "transistor_pnp"}
MOSFET_CLASSES = {"mosfet_n", "mosfet_p"}


# Values are deliberately typical simulation starting points, not guaranteed
# datasheet limits for an unknown physical part.  ``assumed`` is exposed in
# JSON/UI so the user can distinguish them from OCR/manual information.
DEFAULT_COMPONENT_PROFILES = {
    "diode": {
        "model": "1N4148",
        "parameters": {
            "forward_voltage": (0.70, "V"),
            "max_current": (0.20, "A"),
        },
    },
    "led": {
        "model": "LED_RED_GENERIC",
        "parameters": {
            "forward_voltage": (2.0, "V"),
            "max_current": (0.020, "A"),
        },
    },
    "transistor_npn": {
        "model": "BC547",
        "parameters": {
            "hfe": (200.0, None),
            "vce_max": (45.0, "V"),
            "ic_max": (0.10, "A"),
        },
    },
    "transistor_pnp": {
        "model": "BC557",
        "parameters": {
            "hfe": (200.0, None),
            "vce_max": (45.0, "V"),
            "ic_max": (0.10, "A"),
        },
    },
    "mosfet_n": {
        "model": "2N7000",
        "parameters": {
            "vth": (2.1, "V"),
            "rds_on": (5.0, "ohm"),
            "vds_max": (60.0, "V"),
            "id_max": (0.20, "A"),
        },
    },
    "mosfet_p": {
        "model": "BS250",
        "parameters": {
            "vth": (-2.0, "V"),
            "rds_on": (14.0, "ohm"),
            "vds_max": (45.0, "V"),
            "id_max": (0.18, "A"),
        },
    },
    "opamp": {
        "model": "LM358",
        "parameters": {
            "supply_min": (3.0, "V"),
            "supply_max": (32.0, "V"),
            "gain_bandwidth": (1_000_000.0, "Hz"),
        },
    },
}


EDITABLE_PARAMETER_FIELDS = {
    "diode": (
        ("forward_voltage", "Tensão direta", "V"),
        ("max_current", "Corrente máxima", "A"),
    ),
    "led": (
        ("forward_voltage", "Tensão direta", "V"),
        ("max_current", "Corrente máxima", "A"),
    ),
    "transistor_npn": (
        ("hfe", "hFE / β", None),
        ("vce_max", "VCE máximo", "V"),
        ("ic_max", "IC máximo", "A"),
    ),
    "transistor_pnp": (
        ("hfe", "hFE / β", None),
        ("vce_max", "VCE máximo", "V"),
        ("ic_max", "IC máximo", "A"),
    ),
    "mosfet_n": (
        ("vth", "Vth", "V"),
        ("rds_on", "RDS(on)", "ohm"),
        ("vds_max", "VDS máximo", "V"),
        ("id_max", "ID máximo", "A"),
    ),
    "mosfet_p": (
        ("vth", "Vth", "V"),
        ("rds_on", "RDS(on)", "ohm"),
        ("vds_max", "VDS máximo", "V"),
        ("id_max", "ID máximo", "A"),
    ),
    "opamp": (
        ("supply_min", "Alimentação mínima", "V"),
        ("supply_max", "Alimentação máxima", "V"),
        ("gain_bandwidth", "Produto ganho-faixa", "Hz"),
    ),
}


_MODEL_PATTERNS = (
    (re.compile(r"^(?:1N\d{3,5}[A-Z0-9-]*|BAV\d+[A-Z0-9-]*|BAT\d+[A-Z0-9-]*|UF\d+[A-Z0-9-]*|FR\d+[A-Z0-9-]*|BY\d+[A-Z0-9-]*)$"), {"diode"}),
    (re.compile(r"^(?:2N\d{3,5}[A-Z0-9-]*|BC\d{3}[A-Z0-9-]*|BD\d{3}[A-Z0-9-]*|TIP\d{2,3}[A-Z0-9-]*|MPS[A-Z]\d+[A-Z0-9-]*|S8050|S8550|C945|A1015)$"), TRANSISTOR_CLASSES),
    (re.compile(r"^(?:2N7000|BS170|BS250|IRF\d+[A-Z0-9-]*|IRL\d+[A-Z0-9-]*|FQP\d+[A-Z0-9-]*|AO\d+[A-Z0-9-]*|SI\d+[A-Z0-9-]*|STP\d+[A-Z0-9-]*)$"), MOSFET_CLASSES),
    (re.compile(r"^(?:LM\d{3,4}[A-Z0-9-]*|TL\d{2,4}[A-Z0-9-]*|NE\d{3,4}[A-Z0-9-]*|UA\d{3,4}[A-Z0-9-]*|MC\d{3,5}[A-Z0-9-]*|OP\d{2,4}[A-Z0-9-]*|OPA\d{2,4}[A-Z0-9-]*|MCP\d{3,5}[A-Z0-9-]*)$"), {"opamp"}),
    (re.compile(r"^(?:LED(?:RED|GREEN|BLUE|YELLOW|WHITE)[A-Z0-9-]*|[A-Z]{2,5}\d{2,}[A-Z0-9-]*)$"), {"led"}),
)


def _text_key(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


def normalize_model_text(text: str) -> str:
    """Normalize a part number while retaining common dash/package suffixes."""
    cleaned = str(text or "").strip().upper()
    cleaned = cleaned.replace("—", "-").replace("–", "-")
    cleaned = re.sub(r"\s+", "", cleaned)
    cleaned = re.sub(r"[^A-Z0-9._+/-]", "", cleaned)
    # Confusões recorrentes em números manuscritos dentro de part numbers.
    # Só alteramos quando a forma é inequivocamente numérica pelo contexto.
    cleaned = re.sub(r"(?<=\d)[OQ](?=[A-Z0-9])", "0", cleaned)
    cleaned = re.sub(r"(?<=\d)[IL](?=\d)", "1", cleaned)
    original_classes = set().union(*(
        compatible
        for pattern, compatible in _MODEL_PATTERNS
        if pattern.fullmatch(cleaned.replace(".", ""))
    )) if any(
        pattern.fullmatch(cleaned.replace(".", ""))
        for pattern, _ in _MODEL_PATTERNS
    ) else set()
    if original_classes and original_classes != {"led"}:
        return cleaned[:48]
    prefixes = (
        "MCP", "BAV", "BAT", "IRF", "IRL", "FQP", "STP", "OPA",
        "TIP", "MPS", "LED", "BC", "BD", "LM", "TL", "NE", "UA",
        "MC", "OP", "AO", "SI", "UF", "FR", "BY", "1N", "2N",
    )
    prefix = next((item for item in prefixes if cleaned.startswith(item)), "")
    if prefix:
        variants = [""]
        replacements = {"O": "0", "Q": "0", "I": "1", "L": "1", "S": "5", "Z": "2", "B": "8", "G": "6"}
        for character in cleaned[len(prefix):]:
            choices = [character]
            replacement = replacements.get(character)
            if replacement and replacement != character:
                choices.append(replacement)
            variants = [base + choice for base in variants for choice in choices][:128]
        for tail in variants[1:]:
            candidate = prefix + tail
            candidate_classes = set().union(*(
                compatible
                for pattern, compatible in _MODEL_PATTERNS
                if pattern.fullmatch(candidate.replace(".", ""))
            )) if any(
                pattern.fullmatch(candidate.replace(".", ""))
                for pattern, _ in _MODEL_PATTERNS
            ) else set()
            if candidate_classes and candidate_classes != {"led"}:
                return candidate[:48]
    return cleaned[:48]


def model_classes_for_text(text: str) -> set[str]:
    key = normalize_model_text(text).replace(".", "")
    if not key:
        return set()
    if key in {"2N7000", "BS170"}:
        return {"mosfet_n"}
    if key == "BS250":
        return {"mosfet_p"}
    if re.fullmatch(r"(?:BC(?:547|548|549|337)|BD139|TIP(?:31|41)|2N(?:2222|3904)|S8050|C945)[A-Z0-9-]*", key):
        return {"transistor_npn"}
    if re.fullmatch(r"(?:BC(?:557|558|559|327)|BD140|TIP(?:32|42)|2N(?:2907|3906)|S8550|A1015)[A-Z0-9-]*", key):
        return {"transistor_pnp"}
    if re.fullmatch(r"(?:IRF(?:540|Z44)|IRLZ44|FQP30N06)[A-Z0-9-]*", key):
        return {"mosfet_n"}
    if re.fullmatch(r"(?:IRF9540|FQP27P06)[A-Z0-9-]*", key):
        return {"mosfet_p"}
    classes: set[str] = set()
    for pattern, compatible in _MODEL_PATTERNS:
        if pattern.fullmatch(key):
            classes.update(compatible)
    return classes


def parse_hfe_text(text: str) -> float | None:
    """Read gain only when hFE, beta or the Greek beta is explicitly present."""
    raw = str(text or "").strip()
    if not raw:
        return None
    normalized = raw.replace("Β", "β").replace("ß", "β")
    normalized = normalized.replace(",", ".")
    normalized = re.sub(r"\s+", "", normalized)
    match = re.search(
        r"(?i)(?:(?:h[fpt][e3]|beta|β)\s*[:=\-]?\s*|b\s*[:=]\s*)([0-9OQILSZBG]{1,4}(?:\.[0-9OQILSZBG]+)?)",
        normalized,
    )
    if not match:
        return None
    try:
        numeric_text = match.group(1).upper().translate(
            str.maketrans({"O": "0", "Q": "0", "I": "1", "L": "1", "S": "5", "Z": "2", "B": "8", "G": "6"})
        )
        value = float(numeric_text)
    except ValueError:
        return None
    if not math.isfinite(value) or value < 1.0 or value > 2000.0:
        return None
    return value


def is_hfe_prefix(text: str) -> bool:
    key = str(text or "").strip().replace("Β", "β").replace("ß", "β")
    key = re.sub(r"[\s:=\-]", "", key).lower()
    return key in {"hfe", "hf3", "hpe", "beta", "β", "b"}


def _parameter(value, unit: str | None, source: str, confidence=None) -> dict:
    return {
        "value": value,
        "unit": unit,
        "source": source,
        "confidence": confidence,
    }


def generic_metadata(component_type: str) -> dict:
    profile = DEFAULT_COMPONENT_PROFILES.get(str(component_type))
    if not profile:
        return {
            "model": {
                "text": "",
                "normalized": None,
                "source": "not_applicable",
                "confidence": None,
                "assumed": False,
            },
            "parameters": {},
        }
    return {
        "model": {
            "text": profile["model"],
            "normalized": profile["model"],
            "source": "generic_default",
            "confidence": None,
            "assumed": True,
        },
        "parameters": {
            name: _parameter(value, unit, "generic_default")
            for name, (value, unit) in profile.get("parameters", {}).items()
        },
    }


def build_component_metadata(
    component_type: str,
    model_text: str = "",
    model_source: str = "missing",
    model_confidence=None,
    hfe_value: float | None = None,
    hfe_source: str = "ocr",
    hfe_confidence=None,
    existing_parameters: dict | None = None,
    manual_parameters: dict | None = None,
) -> dict:
    """Build canonical metadata and fill only genuinely absent fields."""
    component_type = str(component_type)
    result = generic_metadata(component_type)
    normalized_model = normalize_model_text(model_text)
    if normalized_model and model_source != "generic_default":
        result["model"] = {
            "text": normalized_model,
            "normalized": normalized_model,
            "source": model_source,
            "confidence": model_confidence,
            "assumed": False,
        }

    for name, current in (existing_parameters or {}).items():
        if isinstance(current, dict) and current.get("value") not in (None, ""):
            result["parameters"][str(name)] = copy.deepcopy(current)

    if component_type in TRANSISTOR_CLASSES and hfe_value is not None:
        try:
            parsed_hfe = float(hfe_value)
        except (TypeError, ValueError):
            parsed_hfe = None
        if parsed_hfe is not None and 1.0 <= parsed_hfe <= 2000.0:
            result["parameters"]["hfe"] = _parameter(
                parsed_hfe,
                None,
                hfe_source,
                hfe_confidence,
            )

    field_units = {
        name: unit
        for name, _label, unit in EDITABLE_PARAMETER_FIELDS.get(component_type, ())
    }
    for name, raw in (manual_parameters or {}).items():
        if name not in field_units or raw in (None, ""):
            continue
        try:
            value = float(str(raw).strip().replace(",", "."))
        except ValueError as exc:
            raise ValueError(f"O parâmetro {name} deve ser numérico.") from exc
        if not math.isfinite(value):
            raise ValueError(f"O parâmetro {name} deve ser finito.")
        if name == "hfe" and not 1.0 <= value <= 2000.0:
            raise ValueError("hFE/β deve estar entre 1 e 2000.")
        result["parameters"][name] = _parameter(
            value,
            field_units[name],
            "manual",
            1.0,
        )

    return result


def editable_schema() -> dict:
    return {
        component_type: {
            "accepts_model": component_type in MODEL_COMPONENT_CLASSES,
            "default_model": DEFAULT_COMPONENT_PROFILES.get(component_type, {}).get("model"),
            "parameters": [
                {"name": name, "label": label, "unit": unit}
                for name, label, unit in fields
            ],
        }
        for component_type, fields in EDITABLE_PARAMETER_FIELDS.items()
    }
