"""The JSON schema the language-model extractor is constrained to.

Built from `records.FIELDS` rather than written out by hand, so that adding a
field in one place cannot leave the model emitting a shape the scorer does not
understand. The vocabulary of each categorical field becomes a JSON `enum`,
which the API enforces — a constrained decode cannot produce "double blinded"
where the scorer expects "double-blind", so a whole class of near-miss
mismatches never reaches scoring and never has to be papered over with fuzzy
string matching.

The schema deliberately requires `quote` on every claim. A model that has no
quotation to give has to either invent one — which `ground` catches — or omit
the claim, which is the honest outcome and is scored as an abstention.
"""

from __future__ import annotations

from ..records import FIELDS, FieldKind

#: JSON Schema features the API supports are a subset of the specification.
#: `enum`, `const`, `additionalProperties: false` and the basic types are in;
#: `minLength`, `maxLength` and numeric bounds are not, which is why the length
#: floor on quotations lives in `contracts.CLAIM_CONTRACT` and in `ground`
#: instead of here.
MAX_CLAIMS = len(FIELDS)


def _field_description(name: str) -> str:
    spec = FIELDS[name]
    text = spec.description
    if spec.kind is FieldKind.CATEGORICAL and spec.vocabulary:
        text += " One of: " + ", ".join(spec.vocabulary) + "."
    if spec.kind is FieldKind.INTEGER:
        text += " Report the bare number, with no units or words."
    if spec.kind is FieldKind.NUMBER:
        text += " Report a number of weeks, converting if the paper states another unit."
    return text


def extraction_schema() -> dict:
    """The `output_config.format` schema for a single document."""
    return {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "description": (
                    "One entry per field the article actually reports. Omit a field "
                    "the article does not report — an omission is scored as an "
                    "honest abstention, an invented value is scored as an error."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "field": {
                            "type": "string",
                            "enum": list(FIELDS),
                            "description": "Which characteristic this claim reports.",
                        },
                        "value": {
                            "type": "string",
                            "description": "The extracted value, as text.",
                        },
                        "quote": {
                            "type": "string",
                            "description": (
                                "A span copied verbatim from the article that states "
                                "this value. It must appear in the article character "
                                "for character; a paraphrase or a reconstruction will "
                                "be rejected and the claim discarded."
                            ),
                        },
                        "section": {
                            "type": "string",
                            "description": (
                                "The section heading the quotation came from, if "
                                "identifiable. Advisory only; never scored."
                            ),
                        },
                    },
                    "required": ["field", "value", "quote", "section"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["claims"],
        "additionalProperties": False,
    }


def field_guidance() -> str:
    """Per-field instructions for the prompt, generated from the same source."""
    return "\n".join(f"- {name}: {_field_description(name)}" for name in FIELDS)
