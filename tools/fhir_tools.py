import base64
import logging

import httpx
from google.adk.tools import ToolContext

logger = logging.getLogger(__name__)

_FHIR_TIMEOUT = 15

_DALY_URL = "http://synthetichealth.github.io/synthea/disability-adjusted-life-years"
_QALY_URL = "http://synthetichealth.github.io/synthea/quality-adjusted-life-years"


def _get_fhir_context(tool_context: ToolContext):
    """Read FHIR credentials from session state; return (fhir_url, fhir_token, patient_id) or an error dict."""
    fhir_url   = tool_context.state.get("fhir_url",   "").rstrip("/")
    fhir_token = tool_context.state.get("fhir_token", "")
    patient_id = tool_context.state.get("patient_id", "")

    missing = [
        name for name, val in [
            ("fhir_url",   fhir_url),
            ("patient_id", patient_id),
        ]
        if not val
    ]
    if missing:
        return {
            "status": "error",
            "error_message": (
                f"FHIR context not available — missing: {', '.join(missing)}. "
                "Ensure the caller includes fhir-context in the A2A message metadata."
            ),
        }
    return fhir_url, fhir_token, patient_id


def _fhir_get(fhir_url: str, token: str, path: str, params: dict | None = None) -> dict:
    """Authenticated FHIR GET — returns parsed JSON."""
    headers = {"Accept": "application/fhir+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = httpx.get(
        f"{fhir_url}/{path}",
        params=params,
        headers=headers,
        timeout=_FHIR_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def _http_error_result(exc: httpx.HTTPStatusError) -> dict:
    return {
        "status":        "error",
        "http_status":   exc.response.status_code,
        "error_message": f"FHIR server returned HTTP {exc.response.status_code}",
    }


def _connection_error_result(exc: Exception) -> dict:
    return {
        "status":        "error",
        "error_message": f"Could not reach FHIR server: {exc}",
    }


def _coding_display(codings: list) -> str:
    """Return the first human-readable display text from a FHIR coding list."""
    for c in codings:
        if c.get("display"):
            return c["display"]
    return "Unknown"


def _extract_observation_values(resource: dict) -> list[dict]:
    """Normalise component[], valueQuantity, and valueCodeableConcept into a flat list."""
    if resource.get("component"):
        return [
            {
                "name":  comp.get("code", {}).get("coding", [{}])[0].get("display", "Unknown"),
                "value": comp.get("valueQuantity", {}).get("value"),
                "unit":  comp.get("valueQuantity", {}).get("unit"),
            }
            for comp in resource["component"]
        ]

    name = resource.get("code", {}).get("text") or _coding_display(
        resource.get("code", {}).get("coding", [])
    )

    if resource.get("valueQuantity"):
        return [{
            "name":  name,
            "value": resource["valueQuantity"].get("value"),
            "unit":  resource["valueQuantity"].get("unit"),
        }]

    if resource.get("valueCodeableConcept"):
        return [{
            "name":  name,
            "value": resource["valueCodeableConcept"].get("text"),
            "unit":  None,
        }]

    return [{"name": name, "value": None, "unit": None}]


def get_patient(tool_context: ToolContext) -> dict:
    """Fetch full demographics for the current patient including DALY and QALY scores."""
    ctx = _get_fhir_context(tool_context)
    if isinstance(ctx, dict):
        return ctx
    fhir_url, fhir_token, patient_id = ctx

    logger.info("tool_get_patient patient_id=%s", patient_id)
    try:
        patient = _fhir_get(fhir_url, fhir_token, f"Patient/{patient_id}")
    except httpx.HTTPStatusError as e:
        return _http_error_result(e)
    except Exception as e:
        return _connection_error_result(e)

    name_obj    = patient.get("name", [{}])[0]
    given       = " ".join(name_obj.get("given", []))
    family      = name_obj.get("family", "")
    prefix_list = name_obj.get("prefix", [])
    prefix      = prefix_list[0] if prefix_list else ""
    full_name   = f"{prefix} {given} {family}".strip() if prefix else f"{given} {family}".strip()

    addr    = patient.get("address", [{}])[0]
    address = ", ".join(filter(None, [
        " ".join(addr.get("line", [])),
        addr.get("city", ""),
        addr.get("state", ""),
        addr.get("postalCode", ""),
    ])) or "Not recorded"

    phone          = (patient.get("telecom") or [{}])[0].get("value", "Not recorded")
    marital_status = patient.get("maritalStatus", {}).get("text", "Not recorded")

    extensions = patient.get("extension", [])
    daly = next((e["valueDecimal"] for e in extensions if e.get("url") == _DALY_URL), None)
    qaly = next((e["valueDecimal"] for e in extensions if e.get("url") == _QALY_URL), None)

    return {
        "status":         "success",
        "patient_id":     patient_id,
        "name":           full_name,
        "gender":         patient.get("gender"),
        "birth_date":     patient.get("birthDate"),
        "address":        address,
        "phone":          phone,
        "marital_status": marital_status,
        "daly":           daly,
        "qaly":           qaly,
    }


def get_conditions(tool_context: ToolContext) -> dict:
    """Fetch all conditions for the current patient — clinical diagnoses and SDOH."""
    ctx = _get_fhir_context(tool_context)
    if isinstance(ctx, dict):
        return ctx
    fhir_url, fhir_token, patient_id = ctx

    logger.info("tool_get_conditions patient_id=%s", patient_id)
    try:
        bundle = _fhir_get(fhir_url, fhir_token, "Condition", params={"patient": patient_id})
    except httpx.HTTPStatusError as e:
        return _http_error_result(e)
    except Exception as e:
        return _connection_error_result(e)

    conditions = []
    for entry in bundle.get("entry", []):
        res   = entry.get("resource", {})
        code  = res.get("code", {})
        onset = res.get("onsetDateTime") or (res.get("onsetPeriod") or {}).get("start")
        conditions.append({
            "condition":           code.get("text") or _coding_display(code.get("coding", [])),
            "clinical_status":     (res.get("clinicalStatus")     or {}).get("coding", [{}])[0].get("code"),
            "verification_status": (res.get("verificationStatus") or {}).get("coding", [{}])[0].get("code"),
            "onset":               onset,
            "abatement":           res.get("abatementDateTime"),
            "recorded_date":       res.get("recordedDate"),
        })

    return {
        "status":     "success",
        "patient_id": patient_id,
        "count":      len(conditions),
        "conditions": conditions,
    }


def get_observations(category: str = "vital-signs", tool_context: ToolContext = None) -> dict:
    """Fetch observations for the current patient, newest first.

    Args:
        category: FHIR observation category — 'vital-signs', 'laboratory', or 'social-history'.
    """
    ctx = _get_fhir_context(tool_context)
    if isinstance(ctx, dict):
        return ctx
    fhir_url, fhir_token, patient_id = ctx

    logger.info("tool_get_observations patient_id=%s category=%s", patient_id, category)
    try:
        bundle = _fhir_get(fhir_url, fhir_token, "Observation", params={
            "patient":   patient_id,
            "category":  category,
            "_sort":     "-date",
            "_count":    "50",
        })
    except httpx.HTTPStatusError as e:
        return _http_error_result(e)
    except Exception as e:
        return _connection_error_result(e)

    observations = []
    for entry in bundle.get("entry", []):
        res          = entry.get("resource", {})
        effective    = res.get("effectiveDateTime")
        status       = res.get("status")
        obs_category = (res.get("category") or [{}])[0].get("coding", [{}])[0].get("code")

        for obs in _extract_observation_values(res):
            observations.append({
                "name":           obs["name"],
                "value":          obs["value"],
                "unit":           obs["unit"],
                "effective_date": effective,
                "status":         status,
                "category":       obs_category,
            })

    return {
        "status":       "success",
        "patient_id":   patient_id,
        "category":     category,
        "count":        len(observations),
        "observations": observations,
    }


def get_encounters(tool_context: ToolContext) -> dict:
    """Fetch visit history for the current patient, newest first."""
    ctx = _get_fhir_context(tool_context)
    if isinstance(ctx, dict):
        return ctx
    fhir_url, fhir_token, patient_id = ctx

    logger.info("tool_get_encounters patient_id=%s", patient_id)
    try:
        bundle = _fhir_get(fhir_url, fhir_token, "Encounter", params={
            "patient": patient_id,
            "_sort":   "-date",
        })
    except httpx.HTTPStatusError as e:
        return _http_error_result(e)
    except Exception as e:
        return _connection_error_result(e)

    encounters = []
    for entry in bundle.get("entry", []):
        res         = entry.get("resource", {})
        reason_codes = res.get("reasonCode", [])
        reason      = reason_codes[0].get("coding", [{}])[0].get("display") if reason_codes else None
        encounters.append({
            "id":               res.get("id"),
            "status":           res.get("status"),
            "type":             (res.get("type")            or [{}])[0].get("text"),
            "start":            (res.get("period")          or {}).get("start"),
            "end":              (res.get("period")          or {}).get("end"),
            "clinician":        (res.get("participant")     or [{}])[0].get("individual", {}).get("display"),
            "facility":         (res.get("location")        or [{}])[0].get("location", {}).get("display"),
            "service_provider": (res.get("serviceProvider") or {}).get("display"),
            "reason":           reason,
        })

    return {
        "status":     "success",
        "patient_id": patient_id,
        "count":      len(encounters),
        "encounters": encounters,
    }


def get_medications(tool_context: ToolContext) -> dict:
    """Fetch medication requests for the current patient with drug names resolved."""
    ctx = _get_fhir_context(tool_context)
    if isinstance(ctx, dict):
        return ctx
    fhir_url, fhir_token, patient_id = ctx

    logger.info("tool_get_medications patient_id=%s", patient_id)
    try:
        bundle = _fhir_get(fhir_url, fhir_token, "MedicationRequest", params={"patient": patient_id})
    except httpx.HTTPStatusError as e:
        return _http_error_result(e)
    except Exception as e:
        return _connection_error_result(e)

    entries = bundle.get("entry", [])
    if not entries:
        return {
            "status":      "success",
            "patient_id":  patient_id,
            "count":       0,
            "medications": [],
            "note":        "No medications found for this patient.",
        }

    medications = []
    for entry in entries:
        res = entry.get("resource", {})

        # Drug name lives on the Medication resource, not MedicationRequest.
        med_ref  = res.get("medicationReference", {}).get("reference", "")
        med_id   = med_ref.replace("Medication/", "") if med_ref else ""
        drug_name = "Unknown"
        if med_id:
            try:
                med_resource = _fhir_get(fhir_url, fhir_token, f"Medication/{med_id}")
                drug_name    = med_resource.get("code", {}).get("text", "Unknown")
            except Exception:
                drug_name = "Unknown"

        reason_codes = res.get("reasonCode", [])
        medications.append({
            "drug_name":  drug_name,
            "status":     res.get("status"),
            "intent":     res.get("intent"),
            "authored_on": res.get("authoredOn"),
            "requester":  (res.get("requester") or {}).get("display"),
            "reason":     reason_codes[0].get("text") if reason_codes else None,
        })

    return {
        "status":      "success",
        "patient_id":  patient_id,
        "count":       len(medications),
        "medications": medications,
    }


def get_allergies(tool_context: ToolContext) -> dict:
    """Fetch known allergies for the current patient."""
    ctx = _get_fhir_context(tool_context)
    if isinstance(ctx, dict):
        return ctx
    fhir_url, fhir_token, patient_id = ctx

    logger.info("tool_get_allergies patient_id=%s", patient_id)
    try:
        bundle = _fhir_get(fhir_url, fhir_token, "AllergyIntolerance", params={"patient": patient_id})
    except httpx.HTTPStatusError as e:
        return _http_error_result(e)
    except Exception as e:
        return _connection_error_result(e)

    entries = bundle.get("entry", [])
    if not entries:
        return {
            "status":    "success",
            "patient_id": patient_id,
            "count":     0,
            "allergies": [],
            "note":      "No known allergies found for this patient.",
        }

    allergies = []
    for entry in entries:
        res      = entry.get("resource", {})
        code     = res.get("code", {})
        reaction = (res.get("reaction") or [{}])[0].get("manifestation", [{}])[0].get("text")
        allergies.append({
            "substance":       code.get("text") or _coding_display(code.get("coding", [])),
            "clinical_status": (res.get("clinicalStatus") or {}).get("coding", [{}])[0].get("code"),
            "type":            res.get("type"),
            "criticality":     res.get("criticality"),
            "reaction":        reaction,
        })

    return {
        "status":    "success",
        "patient_id": patient_id,
        "count":     len(allergies),
        "allergies": allergies,
    }


def get_documents(tool_context: ToolContext) -> dict:
    """List all clinical notes for the current patient with metadata. Use document ID with get_document_content to retrieve note text."""
    ctx = _get_fhir_context(tool_context)
    if isinstance(ctx, dict):
        return ctx
    fhir_url, fhir_token, patient_id = ctx

    logger.info("tool_get_documents patient_id=%s", patient_id)
    try:
        bundle = _fhir_get(fhir_url, fhir_token, "DocumentReference", params={"patient": patient_id})
    except httpx.HTTPStatusError as e:
        return _http_error_result(e)
    except Exception as e:
        return _connection_error_result(e)

    documents = []
    for entry in bundle.get("entry", []):
        res          = entry.get("resource", {})
        doc_type     = (res.get("type")     or {}).get("coding", [{}])[0].get("display")
        context      = res.get("context")   or {}
        encounter_ref = context.get("encounter", [{}])[0].get("reference") if context.get("encounter") else None
        period_start = context.get("period", {}).get("start")
        documents.append({
            "id":            res.get("id"),
            "status":        res.get("status"),
            "date":          res.get("date"),
            "type":          doc_type,
            "author":        (res.get("author")    or [{}])[0].get("display"),
            "facility":      (res.get("custodian") or {}).get("display"),
            "encounter_ref": encounter_ref,
            "period_start":  period_start,
        })

    return {
        "status":    "success",
        "patient_id": patient_id,
        "count":     len(documents),
        "documents": documents,
    }


def get_document_content(document_id: str, tool_context: ToolContext) -> dict:
    """Fetch and decode the full text of a clinical note by DocumentReference ID.

    Args:
        document_id: DocumentReference ID obtained from get_documents.
    """
    ctx = _get_fhir_context(tool_context)
    if isinstance(ctx, dict):
        return ctx
    fhir_url, fhir_token, _ = ctx

    logger.info("tool_get_document_content document_id=%s", document_id)
    try:
        resource = _fhir_get(fhir_url, fhir_token, f"DocumentReference/{document_id}")
    except httpx.HTTPStatusError as e:
        return _http_error_result(e)
    except Exception as e:
        return _connection_error_result(e)

    try:
        data    = resource["content"][0]["attachment"]["data"]
        content = base64.b64decode(data).decode("utf-8")
    except (KeyError, IndexError, ValueError) as exc:
        logger.error("document_decode_failed document_id=%s error=%s", document_id, exc)
        return {
            "status":        "error",
            "error_message": "Document content could not be decoded.",
        }

    return {
        "status":      "success",
        "document_id": document_id,
        "content":     content,
    }


def get_procedures(tool_context: ToolContext) -> dict:
    """Fetch procedures performed for the current patient."""
    ctx = _get_fhir_context(tool_context)
    if isinstance(ctx, dict):
        return ctx
    fhir_url, fhir_token, patient_id = ctx

    logger.info("tool_get_procedures patient_id=%s", patient_id)
    try:
        bundle = _fhir_get(fhir_url, fhir_token, "Procedure", params={"patient": patient_id})
    except httpx.HTTPStatusError as e:
        return _http_error_result(e)
    except Exception as e:
        return _connection_error_result(e)

    procedures = []
    for entry in bundle.get("entry", []):
        res          = entry.get("resource", {})
        reason_refs  = res.get("reasonReference", [])
        reason       = reason_refs[0].get("display") if reason_refs else None
        procedures.append({
            "id":            res.get("id"),
            "name":          (res.get("code")            or {}).get("text"),
            "status":        res.get("status"),
            "start":         (res.get("performedPeriod") or {}).get("start"),
            "end":           (res.get("performedPeriod") or {}).get("end"),
            "facility":      (res.get("location")        or {}).get("display"),
            "encounter_ref": (res.get("encounter")       or {}).get("reference"),
            "reason":        reason,
        })

    return {
        "status":     "success",
        "patient_id": patient_id,
        "count":      len(procedures),
        "procedures": procedures,
    }