"""
FreelanceFlow AI
================
A Streamlit web application for freelancers and small-business owners
to organize invoices and expenses using open-weight AI (openai/gpt-oss-20b via Groq).

Built for Hacktoberfest 2026 Week 3 Side Hustle challenge.
"""

import json
import os
import re
from datetime import date
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq


# -----------------------------------------------------------------------------
# 1. Environment & Configuration Setup
# -----------------------------------------------------------------------------
# Load environment variables from a local .env file if available
load_dotenv()

# The designated open-weight model for AI tasks
MODEL_NAME = "openai/gpt-oss-20b"

# Streamlit Page Configuration
st.set_page_config(
    page_title="FreelanceFlow AI",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# 2. AI Helper Function (Groq API Integration)
# -----------------------------------------------------------------------------
def call_groq_model(
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    json_mode: bool = False,
    temperature: float = 0.2,
    max_tokens: int = 2048,
):
    """
    Calls the Groq API using the open-weight model openai/gpt-oss-20b.

    Args:
        api_key (str): The Groq API key.
        system_prompt (str): Instructions for the AI's role and format.
        user_prompt (str): The user's input data or task request.
        json_mode (bool): Whether to enforce structured JSON response format.
        temperature (float): Sampling temperature.
        max_tokens (int): Maximum token allowance for response.

    Returns:
        tuple: (response_text, error_message)
               If successful, error_message is None.
               If failed, response_text is None.
    """
    if not api_key or not api_key.strip():
        return None, "GROQ_API_KEY is not configured. Please add it to your .env file or enter it in the sidebar."

    try:
        # Initialize the official Groq client
        client = Groq(api_key=api_key.strip())

        kwargs = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Apply structured JSON response format if requested
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            completion = client.chat.completions.create(**kwargs)
        except Exception as api_err:
            # Graceful fallback: If structured json mode is rejected by the endpoint, retry without parameter
            if json_mode and "response_format" in str(api_err):
                kwargs.pop("response_format", None)
                completion = client.chat.completions.create(**kwargs)
            else:
                raise api_err

        output_text = completion.choices[0].message.content
        return output_text, None

    except Exception as err:
        # Catch and return actual API errors without pretending or mocking
        return None, f"Groq API Error: {str(err)}"



# -----------------------------------------------------------------------------
# 2.1 Invoice Organizer Data Extraction & Non-Fabrication Logic
# -----------------------------------------------------------------------------
INVOICE_EXTRACTION_SYSTEM_PROMPT = """You are a strict data extraction system for invoices.
Your task is to extract ONLY facts that are explicitly provided in the user's input notes into a valid JSON object.

CRITICAL JSON OUTPUT RULES:
1. You MUST output ONLY a single valid JSON object matching the schema below.
2. Do NOT wrap the JSON in markdown code blocks (do not output ```json or ```).
3. Do NOT include any introductory or concluding conversational text.
4. Ensure all property names and string values use double quotes.
5. Do NOT include trailing commas in arrays or objects.
6. Separate all key-value pairs with commas.

CRITICAL NON-FABRICATION RULES:
1. NEVER invent, assume, or fabricate any information not in the input.
2. If any detail is not explicitly mentioned, you MUST set its value to null.
3. NEVER generate fictional freelancer names, addresses, phone numbers, emails, websites, bank names, account numbers, routing numbers, SWIFT codes, Stripe payment URLs, late fees, or penalty terms.
4. If the input does not describe an invoice or billable project, set "is_valid_invoice": false and provide "error_message".
5. Output ONLY valid JSON matching this schema:

{
  "is_valid_invoice": true,
  "error_message": null,
  "invoice_overview": {
    "invoice_number": null,
    "issue_date": null,
    "due_date": null,
    "payment_terms": null
  },
  "client_info": {
    "name": null,
    "contact_person": null,
    "email": null,
    "project": null,
    "address": null,
    "phone": null
  },
  "freelancer_info": {
    "name": null,
    "email": null,
    "phone": null,
    "address": null,
    "website": null
  },
  "line_items": [
    {
      "description": "Item description",
      "hours_or_qty": 0.0,
      "unit": "hours",
      "rate": 0.0,
      "line_total": 0.0
    }
  ],
  "financial_breakdown": {
    "subtotal": 0.0,
    "discount": 0.0,
    "discount_description": null,
    "tax": 0.0,
    "grand_total": 0.0
  },
  "payment_instructions": {
    "methods_accepted": null,
    "bank_details": null,
    "payment_link": null,
    "late_payment_policy": null,
    "notes": null
  },
  "ai_suggestions": [
    "string suggestion"
  ]
}"""


def safe_float(val):
    """Converts a value to float safely, handling strings with $ or commas."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    try:
        cleaned = str(val).replace("$", "").replace(",", "").strip()
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def format_invoice_field(val, fallback="Not provided"):
    """Returns fallback string if field is None, empty, or placeholder."""
    if val is None:
        return fallback
    s = str(val).strip()
    if not s or s.lower() in ("null", "none", "n/a", "not provided"):
        return fallback
    return s


def clean_and_repair_json(text: str) -> str:
    """
    Cleans and repairs common LLM JSON formatting issues:
    - Markdown code fences (```json ... ```)
    - Preamble or trailing conversational text
    - Trailing commas before closing braces/brackets
    - Missing commas between property definitions
    - Single-line or multi-line comments
    - Unclosed quotes, braces, and brackets from truncated output
    """
    cleaned = text.strip()

    # 1. Strip markdown fences if present
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    elif cleaned.startswith("```json"):
        cleaned = cleaned[7:].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    # 2. Extract outermost JSON structure between first { and last }
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        cleaned = cleaned[first_brace : last_brace + 1]
    elif first_brace != -1 and last_brace == -1:
        cleaned = cleaned[first_brace:]

    # 3. Strip single-line comments (// ...) and multi-line comments (/* ... */)
    cleaned = re.sub(r"//.*?\n", "\n", cleaned)
    cleaned = re.sub(r"/\*[\s\S]*?\*/", "", cleaned)

    # 4. Remove trailing commas before closing braces or brackets (e.g. ", }" or ", ]")
    cleaned = re.sub(r",\s*([\]}])", r"\1", cleaned)

    # 5. Fix missing commas between properties (e.g., '"hours": 30\n"rate": 75')
    cleaned = re.sub(
        r'([0-9"truefalsenull\]\}])\s*\n\s*("[a-zA-Z0-9_]+"\s*:)',
        r'\1,\n\2',
        cleaned,
        flags=re.IGNORECASE,
    )

    # 6. If truncated inside an unescaped string, close the quote
    unescaped_quotes = len(re.findall(r'(?<!\\)"', cleaned))
    if unescaped_quotes % 2 != 0:
        cleaned += '"'

    cleaned = cleaned.rstrip().rstrip(",").rstrip(":")

    # 7. Repair unclosed braces/brackets if response was truncated
    open_brackets = cleaned.count("[") - cleaned.count("]")
    open_braces = cleaned.count("{") - cleaned.count("}")

    if open_brackets > 0 or open_braces > 0:
        cleaned = cleaned.rstrip().rstrip(",")
        if open_brackets > 0:
            cleaned += "]" * open_brackets
        if open_braces > 0:
            cleaned += "}" * open_braces

    # Final pass: remove any trailing comma created before newly added bracket/brace
    cleaned = re.sub(r",\s*([\]}])", r"\1", cleaned)

    return cleaned


def safe_json_loads(text: str):
    """
    Attempts to parse text as JSON. If standard json.loads fails,
    applies cleaning and repair heuristics before retrying.
    Returns (parsed_data, error_string).
    """
    if not text or not text.strip():
        return None, "Empty response received from the model."

    # Fast path: direct parse
    try:
        return json.loads(text.strip()), None
    except Exception:
        pass

    # Repaired path
    repaired = clean_and_repair_json(text)
    try:
        return json.loads(repaired), None
    except Exception as err:
        return None, f"JSON parsing failed: {str(err)}"


def parse_and_validate_invoice(raw_text: str, ai_response_text: str):
    """
    Parses the JSON output from the AI, strictly filters out any hallucinations
    not supported by the user's input, validates mathematical calculations,
    and returns a structured invoice dictionary or an error message.
    """
    data, json_err = safe_json_loads(ai_response_text)
    if json_err:
        return None, f"Could not parse invoice data: {json_err}. The AI response was not in expected JSON format."

    if not isinstance(data, dict):
        return None, "Invalid response format: Expected a JSON object."

    if not data.get("is_valid_invoice", True):
        msg = data.get("error_message") or "The provided text does not contain recognizable invoice or billing details."
        return None, msg


    raw_lower = raw_text.lower()

    def verify_field_supported(val):
        """Discards any value not grounded in user input text."""
        if val is None:
            return None
        val_str = str(val).strip()
        if not val_str or val_str.lower() in ("null", "none", "n/a", "not provided"):
            return None
        words = [w for w in re.findall(r'[a-zA-Z0-9]+', val_str.lower()) if len(w) > 3]
        if words and not any(w in raw_lower for w in words):
            return None
        return val_str

    overview = data.get("invoice_overview") if isinstance(data.get("invoice_overview"), dict) else {}
    client = data.get("client_info") if isinstance(data.get("client_info"), dict) else {}
    freelancer = data.get("freelancer_info") if isinstance(data.get("freelancer_info"), dict) else {}

    payment = data.get("payment_instructions")
    if isinstance(payment, str):
        payment = {"methods_accepted": payment}
    elif not isinstance(payment, dict):
        payment = {}

    fin = data.get("financial_breakdown") if isinstance(data.get("financial_breakdown"), dict) else {}
    line_items = data.get("line_items") if isinstance(data.get("line_items"), list) else []
    ai_suggestions = data.get("ai_suggestions") if isinstance(data.get("ai_suggestions"), list) else []

    # Safeguard against fabricated details
    for k in ["name", "email", "phone", "address", "website"]:
        freelancer[k] = verify_field_supported(freelancer.get(k))

    # Payment link verification: Must be present as a URL in raw text
    if payment.get("payment_link"):
        link_val = str(payment["payment_link"]).strip()
        if not ("http" in raw_lower or "www." in raw_lower or link_val.lower() in raw_lower):
            payment["payment_link"] = None
        else:
            payment["payment_link"] = verify_field_supported(link_val)

    # Bank details verification: Account numbers or bank specifics must be present in raw text
    if payment.get("bank_details"):
        bank_val = str(payment["bank_details"]).strip()
        digits = re.findall(r"\d{3,}", bank_val)
        if digits and not any(d in raw_text for d in digits):
            payment["bank_details"] = None
        else:
            payment["bank_details"] = verify_field_supported(bank_val)

    # Late payment policy verification: Must be explicitly mentioned in raw text
    if payment.get("late_payment_policy"):
        policy_val = str(payment["late_payment_policy"]).strip()
        if not any(term in raw_lower for term in ["late fee", "penalty", "interest", "grace period", "%"]):
            payment["late_payment_policy"] = None
        else:
            payment["late_payment_policy"] = verify_field_supported(policy_val)

    for k in ["address", "phone"]:
        client[k] = verify_field_supported(client.get(k))


    # Calculations validation
    calc_subtotal = 0.0
    valid_items = []
    for item in line_items:
        if not isinstance(item, dict):
            continue
        desc = item.get("description") or "Item"
        qty = safe_float(item.get("hours_or_qty"))
        rate = safe_float(item.get("rate"))
        unit = item.get("unit") or "hours"
        line_total = safe_float(item.get("line_total"))

        computed_line = None
        if qty is not None and rate is not None:
            computed_line = qty * rate
        elif line_total is not None:
            computed_line = line_total

        if computed_line is not None:
            calc_subtotal += computed_line

        valid_items.append({
            "description": desc,
            "hours_or_qty": qty,
            "unit": unit,
            "rate": rate,
            "line_total": computed_line if computed_line is not None else line_total
        })

    stated_subtotal = safe_float(fin.get("subtotal"))
    subtotal_val = stated_subtotal if stated_subtotal is not None else calc_subtotal

    raw_discount = safe_float(fin.get("discount"))
    discount_val = abs(raw_discount) if raw_discount is not None else 0.0
    discount_desc = fin.get("discount_description")

    raw_tax = safe_float(fin.get("tax"))
    tax_val = abs(raw_tax) if raw_tax is not None else 0.0
    stated_grand_total = safe_float(fin.get("grand_total"))

    # Verified calculation: Subtotal - Discount + Tax
    verified_grand_total = subtotal_val - discount_val + tax_val


    math_is_exact = True
    calc_validation_msg = ""
    if stated_grand_total is not None:
        if abs(stated_grand_total - verified_grand_total) > 0.01:
            math_is_exact = False
            calc_validation_msg = (
                f"Calculation discrepancy adjusted: Stated total was ${stated_grand_total:,.2f}, "
                f"but verified math is ${subtotal_val:,.2f} - ${discount_val:,.2f} + ${tax_val:,.2f} = ${verified_grand_total:,.2f}."
            )

    if math_is_exact and (subtotal_val > 0 or discount_val > 0):
        calc_validation_msg = (
            f"Calculations verified: Subtotal (${subtotal_val:,.2f}) "
            f"{f'- Discount (${discount_val:,.2f}) ' if discount_val > 0 else ''}"
            f"{f'+ Tax (${tax_val:,.2f}) ' if tax_val > 0 else ''}"
            f"= Grand Total (${verified_grand_total:,.2f})"
        )

    # Minimum data validation
    if not valid_items and not overview.get("invoice_number") and not client.get("name") and subtotal_val == 0.0:
        return None, "Incomplete invoice data: Unable to identify client, services, or financial totals from the input."

    structured_result = {
        "invoice_overview": overview,
        "client_info": client,
        "freelancer_info": freelancer,
        "line_items": valid_items,
        "financial_breakdown": {
            "subtotal": subtotal_val,
            "discount": discount_val,
            "discount_description": discount_desc,
            "tax": tax_val,
            "grand_total": verified_grand_total,
            "math_is_exact": math_is_exact,
            "calc_validation_msg": calc_validation_msg,
        },
        "payment_instructions": payment,
        "ai_suggestions": ai_suggestions,
    }

    return structured_result, None


def extract_invoice_with_retry(api_key: str, raw_text: str):
    """
    Extracts structured invoice data using Groq with structured JSON mode.
    If parsing fails due to malformed output, retries ONCE with a clear correction prompt.
    Returns:
        tuple: (invoice_data, error_message, was_retried)
    """
    # Attempt 1: Call Groq with JSON mode enabled
    raw_response, call_err = call_groq_model(
        api_key=api_key,
        system_prompt=INVOICE_EXTRACTION_SYSTEM_PROMPT,
        user_prompt=raw_text,
        json_mode=True,
    )
    if call_err:
        return None, call_err, False

    invoice_data, parse_err = parse_and_validate_invoice(raw_text, raw_response)
    if invoice_data is not None:
        return invoice_data, None, False

    # Check whether the error is a JSON format / syntax error
    err_lower = parse_err.lower()
    is_json_syntax_error = (
        "json" in err_lower
        or "parse" in err_lower
        or "delimiter" in err_lower
        or "expecting" in err_lower
        or "unterminated" in err_lower
    )

    if not is_json_syntax_error:
        # It was valid JSON, but the contents were not an invoice or lacked required fields
        return None, parse_err, False

    # Attempt 2: One-shot retry with a focused correction prompt
    correction_prompt = (
        "You are an expert JSON data repair assistant. "
        "The previous invoice extraction produced malformed JSON that failed with this error:\n"
        f"\"{parse_err}\"\n\n"
        "Re-extract the invoice details from the original notes and output ONLY a 100% valid JSON object matching the required schema. "
        "Do not include markdown code fences, comments, or trailing commas. Ensure all strings and keys are enclosed in double quotes."
    )
    retry_user_prompt = (
        f"Original Invoice Notes:\n{raw_text}\n\n"
        f"Malformed Output that failed parsing:\n{raw_response}"
    )

    retry_response, retry_call_err = call_groq_model(
        api_key=api_key,
        system_prompt=correction_prompt,
        user_prompt=retry_user_prompt,
        json_mode=True,
    )
    if retry_call_err:
        return None, f"Automated correction failed: {retry_call_err}", True

    retry_data, retry_parse_err = parse_and_validate_invoice(raw_text, retry_response)
    if retry_data is not None:
        return retry_data, None, True

    # Both attempts failed to produce parseable JSON
    friendly_error = (
        "We could not parse the invoice into a structured format after an automated correction attempt. "
        "Please check that your invoice notes contain readable billing details and click 'Organize Invoice with AI' to try again."
    )
    return None, friendly_error, True


def render_invoice_markdown(data: dict) -> str:
    """Renders structured invoice data into markdown, strictly distinguishing facts from suggestions."""
    overview = data["invoice_overview"]
    client = data["client_info"]
    freelancer = data["freelancer_info"]
    line_items = data["line_items"]
    fin = data["financial_breakdown"]
    payment = data["payment_instructions"]
    suggestions = data["ai_suggestions"]

    md = []
    md.append("### 📋 Organized Invoice Output")
    md.append("")
    md.append("| **Invoice Overview** | |")
    md.append("|:---|:---|")
    md.append(f"| **Invoice #** | {format_invoice_field(overview.get('invoice_number'))} |")
    md.append(f"| **Issue Date** | {format_invoice_field(overview.get('issue_date'))} |")
    md.append(f"| **Due Date** | {format_invoice_field(overview.get('due_date'))} |")
    md.append(f"| **Payment Terms** | {format_invoice_field(overview.get('payment_terms'))} |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("#### 👤 Client & Freelancer Information")
    md.append("")
    md.append("| **Client Details** | |")
    md.append("|:---|:---|")
    md.append(f"| **Client / Company** | {format_invoice_field(client.get('name'))} |")
    md.append(f"| **Contact Person** | {format_invoice_field(client.get('contact_person'))} |")
    md.append(f"| **Email** | {format_invoice_field(client.get('email'))} |")
    md.append(f"| **Project** | {format_invoice_field(client.get('project'))} |")
    md.append(f"| **Address** | {format_invoice_field(client.get('address'))} |")
    md.append(f"| **Phone** | {format_invoice_field(client.get('phone'))} |")
    md.append("")
    md.append("| **Freelancer Details** | |")
    md.append("|:---|:---|")
    md.append(f"| **Freelancer Name** | {format_invoice_field(freelancer.get('name'))} |")
    md.append(f"| **Email** | {format_invoice_field(freelancer.get('email'))} |")
    md.append(f"| **Phone** | {format_invoice_field(freelancer.get('phone'))} |")
    md.append(f"| **Address** | {format_invoice_field(freelancer.get('address'))} |")
    md.append(f"| **Website** | {format_invoice_field(freelancer.get('website'))} |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("#### 📋 Itemized Services")
    md.append("")
    if line_items:
        md.append("| **Description** | **Hours / Qty** | **Rate** | **Line Total** |")
        md.append("|:---|:---|:---|:---|")
        for item in line_items:
            desc = format_invoice_field(item.get("description"), "Service")
            qty = item.get("hours_or_qty")
            unit = item.get("unit") or "hrs"
            qty_str = f"{qty} {unit}" if qty is not None else "Not provided"
            rate = item.get("rate")
            rate_str = f"${float(rate):,.2f}" if rate is not None else "Not provided"
            tot = item.get("line_total")
            tot_str = f"${float(tot):,.2f}" if tot is not None else "Not provided"
            md.append(f"| {desc} | {qty_str} | {rate_str} | {tot_str} |")
    else:
        md.append("*No itemized services detected in the input.*")

    md.append("")
    md.append("---")
    md.append("")
    md.append("#### 💰 Financial Breakdown")
    md.append("")
    subtotal = fin["subtotal"]
    discount = fin["discount"]
    disc_desc = fin["discount_description"]
    tax = fin["tax"]
    grand_total = fin["grand_total"]

    md.append("| **Item** | **Amount** |")
    md.append("|:---|:---|")
    md.append(f"| **Subtotal** | ${subtotal:,.2f} |")
    if discount > 0:
        label = f"Discount ({disc_desc})" if disc_desc else "Discount"
        md.append(f"| **{label}** | -${discount:,.2f} |")
    if tax > 0:
        md.append(f"| **Tax** | +${tax:,.2f} |")
    md.append(f"| **Grand Total** | **${grand_total:,.2f}** |")

    md.append("")
    md.append("---")
    md.append("")
    md.append("#### 💳 Payment Instructions / Notes")
    md.append(f"- **Payment Methods Accepted:** {format_invoice_field(payment.get('methods_accepted'))}")
    md.append(f"- **Bank / Remittance Details:** {format_invoice_field(payment.get('bank_details'))}")
    md.append(f"- **Payment Link:** {format_invoice_field(payment.get('payment_link'))}")
    md.append(f"- **Late Payment Policy:** {format_invoice_field(payment.get('late_payment_policy'))}")
    if payment.get("notes"):
        md.append(f"- **Additional Notes:** {format_invoice_field(payment.get('notes'))}")

    # Separated AI Suggestions section
    md.append("")
    md.append("---")
    md.append("")
    md.append("### 💡 AI Recommendations & Observational Notes")
    md.append("> *Note: The items below are non-contractual advisory suggestions generated by AI, clearly separated from the extracted facts above.*")
    md.append("")

    notes_count = 0
    if not freelancer.get("name"):
        md.append("- ℹ️ **Freelancer Contact Information:** Not detected in the input. Add your name, email, and phone number before issuing to the client.")
        notes_count += 1
    if not payment.get("bank_details") and not payment.get("payment_link"):
        md.append("- ℹ️ **Remittance Destination:** Specific bank account numbers or direct payment URLs were not provided in your notes. Ensure you provide payment details.")
        notes_count += 1
    if not payment.get("late_payment_policy"):
        md.append("- ℹ️ **Late Payment Policy:** No late fee terms were specified. Consider defining standard payment grace terms.")
        notes_count += 1

    for sug in suggestions:
        if sug and isinstance(sug, str):
            md.append(f"- 💡 {sug}")
            notes_count += 1

    if notes_count == 0:
        md.append("- All standard invoice details were verified and supported by your input notes.")

    return "\n".join(md)


# -----------------------------------------------------------------------------
# 2.2 Feature Helper Functions (Expense, Reminder, Monthly Summary)
# -----------------------------------------------------------------------------
def build_expense_categorizer_prompts(
    raw_expense_text: str,
    tax_jurisdiction: str = "Not specified",
) -> tuple[str, str]:
    """
    Builds the system and user prompts for the Expense Categorizer.
    Enforces jurisdiction-awareness, labels deductibility as 'Needs verification'
    when jurisdiction is unspecified, and mandates a non-definitive tax disclaimer.
    """
    jurisdiction_clean = tax_jurisdiction.strip() if tax_jurisdiction else "Not specified"
    is_specified = bool(jurisdiction_clean and jurisdiction_clean.lower() != "not specified")

    if is_specified:
        jurisdiction_rules = (
            f"The user has specified their tax jurisdiction as: '{jurisdiction_clean}'.\n"
            "- In the Potential Deductibility column, provide standard general guidelines relevant to this jurisdiction.\n"
            "- CRITICAL NON-LEGAL RULE: Do NOT make definitive tax claims such as 'fully deductible', '100% deductible', '50% deductible', or guaranteed qualification for specific tax provisions (such as Section 179). Actual eligibility depends on individual business structure, filing thresholds, and local tax rules."
        )
    else:
        jurisdiction_rules = (
            "The user has NOT specified a tax jurisdiction.\n"
            "- In the Potential Deductibility column, you MUST label every expense strictly as 'Needs verification'.\n"
            "- CRITICAL RULE: Do NOT make definitive tax deductibility claims (such as 'fully deductible', '50% deductible', or citing specific national tax codes like Section 179) because deductibility rules differ fundamentally across jurisdictions."
        )

    system_prompt = (
        "You are an expense organization and bookkeeping assistant for freelancers and independent contractors.\n"
        "Examine the list of transactions and present them in a clean markdown table with these exact columns:\n"
        "- Date\n"
        "- Description\n"
        "- Amount ($)\n"
        "- Expense Category (e.g., Software & Subscriptions, Office Supplies, Meals & Client Meetings, Travel, Equipment, Hosting/Infrastructure)\n"
        "- Potential Deductibility (Follow jurisdiction guidelines below)\n"
        "- Documentation Recommendation (e.g., 'Keep itemized receipt', 'Log business purpose')\n\n"
        f"{jurisdiction_rules}\n\n"
        "After the table, provide:\n"
        "1. Total Expenses ($) - mathematical sum of all listed items\n"
        "2. Spending Breakdown by Category\n"
        "3. Practical Bookkeeping Guidance (receipt retention, business/personal separation).\n\n"
        "MANDATORY DISCLAIMER:\n"
        "Conclude your response with this exact disclaimer:\n"
        "> ⚠️ **General Disclaimer:** Expense categorizations and potential deductibility notes are for informational and organizational purposes only and do not constitute formal tax, accounting, or legal advice. Deductibility varies based on your specific jurisdiction, filing status, and business structure. Consult a certified public accountant (CPA) or local tax professional."
    )

    user_prompt = f"Tax Jurisdiction: {jurisdiction_clean}\n\nRaw Expense Records:\n{raw_expense_text}"
    return system_prompt, user_prompt


def get_payment_reminder_status(
    due_date: date,
    current_date: date | None = None,
) -> tuple[str, int, str]:
    """
    Computes the invoice status based on due date and current reference date.
    Returns:
        tuple: (status_category, days_count, status_description)
               status_category: 'upcoming', 'due_today', or 'overdue'
               days_count: absolute day difference
               status_description: readable status label
    """
    if current_date is None:
        current_date = date.today()

    delta_days = (due_date - current_date).days

    if delta_days > 0:
        day_str = f"{delta_days} day" if delta_days == 1 else f"{delta_days} days"
        return "upcoming", delta_days, f"Upcoming (Due in {day_str})"
    elif delta_days == 0:
        return "due_today", 0, "Due Today"
    else:
        overdue_days = abs(delta_days)
        day_str = f"{overdue_days} day" if overdue_days == 1 else f"{overdue_days} days"
        return "overdue", overdue_days, f"Overdue by {day_str}"


def build_payment_reminder_prompts(
    client_name: str,
    invoice_id: str,
    amount_due: float,
    invoice_due_date: date,
    reminder_tone: str,
    additional_context: str = "",
    current_date: date | None = None,
) -> tuple[str, str, dict]:
    """
    Builds prompts for Payment Reminder.
    Enforces strict date-awareness (never describes an upcoming invoice as overdue)
    and strictly forbids inventing bank details, payment links, or late fees.
    """
    ref_date = current_date if current_date is not None else date.today()
    status_cat, days_count, status_label = get_payment_reminder_status(invoice_due_date, ref_date)

    system_prompt = (
        "You are a professional business communications specialist for independent contractors and freelancers.\n"
        "Draft an effective, courteous payment reminder message based strictly on the provided details.\n\n"
        "CRITICAL DATE & TIMING RULES:\n"
        f"- Reference Date (Today): {ref_date.strftime('%B %d, %Y')}\n"
        f"- Invoice Due Date: {invoice_due_date.strftime('%B %d, %Y')}\n"
        f"- Calculated Invoice Status: {status_label}\n"
        "- If the invoice is 'Upcoming' (due in the future), do NOT describe it as overdue, past due, or late! "
        "Frame the message as a courteous advance reminder of the upcoming due date.\n"
        "- If the invoice is 'Due Today', state clearly that payment is due today.\n"
        "- If the invoice is 'Overdue', politely note that the invoice is overdue by the calculated number of days.\n\n"
        "CRITICAL NON-FABRICATION & PLACEHOLDER RULES:\n"
        "- Do NOT invent bank details, account numbers, routing numbers, Stripe/PayPal payment URLs, client contact details, phone numbers, email addresses, or late-fee policies.\n"
        "- If payment methods, payment links, contact numbers, or sender name are not provided in the inputs, use clear bracketed placeholders: "
        "[Your Name], [Your Email / Phone], [Payment Link / Bank Account Details], [Late Fee Terms if applicable].\n\n"
        "OUTPUT STRUCTURE:\n"
        "1. Recommended Subject Line\n"
        "2. Complete Message / Email Body (with clear placeholders for any missing details)\n"
        "3. Recommended Next Steps (when to follow up next based on the due date status)."
    )

    user_prompt = (
        f"Client Name: {client_name}\n"
        f"Invoice Number: {invoice_id}\n"
        f"Amount Due: ${amount_due:,.2f}\n"
        f"Due Date: {invoice_due_date.strftime('%B %d, %Y')}\n"
        f"Calculated Status: {status_label}\n"
        f"Tone: {reminder_tone}\n"
        f"Additional Context: {additional_context.strip() if additional_context.strip() else 'None provided'}"
    )

    meta = {
        "status_category": status_cat,
        "days_count": days_count,
        "status_label": status_label,
    }

    return system_prompt, user_prompt, meta


def calculate_monthly_financials(
    rev_collected: float,
    rev_pending: float,
    total_exp: float,
    tax_rate_est: float,
) -> dict:
    """
    Calculates monthly financial totals, margins, and estimated tax reserves
    strictly from actual user-supplied inputs.
    """
    total_billed = rev_collected + rev_pending
    net_profit = rev_collected - total_exp
    profit_margin_pct = (net_profit / rev_collected * 100.0) if rev_collected > 0 else 0.0
    estimated_tax = max(0.0, net_profit * (tax_rate_est / 100.0))
    take_home = max(0.0, net_profit - estimated_tax)
    collection_rate_pct = (rev_collected / total_billed * 100.0) if total_billed > 0 else 100.0
    expense_ratio_pct = (total_exp / rev_collected * 100.0) if rev_collected > 0 else 0.0

    return {
        "rev_collected": rev_collected,
        "rev_pending": rev_pending,
        "total_billed": total_billed,
        "total_exp": total_exp,
        "net_profit": net_profit,
        "profit_margin_pct": profit_margin_pct,
        "tax_rate_est": tax_rate_est,
        "estimated_tax": estimated_tax,
        "take_home": take_home,
        "collection_rate_pct": collection_rate_pct,
        "expense_ratio_pct": expense_ratio_pct,
    }


def build_monthly_summary_prompts(fin: dict) -> tuple[str, str]:
    """
    Builds prompts for Monthly Summary using strictly user-supplied figures.
    Forbids universal benchmarks or guaranteed health claims.
    """
    system_prompt = (
        "You are an analytical financial reviewer for freelancers and independent businesses.\n"
        "Evaluate the monthly performance strictly based on the supplied actual figures.\n\n"
        "STRICT EVALUATION GUIDELINES:\n"
        "- Base all analysis ONLY on the supplied numbers. Do NOT invent figures or cite fictional industry averages.\n"
        "- Do NOT claim universal profit margin benchmarks (e.g., claiming 'all freelancers must have 70% margins'). Profit margins vary widely by industry and business model.\n"
        "- Do NOT guarantee financial health or outcomes. Provide objective observations on cash flow liquidity and outstanding collections.\n"
        f"- Explicitly state that the tax reserve (${fin['estimated_tax']:,.2f}) is an estimate based strictly on the user-selected rate of {fin['tax_rate_est']}%.\n\n"
        "STRUCTURE YOUR REPORT:\n"
        "1. Cash Flow & Revenue Analysis (Collected vs Pending amounts)\n"
        "2. Expense & Margin Observations (Operating expenses relative to collected revenue)\n"
        "3. Tax Reserve Planning Note (Emphasizing this is a user-chosen estimate and subject to local tax rules)\n"
        "4. Practical Observations to improve invoice collection speed.\n\n"
        "MANDATORY DISCLAIMER:\n"
        "Conclude with:\n"
        "> ⚠️ **Disclaimer:** All totals, margins, and tax reserves are calculated from user-provided inputs and selected percentages. This summary is for organizational review only and does not constitute formal accounting, tax, or investment advice."
    )

    user_prompt = (
        "User-Supplied Monthly Financial Performance:\n"
        f"- Revenue Collected: ${fin['rev_collected']:,.2f}\n"
        f"- Invoices Outstanding / Pending: ${fin['rev_pending']:,.2f}\n"
        f"- Total Billed (Collected + Pending): ${fin['total_billed']:,.2f}\n"
        f"- Total Operating Expenses: ${fin['total_exp']:,.2f}\n"
        f"- Net Profit (Collected - Expenses): ${fin['net_profit']:,.2f}\n"
        f"- Calculated Profit Margin: {fin['profit_margin_pct']:.1f}%\n"
        f"- User-Selected Tax Reserve Rate: {fin['tax_rate_est']}%\n"
        f"- Estimated Tax Reserve: ${fin['estimated_tax']:,.2f} (Estimated based on user-selected {fin['tax_rate_est']}%)\n"
        f"- Estimated Take-Home: ${fin['take_home']:,.2f}\n"
        f"- Collection Rate: {fin['collection_rate_pct']:.1f}% of billed revenue collected so far."
    )

    return system_prompt, user_prompt



# -----------------------------------------------------------------------------
# 3. Sidebar: Configuration & App Information
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("💼 FreelanceFlow AI")
    st.caption("Side Hustle & Freelance Finance Assistant")
    st.markdown("---")

    # API Key Configuration
    st.subheader("🔑 Groq API Settings")
    env_api_key = os.getenv("GROQ_API_KEY", "")

    # Allow user to supply or override the API key securely
    user_api_key = st.text_input(
        label="Groq API Key",
        value=env_api_key,
        type="password",
        placeholder="gsk_...",
        help="Reads from GROQ_API_KEY in .env by default. You can also paste your key here.",
    )

    # Active API Key in session
    active_api_key = user_api_key.strip() if user_api_key else env_api_key.strip()

    if active_api_key:
        st.success("API Key configured", icon="✅")
    else:
        st.warning("API Key missing", icon="⚠️")
        st.info(
            "Get a free API key at [console.groq.com/keys](https://console.groq.com/keys) "
            "and add `GROQ_API_KEY=your_key` to a `.env` file."
        )

    st.markdown("---")
    st.subheader("🤖 Model Info")
    st.code(MODEL_NAME, language="text")
    st.caption("Running open-weight model via ultra-fast Groq LPU inference.")

    st.markdown("---")
    st.caption("🎃 **Hacktoberfest 2026**")
    st.caption("Week 3: Side Hustle Challenge")


# -----------------------------------------------------------------------------
# 4. Main Application Interface
# -----------------------------------------------------------------------------
st.title("💼 FreelanceFlow AI")
st.markdown(
    "A clean, AI-powered financial workspace for freelancers and small-business owners. "
    "Organize messy invoices, categorize expenses, draft polite payment reminders, and analyze monthly cash flow."
)

# Tab Navigation for the four required core sections
tab_invoice, tab_expense, tab_reminder, tab_summary = st.tabs([
    "📄 Invoice Organizer",
    "🏷️ Expense Categorizer",
    "⏰ Payment Reminder",
    "📊 Monthly Summary",
])


# =============================================================================
# SECTION 1: INVOICE ORGANIZER
# =============================================================================
with tab_invoice:
    st.header("📄 Invoice Organizer")
    st.write(
        "Paste unorganized invoice details, email threads, or freelance billing notes. "
        "The open-weight AI will parse and organize them into a clean, structured invoice format."
    )

    # Sample data loader for quick college project demonstrations
    sample_invoice_text = (
        "Client: Apex Digital Studios (Contact: Sarah Jenkins - billing@apexstudios.io)\n"
        "Project: React & Next.js Website Redesign for Fall 2026 launch.\n"
        "Work done: 30 hours of frontend development at $75/hr ($2,250), "
        "plus 10 hours of backend API integration at $85/hr ($850).\n"
        "Subtotal: $3,100. Applied client referral discount: -$100.\n"
        "Invoice Number: INV-2026-104. Date issued: October 5, 2026. Payment terms: Net 15 (Due Oct 20, 2026).\n"
        "Payment methods accepted: Wire transfer or Stripe link."
    )

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        load_sample_inv = st.button("Load Sample Invoice", key="load_inv_btn")

    if "invoice_input_text" not in st.session_state:
        st.session_state["invoice_input_text"] = ""

    if load_sample_inv:
        st.session_state["invoice_input_text"] = sample_invoice_text

    raw_invoice_text = st.text_area(
        label="Raw Invoice Notes / Details",
        value=st.session_state["invoice_input_text"],
        height=180,
        placeholder="Paste rough client billing notes, hours worked, rates, due dates, or email transcripts here...",
    )

    if st.button("Organize Invoice with AI", type="primary", key="btn_organize_inv"):
        if not raw_invoice_text.strip():
            st.warning("Please paste or type invoice notes above before running the organizer.")
        elif not active_api_key:
            st.error("Cannot process: GROQ_API_KEY is not configured. Please add your key in the sidebar or `.env`.")
        else:
            with st.spinner("Analyzing and organizing invoice with openai/gpt-oss-20b..."):
                invoice_data, error_msg, was_retried = extract_invoice_with_retry(
                    api_key=active_api_key,
                    raw_text=raw_invoice_text,
                )

                if error_msg:
                    st.error(f"❌ {error_msg}")
                    st.info(
                        "💡 Tip: Ensure your notes include readable billing details (client name, "
                        "services rendered, rates, or amounts) and click 'Organize Invoice with AI' to try again."
                    )
                else:
                    if was_retried:
                        st.info("ℹ️ The invoice structure was automatically repaired and verified after formatting adjustment.")
                    st.success("Invoice successfully organized!")

                    # Display mathematical calculation verification badge
                    calc_msg = invoice_data["financial_breakdown"]["calc_validation_msg"]
                    if calc_msg:
                        if invoice_data["financial_breakdown"]["math_is_exact"]:
                            st.info(f"🧮 {calc_msg}")
                        else:
                            st.warning(f"⚠️ {calc_msg}")

                    # Render structured invoice with clear distinction between facts and suggestions
                    rendered_md = render_invoice_markdown(invoice_data)
                    st.markdown(rendered_md)



# =============================================================================
# SECTION 2: EXPENSE CATEGORIZER
# =============================================================================
with tab_expense:
    st.header("🏷️ Expense Categorizer")
    st.write(
        "Paste your receipts, bank statement line-items, or rough expense logs. "
        "The AI will classify each item into standardized freelance tax categories."
    )

    sample_expenses_text = (
        "2026-10-01: GitHub Copilot Pro subscription - $19.00\n"
        "2026-10-03: Coffee and pastries during client onboarding meeting at Blue Bottle - $24.50\n"
        "2026-10-06: AWS EC2 hosting and S3 storage fees - $78.30\n"
        "2026-10-09: Ergonomic monitor stand and cables from OfficeSupply Co - $89.99\n"
        "2026-10-11: Uber ride to meet prospective marketing client downtown - $32.40\n"
        "2026-10-12: Domain renewal for portfolio website (Namecheap) - $14.88"
    )

    col_exp1, col_exp2 = st.columns([1, 4])
    with col_exp1:
        load_sample_exp = st.button("Load Sample Expenses", key="load_exp_btn")

    if "expense_input_text" not in st.session_state:
        st.session_state["expense_input_text"] = ""

    if load_sample_exp:
        st.session_state["expense_input_text"] = sample_expenses_text

    raw_expense_text = st.text_area(
        label="Raw Expense Records",
        value=st.session_state["expense_input_text"],
        height=180,
        placeholder="e.g.,\n2026-10-02: Figma Subscription - $15.00\n2026-10-04: Client Lunch - $45.00\n2026-10-08: USB-C Hub - $35.00",
    )

    tax_jurisdiction = st.selectbox(
        label="Tax Jurisdiction (Optional)",
        options=[
            "Not specified",
            "United States (IRS)",
            "United Kingdom (HMRC)",
            "Canada (CRA)",
            "Australia (ATO)",
            "European Union",
            "India",
            "Other",
        ],
        help="Tax deductibility rules vary significantly by country and business type. Leave as 'Not specified' if unsure.",
    )

    if st.button("Categorize Expenses with AI", type="primary", key="btn_categorize_exp"):
        if not raw_expense_text.strip():
            st.warning("Please enter or load expense records first.")
        elif not active_api_key:
            st.error("Cannot process: GROQ_API_KEY is not configured. Please add your key in the sidebar or `.env`.")
        else:
            with st.spinner("Classifying expenses with openai/gpt-oss-20b..."):
                system_prompt, user_prompt = build_expense_categorizer_prompts(
                    raw_expense_text=raw_expense_text,
                    tax_jurisdiction=tax_jurisdiction,
                )
                result, error = call_groq_model(
                    api_key=active_api_key,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )

                if error:
                    st.error(error)
                else:
                    st.success("Expenses successfully categorized!")
                    st.markdown("### 📊 Categorized Expense Report")
                    st.markdown(result)



# =============================================================================
# SECTION 3: PAYMENT REMINDER
# =============================================================================
with tab_reminder:
    st.header("⏰ Payment Reminder Generator")
    st.write(
        "Late payments hurt freelance cash flow. Generate customized, professional, "
        "and diplomatic payment reminder messages tailored to your client's payment status."
    )

    with st.form("reminder_form"):
        col_r1, col_r2 = st.columns(2)

        with col_r1:
            client_name = st.text_input(
                label="Client / Company Name",
                placeholder="e.g., Acme Design Co.",
                value="Acme Design Co.",
            )
            invoice_id = st.text_input(
                label="Invoice Number",
                placeholder="e.g., INV-2026-088",
                value="INV-2026-088",
            )
            amount_due = st.number_input(
                label="Amount Due ($)",
                min_value=0.0,
                value=1450.00,
                step=50.0,
                format="%.2f",
            )

        with col_r2:
            invoice_due_date = st.date_input(
                label="Original Invoice Due Date",
                value=date.today(),
            )

            # Determine dynamic invoice timing status
            status_cat, days_count, status_label = get_payment_reminder_status(invoice_due_date)

            if status_cat == "upcoming":
                tone_options = [
                    "Friendly / Advance Notice (Upcoming)",
                    "Gentle Reminder (Upcoming)",
                ]
            elif status_cat == "due_today":
                tone_options = [
                    "Friendly & Courteous (Due Today)",
                    "Professional & Direct (Due Today)",
                ]
            else:
                tone_options = [
                    "Polite & Direct (Overdue Follow-Up)",
                    "Firm & Professional (Overdue)",
                    "Urgent Notice / Final Follow-Up (Overdue)",
                ]

            reminder_tone = st.selectbox(
                label="Reminder Tone",
                options=tone_options,
            )
            additional_context = st.text_input(
                label="Additional Context (Optional)",
                placeholder="e.g., Client previously promised payment by Friday, project delivered on time.",
            )

        # Show status indicator
        if status_cat == "upcoming":
            st.info(f"📅 **Invoice Timing Status:** {status_label}")
        elif status_cat == "due_today":
            st.warning(f"📅 **Invoice Timing Status:** {status_label}")
        else:
            st.error(f"📅 **Invoice Timing Status:** {status_label}")

        submit_reminder = st.form_submit_button("Generate Payment Reminder with AI", type="primary")

    if submit_reminder:
        if not client_name.strip() or not invoice_id.strip():
            st.warning("Please provide at least a Client Name and Invoice Number.")
        elif not active_api_key:
            st.error("Cannot process: GROQ_API_KEY is not configured. Please add your key in the sidebar or `.env`.")
        else:
            with st.spinner("Drafting payment reminder with openai/gpt-oss-20b..."):
                system_prompt, user_prompt, meta = build_payment_reminder_prompts(
                    client_name=client_name,
                    invoice_id=invoice_id,
                    amount_due=amount_due,
                    invoice_due_date=invoice_due_date,
                    reminder_tone=reminder_tone,
                    additional_context=additional_context,
                )

                result, error = call_groq_model(
                    api_key=active_api_key,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )

                if error:
                    st.error(error)
                else:
                    st.success("Payment reminder drafted!")
                    st.markdown("### ✉️ Drafted Payment Reminder")
                    st.markdown(result)



# =============================================================================
# SECTION 4: MONTHLY SUMMARY & INSIGHTS
# =============================================================================
with tab_summary:
    st.header("📊 Monthly Summary & Financial Insights")
    st.write(
        "Track monthly freelance income, review expenses, and get AI-driven advice "
        "on profit margins, cash flow management, and tax set-asides."
    )

    # Input metrics
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    with col_m1:
        rev_collected = st.number_input(
            label="Revenue Collected ($)",
            min_value=0.0,
            value=4850.00,
            step=100.0,
            format="%.2f",
            help="Income already received in your account this month.",
        )
    with col_m2:
        rev_pending = st.number_input(
            label="Pending / Unpaid Invoices ($)",
            min_value=0.0,
            value=1600.00,
            step=100.0,
            format="%.2f",
            help="Total amount from issued invoices awaiting client payment.",
        )
    with col_m3:
        total_exp = st.number_input(
            label="Total Expenses ($)",
            min_value=0.0,
            value=820.00,
            step=50.0,
            format="%.2f",
            help="Total business expenses incurred this month.",
        )
    with col_m4:
        tax_rate_est = st.slider(
            label="Estimated Tax Reserve (%)",
            min_value=10,
            max_value=40,
            value=25,
            help="Percentage of net profit recommended to save for self-employment taxes (user-selected estimate).",
        )

    # Calculations strictly derived from actual supplied inputs
    fin = calculate_monthly_financials(
        rev_collected=rev_collected,
        rev_pending=rev_pending,
        total_exp=total_exp,
        tax_rate_est=tax_rate_est,
    )

    # Display KPI Cards
    st.markdown("#### 📈 Key Financial Metrics")
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    kpi_col1.metric("Revenue Collected", f"${fin['rev_collected']:,.2f}")
    kpi_col2.metric("Total Expenses", f"${fin['total_exp']:,.2f}")
    kpi_col3.metric(
        "Net Profit",
        f"${fin['net_profit']:,.2f}",
        delta=f"{fin['profit_margin_pct']:.1f}% margin" if fin['rev_collected'] > 0 else None,
    )
    kpi_col4.metric(
        f"Tax Reserve ({fin['tax_rate_est']}% estimate)",
        f"${fin['estimated_tax']:,.2f}",
        help=f"Estimated tax reserve based strictly on your selected rate of {fin['tax_rate_est']}% on net profit.",
    )

    # Visual Comparison Chart using pandas & Streamlit
    st.markdown("#### 📊 Cash Flow Breakdown")
    chart_data = pd.DataFrame(
        {
            "Financial Metric": [
                "Revenue Collected",
                "Pending Invoices",
                "Total Expenses",
                "Net Profit",
                f"Tax Reserve ({fin['tax_rate_est']}%)",
                "Estimated Take-Home",
            ],
            "Amount ($)": [
                fin["rev_collected"],
                fin["rev_pending"],
                fin["total_exp"],
                fin["net_profit"],
                fin["estimated_tax"],
                fin["take_home"],
            ],
        }
    )
    # Streamlit bar chart indexed by metric name
    st.bar_chart(chart_data.set_index("Financial Metric"), height=300)

    # AI Financial Advice Button
    st.markdown("#### 🧠 AI Financial Advisor")
    st.caption("Generate actionable tips, risk analysis, and cash flow recommendations based strictly on your numbers.")

    if st.button("Generate Financial Insights with AI", type="primary", key="btn_summary_ai"):
        if not active_api_key:
            st.error("Cannot process: GROQ_API_KEY is not configured. Please add your key in the sidebar or `.env`.")
        else:
            with st.spinner("Analyzing financial figures with openai/gpt-oss-20b..."):
                system_prompt, user_prompt = build_monthly_summary_prompts(fin)
                result, error = call_groq_model(
                    api_key=active_api_key,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )

                if error:
                    st.error(error)
                else:
                    st.success("Financial evaluation complete!")
                    st.markdown("### 💡 AI Financial Advisory Report")
                    st.markdown(result)



# -----------------------------------------------------------------------------
# 5. Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "FreelanceFlow AI • Built with Streamlit, Groq SDK & openai/gpt-oss-20b • Hacktoberfest 2026"
    "</div>",
    unsafe_allow_html=True,
)
