"""
Tests for Invoice Organizer JSON Parsing, Validation, and Non-Fabrication Logic
"""

import sys
import unittest
from pathlib import Path

# Ensure root directory is on sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from app import (
    clean_and_repair_json,
    safe_json_loads,
    parse_and_validate_invoice,
    render_invoice_markdown,
    format_invoice_field,
    safe_float,
)


class TestInvoiceJSONRepairAndParsing(unittest.TestCase):
    """Test suite for robust JSON repair and safe loading."""

    def test_valid_json(self):
        valid_json = '{"is_valid_invoice": true, "invoice_overview": {"invoice_number": "INV-100"}}'
        data, err = safe_json_loads(valid_json)
        self.assertIsNone(err)
        self.assertIsNotNone(data)
        self.assertEqual(data.get("invoice_overview", {}).get("invoice_number"), "INV-100")

    def test_malformed_json_trailing_commas(self):
        """Tests that trailing commas before } and ] are repaired without error."""
        malformed = """
        {
            "is_valid_invoice": true,
            "line_items": [
                {"description": "Web Dev", "rate": 75,},
            ],
            "financial_breakdown": {
                "subtotal": 3100,
                "discount": 100,
            },
        }
        """
        data, err = safe_json_loads(malformed)
        self.assertIsNone(err, f"Expected safe parse, got error: {err}")
        self.assertIsNotNone(data)
        self.assertEqual(data["financial_breakdown"]["subtotal"], 3100)

    def test_malformed_json_missing_commas(self):
        """Tests that missing commas between consecutive lines are repaired."""
        malformed = """
        {
            "is_valid_invoice": true
            "invoice_overview": {
                "invoice_number": "INV-2026-104"
                "payment_terms": "Net 15"
            }
        }
        """
        data, err = safe_json_loads(malformed)
        self.assertIsNone(err, f"Expected repaired JSON, got error: {err}")
        self.assertIsNotNone(data)
        self.assertEqual(data["invoice_overview"]["invoice_number"], "INV-2026-104")

    def test_markdown_code_fences_and_commentary(self):
        """Tests stripping of markdown code fences and conversational preamble/postamble."""
        wrapped = """
        Here is the structured invoice data you requested:
        ```json
        {
            "is_valid_invoice": true,
            "invoice_overview": {
                "invoice_number": "INV-2026-104"
            }
        }
        ```
        Hope this helps!
        """
        data, err = safe_json_loads(wrapped)
        self.assertIsNone(err)
        self.assertIsNotNone(data)
        self.assertEqual(data["invoice_overview"]["invoice_number"], "INV-2026-104")

    def test_incomplete_and_truncated_json(self):
        """Tests that truncated JSON missing closing quotes and braces is safely balanced."""
        truncated = '{"is_valid_invoice": true, "invoice_overview": {"invoice_number": "INV-TRUNC'
        data, err = safe_json_loads(truncated)
        self.assertIsNone(err, f"Expected safe repair of truncated JSON, got: {err}")
        self.assertIsNotNone(data)
        self.assertEqual(data["invoice_overview"]["invoice_number"], "INV-TRUNC")

    def test_comments_in_json(self):
        """Tests that inline and block comments are stripped safely."""
        with_comments = """
        {
            // Invoice status
            "is_valid_invoice": true,
            /* Line items block */
            "line_items": []
        }
        """
        data, err = safe_json_loads(with_comments)
        self.assertIsNone(err)
        self.assertIsNotNone(data)
        self.assertTrue(data["is_valid_invoice"])

    def test_empty_or_whitespace_response(self):
        """Tests that empty input returns clean error string without exception."""
        data, err = safe_json_loads("")
        self.assertIsNone(data)
        self.assertIn("Empty", err)


class TestInvoiceValidationAndNonFabrication(unittest.TestCase):
    """Test suite for grounding, non-fabrication, and mathematical validation."""

    def setUp(self):
        self.sample_raw_invoice = (
            "Client: Apex Digital Studios (Contact: Sarah Jenkins - billing@apexstudios.io)\n"
            "Project: React & Next.js Website Redesign for Fall 2026 launch.\n"
            "Work done: 30 hours of frontend development at $75/hr ($2,250), "
            "plus 10 hours of backend API integration at $85/hr ($850).\n"
            "Subtotal: $3,100. Applied client referral discount: -$100.\n"
            "Invoice Number: INV-2026-104. Date issued: October 5, 2026. Payment terms: Net 15 (Due Oct 20, 2026).\n"
            "Payment methods accepted: Wire transfer or Stripe link."
        )

    def test_sample_invoice_calculations(self):
        """Verifies calculation: Subtotal $3,100 - Discount $100 = Grand Total $3,000."""
        ai_json = """
        {
            "is_valid_invoice": true,
            "invoice_overview": {
                "invoice_number": "INV-2026-104",
                "issue_date": "October 5, 2026",
                "due_date": "October 20, 2026",
                "payment_terms": "Net 15"
            },
            "client_info": {
                "name": "Apex Digital Studios",
                "contact_person": "Sarah Jenkins",
                "email": "billing@apexstudios.io",
                "project": "React & Next.js Website Redesign",
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
                {"description": "frontend development", "hours_or_qty": 30, "rate": 75, "line_total": 2250},
                {"description": "backend API integration", "hours_or_qty": 10, "rate": 85, "line_total": 850}
            ],
            "financial_breakdown": {
                "subtotal": 3100,
                "discount": 100,
                "discount_description": "client referral discount",
                "tax": 0,
                "grand_total": 3000
            },
            "payment_instructions": {
                "methods_accepted": "Wire transfer or Stripe link",
                "bank_details": null,
                "payment_link": null,
                "late_payment_policy": null,
                "notes": null
            },
            "ai_suggestions": []
        }
        """
        data, err = parse_and_validate_invoice(self.sample_raw_invoice, ai_json)
        self.assertIsNone(err)
        self.assertIsNotNone(data)

        fin = data["financial_breakdown"]
        self.assertEqual(fin["subtotal"], 3100.0)
        self.assertEqual(fin["discount"], 100.0)
        self.assertEqual(fin["grand_total"], 3000.0)
        self.assertTrue(fin["math_is_exact"])
        self.assertIn("Calculations verified", fin["calc_validation_msg"])

    def test_non_fabrication_defaults_to_not_provided(self):
        """Verifies missing details default to 'Not provided' and are not fabricated."""
        ai_json_with_hallucinations = """
        {
            "is_valid_invoice": true,
            "invoice_overview": {"invoice_number": "INV-2026-104"},
            "client_info": {
                "name": "Apex Digital Studios",
                "address": "999 Fictional Boulevard",
                "phone": "555-0199"
            },
            "freelancer_info": {
                "name": "Fake Freelancer",
                "email": "fake@freelance.org",
                "phone": "555-9999",
                "address": "123 Fake Street",
                "website": "https://fakewebsite.dev"
            },
            "line_items": [{"description": "Dev", "hours_or_qty": 10, "rate": 50, "line_total": 500}],
            "financial_breakdown": {"subtotal": 500, "discount": 0, "tax": 0, "grand_total": 500},
            "payment_instructions": {
                "methods_accepted": "Wire transfer",
                "bank_details": "Fake Bank Routing 987654321",
                "payment_link": "https://buy.stripe.com/fake_link",
                "late_payment_policy": "2% late penalty per month"
            }
        }
        """
        data, err = parse_and_validate_invoice(self.sample_raw_invoice, ai_json_with_hallucinations)
        self.assertIsNone(err)

        # Grounding check: fabricated info not in source must be discarded
        freelancer = data["freelancer_info"]
        self.assertIsNone(freelancer.get("name"))
        self.assertIsNone(freelancer.get("email"))
        self.assertIsNone(freelancer.get("phone"))
        self.assertIsNone(freelancer.get("address"))

        payment = data["payment_instructions"]
        self.assertIsNone(payment.get("bank_details"))
        self.assertIsNone(payment.get("payment_link"))
        self.assertIsNone(payment.get("late_payment_policy"))

        # Render check
        md = render_invoice_markdown(data)
        self.assertIn("Freelancer Name** | Not provided", md)
        self.assertIn("Bank / Remittance Details:** Not provided", md)
        self.assertIn("Payment Link:** Not provided", md)
        self.assertIn("Late Payment Policy:** Not provided", md)

    def test_non_invoice_input_rejection(self):
        """Verifies non-invoice input is flagged as invalid without generating fake data."""
        non_invoice_text = "Good morning! Just writing to check in on our lunch plans."
        ai_resp = '{"is_valid_invoice": false, "error_message": "No invoice details found in input."}'
        data, err = parse_and_validate_invoice(non_invoice_text, ai_resp)
        self.assertIsNone(data)
        self.assertIsNotNone(err)
        self.assertIn("No invoice details found", err)


if __name__ == "__main__":
    unittest.main()
