"""
Tests for Invoice Organizer JSON Parsing, Validation, and Non-Fabrication Logic
"""

import sys
import unittest
from pathlib import Path

# Ensure root directory is on sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from datetime import date
from app import (
    clean_and_repair_json,
    safe_json_loads,
    parse_and_validate_invoice,
    render_invoice_markdown,
    format_invoice_field,
    safe_float,
    build_expense_categorizer_prompts,
    get_payment_reminder_status,
    build_payment_reminder_prompts,
    calculate_monthly_financials,
    build_monthly_summary_prompts,
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


class TestExpenseCategorizerPrompts(unittest.TestCase):
    """Test suite for Expense Categorizer jurisdiction awareness and disclaimers."""

    def test_unspecified_jurisdiction_requires_needs_verification(self):
        sys_p, user_p = build_expense_categorizer_prompts("2026-10-01: Coffee - $5.00", "Not specified")
        self.assertIn("Needs verification", sys_p)
        self.assertIn("General Disclaimer", sys_p)
        self.assertIn("Tax Jurisdiction: Not specified", user_p)
        self.assertIn("Do NOT make definitive tax deductibility claims", sys_p)

    def test_specified_jurisdiction_includes_jurisdiction_and_disclaimer(self):
        sys_p, user_p = build_expense_categorizer_prompts("2026-10-01: Software - $20.00", "United States (IRS)")
        self.assertIn("United States (IRS)", user_p)
        self.assertIn("United States (IRS)", sys_p)
        self.assertIn("Do NOT make definitive tax claims", sys_p)
        self.assertIn("General Disclaimer", sys_p)


class TestPaymentReminderLogic(unittest.TestCase):
    """Test suite for Payment Reminder date timing, non-fabrication, and placeholders."""

    def test_upcoming_due_date_status(self):
        today = date(2026, 10, 15)
        future_date = date(2026, 10, 20)
        cat, days, label = get_payment_reminder_status(future_date, current_date=today)
        self.assertEqual(cat, "upcoming")
        self.assertEqual(days, 5)
        self.assertEqual(label, "Upcoming (Due in 5 days)")

    def test_due_today_status(self):
        today = date(2026, 10, 15)
        cat, days, label = get_payment_reminder_status(today, current_date=today)
        self.assertEqual(cat, "due_today")
        self.assertEqual(days, 0)
        self.assertEqual(label, "Due Today")

    def test_overdue_status(self):
        today = date(2026, 10, 15)
        past_date = date(2026, 10, 10)
        cat, days, label = get_payment_reminder_status(past_date, current_date=today)
        self.assertEqual(cat, "overdue")
        self.assertEqual(days, 5)
        self.assertEqual(label, "Overdue by 5 days")

    def test_prompts_forbid_describing_upcoming_as_overdue(self):
        today = date(2026, 10, 15)
        future_date = date(2026, 10, 22)
        sys_p, user_p, meta = build_payment_reminder_prompts(
            client_name="Test Corp",
            invoice_id="INV-99",
            amount_due=500.0,
            invoice_due_date=future_date,
            reminder_tone="Friendly / Advance Notice (Upcoming)",
            current_date=today,
        )
        self.assertEqual(meta["status_category"], "upcoming")
        self.assertIn("do NOT describe it as overdue, past due, or late", sys_p)
        self.assertIn("Do NOT invent bank details", sys_p)
        self.assertIn("[Payment Link / Bank Account Details]", sys_p)
        self.assertIn("[Your Name]", sys_p)


class TestMonthlySummaryCalculations(unittest.TestCase):
    """Test suite for Monthly Summary actual calculations and tax reserve estimates."""

    def test_calculation_from_supplied_inputs(self):
        fin = calculate_monthly_financials(
            rev_collected=5000.0,
            rev_pending=2000.0,
            total_exp=1500.0,
            tax_rate_est=25.0,
        )
        self.assertEqual(fin["total_billed"], 7000.0)
        self.assertEqual(fin["net_profit"], 3500.0)
        self.assertEqual(fin["profit_margin_pct"], 70.0)
        self.assertEqual(fin["estimated_tax"], 875.0)
        self.assertEqual(fin["take_home"], 2625.0)
        self.assertAlmostEqual(fin["collection_rate_pct"], (5000.0 / 7000.0) * 100.0)

    def test_prompts_require_user_selected_estimates_and_forbid_universal_benchmarks(self):
        fin = calculate_monthly_financials(
            rev_collected=4000.0,
            rev_pending=1000.0,
            total_exp=1000.0,
            tax_rate_est=30.0,
        )
        sys_p, user_p = build_monthly_summary_prompts(fin)
        self.assertIn("Do NOT claim universal profit margin benchmarks", sys_p)
        self.assertIn("Do NOT guarantee financial health or outcomes", sys_p)
        self.assertIn("estimate based strictly on the user-selected rate of 30.0%", sys_p)
        self.assertIn("Revenue Collected: $4,000.00", user_p)
        self.assertIn("Operating Expenses: $1,000.00", user_p)
        self.assertIn("Net Profit (Collected - Expenses): $3,000.00", user_p)


if __name__ == "__main__":
    unittest.main()
