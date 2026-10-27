from types import SimpleNamespace
import unittest

from backend.offer_discount_agent import OfferDiscountAgentLLM


class OfferDiscountAgentTests(unittest.TestCase):
    def test_compute_discounted_rate_returns_policy_result(self):
        profile = SimpleNamespace(
            gender="female",
            age=25,
            cibil_score=790,
            monthly_income=50000,
        )

        final_rate, result = OfferDiscountAgentLLM().compute_discounted_rate(profile)

        self.assertEqual(final_rate, 10.85)
        self.assertEqual(result.total_discount, 0.65)
        self.assertEqual(
            [concession.policy_name for concession in result.concessions],
            [
                "Women Borrowers Scheme",
                "Youth Career Starter Program",
                "Super Prime Credit Tier",
            ],
        )


if __name__ == "__main__":
    unittest.main()
