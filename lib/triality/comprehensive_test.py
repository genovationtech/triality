#!/usr/bin/env python3
"""
Comprehensive Test Suite - Single File to Run Everything

This runs all test suites:
1. Basic feature tests
2. Production verification tests (MMS, convergence, conservation, regression)
3. Negative tests (things that should fail)

Usage:
    python comprehensive_test.py
    python comprehensive_test.py --save-benchmarks
    python comprehensive_test.py --plots
"""

import sys
import os
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import test suites
from triality.verification.test_verification import ProductionTestSuite
from triality.verification.negative_tests import NegativeTestSuite


class ComprehensiveTestRunner:
    """Runs all test suites and provides unified reporting"""

    def __init__(self, save_plots=False, save_benchmarks=False):
        self.save_plots = save_plots
        self.save_benchmarks = save_benchmarks
        self.results = {}

    def run_all(self):
        """Run all test suites"""
        print("\n" + "="*70)
        print("  TRIALITY COMPREHENSIVE TEST SUITE")
        print("  Testing: Correctness + Robustness + Edge Cases")
        print("="*70)

        # Run verification tests (positive tests - things that should work)
        print("\n" + "█"*70)
        print("  PART 1: PRODUCTION VERIFICATION (Positive Tests)")
        print("█"*70)
        verification_suite = ProductionTestSuite(
            save_plots=self.save_plots,
            save_benchmarks=self.save_benchmarks
        )
        verification_suite.run_all_tests()
        self.results['verification'] = verification_suite.results

        # Run negative tests (things that should fail gracefully)
        print("\n" + "█"*70)
        print("  PART 2: NEGATIVE TESTS (Edge Cases & Failures)")
        print("█"*70)
        negative_suite = NegativeTestSuite()
        negative_suite.run_all_tests()
        self.results['negative'] = negative_suite.results

        # Final summary
        self.print_comprehensive_summary()

    def print_comprehensive_summary(self):
        """Print overall summary of all tests"""
        print("\n" + "="*70)
        print("  COMPREHENSIVE TEST SUMMARY")
        print("="*70)

        # Verification tests
        ver = self.results['verification']
        print("\n━━━ PART 1: PRODUCTION VERIFICATION ━━━")

        # MMS
        mms_passed = sum(1 for _, passed in ver['mms'] if passed)
        mms_total = len(ver['mms'])
        print(f"\n  MMS Tests: {mms_passed}/{mms_total} ✓")

        # Convergence
        conv_passed = sum(1 for _, passed in ver['convergence'] if passed)
        conv_total = len(ver['convergence'])
        print(f"  Grid Convergence: {conv_passed}/{conv_total} ✓")

        # Conservation
        cons_passed = sum(1 for _, passed in ver['conservation'] if passed)
        cons_total = len(ver['conservation'])
        print(f"  Conservation Laws: {cons_passed}/{cons_total} ✓")

        # Regression
        reg_report = ver['regression']
        if reg_report:
            print(f"  Regression: {reg_report.passed}/{reg_report.total_tests} ✓")

        verification_total_passed = mms_passed + conv_passed + cons_passed
        verification_total = mms_total + conv_total + cons_total
        if reg_report:
            verification_total_passed += reg_report.passed
            verification_total += reg_report.total_tests

        # Negative tests
        neg = self.results['negative']
        print("\n━━━ PART 2: NEGATIVE TESTS ━━━")

        ill_posed_passed = sum(1 for _, p in neg['ill_posed'] if p)
        ill_posed_total = len(neg['ill_posed'])
        print(f"\n  Ill-Posed Problems: {ill_posed_passed}/{ill_posed_total} 🛡️")

        ambiguous_passed = sum(1 for _, p in neg['ambiguous'] if p)
        ambiguous_total = len(neg['ambiguous'])
        print(f"  Ambiguous Problems: {ambiguous_passed}/{ambiguous_total} 🛡️")

        invalid_passed = sum(1 for _, p in neg['invalid'] if p)
        invalid_total = len(neg['invalid'])
        print(f"  Invalid Inputs: {invalid_passed}/{invalid_total} 🛡️")

        negative_total_passed = ill_posed_passed + ambiguous_passed + invalid_passed
        negative_total = ill_posed_total + ambiguous_total + invalid_total

        # Overall
        print("\n" + "="*70)
        overall_passed = verification_total_passed + negative_total_passed
        overall_total = verification_total + negative_total

        print(f"  OVERALL: {overall_passed}/{overall_total} tests passed")
        print(f"\n  Verification (Correctness): {verification_total_passed}/{verification_total}")
        print(f"  Negative (Robustness):      {negative_total_passed}/{negative_total}")

        # Final verdict
        print("\n" + "="*70)
        if overall_passed == overall_total:
            print("  🎉🎉🎉 ALL TESTS PASSED 🎉🎉🎉")
            print("\n  Status: PRODUCTION-READY")
            print("  • Mathematically correct (verified)")
            print("  • Robust error handling")
            print("  • Graceful failure modes")
        elif verification_total_passed == verification_total:
            print("  ✅ VERIFICATION PASSED, ⚠️  SOME EDGE CASES NEED WORK")
            print("\n  Status: FUNCTIONALLY CORRECT")
            print("  • Core algorithms verified")
            print("  • Some edge cases need improvement")
        else:
            print("  ⚠️  SOME TESTS FAILED")
            print("\n  Status: NEEDS ATTENTION")
            failed = overall_total - overall_passed
            print(f"  • {failed} test(s) need fixing")

        print("="*70 + "\n")

        # Detailed breakdown if any failures
        if overall_passed < overall_total:
            print("\n" + "━"*70)
            print("  FAILED TESTS")
            print("━"*70)

            if mms_passed < mms_total:
                print("\n  MMS Tests:")
                for name, passed in ver['mms']:
                    if not passed:
                        print(f"    ✗ {name}")

            if conv_passed < conv_total:
                print("\n  Convergence Tests:")
                for name, passed in ver['convergence']:
                    if not passed:
                        print(f"    ✗ {name}")

            if cons_passed < cons_total:
                print("\n  Conservation Tests:")
                for name, passed in ver['conservation']:
                    if not passed:
                        print(f"    ✗ {name}")

            if ill_posed_passed < ill_posed_total:
                print("\n  Ill-Posed Problems:")
                for name, passed in neg['ill_posed']:
                    if not passed:
                        print(f"    ✗ {name}")

            if ambiguous_passed < ambiguous_total:
                print("\n  Ambiguous Problems:")
                for name, passed in neg['ambiguous']:
                    if not passed:
                        print(f"    ✗ {name}")

            if invalid_passed < invalid_total:
                print("\n  Invalid Inputs:")
                for name, passed in neg['invalid']:
                    if not passed:
                        print(f"    ✗ {name}")

            print("━"*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Triality Comprehensive Test Suite - Run Everything")
    parser.add_argument('--plots', action='store_true',
                       help='Generate convergence plots')
    parser.add_argument('--save-benchmarks', action='store_true',
                       help='Save new regression benchmarks')
    args = parser.parse_args()

    runner = ComprehensiveTestRunner(
        save_plots=args.plots,
        save_benchmarks=args.save_benchmarks
    )
    runner.run_all()
