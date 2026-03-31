#!/usr/bin/env python3
import sys

import requests
from packaging.requirements import Requirement
from pathlib import Path
import json
from typing import Dict, List, Optional, Set, Tuple
import configparser
import re
from dataclasses import dataclass


@dataclass
class PackageLicense:
    name: str
    version: Optional[str]
    license_type: Optional[str]
    is_authorized: bool
    reason: str


def main():
    # req_file = "../../requirements.txt" ## LOCAL TESTING
    req_file = "./requirements.txt"

    # Check requirements
    if not checker.check_requirements(Path(req_file)):
        # Get lists of problematic packages
        unverified = [p for p in checker.package_results if not p.license_type]
        invalid = [
            p for p in checker.package_results if p.license_type and not p.is_authorized
        ]

        # Print detailed information about problematic packages
        if unverified:
            print("\n❌ Packages with unknown licenses:")
            for pkg in unverified:
                version_str = f" ({pkg.version})" if pkg.version else ""
                print(f"- {pkg.name}{version_str}")

        if invalid:
            print("\n❌ Packages with unauthorized licenses:")
            for pkg in invalid:
                version_str = f" ({pkg.version})" if pkg.version else ""
                print(f"- {pkg.name}{version_str}: {pkg.license_type}")

        # Only error if there are packages that aren't explicitly authorized
        unhandled_packages = [
            p
            for p in (unverified + invalid)
            if p.name.lower() not in checker.authorized_packages
        ]

        if unhandled_packages:
            print("\n❌ Error: Found packages that need verification:")
            for pkg in unhandled_packages:
                version_str = f" ({pkg.version})" if pkg.version else ""
                license_str = (
                    f" - {pkg.license_type}"
                    if pkg.license_type
                    else " - Unknown license"
                )
                print(f"- {pkg.name}{version_str}{license_str}")
            print(
                "\nAdd these packages to the [Authorized Packages] section in liccheck.ini with a comment about their license verification."
            )
            print("Example:")
            print("package-name: >=1.0.0  # MIT license manually verified")
            sys.exit(1)
    else:
        print("\n✅ All dependencies have acceptable licenses.")

    sys.exit(0)


if __name__ == "__main__":
    main()
