import sys
import os
import pandas as pd

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import DataLoader
from src.config import settings

def test_ingestion():
    loader = DataLoader()
    data_dir = settings.DATA_DIR
    
    print(f"Testing ingestion from: {data_dir}")
    
    # Test 1: Employees (CSV)
    try:
        csv_path = os.path.join(data_dir, "employee_master.csv")
        employees = loader.load_employees(csv_path)
        print(f"[SUCCESS] Loaded {len(employees)} employees.")
        print(employees.head(2))
    except Exception as e:
        print(f"[FAILURE] CSV Load failed: {e}")

    # Test 2: Leaves (Excel)
    try:
        xlsx_path = os.path.join(data_dir, "leave_intelligence.xlsx")
        leaves = loader.load_leaves(xlsx_path)
        print(f"[SUCCESS] Loaded {len(leaves)} leave records.")
        print(leaves.head(2))
    except Exception as e:
        print(f"[FAILURE] Excel Load failed: {e}")

    # Test 3: Attendance (JSON)
    try:
        json_path = os.path.join(data_dir, "attendance_logs_detailed.json")
        attendance = loader.load_attendance(json_path)
        print(f"[SUCCESS] Loaded {len(attendance)} attendance records.")
        print(attendance.head(2))
    except Exception as e:
        print(f"[FAILURE] JSON Load failed: {e}")

    # Test 4: Policy (PDF)
    try:
        pdf_path = os.path.join(data_dir, "Helix_Pro_Policy_v2.pdf")
        policies = loader.load_policies(pdf_path)
        print(f"[SUCCESS] Extracted {len(policies)} policy chunks.")
        if policies:
            print(f"Sample chunk: {policies[0]['text'][:100]}...")
    except Exception as e:
        print(f"[FAILURE] PDF Load failed: {e}")

if __name__ == "__main__":
    test_ingestion()
