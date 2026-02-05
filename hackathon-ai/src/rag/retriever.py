from typing import List, Dict, Any, Optional
import pandas as pd
from src.data_loader import DataLoader
from src.rag.vector_store import VectorDB
from src.llm.client import LLMInterface

class HybridRetriever:
    """
    Orchestrates retrieval from Structured (Pandas) and Unstructured (VectorDB) sources.
    """
    
    def __init__(self, data_loader: DataLoader, vector_db: VectorDB, llm_client: LLMInterface):
        self.employees_df = None
        self.leaves_df = None
        self.attendance_df = None
        self.vector_db = vector_db
        self.llm_client = llm_client
        
        # Load all structured data into memory
        self._load_structured_data(data_loader)

    def _load_structured_data(self, loader: DataLoader):
        # Hardcoded paths based on project structure
        # In production, pass paths via config
        base_dir = "data" 
        import os
        self.employees_df = loader.load_employees(os.path.join(base_dir, "employee_master.csv"))
        self.leaves_df = loader.load_leaves(os.path.join(base_dir, "leave_intelligence.xlsx"))
        self.attendance_df = loader.load_attendance(os.path.join(base_dir, "attendance_logs_detailed.json"))

    def search_policies(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Semantic search on policies.
        """
        return self.vector_db.search(query, k, self.llm_client)

    def get_employee_info(self, emp_id: str) -> Dict[str, Any]:
        """
        Retrieves comprehensive info for a specific employee.
        """
        if self.employees_df is None:
            return {"error": "Structured data not loaded"}
            
        # Employee Details
        emp = self.employees_df[self.employees_df['emp_id'] == emp_id]
        if emp.empty:
            return {"error": f"Employee {emp_id} not found"}
        
        emp_data = emp.iloc[0].to_dict()
        
        # Leaves
        leaves = self.leaves_df[self.leaves_df['emp_id'] == emp_id]
        emp_data['leaves'] = leaves.to_dict(orient='records')
        
        # Attendance Summary (Example: Count of days present)
        # Assuming attendance_df has 'emp_id'
        attendance = self.attendance_df[self.attendance_df['emp_id'] == emp_id]
        emp_data['attendance_count'] = len(attendance)
        emp_data['recent_attendance'] = attendance.tail(5).to_dict(orient='records')

        return emp_data
    
    def calculate_tenure(self, emp_id: str) -> Dict[str, Any]:
        """Calculate tenure and eligibility for an employee."""
        from datetime import datetime
        
        if self.employees_df is None:
            return {"error": "Employee data not loaded"}
        
        emp = self.employees_df[self.employees_df['emp_id'] == emp_id]
        if emp.empty:
            return {"error": f"Employee {emp_id} not found"}
        
        joining_date = emp.iloc[0].get('Joining_Date')
        if pd.isna(joining_date):
            return {"error": "Joining date not available"}
        
        # Calculate tenure
        if isinstance(joining_date, str):
            joining_date = pd.to_datetime(joining_date)
        
        current_date = datetime(2026, 2, 5)  # As per current context
        tenure_days = (current_date - joining_date).days
        tenure_years = tenure_days / 365.25
        
        # Eligibility checks
        sabbatical_eligible = tenure_years >= 5
        senior_benefits = tenure_years >= 3
        
        return {
            "emp_id": emp_id,
            "joining_date": str(joining_date.date()),
            "tenure_days": tenure_days,
            "tenure_years": round(tenure_years, 2),
            "sabbatical_eligible": sabbatical_eligible,
            "senior_benefits_eligible": senior_benefits
        }


    def query_structured_data(self, query_filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Executes structured filtering on Employee data.
        Example filters: {'dept': 'Engineering', 'location': 'Sydney'}
        """
        df = self.employees_df.copy()
        
        for key, value in query_filters.items():
            if key in df.columns:
                df = df[df[key] == value]
        
        return df.to_dict(orient='records')
