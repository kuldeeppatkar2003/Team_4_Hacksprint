import pandas as pd
import json
import os
from typing import List, Dict, Any
from pypdf import PdfReader
from src.utils import normalize_date, clean_dataframe

class DataLoader:
    """
    Handles loading and processing of various data formats for Helix Corp.
    """

    def load_employees(self, file_path: str) -> pd.DataFrame:
        """
        Loads and cleans employee master data from CSV.
        """
        print(f"Loading employees from {file_path}...")
        df = pd.read_csv(file_path)
        
        # Standardize dates
        if 'Joining Date' in df.columns:
            df['Joining_Date'] = pd.to_datetime(df['Joining Date'], errors='coerce')
        
        # General cleaning
        df = clean_dataframe(df)
        
        # Handle missing fields (example: fill generic missing values)
        df.fillna("Unknown", inplace=True)
        
        return df

    def load_leaves(self, file_path: str) -> pd.DataFrame:
        """
        Loads leave data from Excel.
        """
        print(f"Loading leaves from {file_path}...")
        df = pd.read_excel(file_path)
        df = clean_dataframe(df)
        return df

    def load_attendance(self, file_path: str) -> pd.DataFrame:
        """
        Loads and flattens attendance logs from JSON.
        Structure: Dictionary where keys are EmpIDs, containing a 'records' list.
        """
        print(f"Loading attendance from {file_path}...")
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        all_records = []
        for emp_id, content in data.items():
            records = content.get('records', [])
            for record in records:
                record['emp_id'] = emp_id  # Add metadata
                all_records.append(record)
        
        df = pd.DataFrame(all_records)
        df = clean_dataframe(df)
        
        if 'date' in df.columns:
             df['date'] = pd.to_datetime(df['date'], errors='coerce')

        return df

    def load_policies(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extracts text from Policy PDF and chunks it.
        Returns a list of dicts: {'text': chunk, 'source': source, 'page': page_num}
        """
        print(f"Loading policies from {file_path}...")
        reader = PdfReader(file_path)
        chunks = []
        
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                # Simple chunking by page for now
                # In a real scenario, use a text splitter (recursive char splitter)
                chunks.append({
                    "text": text,
                    "source": os.path.basename(file_path),
                    "page": i + 1
                })
        
        return chunks
