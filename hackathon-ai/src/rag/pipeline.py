import re
from typing import Dict, Any, List
from src.llm.client import LLMInterface
from src.rag.retriever import HybridRetriever

class RAGPipeline:
    """
    Orchestrates the RAG flow: Query -> Intent -> Retrieve -> Generate.
    """
    
    def __init__(self, retriever: HybridRetriever, llm_client: LLMInterface):
        self.retriever = retriever
        self.llm_client = llm_client

    def detect_intent(self, query: str) -> str:
        """
        Simple regex-based intent detection.
        Returns: 'EMPLOYEE_INFO' or 'POLICY_SEARCH'
        """
        # Matches EMP followed by digits (e.g., EMP1001)
        if re.search(r"EMP\d+", query, re.IGNORECASE):
            return "EMPLOYEE_INFO"
        return "POLICY_SEARCH"

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Main entry point for processing a user query.
        """
        # 1. Attempt to gather Employee Context
        context_parts = []
        citations = []
        intent = "POLICY_SEARCH" # Default

        match = re.search(r"(EMP\d+)", query, re.IGNORECASE)
        if match:
            intent = "EMPLOYEE_INFO_HYBRID"
            emp_id = match.group(1).upper()
            
            data = self.retriever.get_employee_info(emp_id)
            if "error" not in data:
                emp_context = f"--- User Profile ({emp_id}) ---\n"
                emp_context += f"Name: {data.get('name')}\n"
                emp_context += f"Department: {data.get('dept')}\n"
                emp_context += f"Location: {data.get('location')}\n"
                emp_context += f"Joining Date: {data.get('Joining_Date', 'Unknown')}\n"
                
                # Calculate Tenure
                tenure_info = self.retriever.calculate_tenure(emp_id)
                if "error" not in tenure_info:
                    emp_context += f"Tenure: {tenure_info['tenure_years']} years ({tenure_info['tenure_days']} days)\n"
                    emp_context += f"Sabbatical Eligible: {'Yes' if tenure_info['sabbatical_eligible'] else 'No'}\n"
                
                # Leaves
                emp_context += f"Leaves Taken: {len(data.get('leaves', []))} records\n"
                
                # Attendance bits
                emp_context += f"Attendance Days: {data.get('attendance_count')}\n"
                
                context_parts.append(emp_context)
                citations.append(f"Employee DB: {emp_id}")
            else:
                 print(f"Warning: Could not find data for {emp_id}")

        # 2. Always Search Policies (Contextual Grounding)
        # Even if looking up an employee, we need policies to answer questions like "Am I eligible?"
        results = self.retriever.search_policies(query, k=3)
        if results:
            policy_context = "--- Relevant HR Policies ---\n"
            for i, res in enumerate(results):
                policy_context += f"Source: {res['metadata']['source']} (Page {res['metadata']['page']})\n"
                policy_context += f"Content: {res['text']}\n\n"
                citations.append(f"{res['metadata']['source']} (Page {res['metadata']['page']})")
            context_parts.append(policy_context)

        if not context_parts:
             return {
                "response": "I couldn't find any relevant policy information or employee data for your query.",
                "context": "",
                "intent": intent,
                "citations": []
            }
            
        full_context = "\n".join(context_parts)

        # Generate Response
        prompt = self._construct_prompt(query, full_context)
        response_text = self.llm_client.generate_text(prompt)
        
        # Calculate Confidence Score
        confidence = self._calculate_confidence(query, context_parts, results, match)
        
        return {
            "response": response_text,
            "citations": citations,
            "intent": intent,
            "context": full_context,
            "confidence": confidence
        }
    
    def _calculate_confidence(self, query: str, context_parts: list, policy_results: list, emp_match) -> Dict[str, Any]:
        """Calculate confidence score based on retrieval quality."""
        score = 0.0
        reasons = []
        
        # Factor 1: Employee data found (if EMP ID mentioned)
        if emp_match:
            if len(context_parts) > 0 and "User Profile" in context_parts[0]:
                score += 30
                reasons.append("Employee data found")
            else:
                score += 10
                reasons.append("Employee ID mentioned but data incomplete")
        
        # Factor 2: Policy documents retrieved
        if policy_results:
            score += min(len(policy_results) * 15, 40)
            reasons.append(f"{len(policy_results)} relevant policy documents found")
        
        # Factor 3: Query clarity (simple heuristic)
        if len(query.split()) > 5:
            score += 15
            reasons.append("Detailed query")
        else:
            score += 5
            reasons.append("Brief query")
        
        # Factor 4: Context richness
        if len(context_parts) >= 2:
            score += 15
            reasons.append("Multiple context sources")
        
        # Normalize to 0-100
        score = min(score, 100)
        
        # Determine level
        if score >= 75:
            level = "High"
        elif score >= 50:
            level = "Medium"
        else:
            level = "Low"
        
        return {
            "score": round(score, 1),
            "level": level,
            "reasons": reasons
        }

    def _construct_prompt(self, query: str, context: str) -> str:
        return f"""
You are the Helix Corp HR Intelligence Bot. 
Answer the user's question strictly based on the provided context.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {query}

Answer:
"""
